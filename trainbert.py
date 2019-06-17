# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict


import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler

from pytorch_pretrained_bert import BertTokenizer, BertAdam, WEIGHTS_NAME, CONFIG_NAME
from modeling import BertNQA


from msdata import get_data_loaders_ms_nqstyle, create_stopper


logger = logging.getLogger(__file__)

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()






def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dc_ms_nqpipe', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="savedmodel", help="Path, url or short name of the model")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--warmup_proportion", type=float, default=0.1, help="warmup ratio")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer ")

    # tokenizer_class = GPT2Tokenizer if "gpt2" in args.model_checkpoint else OpenAIGPTTokenizer
    # tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    # model_class = GPT2DoubleHeadsModel if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel
    # model = model_class.from_pretrained(args.model_checkpoint)
    # tokenizer.set_special_tokens(SPECIAL_TOKENS)
    # tokenizer.max_len = 384 # if we want to change the maxlen, we need to delete the last part of the dataset cache
    # model.set_num_special_tokens(len(SPECIAL_TOKENS))
    # model.to(args.device)
    # optimizer = OpenAIAdam(model.parameters(), lr=args.lr)

    stopper = create_stopper()
    tokenizer = BertTokenizer.from_pretrained(args.model_checkpoint, never_split = stopper)
    tokenizer.max_len = 384 # if we want to change the maxlen, we need to delete the last part of the dataset cache
    model = BertNQA.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    model.train()

    logger.info("Prepare datasets")
    #train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)
    train_loader, train_sampler = get_data_loaders_ms_nqstyle(args, tokenizer, mode="train")
    val_loader, val_sampler = get_data_loaders_ms_nqstyle(args, tokenizer, mode="valid")

    num_train_optimization_steps = int(
            len(train_loader)  / args.gradient_accumulation_steps) * args.n_epochs
    optimizer = BertAdam(model.parameters(),
                                 lr=args.lr,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)


    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16, loss_scale = 128)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)


    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        loss = model(*batch)
        loss = loss * args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            # input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
            input_ids, token_type_ids ,attention_mask, start_positions, end_positions, answer_type = batch
            # if engine.state.iteration % 500 == 0:
            #     logger.info(tokenizer.decode(input_ids[0, 0, :].tolist()))
            # model_outputs = model(input_ids, mc_token_ids, token_type_ids=token_type_ids)
            # lm_logits, mc_logits = model_outputs[0], model_outputs[1]  # So we can also use GPT2 outputs
            # lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            # lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            #(lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)
            start_logits, end_logits, answer_type_logits = model(input_ids, token_type_ids, attention_mask)
            return (start_logits, end_logits, answer_type_logits), (start_positions, end_positions, answer_type)
    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero

    # scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    # trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics 
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll_start": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0][0], x[1][0])),
               "nll_end": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0][1], x[1][1])),
               "accuracy": Accuracy(output_transform=lambda x: (x[0][2], x[1][2]))}
    metrics.update({"average_nll_start": MetricsLambda(average_distributed_scalar, metrics["nll_start"], args),
                    "average_nll_end": MetricsLambda(average_distributed_scalar, metrics["nll_end"], args),
                    "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], args)})
    metrics["average_ppl_start"] = MetricsLambda(math.exp, metrics["average_nll_start"])
    metrics["average_ppl_end"] = MetricsLambda(math.exp, metrics["average_nll_end"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        tb_logger = TensorboardLogger(log_dir=None)
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(tb_logger.writer.log_dir, 'checkpoint', save_interval=2000, n_saved=3)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

        torch.save(args, tb_logger.writer.log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.log_dir, CONFIG_NAME))
        tokenizer.save_vocabulary(tb_logger.writer.log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(tb_logger.writer.log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()

if __name__ == "__main__":
    train()
