
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import random
from tqdm import tqdm, trange

import numpy as np

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss

from tensorboardX import SummaryWriter

from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from modelingclassbert import BertForMetaClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from run_classifier_dataset_utils import processors, output_modes, convert_examples_to_features, compute_metrics

from collections import defaultdict

import pickle

from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=384,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=2,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir',
                        action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    args.device = device

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()

    num_binary_labels = len(label_list[0])
    num_span_labels = len(label_list[1])
    num_multi_labels = len(label_list[2])

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    model = BertForMetaClassification.from_pretrained(args.bert_model, num_binary_labels=num_binary_labels, num_span_labels=num_span_labels, num_multi_labels=num_multi_labels)

    if args.local_rank == 0:
        torch.distributed.barrier()

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if args.do_train:
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()

        # Prepare data loader
        train_features = processor.get_train_examples(args.data_dir)
        train_examples = train_features

        all_input_ids = torch.tensor([f["input_ids"] for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f["input_mask"] for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f["segment_ids"] for f in train_features], dtype=torch.long)

        input_len = all_input_ids.size(1)

        def index2onehot(features, keyname):
            returnlist = []
            for f in features:
                cur_list = []
                for position_i in range(input_len):
                    if position_i in f[keyname]:
                        cur_list.append(1)
                    else:
                        cur_list.append(0)
                returnlist.append(cur_list)
            return returnlist

        def labelindex2binary(label, newline_mask, input_len, ignore=0):
            zeros = [ignore]* input_len

            for maskid, mask in enumerate(newline_mask):
                zeros[mask] = label[maskid]
            return zeros



        
        newline_mask = torch.tensor(index2onehot(train_features, "[Newline]"), dtype=torch.long)
        #newline_mask = torch.tensor([f["Newline"] for f in train_features], dtype=torch.long)

        list_binary_labels = []
        for lb in label_list[0]:
            list_binary_labels.append(torch.tensor([labelindex2binary(f[lb], f["[Newline]"], input_len=input_len) for f in train_features], dtype=torch.long))

        pos_weights = []
        for lb in label_list[0]:
            pos_cases = 0
            neg_cases = 0
            for example in train_features:
                cur_arr = np.array(example[lb])
                size = cur_arr.size
                pos = cur_arr.sum()
                pos_cases += pos
                neg_cases = neg_cases + size - pos
            if pos_cases > 0:
                ratio = neg_cases / pos_cases
            else:
                ratio = 1.0
            pos_weights.append(ratio)
        pos_weights = torch.tensor(pos_weights).cuda()

        model.pos_weights = pos_weights

        list_span_labels = []
        for lb in label_list[1]:
            list_span_labels.append(torch.tensor([f[lb] for f in train_features], dtype=torch.long))


        list_multi_labels = []
        for lb in label_list[2]:

            list_multi_labels.append(torch.tensor([labelindex2binary(f[lb[0]], f["[Newline]"], input_len=input_len, ignore=-1) for f in train_features], dtype=torch.long))



        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, newline_mask, *list_binary_labels, *list_span_labels, *list_multi_labels)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                 t_total=num_train_optimization_steps)


        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, newline_mask, *label_id_list = batch

                # define a new function to compute loss values for both output_modes
                logits, loss, loss_list = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, newline_mask= newline_mask, labels=label_id_list)

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if args.local_rank in [-1, 0]:
                        tb_writer.add_scalar('lr', optimizer.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', loss.item(), global_step)

    ### Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    ### Example:
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForMetaClassification.from_pretrained(args.output_dir, num_binary_labels=num_binary_labels, num_span_labels=num_span_labels, num_multi_labels=num_multi_labels)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

        # Good practice: save your training arguments together with the trained model
        output_args_file = os.path.join(args.output_dir, 'training_args.bin')
        torch.save(args, output_args_file)
    else:
        model = BertForMetaClassification.from_pretrained(args.bert_model, num_binary_labels=num_binary_labels, num_span_labels=num_span_labels, num_multi_labels=num_multi_labels)

    model.to(device)

    ### Evaluation
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features =  eval_examples
        

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        
        

        
        all_input_ids = torch.tensor([f["input_ids"] for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f["input_mask"] for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f["segment_ids"] for f in eval_features], dtype=torch.long)

        input_len = all_input_ids.size(1)

        def index2onehot(features, keyname):
            returnlist = []
            for f in features:
                cur_list = []
                for position_i in range(input_len):
                    if position_i in f[keyname]:
                        cur_list.append(1)
                    else:
                        cur_list.append(0)
                returnlist.append(cur_list)
            return returnlist

        def labelindex2binary(label, newline_mask, input_len, ignore=0):
            zeros = [ignore]* input_len

            for maskid, mask in enumerate(newline_mask):
                zeros[mask] = label[maskid]
            return zeros



        
        newline_mask = torch.tensor(index2onehot(eval_features, "[Newline]"), dtype=torch.long)
        #newline_mask = torch.tensor([f["Newline"] for f in train_features], dtype=torch.long)

        list_binary_labels = []
        for lb in label_list[0]:
            list_binary_labels.append(torch.tensor([labelindex2binary(f[lb], f["[Newline]"], input_len=input_len) for f in eval_features], dtype=torch.long))

        list_span_labels = []
        for lb in label_list[1]:
            list_span_labels.append(torch.tensor([f[lb] for f in eval_features], dtype=torch.long))


        list_multi_labels = []
        for lb in label_list[2]:
            list_multi_labels.append(torch.tensor([labelindex2binary(f[lb[0]], f["[Newline]"], input_len=input_len, ignore=-1) for f in eval_features], dtype=torch.long))



        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, newline_mask, *list_binary_labels, *list_span_labels, *list_multi_labels)

        # Run prediction for full data
        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)  # Note that this sampler samples randomly
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        bce_loss = 0
        cross_loss = 0
        token_loss = 0
        nb_eval_steps = 0
        preds = []
        out_label_ids = None
        result = defaultdict(float)
        len_bce = len(label_list[0])

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, newline_mask, *label_id_list = batch
            with torch.no_grad():
                logits, loss, loss_list = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, newline_mask= newline_mask, labels=label_id_list)

            
            eval_loss += loss.mean().item()
            bce_loss += loss_list[0].mean().item()
            cross_loss += loss_list[1].mean().item()
            token_loss += loss_list[2].mean().item()

            bce_logits = logits[0]
        
            for l_id, label in enumerate(label_list[0]):
                cur_labels = label_id_list[l_id]

                cur_logits = bce_logits[:, :, l_id]
                
                
                for thresh in [0.5, 0.0, -0.5, -1.0, -2.0, -2.1, -2.15, -2.2, -2.3, -3.0, -0.1 , -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8,-0.9]:
                    threshed_logs = cur_logits > thresh
                    threshed_logs = ((cur_logits == 0).float() * -100).float() + (cur_logits != 0).float() * threshed_logs.float()
                    
                    
                    cur_labels_cpu = ((cur_logits == 0).float() * -100.0).float() + (cur_logits != 0).float() * cur_labels.float()
                    cur_labels_cpu = cur_labels_cpu.detach().cpu().numpy()
                    threshed_logs = threshed_logs.detach().cpu().numpy()
                    ignoring= (cur_logits == 0).sum().detach().cpu().numpy()
                    threshed_logs = threshed_logs[threshed_logs != -100]
                    cur_labels_cpu = cur_labels_cpu[cur_labels_cpu != -100]
                    # acc = ((cur_labels_cpu == threshed_logs).sum() - ignoring) / (cur_labels_cpu.size - ignoring)
                    acc = (cur_labels_cpu == threshed_logs).sum() / cur_labels_cpu.size
                    f1= f1_score(cur_labels_cpu, threshed_logs)
                    #import pdb; pdb.set_trace()
                    #acc = compute_metrics("meta", threshed_logs, cur_labels_cpu, ignoring)
                    # if label == "new_topic" and thresh == -1.0:
                    #import pdb; pdb.set_trace()
                    result[str(thresh)+ "_" + label + "_f1"] += f1 / len_bce
                    result[str(thresh)+ "_" + label + "_acc"] += acc / len_bce





            nb_eval_steps += 1
        #     if len(preds) == 0:
        #         preds.append(logits.detach().cpu().numpy())
        #         out_label_ids = label_ids.detach().cpu().numpy()
        #     else:
        #         preds[0] = np.append(
        #             preds[0], logits.detach().cpu().numpy(), axis=0)
        #         out_label_ids = np.append(
        #             out_label_ids, label_ids.detach().cpu().numpy(), axis=0)

        # eval_loss = eval_loss / nb_eval_steps
        # preds = preds[0]
        # if output_mode == "classification":
        #     preds = np.argmax(preds, axis=1)
        # elif output_mode == "regression":
        #     preds = np.squeeze(preds)
        # result = compute_metrics(task_name, preds, out_label_ids)

        loss = tr_loss/global_step if args.do_train else None
        #result = {}
        result['eval_loss'] = eval_loss
        result["bce_loss"] = bce_loss
        result["cross_loss"] = cross_loss
        result["token_loss"] = token_loss
        result['global_step'] = global_step
        result['loss'] = loss

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        
        
if __name__ == "__main__":
    main()