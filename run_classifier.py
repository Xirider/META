
from __future__ import absolute_import, division, print_function

# import comet_ml in the top of your file
from comet_ml import Experiment
    
from sklearn.metrics import confusion_matrix

import argparse
import logging
import os
import sys
import random
from tqdm import tqdm, trange

import numpy as np

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Sampler)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss

from tensorboardX import SummaryWriter

from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from modelingclassbert import BertForMetaClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from run_classifier_dataset_utils import processors, output_modes, convert_examples_to_features, compute_metrics

from collections import defaultdict

from functools import partial

from tokenlist import stopper



import pickle

from sklearn.metrics import f1_score

from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score


logger = logging.getLogger(__name__)

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="zMVSRiUzF89hdX5u7uWrSW5og",
                        project_name="general", workspace="xirider")


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
                        default="meta",
                        type=str,
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
    parser.add_argument('--random_sampling',
                        action='store_true',
                        help="random sampling instead of balanced sampling")
    parser.add_argument('--active_sampling',
                        action='store_true',
                        help="uses active sampling instead of balanced sampling")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    experiment.log_parameters(vars(args))

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

    full_label_list = label_list[0] + label_list[1] + label_list[2]

    num_binary_labels = len(label_list[0])
    num_span_labels = len(label_list[1])
    num_multi_labels = len(label_list[2])

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    tokenizer = BertTokenizer("savedmodel/vocab.txt", never_split = stopper)


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


    def save_model(model, outputdir, threshs, score):

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(outputdir, WEIGHTS_NAME)
        output_config_file = os.path.join(outputdir, CONFIG_NAME)
        TRESH_NAME = "thresholds.txt"
        output_thresh_file = os.path.join(outputdir, TRESH_NAME)
        print(f"Saving model with score of {score}")
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(outputdir)



        with open(output_thresh_file, "w") as text_file:
            text_file.write(str(score))
            for thresh in threshs:
                text_file.write("\n")
                text_file.write(str(thresh))
        

    def sigmoid(x):
        sigm = 1. / (1. + np.exp(-x))
        return sigm

    class UnderSampler(Sampler):


        def __init__(self, label_mins, label_type_list, label_number_list, order_index_list = None):
            self.label_mins = label_mins
            self.label_type_list = label_type_list
            self.label_number_list = label_number_list
            self.order_index_list = order_index_list

            label_mins = self.label_mins
            label_type_list = self.label_type_list
            label_number_list = self.label_number_list


            index_list = []
            counter_dict = defaultdict(int)

            if not order_index_list:
                randomlist = list(range(len(label_type_list)))
                random.shuffle(randomlist)
            else:
                randomlist = order_index_list

            for i in randomlist:

                current_label = label_type_list[i]
                current_label_number = label_number_list[i]
                current_label_min = label_mins[current_label]

                if current_label_min > counter_dict[str(current_label)+ "_" + str(current_label_number)]:
                     counter_dict[str(current_label)+"_" + str(current_label_number)] += 1
                     index_list.append(i)
            
            random.shuffle(index_list)

            self.index_list_len = len(index_list)

        def __iter__(self):
            

            label_mins = self.label_mins
            label_type_list = self.label_type_list
            label_number_list = self.label_number_list
            order_index_list = self.order_index_list


            index_list = []
            counter_dict = defaultdict(int)

            if not order_index_list:
                randomlist = list(range(len(label_type_list)))
                random.shuffle(randomlist)
            else:
                randomlist = order_index_list

            for i in randomlist:

                current_label = label_type_list[i]
                current_label_number = label_number_list[i]
                current_label_min = label_mins[current_label]

                if current_label_min > counter_dict[str(current_label)+ "_" + str(current_label_number)]:
                     counter_dict[str(current_label)+"_" + str(current_label_number)] += 1
                     index_list.append(i)
            
            random.shuffle(index_list)

            return iter(index_list)

        def __len__(self):
            return self.index_list_len


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

        def labelindex2binary(label, newline_mask, input_len, ignore= -1):
            zeros = [ignore]* input_len

            for maskid, mask in enumerate(newline_mask):
                zeros[mask] = label[maskid]
            return zeros



        
        newline_mask = torch.tensor(index2onehot(eval_features, "[newline]"), dtype=torch.long)
        #newline_mask = torch.tensor([f["Newline"] for f in train_features], dtype=torch.long)

        list_binary_labels = []
        for lb in label_list[0]:
            list_binary_labels.append(torch.tensor([labelindex2binary(f[lb], f["[newline]"], input_len=input_len, ignore=-1) for f in eval_features], dtype=torch.long))

        list_span_labels = []
        for lb in label_list[1]:
            list_span_labels.append(torch.tensor([f[lb] for f in eval_features], dtype=torch.long))


        list_multi_labels = []
        for lb in label_list[2]:
            list_multi_labels.append(torch.tensor([labelindex2binary(f[lb[0]], f["[newline]"], input_len=input_len, ignore=-1) for f in eval_features], dtype=torch.long))



        

        pos_weights = []
        for lb in label_list[0]:
            pos_cases = 0
            neg_cases = 0
            for example in eval_features:
                cur_arr = np.array(example[lb])
                cur_arr = cur_arr[cur_arr != -1]
                size = cur_arr.size
                pos = cur_arr.sum()
                pos_cases += pos
                neg_cases = neg_cases + size - pos
            if pos_cases > 0:
                ratio = neg_cases / pos_cases
            else:
                ratio = 1.0
            pos_weights.append(ratio)
            experiment.log_metric(f"positive test labels for class: {lb}",pos_cases)
            experiment.log_metric(f"negative test labels for class: {lb}",neg_cases)
        pos_weights = torch.tensor(pos_weights)
        #pos_weights = [pos_weights] * len(eval_features)
        pos_weights = pos_weights.expand(all_input_ids.size(0), -1)
            
        # prepare label information for undersampler
        label_mins = [defaultdict(int)] * len(full_label_list)
        label_type_list = ["x"] * len(eval_features)
        label_number_list = ["x"] * len(eval_features)

        for lbid, lb in enumerate(full_label_list):
            if type(lb) == list:
                lb = lb[0]
            for exid, example in enumerate(eval_features):
                cur_arr = example[lb]
                arr_number = sum(cur_arr) + len(cur_arr) 
                if arr_number != 0:
                    label_number_list[exid] = arr_number
                    label_type_list[exid] = lbid
                    label_mins[lbid][arr_number] += 1
        assert ("x" not in label_type_list)
        assert ("x" not in label_number_list)

        for lid, lm in enumerate(label_mins):
            label_mins[lid] = min(lm.values())

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, newline_mask, pos_weights, *list_binary_labels, *list_span_labels, *list_multi_labels)

        # Run prediction for full data 

        if args.local_rank == -1:
            # if args.random_sampling:
            eval_sampler = SequentialSampler(eval_data)
            # else:
            #eval_sampler = UnderSampler( label_mins, label_type_list, label_number_list)
        else:
            eval_sampler = DistributedSampler(eval_data)  # Note that this sampler samples randomly
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        def f1_calculate(precision, recall):

            num =(2 * precision * recall).astype(float) 
            den = (precision + recall).astype(float)
            try:
                f1 = np.divide(num, den, out=np.zeros_like(num), where=den>0.0001)
            except:
                import pdb; pdb.set_trace()
            return f1



        def evaluate(number_of_epochs=0, show_examples=True, best_sum_of_scores=0.0):

                model.eval()
                eval_loss = 0
                bce_loss = 0
                cross_loss = 0
                token_loss = 0
                nb_eval_steps = 0
                preds = []
                out_label_ids = None
                result = 0
                result = defaultdict(float)
                len_bce = len(eval_dataloader)

                evaldict = defaultdict(partial(np.ndarray, 0))

                
                thresholds = np.around(np.arange(-10,10, 0.1), decimals=1).tolist()
                thresholds= list(dict.fromkeys(thresholds))
                bnum = 0
                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    bnum += 1
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, newline_mask, pos_weights,  *label_id_list = batch
                    with torch.no_grad():
                        logits,loss, loss_list = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, newline_mask= newline_mask, labels=label_id_list, pos_weights=pos_weights)

                    
                    eval_loss += loss.mean().item()
                    bce_loss += loss_list[0].mean().item()
                    cross_loss += loss_list[2].mean().item()
                    token_loss += loss_list[1].mean().item()

                    bce_logits = logits[0]

                    token_logits = logits[1]

                    evaldict["binary_mask"] = np.append(evaldict["binary_mask"], newline_mask.detach().cpu().numpy())
                
                    for l_id, label in enumerate(label_list[0]):
                        cur_labels = label_id_list[l_id]
                        bin_label_len = len(cur_labels)
                        cur_logits = bce_logits[:, :, l_id]
                        
                        cur_logits_n = cur_logits.detach().cpu().numpy()
                        cur_labels_n = cur_labels.detach().cpu().numpy()

                        evaldict[label +"logits"] = np.append(evaldict[label +"logits"], cur_logits_n)
                        evaldict[label +"labels"]= np.append(evaldict[label +"labels"], cur_labels_n)

                        if (l_id == 0 or l_id == 1) and bnum < 5:
                            mask = newline_mask[0].detach().cpu().numpy()
                            text = " ".join(tokenizer.convert_ids_to_tokens(input_ids.cpu().numpy().tolist()[0]))
                            print("\n\n1. TEXT:\n")
                            print(text)
                            print("\n\n2. LOGITS: \n")
                            print(sigmoid(cur_logits[0].cpu().numpy())[mask == 1])
                            print("\n\n3. LABELS: \n")
                            print(cur_labels[0].cpu().numpy()[mask == 1])
                            print("\n\n\n")


                        # for thresh in thresholds:
                        #     threshed_logs = cur_logits > thresh

                        #     threshed_logs = ((cur_logits == 0).float() * -100).float() + (cur_logits != 0).float() * threshed_logs.float()
                            
                            
                        #     cur_labels_cpu = ((cur_logits == 0).float() * -100.0).float() + (cur_logits != 0).float() * cur_labels.float()
                        #     cur_labels_cpu = cur_labels_cpu.detach().numpy()
                        #     threshed_logs = threshed_logs.detach().numpy()
                        #     ignoring= (cur_logits == 0).sum().detach().numpy()
                        #     threshed_logs = threshed_logs[threshed_logs != -100]
                        #     cur_labels_cpu = cur_labels_cpu[cur_labels_cpu != -100]
                        #     # acc = ((cur_labels_cpu == threshed_logs).sum() - ignoring) / (cur_labels_cpu.size - ignoring)
                        #     acc = (cur_labels_cpu == threshed_logs).sum() / cur_labels_cpu.size
                        #     f1= f1_score(cur_labels_cpu, threshed_logs)
                        #     #import pdb; pdb.set_trace()
                        #     #acc = compute_metrics("meta", threshed_logs, cur_labels_cpu, ignoring)
                        #     # if label == "new_topic" and thresh == -1.0:
                        #     #import pdb; pdb.set_trace()
                        #     result[str(thresh)+ "_" + label + "_f1"] += f1 / len_bce
                        #     result[str(thresh)+ "_" + label + "_acc"] += acc / len_bce
                    
                    for l_id, label in enumerate(label_list[1]):

                                            cur_labels = label_id_list[l_id +len(label_list[0])]

                                            cur_logits = token_logits[:, :, l_id]
                                            
                                            cur_logits_n = cur_logits.detach().cpu().numpy()
                                            cur_labels_n = cur_labels.detach().cpu().numpy()

                                            evaldict[label +"logits"] = np.append(evaldict[label +"logits"], cur_logits_n)
                                            evaldict[label +"labels"]= np.append(evaldict[label +"labels"], cur_labels_n)

                                            if (l_id == 0) and bnum < 5:
                                                
                                                text = " ".join(tokenizer.convert_ids_to_tokens(input_ids.cpu().numpy().tolist()[0]))
                                                print("\n\n1. TEXT:\n")
                                                print(text)
                                                print("\n\n2. LOGITS: \n")
                                                print(sigmoid(cur_logits[0].cpu().numpy()))
                                                print("\n\n3. LABELS: \n")
                                                print(cur_labels[0].cpu().numpy())
                                                print("\n\n\n")




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
 
                # bestf1 = 0
                # bestf1name = ""
                # for key in sorted(result.keys()):
                #     if "f1" in key and "new_topic" in key:
                #         if result[key] > bestf1:
                #             bestf1 = result[key]
                #             bestf1name = float(key.replace("_f1", "").replace("new_topic", "").replace("_", ""))

                # result = 0
                # result = defaultdict(float)
                # result["zbestf1_"] = bestf1

                # result["zbestf1_threshold"] = bestf1name

                for l_id, label in enumerate(label_list[0]):
                    binary_mask = evaldict["binary_mask"]
                    cur_labels = evaldict[label +"labels"]
                    cur_preds = evaldict[label+"logits"]
                    
                    cur_labels = cur_labels[binary_mask == 1]
                    cur_preds = cur_preds[binary_mask == 1]

                    cur_ignore = cur_labels != -1
                    cur_labels = cur_labels[cur_ignore]
                    cur_preds = cur_preds[cur_ignore]

                    cur_preds = sigmoid(cur_preds)
                    try:
                        if len(cur_labels) == 0:
                            precision = np.array([0.0])
                            recall = np.array([0.0])
                            thresh = np.array([0.0])
                        else:
                            precision, recall, thresh = precision_recall_curve(cur_labels, cur_preds)
                    except:
                        import pdb; pdb.set_trace()

                    all_f1 = f1_calculate(precision, recall)

                    maxindex = np.argmax(all_f1)

                    result[label+"_best_thresh"] = thresh[maxindex]

                    best_tresh = thresh[maxindex]

                    if len(cur_labels) > 0:
                        threshed_val = cur_preds > best_tresh
                        conf = confusion_matrix(cur_labels, threshed_val)
                        print(f"Confusion Matrix for {label}\n")
                        print(conf)

                    result[label+"_best_f1"] = all_f1[maxindex]

                    result[label+"atbf1_best_precision"] = precision[maxindex]
                    result[label+"atbf1_best_recall"] = recall[maxindex]
                    if len(cur_labels) == 0:
                        result[label +"_pr_auc_score"] = 0.0
                    else:
                        result[label +"_pr_auc_score"]  = auc(recall, precision)

                # token metrics
                
                for l_id, label in enumerate(label_list[1]):
                    
                    cur_labels = evaldict[label +"labels"]
                    cur_preds = evaldict[label+"logits"]
                    
                    cur_ignore = cur_labels != -1
                    cur_labels = cur_labels[cur_ignore]
                    cur_preds = cur_preds[cur_ignore]

                    cur_preds = sigmoid(cur_preds)
                    try:
                        if len(cur_labels) == 0:
                            precision = np.array([0.0])
                            recall = np.array([0.0])
                            thresh = np.array([0.0])
                        else:
                            precision, recall, thresh = precision_recall_curve(cur_labels, cur_preds)
                    except:
                        import pdb; pdb.set_trace()

                    all_f1 = f1_calculate(precision, recall)

                    maxindex = np.argmax(all_f1)

                    result[label+"_best_thresh"] = thresh[maxindex]

                    best_tresh = thresh[maxindex]

                    if len(cur_labels) > 0:
                        threshed_val = cur_preds > best_tresh
                        conf = confusion_matrix(cur_labels, threshed_val)
                        print(f"Confusion Matrix for {label}\n")
                        print(conf)

                    result[label+"_best_f1"] = all_f1[maxindex]

                    result[label+"_f1_best_precision"] = precision[maxindex]
                    result[label+"_f1_best_recall"] = recall[maxindex]
                    if len(cur_labels) == 0:
                        result[label +"_pr_auc_score"] = 0.0
                    else:
                        result[label +"_pr_auc_score"]  = auc(recall, precision)






                if global_step == 0:
                    loss = tr_loss/1
                else:
                    loss = tr_loss/global_step if args.do_train else None
                #result = {}
                result['eval_loss'] = eval_loss
                result["bce_loss"] = bce_loss
                result["cross_loss"] = cross_loss
                result["token_loss"] = token_loss
                result['global_step'] = global_step
                result['loss'] = loss

                for key in sorted(result.keys()):
                    experiment.log_metric(key,result[key], number_of_epochs)

                # output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                # with open(output_eval_file, "w") as writer:
                #     logger.info("***** Eval results *****")
                #     for key in sorted(result.keys()):
                #         logger.info("  %s = %s", key, str(result[key]))
                #         writer.write("%s = %s\n" % (key, str(result[key])))



                important_keys = ["self_con","secondary_relevance", "topic_words"]

                sum_of_scores = 0.0
                for ikd, ik in enumerate(important_keys):
                    sum_of_scores += result[ik + "_pr_auc_score"]
                    if ikd == 0:
                        sum_of_scores += result[ik + "_pr_auc_score"]
                if sum_of_scores > best_sum_of_scores:
                    threshs = [ result[ts+"_best_thresh"] for ts in important_keys]

                    save_model(model, args.output_dir, threshs, sum_of_scores/4)
                    best_sum_of_scores = sum_of_scores

                return best_sum_of_scores



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

        def labelindex2binary(label, newline_mask, input_len, ignore=-1):
            zeros = [ignore]* input_len

            for maskid, mask in enumerate(newline_mask):
                zeros[mask] = label[maskid]
            return zeros



        
        newline_mask = torch.tensor(index2onehot(train_features, "[newline]"), dtype=torch.long)
        #newline_mask = torch.tensor([f["Newline"] for f in train_features], dtype=torch.long)

        list_binary_labels = []
        for lb in label_list[0]:
            list_binary_labels.append(torch.tensor([labelindex2binary(f[lb], f["[newline]"], input_len=input_len) for f in train_features], dtype=torch.long))

        pos_weights = []
        for lbid, lb in enumerate(label_list[0]):
            pos_cases = 0
            neg_cases = 0
            for example in train_features:
                cur_arr = np.array(example[lb])
                cur_arr = cur_arr[cur_arr != -1]
                size = cur_arr.size
                pos = cur_arr.sum()
                pos_cases += pos
                neg_cases = neg_cases + size - pos

            if pos_cases > 0:
                ratio = neg_cases / pos_cases
            else:
                ratio = 1.0


            
            pos_weights.append(ratio)
            experiment.log_metric(f"positive training labels for class: {lb}",pos_cases)
            experiment.log_metric(f"negative training labels for class: {lb}",neg_cases)
            
        pos_weights = torch.tensor(pos_weights)
        #pos_weights = [pos_weights] * len(train_features)
        #pos_weights = None
        pos_weights = pos_weights.expand(all_input_ids.size(0), -1)
        

        list_span_labels = []
        for lb in label_list[1]:
            list_span_labels.append(torch.tensor([f[lb] for f in train_features], dtype=torch.long))


        list_multi_labels = []
        for lb in label_list[2]:

            list_multi_labels.append(torch.tensor([labelindex2binary(f[lb[0]], f["[newline]"], input_len=input_len, ignore=-1) for f in train_features], dtype=torch.long))


        # prepare label information for undersampler
        label_mins = [defaultdict(int)] * len(full_label_list)
        label_type_list = ["x"] * len(train_features)
        label_number_list = ["x"] * len(train_features)

        for lbid, lb in enumerate(full_label_list):
            if type(lb) == list:
                lb = lb[0]
            for exid, example in enumerate(train_features):
                cur_arr = example[lb]
                arr_number = sum(cur_arr) + len(cur_arr) 
                if arr_number != 0:
                    label_number_list[exid] = arr_number
                    label_type_list[exid] = lbid
                    label_mins[lbid][arr_number] += 1
        assert ("x" not in label_type_list)
        assert ("x" not in label_number_list)

        for lid, lm in enumerate(label_mins):
            label_mins[lid] = min(lm.values())

            








        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, newline_mask, pos_weights, *list_binary_labels, *list_span_labels, *list_multi_labels)
        if args.local_rank == -1:
            # if args.weighted_sampling:
            #     class_weightings = []
            #     for f in train_features:
            #         for lblist in label_list:
            #             for lb in lblist:
            #                 if type(lb) == list:
            #                     lb = lb[0]


            #                 f["activelist"]
                    


                #train_sampler = RandomWeightedSampler(train_data)
            #else:

            if args.random_sampling:
                train_sampler = RandomSampler(train_data)
            else:

                train_sampler = UnderSampler( label_mins, label_type_list, label_number_list)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


        def sample_active(label_mins, label_type_list, label_number_list ,train_data,model):
            """ Goes through each train example and evaluates them. The indices of the ranking are then used to create a
                new Undersampler and then a new dataloader is returned """
            resultlist = []
            sample_dataloader = train_dataloader = DataLoader(train_data, sampler=SequentialSampler(train_data), batch_size=1)

            for sampleid, batch in enumerate(tqdm(sample_dataloader, desc="Iteration")):
                
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, newline_mask, pos_weights, *label_id_list = batch
                with torch.no_grad():

                    logits, loss, loss_list = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, newline_mask= newline_mask, labels=label_id_list, pos_weights=pos_weights)
                
                loss = loss.detach().cpu().numpy()
                resultlist.append(loss)

            sample_dataloader = 0

            resultlist = np.array(resultlist)

            sorted_resultlist = np.argsort(resultlist).tolist()
            sorted_resultlist.reverse()

            new_sampler = UnderSampler( label_mins, label_type_list, label_number_list, sorted_resultlist)
            return_dataloader = DataLoader(train_data, sampler=new_sampler,  batch_size=args.train_batch_size)
            return return_dataloader

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
        number_of_epochs = -1
        best_sum_of_scores = 0.0
        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]):
            tr_loss = 0
            number_of_epochs += 1
            nb_tr_examples, nb_tr_steps = 0, 0
            if number_of_epochs % 1 == 0:
                best_sum_of_scores =evaluate(number_of_epochs=number_of_epochs ,best_sum_of_scores = best_sum_of_scores)
            if number_of_epochs > 0 and args.active_sampling:
                train_dataloader = sample_active(label_mins, label_type_list, label_number_list ,train_data, model)
            
            model.train()
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, newline_mask, pos_weights, *label_id_list = batch

                # define a new function to compute loss values for both output_modes
                logits, loss, loss_list = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, newline_mask= newline_mask, labels=label_id_list, pos_weights=pos_weights)

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
                        experiment.log_metric("lr",optimizer.get_lr()[0], global_step)
                        experiment.log_metric("train_loss",loss.item(), global_step)
                        

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
        tokenizer = BertTokenizer("savedmodel/vocab.txt", never_split = stopper)

        # Good practice: save your training arguments together with the trained model
        output_args_file = os.path.join(args.output_dir, 'training_args.bin')
        torch.save(args, output_args_file)
    else:
        model = BertForMetaClassification.from_pretrained(args.bert_model, num_binary_labels=num_binary_labels, num_span_labels=num_span_labels, num_multi_labels=num_multi_labels)

    model.to(device)


    

        
        
if __name__ == "__main__":
    main()