# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
import os
import tarfile
import tempfile
from collections import defaultdict
from itertools import chain
import gzip
import random

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel

from pytorch_pretrained_bert import cached_path

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz"

MSMARCO_TRAIN_URL = "https://msmarco.blob.core.windows.net/msmarco/train_v2.1.json.gz"
MSMARCO_DEV_URL = "https://msmarco.blob.core.windows.net/msmarco/dev_v2.1.json.gz"

logger = logging.getLogger(__file__)


PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
SPECIAL_TOKENS = ["<paragraph>", "<question>", "<answer>", "<eos>", "<clas>", "<emb_para>",  "<emb_question>",  "<emb_answer>", "<pad>"]
# missing tokens
textcounter = 0

def download_pretrained_model():
    """ Download and extract finetuned model from S3 """
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
    tempdir = tempfile.mkdtemp()

    logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        archive.extractall(tempdir)
    return tempdir



def get_dataset_ms(tokenizer, dataset_path, dataset_cache=None, mode = "train"):
    """ get ms marco """
    if mode == "train":
        dataset_path = dataset_path or MSMARCO_TRAIN_URL
    elif mode == "valid":
        dataset_path = dataset_path or MSMARCO_DEV_URL

    dataset_cache = dataset_cache + '_' + "msmarco_" + mode + type(tokenizer).__name__  # Do avoid using GPT cache for GPT-2 and vice-versa
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
        print("dataset loaded")
    else:
        logger.info("Download dataset from %s", dataset_path)
        ms_marco_file = cached_path(dataset_path)

        with gzip.open(ms_marco_file, "rt", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        
        logger.info("Tokenize and encode the dataset")


        def tokenize(obj):
            global textcounter
            if isinstance(obj, str):
                toks = tokenizer.tokenize(obj)
                if len(toks) > tokenizer.max_len:
                    toks = toks[- tokenizer.max_len:].copy()
                textcounter += 1
                if textcounter % 10000 == 0:
                    print(textcounter)
                    print(obj)
                return tokenizer.convert_tokens_to_ids(toks)
                # except:
                #     import pdb; pdb.set_trace()
                #return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            if isinstance(obj, int):
                return obj
            return list(tokenize(o) for o in obj)


        dataset = tokenize(dataset)
        if dataset_cache:
            torch.save(dataset, dataset_cache, pickle_protocol=4)
            print("dataset saved")
    return dataset



def pad_data(data, maxlen, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    # need to change: index 0 and 1 for each example and pad it
    out = []
    for x in data:
        x = x + [padding] * (maxlen - len(x))
        out.append(x)
    assert(len(out) == 2)
    return out


def build_input_from_segments_ms(query, context1, context2, answer1, tokenizer, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """

    #SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]

    

    para, ques, answ, eos, clas, emb_para, emb_question, emb_answer, pad = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    lenquery = len(query)
    lencontext1 = len(context1)
    lencontext2 = len(context2)
    lenanswer1 = len(answer1)

    # shorten too long paragraphs

    if ((lenquery + lencontext1 + lenanswer1 + 5) > tokenizer.max_len):
        reduced = tokenizer.max_len - (lenquery + lenanswer1 + 5)
        context1 = context1[:reduced].copy()
        lencontext1 = reduced
        assert (lencontext1 == len(context1))
        
    if ((lenquery + lencontext2 + lenanswer1 + 5) > tokenizer.max_len):
        reduced = tokenizer.max_len - (lenquery + lenanswer1 + 5)
        context2 = context2[:reduced].copy()
        lencontext2 = reduced
        assert (lencontext2 == len(context2))
    



    position_cls_pos = lencontext1 + lenquery + 3 - 1
    position_cls_neg = lencontext2 + lenquery + 3 - 1

    pos_para = [[para] + context1 + [ques] + query + [answ] + [clas] + answer1 + [eos] ]
    neg_para = [[para] + context2 + [ques] + query + [answ] + [clas] + answer1 + [eos] ]


    input_ids = [list(chain(*pos_para)) , list(chain(*neg_para)) ]

    token_type_ids_pos = [(1 + lencontext1)* [emb_para] + (3 + lenquery) * [emb_question] + (1 + lenanswer1) * [emb_answer] ]
    token_type_ids_neg = [(1 + lencontext2)* [emb_para] + (3 + lenquery) * [emb_question] + (1 + lenanswer1) * [emb_answer] ]

    token_type_ids = [ list(chain(*token_type_ids_pos)) , list(chain(*token_type_ids_neg)) ]

    mc_token_ids = [position_cls_pos, position_cls_neg]

    lm_labels_pos =  (lenquery + lencontext1 + 3) * [-1]  +  [clas] + answer1 + [eos]
    lm_labels_neg = (lenquery + lencontext2 + lenanswer1 +  5) * [-1]

    lm_labels = [lm_labels_pos, lm_labels_neg]

    assert (len(lm_labels_pos) == len(input_ids[0]))
    assert (len(lm_labels_neg) == len(input_ids[1]))
    assert (input_ids[0][position_cls_pos] == clas)
    assert (input_ids[1][position_cls_neg] == clas)

    mc_label = [0]
    maxlen = tokenizer.max_len

    input_ids = pad_data(input_ids, maxlen=maxlen, padding = pad )
    token_type_ids = pad_data(token_type_ids, maxlen=maxlen, padding = pad )
    lm_labels = pad_data(lm_labels, maxlen=maxlen, padding = -1 )
    assert (len(input_ids[1]) == maxlen)


    # instance = {}
    # sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    # sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]

    # instance["input_ids"] = list(chain(*sequence))
    # instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    # instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    # instance["lm_labels"] = [-1] * len(instance["input_ids"])
    # if lm_labels:
    #     instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

    return input_ids, token_type_ids, mc_token_ids, lm_labels, mc_label

def get_data_loaders_ms(args, tokenizer, mode = "train", no_answer = False, rebuild=False):
    """ Prepare the dataset for training and evaluation """


    dataset_cache = args.dataset_cache

    dataset_cache = dataset_cache + '_' + "msmarco_" + mode + type(tokenizer).__name__  # Do avoid using GPT cache for GPT-2 and vice-versa
    dataset_cache_e = dataset_cache + "_after_rem_paras"
    dataset_cache_final = dataset_cache + "_final"

    if dataset_cache_final and os.path.isfile(dataset_cache_final) and rebuild == False:
        logger.info("Load few paragraph dataset from cache at %s", dataset_cache_final)
        tdataset = torch.load(dataset_cache_final)
        print("few paragraph dataset loaded")

    else:

        if dataset_cache_e and os.path.isfile(dataset_cache_e) and rebuild == False:
            logger.info("Load few paragraph dataset from cache at %s", dataset_cache_e)
            ms = torch.load(dataset_cache_e)
            print("few paragraph dataset loaded")

        else:

            ms = get_dataset_ms(tokenizer, args.dataset_path, args.dataset_cache, mode = mode)

            # remove no-answer questions
            nq = len(ms["query"])

            noanswtoks = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("No Answer present."))

            removed_counter = 0
            
            keyslist = [*ms["query"]]

            for i in keyslist:
                istr = str(i)
                passages_obj = ms["passages"][istr]
                poscounter = False
                for pas in passages_obj:
                    if pas["is_selected"] == 1:
                        poscounter = True
                if ms["answers"][istr] == [noanswtoks]:
                    for elem in ms:
                        del ms[elem][istr]
                    removed_counter += 1
                elif len(passages_obj) < 2:
                    for elem in ms:
                        del ms[elem][istr]
                    removed_counter += 1
                elif poscounter == False:
                    for elem in ms:
                        del ms[elem][istr]
                    removed_counter += 1

                if int(i) % 10000 == 0:
                    print(f"Removing paragraphs step: {i}")

            logger.info(f"Previous dataset size: {nq}")
            logger.info(f"Datapoints removed: {removed_counter}")

            if dataset_cache_e:
                torch.save(ms, dataset_cache_e, pickle_protocol=4)
                print("removed paragraph dataset saved")

        nq = len(ms["query"])
        logger.info(f"New dataset size after removing No-Answers: {nq}")

        logger.info("Build inputs and labels")

        number_questions = len(ms["query"])
        # half_questions = int((number_questions - (number_questions % 2)) / 2)
        # logger.info(f"Number of question pairs: {half_questions}")

        datadict = defaultdict(list)
        #for i in range(number_questions):
        qcounter = 0
        for i in ms["query"]:
            istr = str(i)

            
            query = ms["query"][istr]
            passages_obj = ms["passages"][istr]
            number_passages = len(passages_obj)
            pos_passage_list = []
            for ids, pas in enumerate(passages_obj):
                if pas["is_selected"] == 1:
                    pos_passage_list.append(ids)

            #pos_pass = passages_obj[random.randint(0, len(pos_passage_list))]
            if len(pos_passage_list) > 0:


                pos_pass = passages_obj[random.choice(pos_passage_list)]

                assert (pos_pass["is_selected"] == 1)

                neg_pass_list = [x for x in range(number_passages) if x not in pos_passage_list]
                assert (len(neg_pass_list) > 0)

                neg_pass = passages_obj[random.choice(neg_pass_list)]

                
                context1 = pos_pass["passage_text"]
                context2 = neg_pass["passage_text"]

                answer1 = ms["answers"][istr][0]

                input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels = build_input_from_segments_ms(query, context1, 
                                                                                context2, answer1, tokenizer, with_eos=True)

                datadict["input_ids"].append(input_ids)
                datadict["mc_token_ids"].append(mc_token_ids)
                datadict["lm_labels"].append(lm_labels)
                datadict["mc_labels"].append(mc_labels)
                datadict["token_type_ids"].append(token_type_ids)
                
                qcounter += 1
                if qcounter % 10000 == 0:
                    print(f"Input lists building step: {qcounter}")

            else:
                print("Empty pos list skipped")

        ms = 0
        # tensor_dataset = []
        print("creating tensor dataset")
        # for input_type in MODEL_INPUTS:

        #     tensor = torch.tensor(datadict[input_type])
            
        #     tensor_dataset.append(tensor)

        #     tensor = None
        #     print(f"model input tensor finished: {input_type}")


        tensor1 = torch.tensor(datadict["input_ids"])
        del datadict["input_ids"]
        print(f"model input tensor finished")
        tensor2 = torch.tensor(datadict["mc_token_ids"])
        del datadict["mc_token_ids"]
        print(f"model input tensor finished")
        tensor3 = torch.tensor(datadict["lm_labels"])
        del datadict["lm_labels"]
        print(f"model input tensor finished")
        tensor4 = torch.tensor(datadict["mc_labels"])
        del datadict["mc_labels"]
        print(f"model input tensor finished")
        tensor5 = torch.tensor(datadict["token_type_ids"])
        del datadict["token_type_ids"]
        print(f"model input tensor finished")

        datadict = 0

        

        tdataset = TensorDataset(tensor1, tensor2, tensor3, tensor4, tensor5)

        if dataset_cache_final:
            torch.save(tdataset, dataset_cache_final, pickle_protocol=4)
            print("final paragraph dataset saved")

        # tdataset = TensorDataset(*tensor_dataset)
    # tensor_dataset = None
    sampler = torch.utils.data.distributed.DistributedSampler(tdataset) if args.distributed else None
    loader = DataLoader(tdataset, sampler=sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))

    #logger.info("Msmarco dataset (Batch, Candidates, Seq length): {}".format(tdataset.tensors[0].shape))
    print("dataloader finished")

    return loader, sampler




















def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for dataset_name, dataset in personachat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(args.num_candidates, num_candidates)
        for dialog in dataset:
            persona = dialog["personality"].copy()
            for _ in range(args.personality_permutations):
                for utterance in dialog["utterances"]:
                    history = utterance["history"][-(2*args.max_history+1):]
                    for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                        lm_labels = bool(j == num_candidates-1)
                        instance, _ = build_input_from_segments(persona, history, candidate, tokenizer, lm_labels)
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)
                    datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                    datasets[dataset_name]["n_candidates"] = num_candidates
                persona = [persona[-1]] + persona[:-1]  # permuted personalities

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler









def get_dataset(tokenizer, dataset_path, dataset_cache=None):
    """ Get PERSONACHAT from S3 """
    dataset_path = dataset_path or PERSONACHAT_URL
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # Do avoid using GPT cache for GPT-2 and vice-versa
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(dataset)
        if dataset_cache:
            torch.save(dataset, dataset_cache)
    return dataset

def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset

def pad_dataset_ms(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    # need to change: index 0 and 1 for each example and pad it
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset



def get_dataset_personalities(tokenizer, dataset_path, dataset_cache=None):
    """ Get personalities from PERSONACHAT """
    dataset_path = dataset_path or PERSONACHAT_URL
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # Do avoid using GPT cache for GPT-2 and vice-versa
    if os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        personachat = torch.load(dataset_cache)
    else:
        logger.info("Download PERSONACHAT dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            personachat = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        personachat = tokenize(personachat)
        torch.save(personachat, dataset_cache)

    logger.info("Filter personalities")
    personalities = []
    for dataset in personachat.values():
        for dialog in dataset:
            personalities.append(dialog["personality"])

    logger.info("Gathered {} personalities".format(len(personalities)))
    return personalities

def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])

    instance = {}
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]

    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-1] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
    return instance, sequence



class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

from pytorch_pretrained_bert import (OpenAIAdam, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                     GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)
from argparse import ArgumentParser

if __name__ == "__main__":

    parser = ArgumentParser()
    args = parser.parse_args()

    args.dataset_path = None
    args.dataset_cache="./dataset_cache"
    args.distributed = False
    args.train_batch_size = 4

    tokenizer =  OpenAIGPTTokenizer.from_pretrained("openai-gpt")
    tokenizer.set_special_tokens(SPECIAL_TOKENS)


    #"./train_v2.1.json.gz"
    print("getting dataset")
    # h = get_dataset_ms(tokenizer = tokenizer, dataset_path = None, dataset_cache="./dataset_cache", mode = "train")

    # print(len(h["query"]))

    #train_loader, train_sampler = get_data_loaders_ms(args, tokenizer, mode = "train")

    train_loader, train_sampler = get_data_loaders_ms(args, tokenizer, mode = "valid")






















