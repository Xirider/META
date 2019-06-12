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
import collections
import numpy as np

from nqdata import convert_single_example

from pytorch_pretrained_bert import cached_path


MSMARCO_TRAIN_URL = "https://msmarco.blob.core.windows.net/msmarco/train_v2.1.json.gz"
MSMARCO_DEV_URL = "https://msmarco.blob.core.windows.net/msmarco/dev_v2.1.json.gz"

logger = logging.getLogger(__file__)


# missing tokens
textcounter = 0

def create_stopper():

    stopper = ["[UNK]", "[SEP]", "[Q]", "[CLS]", "[ContextId=-1]", "[NoLongAnswer]"]
    for i in range(50):
        stopper.extend([f"[ContextId={i}]",f"[Paragraph={i}]",f"[Table={i}]",f"[List={i}]" ])
    stopper = tuple(stopper)
    return stopper

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

    dataset_cache = dataset_cache + 'posttokenization_' + "msmarco_" + mode + type(tokenizer).__name__  # Do avoid using GPT cache for GPT-2 and vice-versa
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
                if textcounter < 100:
                    print(textcounter)
                    print(obj)
                return toks
                                # except:
                #     import pdb; pdb.set_trace()
                #return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            if isinstance(obj, int):
                return obj
            return list(tokenize(o) for o in obj)


        # dataset = tokenize(dataset)
        # with open(dataset_cache, 'w') as json_file:
        #     json.dump(dataset, json_file)
        # print("json saved")
        # import pdb; pdb.set_trace()
        # if dataset_cache:
        #     torch.save(dataset, dataset_cache, pickle_protocol=3)
        #     print("dataset saved")
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


def findspanmatch(context, answer, maxlen = 50, overlap = 20, max_misses = 10, min_real_content = 0.7, answer_overlap = 0.7):
    """ Takes in an context and answer, marks all exact matches, seperates in overlaping parts,takes all possible spans, 
    rates them acc to most pos word, cutting if more than a number of nun content words, returning the original span, 
    maybe count each word only once  """

    poslist = []
    for tok in context:
        if tok in answer:
            poslist.append(1)
        else:
            poslist.append(0)

    poslistarray = np.asarray(poslist, dtype=np.int64)

    _DocSpan = collections.namedtuple("DocSpan", ["start", "length", "tokens", "pos", "posarray"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(context):
        length = len(context) - start_offset
        length = min(length, maxlen)
        doc_spans.append(_DocSpan(start=start_offset, length=length, tokens=context[start_offset:start_offset+length],
         pos=poslist[start_offset:start_offset+length],posarray=poslistarray[start_offset:start_offset+length] ))
        
        if start_offset + length == len(context):
            break
        start_offset += min(length, overlap)

    beststart = None
    bestend = None
    bestoverlap = 0
    bestratio = 0


    for (doc_span_index, doc_span) in enumerate(doc_spans):

        
        for start_index, startbin in enumerate(doc_span.pos):

            if startbin:
                for end_index, endbin in enumerate(doc_span.pos):
                    if endbin:
                        if end_index < start_index:
                            #print("skipped, wrong order")
                            continue
                        if (doc_span.posarray[start_index:end_index + 1].sum() + max_misses) < (end_index - start_index):
                            #print("skipped, to few pos")
                            continue

                        # test for real overlap
                        span_tokens = doc_span.tokens[start_index:end_index + 1]
                        copied_answer = answer.copy()
                        realoverlap = 0
                        for ele in span_tokens:
                            if ele in copied_answer:
                                ele_index = copied_answer.index(ele)
                                del copied_answer[ele_index]

                                realoverlap += 1
                        
                        # if start_index == 11 and end_index == 14:
                        #     import pdb; pdb.set_trace()


                        realratio = realoverlap / (end_index - start_index + 1)

                        if realratio < min_real_content:
                            #print("too low ratio")
                            continue
                        if (realoverlap + max_misses) < (end_index - start_index):
                            #print("skipped, to few real pos")
                            continue
                        
                        if len(answer) * answer_overlap > realoverlap:
                            continue


                        if realoverlap >= bestoverlap:
                            if realoverlap == bestoverlap:
                                if realratio < bestratio:
                                    continue
                            bestoverlap = realoverlap
                            bestratio = realoverlap
                            beststart = start_index + doc_span.start
                            bestend = end_index + doc_span.start
                            besttokens = doc_span.tokens



                        #import pdb; pdb.set_trace()
                        # print(doc_span.tokens[start_index:end_index + 1])
                        # print(start_index)
                        # print(end_index)
                        # if end_index == 14:
                        #     import pdb; pdb.set_trace()
        
        
        
        
        
        
        # np.sum(doc_span.pos)

    def check_overlap(besttokens, beststart, bestend, answer):
        span_tokens = besttokens[beststart: bestend+ 1]
        copied_answer = answer.copy()
        realoverlap = 0
        for ele in span_tokens:
            if ele in copied_answer:
                ele_index = copied_answer.index(ele)
                del copied_answer[ele_index]

                realoverlap += 1
        return realoverlap


    if bestoverlap > 0:
        for x in range(bestend-beststart+ 1):
            overlap = check_overlap(besttokens, beststart + 1, bestend, answer)
            if overlap == bestoverlap:
                beststart += 1
            else:
                break




        return beststart, bestend
    else:
        return None, None



def convert_to_full_text(spanstart, spanend, context, contextid, neg_pass_list, passages_obj, answerable, maxlen):
    """ takes in the important context and adds more negative passages to both sides randomly, also cuts them off if they are too long  """

    def create_cp_token(number, token_type):
        if number < 50:
            if token_type == "para":
                return f"[Paragraph={number}]"
            elif token_type == "con":
                return f"[ContextId={number}]"
            else: raise Exception("need para or con as token type")
        else:
            return "[UNK]"




    full_text = []
    left_ids = []
    right_ids = []
    initial_context_len = len(context)
    if spanstart == -1:
        fixatespans = True
        
    else:
        fixatespans = False

    # modify all texts with paragraph tags
    paracounter = 0
    contextfinished = False
    for pasid in neg_pass_list:
        paracounter += 1
        if contextid < pasid and not contextfinished:
            context = [create_cp_token(paracounter -1, "con")] + [create_cp_token(paracounter, "para")] + context
            paracounter += 1
            contextfinished = True
        passages_obj[pasid]["passage_text"] = tokenizer.tokenize(passages_obj[pasid]["passage_text"])
        passages_obj[pasid]["passage_text"] = [create_cp_token(paracounter -1, "con")] + [create_cp_token(paracounter, "para")] + passages_obj[pasid]["passage_text"]
    if not contextfinished:
        paracounter += 1
        context = [create_cp_token(paracounter -1, "con")] + [create_cp_token(paracounter, "para")] + context
            
    if 0 in neg_pass_list:
        passages_obj[pasid]["passage_text"] = ["[ContextId=-1]", "[NoLongAnswer]"] + passages_obj[pasid]["passage_text"]
    elif contextid == 0:
        context =  ["[ContextId=-1]", "[NoLongAnswer]"] + context

    # fulltext = ["[ContextId=-1]", "[NoLongAnswer]", f"[ContextId={paracounter -1}]",f"[Paragraph={paracounter}]"]

    context_len_increase = len(context) - initial_context_len
    # loop through neglist and add it to both sides
    for pasid in neg_pass_list:
        if pasid < contextid:
            left_ids.append(pasid)
        else:
            right_ids.append(pasid)
    
    tokens_added_left = 0
    for left in left_ids:
        text_to_add = passages_obj[left]["passage_text"]
        lentext = len(text_to_add)
        full_text.extend(text_to_add)
        tokens_added_left += lentext

    contextlen = len(context)
    full_text.extend(context)

    tokens_added_right = 0
    for right in right_ids:
        text_to_add = passages_obj[right]["passage_text"]
        lentext = len(text_to_add)
        full_text.extend(text_to_add)
        tokens_added_right += lentext

    maxlen = maxlen - 34
    startrangepoint = max(0, tokens_added_left - (maxlen - contextlen))

    if tokens_added_left == 0:
        startrange = 0
    else:
        startrange = random.randrange(startrangepoint, tokens_added_left)

    spanmovement = tokens_added_left - startrange
    spanstart += spanmovement
    spanend += spanmovement

    endrange = min(startrange + maxlen + 1, len(full_text) + 1)
    immitext = full_text.copy()
    full_text = full_text[startrange:endrange]

    spanstart += context_len_increase
    spanend += context_len_increase

    if fixatespans:
        spanstart = -1
        spanend = -1

    start_position = spanmovement

    if full_text == []:
        import pdb; pdb.set_trace()

    return spanstart, spanend, start_position ,full_text



def get_data_loaders_ms_nqstyle(args, tokenizer, mode = "train", no_answer = False, rebuild=False,):
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

            noanswtoks ="No Answer Present."

            removed_counter = 0
            
            keyslist = [*ms["query"]]

            for i in keyslist:
                istr = str(i)

                passages_obj = ms["passages"][istr]
                poscounter = False
                for pas in passages_obj:
                    if pas["is_selected"] == 1:
                        poscounter = True
                if not ms["answers"][istr] == noanswtoks and poscounter == False:
                    for elem in ms:
                        del ms[elem][istr]
                    removed_counter += 1
                elif len(passages_obj) < 2:
                    for elem in ms:
                        try:
                            del ms[elem][istr]
                        except:
                            import pdb; pdb.set_trace()
                    removed_counter += 1
# '                elif poscounter == False:
#                     for elem in ms:
#                         del ms[elem][istr]
#                     removed_counter += 1'

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

        positive_count = 0
        negative_count = 0

        number_questions = len(ms["query"])
        # half_questions = int((number_questions - (number_questions % 2)) / 2)
        # logger.info(f"Number of question pairs: {half_questions}")

        datadict = defaultdict(list)
        #for i in range(number_questions):
        qcounter = 0
        passcounter = 0
        for i in ms["query"]:
            istr = str(i)

            
            query = ms["query"][istr]
            query = tokenizer.tokenize(query)
            passages_obj = ms["passages"][istr]
            number_passages = len(passages_obj)
            pos_passage_list = []
            for ids, pas in enumerate(passages_obj):
                if pas["is_selected"] == 1:
                    pos_passage_list.append(ids)

            #pos_pass = passages_obj[random.randint(0, len(pos_passage_list))]
            if len(pos_passage_list) > 0:

                answerable = True
                
                pospasid = random.choice(pos_passage_list)

                pos_pass = passages_obj[pospasid]

                assert (pos_pass["is_selected"] == 1)

                neg_pass_list = [x for x in range(number_passages) if x not in pos_passage_list]
                assert (len(neg_pass_list) > 0)
                passage = pos_pass
                
                context = passage["passage_text"]
                
                context = tokenizer.tokenize(context)
                answer1 = ms["answers"][istr][0]
                answer1 = tokenizer.tokenize(answer1)
                spanstart, spanend = findspanmatch(context, answer1)
                if spanstart and positive_count < 1000:
                    print("answer span that was found")
                    print(context[spanstart: spanend+ 1])
                    print("context for this text")
                    print(context)
                contextid = pospasid
                if answer1 in ["Yes", "YES", "yes"]:
                    answer_type = 0

                    spanstart = -1
                    spanend = -1
                    if positive_count < 1000:
                        print("Yes happended")
                        print(answer1)
                        print(spanstart)
                elif answer1 in ["No", "no", "NO"]:
                    answer_type = 1
                    spanstart = -1
                    spanend = -1
                    if positive_count < 1000:
                        print("No happended")
                        print(answer1)
                        print(spanstart)
                else:
                    answer_type = 3
                    if not spanstart:
                        continue
                    if positive_count < 1000:
                        print(f"Here should be not none {spanstart}")
                        print(spanstart)
                        print(answer1)

                positive_count += 1

            else:
                answerable = False
                neg_pass_list = list(range(0, number_passages))
                negpasid = random.choice(neg_pass_list)
                neg_pass_list.remove(negpasid)
                neg_pass = passages_obj[negpasid]

                contextid = negpasid

                passage = neg_pass

                context = passage["passage_text"]
                context = tokenizer.tokenize(context)
                answer1 = None

                answer_type = 2
                spanstart = -1
                spanend = -1
            
                negative_count += 1

            if len(context) + 34 > tokenizer.max_len:
                if positive_count < 1000:
                    print("skipped example because it was too long")
                    print(len(context))
                continue
            # add all the contexts with their type, and move the start and end spans accordingly
            spanstart, spanend, start_position, full_text = convert_to_full_text(spanstart, spanend, context, contextid, neg_pass_list, passages_obj, answerable, maxlen=tokenizer.max_len)
            if full_text ==  []:
                import pdb; pdb.set_trace()
            if positive_count < 1000:
                print("after convert to full text")
                print(spanstart)
                print(spanend)
                print(start_position)
                print(full_text)


            # if ((len(context1) + len(answer1) + len(query) + 5 - 10) < tokenizer.max_len) and ((len(context2) + len(answer1) + len(query) + 5 - 10) < tokenizer.max_len) :
            
            single_example = convert_single_example(full_text, query, tokenizer, None,  already_tokenized = True, answer_start = spanstart, answer_end=spanend, answer_type = answer_type, mode = "train")
            
            single_example = single_example[0]

            if positive_count < 1000:
                print("After converting to single example")
                print(tokenizer.convert_ids_to_tokens(single_example.input_ids))
                print("here is the answer again:")
                print(tokenizer.convert_ids_to_tokens(single_example.input_ids[single_example.answer_start:single_example.answer_end + 1]))
                print(single_example.answer_start)
                print(single_example.answer_end)
                print(single_example.answer_type)
                print(single_example.segment_ids)
                print(single_example.input_mask)
                print("single example finished")
                # input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels = build_input_from_segments_ms(query, context1, 
                #                                                                     context2, answer1, tokenizer, with_eos=True)

            datadict["input_ids"].append(single_example.input_ids)
            datadict["start_label"].append(single_example.answer_start)
            datadict["end_label"].append(single_example.answer_end)
            datadict["answer_type_label"].append(single_example.answer_type)
            datadict["token_type_ids"].append(single_example.segment_ids)
            datadict["input_mask"].append(single_example.input_mask)
            


            qcounter += 1
            if qcounter % 10000 == 0:
                print(f"Input lists building step: {qcounter}")
            else:
                passcounter += 1

        print(f"positive count: {positive_count}")
        print(f"negative count: {negative_count}")

        print(f"Context too long were deleted, number: {passcounter}")
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
        tensor2 = torch.tensor(datadict["token_type_ids"])
        del datadict["token_type_ids"]
        print(f"model input tensor finished")
        tensor3 = torch.tensor(datadict["input_mask"])
        del datadict["input_mask"]
        print(f"model input tensor finished")
        tensor4 = torch.tensor(datadict["start_label"])
        del datadict["start_label"]
        print(f"model input tensor finished")
        tensor5 = torch.tensor(datadict["end_label"])
        del datadict["end_label"]
        print(f"model input tensor finished")
        tensor6 = torch.tensor(datadict["answer_type_label"])
        del datadict["answer_type_label"]
        print(f"model input tensor finished")

        datadict = 0

        tdataset = TensorDataset(tensor1, tensor2, tensor3, tensor4, tensor5, tensor6)

        if dataset_cache_final:
            torch.save(tdataset, dataset_cache_final, pickle_protocol=4)
            print("final paragraph dataset saved")

        # tdataset = TensorDataset(*tensor_dataset)
    # tensor_dataset = None

    sampler = torch.utils.data.distributed.DistributedSampler(tdataset) if args.distributed else None
    loader = DataLoader(tdataset, sampler=sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    tdataset = 0
    #logger.info("Msmarco dataset (Batch, Candidates, Seq length): {}".format(tdataset.tensors[0].shape))
    print("dataloader finished")

    return loader, sampler










def pad_dataset_ms(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    # need to change: index 0 and 1 for each example and pad it
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset




class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

from pytorch_pretrained_bert import (OpenAIAdam, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                     GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME, BertTokenizer)
from argparse import ArgumentParser

if __name__ == "__main__":

    print("testing dataloaders")
    parser = ArgumentParser()
    args = parser.parse_args()

    args.dataset_path = None
    args.dataset_cache="./dc_ms_nqpipe"
    args.distributed = False
    args.train_batch_size = 4
    stopper = create_stopper()
    tokenizer =  BertTokenizer.from_pretrained("savedmodel", never_split = stopper)
    # tokenizer.set_special_tokens(SPECIAL_TOKENS)
    tokenizer.max_len = 384

    #"./train_v2.1.json.gz"
    print("getting dataset")
    # h = get_dataset_ms(tokenizer = tokenizer, dataset_path = None, dataset_cache="./dataset_cache", mode = "train")

    # print(len(h["query"]))

    #train_loader, train_sampler = get_data_loaders_ms(args, tokenizer, mode = "train")

    train_loader, train_sampler = get_data_loaders_ms_nqstyle(args, tokenizer, mode = "train")
    print("train set finished")
    train_loader, train_sampler = get_data_loaders_ms_nqstyle(args, tokenizer, mode = "valid")



    # tokenizer =  OpenAIGPTTokenizer.from_pretrained("openai-gpt")
    
    #context =  ['the','theory', 'behind', 'the', '85', '##th', 'percent', '##ile', 'rules', 'is', ',', 'that', 'as', 'a', 'policy', ',', 'most','citizens', 'should', 'be', 'deemed', 'reasonable',  'a', 'policy','and', 'pr', '##ude', '##nt', ',', 'and', 'limits', 'must', 'be', 'practical', 'to', 'enforce', '.', 'however', ',', 'there', 'are', 'some', 'circumstances', 'where', 'motor', '##ists', 'do', 'not', 'tend', 'to', 'process', 'all', 'the', 'risks', 'involved', ',', 'and', 'as', 'a', 'mass', 'choose', 'a', 'poor', '85', '##th', 'percent', '##ile', 'speed', '.', 'this', 'rule', 'in', 'substance', 'is', 'a', 'process', 'for', 'voting', 'the', 'speed', 'limit', 'by', 'driving', ';', 'and', 'in', 'contrast', 'to', 'del', '##ega', '##ting', 'the', 'speed', 'limit', 'to', 'an', 'engineering', 'expert', '.', '[ContextId=28]']
    #answer = ['that', 'policy', 'as']  


    # context = f"Shaking is a symptom in which a person has tremors (shakiness or small back and forth movements) in part or all of his body. Shaking can be due to cold body temperatures, rising fever (such as with infections), neurological problems, medicine effects, drug abuse, etc. ...Read more"
    # answer = "Shaking can be due to cold body temperatures, rising fever (such as with infections), neurological problems, medicine effects, drug abuse, etc."

    # context = tokenizer.tokenize(context)
    # answer = tokenizer.tokenize(answer)
    # print(context)
    # print(answer)
    # a = findspanmatch(context, answer)





















