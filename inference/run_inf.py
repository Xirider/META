import sys
sys.path.append(r"C:\Users\peter\Desktop\dl\qa")
sys.path.append(r"C:\Users\sophi\Desktop\peter\qa")




import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import time
import torch
import torch.nn.functional as F
from collections import defaultdict
import pickle
import copy
import numpy as np
import cProfile

from scrape import Searcher
#from pytorch_pretrained_bert import OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer
from utils import SPECIAL_TOKENS, build_input_from_segments_ms
from nqdata import build_input_batch
from modelingclassbert import BertForMetaClassification
from pytorch_pretrained_bert import BertTokenizer

from labels import binary_labels, span_labels, multi_labels


stopper = ["[Newline]" , "[UNK]" , "[SEP]" , "[Q]" , "[CLS]" , "[WebLinkStart]" , "[LocalLinkStart]" , "[RelativeLinkStart]" ,
    "[WebLinkEnd]" , "[LocalLinkEnd]" , "[RelativeLinkEnd]" , "[VideoStart]" , "[VideoEnd]" , "[TitleStart]" , 
    "[NavStart]" , "[AsideStart]" , "[FooterStart]" , "[IframeStart]" , "[IframeEnd]" , "[NavEnd]" , "[AsideEnd]" , 
    "[FooterEnd]" , "[CodeStart]" , "[H1Start]" , "[H2Start]" , "[H3Start]" , "[H4Start]" , "[H5Start]" , "[H6Start]" ,
    "[CodeEnd]" , "[UnorderedList=1]" , "[UnorderedList=2]" , "[UnorderedList=3]" , "[UnorderedList=4]" , "[OrderedList]"
    , "[UnorderedListEnd=1]" , "[UnorderedListEnd=2]" , "[UnorderedListEnd=3]" , "[UnorderedListEnd=4]" , 
    "[OrderedListEnd]" , "[TableStart]" , "[RowStart]" , "[CellStart]" , "[TableEnd]" , "[RowEnd]" , "[CellEnd]" ,
    "[LineBreak]" , "[Paragraph]" , "[StartImage]" , "[EndImage]" , "[Segment=00]" , "[Segment=01]" , "[Segment=02]" ,
        "[Segment=03]" , "[Segment=04]" , "[Segment=05]" , "[Segment=06]" , "[Segment=07]" , "[Segment=08]" ,
        "[Segment=09]" , "[Segment=10]" , "[Segment=11]" , "[Segment=12]" , "[Segment=13]" , "[Segment=14]" ,
        "[Segment=15]" , "[Segment=16]" , "[Segment=17]" , "[Segment=18]" , "[Segment=19]" , "[Segment=20]" , 
        "[Segment=21]" , "[Segment=22]" , "[Segment=23]" , "[Segment=24]" , "[Segment=25]" , "[Segment=26]" , 
        "[Segment=27]" , "[Segment=28]" , "[Segment=29]" , "[Segment=30]" , "[Segment=XX]", "\n"]


segment_tokens = ["[Segment=00]" , "[Segment=01]" , "[Segment=02]" ,
        "[Segment=03]" , "[Segment=04]" , "[Segment=05]" , "[Segment=06]" , "[Segment=07]" , "[Segment=08]" ,
        "[Segment=09]" , "[Segment=10]" , "[Segment=11]" , "[Segment=12]" , "[Segment=13]" , "[Segment=14]" ,
        "[Segment=15]" , "[Segment=16]" , "[Segment=17]" , "[Segment=18]" , "[Segment=19]" , "[Segment=20]" , 
        "[Segment=21]" , "[Segment=22]" , "[Segment=23]" , "[Segment=24]" , "[Segment=25]" , "[Segment=26]" , 
        "[Segment=27]" , "[Segment=28]" , "[Segment=29]" , "[Segment=30]" , "[Segment=XX]"]

headline_tokens = ["[H1Start]" , "[H2Start]" , "[H3Start]" , "[H4Start]" , "[H5Start]" , "[H6Start]"]

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

# 

def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func

def sample_sequence(query, para, tokenizer, model, args, current_output=None, threshold=0.5):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)




    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        input_ids, token_type_ids, mc_token_ids, _, _ = build_input_from_segments_ms(query = query, context1 = para, context2= [] ,
        answer1 = current_output, tokenizer=tokenizer, with_eos=False, inference = True)

        input_ids = torch.tensor(input_ids, device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(token_type_ids, device=args.device).unsqueeze(0)
        mc_token_ids = torch.tensor(mc_token_ids, device=args.device).unsqueeze(0)

        lm_logits, mc_logits = model(input_ids = input_ids, mc_token_ids = mc_token_ids, token_type_ids=token_type_ids)

        logits = lm_logits[0, 0,-1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        if i == 0:
            mc = torch.sigmoid(mc_logits[0,0])

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())
        
        if mc < threshold:
            break

    return current_output, mc


def get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(
      enumerate(logits[1:], 1), key=lambda x: x[1], reverse=True)
  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes



def compute_best_predictions(prediction_list, stopper, topk = 5,threshold = 0, span_threshold = 0.22208, con_threshold=0.89185):
    funcstart = time.time()
    """ takes in list of predictions, creates list of prediction spans, returns the best span for the top spans """
    #articles = defaultdict(list)
    score_list = []
    for batch in prediction_list:
        loopstart = time.time()
        print("new batch")
        [binary_logits, span_logits, batch_article] = batch
        batch_size = len(batch_article)
        
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # start.record()
    
    

        binary_logits = binary_logits.cpu().numpy()
        span_logits = span_logits.cpu().numpy()
        

        # end.record()

        # torch.cuda.synchronize()

        # elapsed = start.elapsed_time(end)
        justcpu = time.time() - loopstart
        print(f"single batch best answer JUST to cpu time {justcpu}")
        # print(f"elapsed time acc to cuda sync {elapsed}")
        # start_logits = start_logits.tolist()
        # end_logits = end_logits.tolist()
        # answer_type_logits = answer_type_logits.tolist()
        # start_logits = start_logits.data.cpu().numpy()
        # end_logits = end_logits.data.cpu().numpy()
        # answer_type_logits = answer_type_logits.data.cpu().numpy()
        print("batch at cpu")
        loopfinmiddle = time.time() - loopstart
        print(f"single batch best answer to cpu time {loopfinmiddle}")
        for b in range(batch_size):
            example = batch_article[b]
            example_binary_logits = binary_logits[b]
            example_span_logits = span_logits[b]
            newlinelist, spanslist = score_logits(example,example_binary_logits, example_span_logits, n_best_size=35, span_threshold=span_threshold)
            
            score_list.append((newlinelist, spanslist))
        loopfin = time.time() - loopstart
        print(f"single batch best answer getting time {loopfin}")
    print("finished putting examples into lists")
    ranked_list = do_ranking(score_list, score_threshold=threshold, con_threshold=con_threshold)
    finaltime = time.time() - funcstart
    print(f"Processing finished of all batches before cutting them out {finaltime}")
    # for ex in score_list:
    #     print(ex.score)
    #     print(ex.doc_start)
    #     print(ex.doc_end)
    #     #print(ex.short_text)
    # import pdb; pdb.set_trace()
    lenlist = len(score_list)
    print("number of candidates")
    print(lenlist)




    return ranked_list

def check_index(target_list, index):
    if 0 <= index < len(target_list):
        return True, target_list[index]
    else:
        return False, 0


def score_logits(example, example_binary_logits, example_span_logits, n_best_size=35, span_threshold=0.25):
    
    # logits, tokens -> scores per newline, newline ranges
    # for spans: list of spans, with modulated scores, tokid, newline token number

    # binary
    tokens = example.tokens
    newlinestartlist = []
    newlinelist = []
    active_newline = []
    threshold = span_threshold
    
    bin_num = example_binary_logits.shape[1]
    cur_line = []
    toklen = len(tokens)
    for tokid, token in enumerate(tokens):
        if token == "[Newline]":
            cur_line = [tokid]
            cur_bin = example_binary_logits[tokid]
            cur_dict = {"tokid": tokid}
            for i in range(bin_num):
                cur_dict[i] = cur_bin[i].item()
            newlinestartlist.append(cur_dict)
        



        if (toklen - 1) == tokid:
            cur_line.append(tokid + 1)
            cur_line_dict = { "ranges" :cur_line , "tokens" : tokens[cur_line[0]:cur_line[1]], "score_dict": cur_dict}
            newlinelist.append(cur_line_dict)


        elif tokens[tokid + 1] == "[Newline]":
            if len(cur_line) > 0:
                cur_line.append(tokid + 1)
                cur_line_dict = { "ranges" :cur_line , "tokens" : tokens[cur_line[0]:cur_line[1]],"score_dict": cur_dict}
                newlinelist.append(cur_line_dict)

    
    # spans
    span_num = example_span_logits.shape[1]
    span_type_list = []
    for span_type in range(span_num):

        span_logits = example_span_logits[:,span_type].tolist()
        toplist = get_best_indexes(span_logits, n_best_size)
        scorelist = []
        for ind in toplist:
            scorelist.append(span_logits[ind])
        spanslist = []
        
        used_tokids = []
        for top in toplist:
            if span_logits[top] < threshold:
                continue
            cur_minus = 0
            cur_plus = 0
            while True:
                cond, value = check_index(span_logits, top + cur_minus - 1)
                if cond and value > threshold:
                    cur_minus -= 1
                else:
                    break
            while True:
                cond, value = check_index(span_logits, top + cur_plus + 1)
                if cond and value > threshold:
                    cur_plus += 1
                else:
                    break

            span_range = [top + cur_minus, top + cur_plus + 1]
            span_tokids = list(range(top + cur_minus,  cur_plus +top + 1))

            cur_span_score_list = [span_logits[tokids] for tokids in span_tokids]

            average_score = sum(cur_span_score_list)/ len(cur_span_score_list)

            max_score = max(cur_span_score_list)


            cur_span = {"span_range": span_range, "span_tokids": span_tokids, "cur_span_score_list": cur_span_score_list,
                        "average_score": average_score, "max_score": max_score, "tokens": tokens[span_range[0]: span_range[1]]}
            

            if not any(tokid in used_tokids for tokid in span_tokids):
                spanslist.append(cur_span)
            
            used_tokids.extend(span_tokids)

        span_type_list.append(spanslist)
    
    return  newlinelist, span_type_list


def do_ranking(score_list, score_threshold= 0.25, con_threshold = 0.25,  sep_type="self_con", top_k = 100, max_continuations= 10, max_headline = 15, headline_finding_range= 30, headline_before =True, headline_after =True):
    """ Takes in a list of tuples (newlinestartlist, newlinelist, span_type_list), returns list of para groups with scores  """

    #(newlinelist, spanslist) 

    para_groups = []
    cur_worst_score = score_threshold
    main_counter = 0
    score_len = len(score_list)
    for exid, example in enumerate(score_list):
        (newlinelist, spanslist) = example
        skipping_list = []
        for nid, newline in enumerate(newlinelist):

            # create group by looking around the highest scoring newline
            # active_score = newline["score_dict"][1]
            # if active_score < cur_worst_score:
            #     continue
            # look_back = False
            # look_forward = False
            # cur_minus = 0
            # cur_plus = 0
            # skip = False
            # while True:
            #     cond, value = check_index(newlinelist, nid + cur_minus - 1)
            #     if cond and value["score_dict"][1] > score_threshold:
            #         cur_minus -= 1
            #         if value["score_dict"][1] > active_score:
            #             skip = True
            #     elif not cond:
            #         look_back = True
            #         break
            #     else:
            #         break
            # while True:
            #     cond, value = check_index(newlinelist, nid + cur_plus + 1)
            #     if cond and value["score_dict"][1] > score_threshold:
            #         cur_plus += 1
            #         if value["score_dict"][1] > active_score:
            #             skip = True
            #     elif not cond:
            #         look_forward = True
            #         break
            #     else:
            #         break
            
            # if skip:
            #     continue


            # span_range = [nid + cur_minus, nid + cur_plus + 1]
            # span_tokids = list(range(nid + cur_minus,  cur_plus +nid + 1))

            if sep_type == "score":

                if nid in skipping_list:
                    continue
                cur_plus = -1
                condition = True
                look_back = False
                look_forward = False
                if nid == 0:
                    look_back = True

                while condition:
                    cur_plus += 1
                    cond, value = check_index(newlinelist, nid + cur_plus)
                    if not cond and cur_plus > 0:
                        look_forward = True
                        cur_plus -= 0
                        condition = False
                    if not cond and cur_plus <= 0:
                        condition = False

                    if cond:
                        if value["score_dict"][1] > score_threshold:
                            skipping_list.append(nid + cur_plus)
                            continue
                        else:
                            condition = False


                if cur_plus < 1:
                    continue

            elif sep_type == "self_con":

                # loop through all newlines, check if a new self_con starts, if yes, save the old param_group, also create a new param_group and add the first line, if not, just add the newline data
                if nid in skipping_list:
                    continue

                con_score = newline["score_dict"][0]
                #if con_score > con_threshold:
                if con_score < con_threshold:
                    continue
                
                
                cur_plus = 0
                condition = True
                look_back = False
                look_forward = False
                if nid == 0:
                    look_back = True

                while condition:
                    cur_plus += 1
                    cond, value = check_index(newlinelist, nid + cur_plus)
                    if not cond and cur_plus > 0:
                        look_forward = True
                        cur_plus -= 0
                        condition = False
                    if not cond and cur_plus <= 0:
                        condition = False

                    if cond:
                        if value["score_dict"][0] < con_threshold:
                            skipping_list.append(nid + cur_plus)
                            continue
                        else:
                            condition = False





            else:
                raise Exception("do_ranking needs either score or self_con as sep_type")

            span_range = [nid, nid + cur_plus]
            span_tokids = list(range(nid,  cur_plus +nid))


            cur_span_score_list = [newlinelist[tokids]["score_dict"][1] for tokids in span_tokids]

            average_score = sum(cur_span_score_list)/ len(cur_span_score_list)
            if average_score < score_threshold:
                continue
            max_score = max(cur_span_score_list)

            token_list = []
            original_ranges = []
            for tokid in span_tokids:
                token_list.extend(newlinelist[tokid]["tokens"])
                original_ranges.append(newlinelist[tokid]["ranges"][0])
                original_ranges.append(newlinelist[tokid]["ranges"][1])
            new_original_ranges = [original_ranges[0], original_ranges[-1]]


            para_group = {"exid": exid, "exid_list": [exid], "nid": nid, "main_counter": main_counter, "span_range":span_range,
            "cur_span_score_list": cur_span_score_list, "average_score": average_score, "max_score": max_score,
            "token_list":token_list , "look_forward": look_forward, "look_back": look_back , "original_ranges":new_original_ranges}
            
            main_counter += 1

            para_groups.append(para_group)

    # check for example overlapping and fuse para_groups
    
    group_len = len(para_groups)
    skip_list = []
    for conti in range(max_continuations):
        for pid ,para_group in enumerate(para_groups):
            if pid in skip_list:
                continue
            next_id = pid
            while next_id < group_len - 1:
                next_id += 1
                if next_id in skip_list:
                    continue
                else:
                    break
            if next_id == group_len:
                continue
            if para_group["look_forward"]:
                cond, value = check_index(para_groups, next_id)
                if cond:
                    if value["look_back"]:
                        new_start = False
                        for tok in value["token_list"][:3]:
                            if tok in segment_tokens:
                                new_start = True
                        if not new_start:

                            para_group["span_range"][1] = value["span_range"][1]
                            para_group["original_ranges"][1] = value["original_ranges"][1]
                            para_group["cur_span_score_list"].extend(value["cur_span_score_list"])
                            para_group["average_score"] =  sum(para_group["cur_span_score_list"]) / len(para_group["cur_span_score_list"])
                            para_group["max_score"] = max(para_group["cur_span_score_list"])
                            para_group["token_list"].extend(value["token_list"])
                            para_group["look_forward"] = value["look_forward"]
                            para_group["exid_list"].extend(value["exid_list"])
                            skip_list.append(next_id)                       

    skip_list.sort(reverse=True)
    for skipid, skipindex in enumerate(skip_list):
        del para_groups[skipindex]



    # check for headlines
    
    for group in para_groups:
        headline_found = False

        if headline_before:
            nid = group["nid"]
            for before in range(3):
                before += 1
                before_pos = nid - before
                if before_pos < 0:
                    break
                cur_exid = group["exid"]

                before_tokens = score_list[cur_exid][0][before_pos]["tokens"]


                for tokid, tok in enumerate(before_tokens[-headline_finding_range:]):
                    if tok in headline_tokens:
                        
                        headline_found = True
                        headline_start = tokid + 1
                        rest_tokens = before_tokens[headline_start:]
                        rest_tok_len = len(rest_tokens)
                        for nltokid, nltok in enumerate(rest_tokens):
                            if nltok == "[Newline]" or (nltokid == (rest_tok_len - 1)):
                                headline_end = nltokid + headline_start
                                break
                        headline = before_tokens[headline_start:headline_end]
                        break

            

        if headline_after:
            for tokid, tok in enumerate(group["token_list"][:headline_finding_range]):
                if tok in headline_tokens:
                    
                    headline_found = True
                    headline_start = tokid + 1
                    rest_tokens = group["token_list"][headline_start:]
                    rest_tok_len = len(rest_tokens)
                    for nltokid, nltok in enumerate(rest_tokens):
                        if nltok == "[Newline]" or (nltokid == (rest_tok_len - 1)):
                            headline_end = nltokid + headline_start
                            break
                    headline = group["token_list"][headline_start:headline_end]
                    break
        # look in spans if there is a headline
        
        if not headline_found:
            exidlist = group["exid_list"]
            found_list = []
            for exid in exidlist:
                labelid = 0
                spanlist = score_list[exid][1][labelid]

                for span in spanlist:
                    if span["span_range"][0] >= group["original_ranges"][0] and span["span_range"][1] <= group["original_ranges"][1]:
                        if len(span["tokens"]) == 1 and span["tokens"][0] in stopper:
                            continue
                        found_list.append(span)
            
            
            if len(found_list) == 0:
                headline = ["headline"]
            else:

                headline = sorted(found_list, key=lambda x: x["max_score"], reverse=True)[0]["tokens"]
            
        group["headline"] = headline

    
    #para_groups.sort(key= lambda x : x["max_score"], reverse= True)

    
    for i, p in enumerate(para_groups):
        print("\n\n")
        # print("Headline here: ")
        # print(" ".join(p["headline"]))
        # print("\n")
        # print(i)
        # print(" Text here: ")
        spanr0 = p["span_range"][0]
        spanr1 = p["span_range"][1]
        maxscore = p["max_score"]
        print(f"Max score: {maxscore:.2} , Spanrange: {spanr0} - {spanr1}\n")
        print(" ".join(p["token_list"]))
    print("\n\nnumber of para_groups: ")
    print(len(para_groups))
    import pdb; pdb.set_trace()

    return para_groups


    
    





    # for start_index in start_indexes:
    #     for end_index in end_indexes:
    #         if end_index < start_index:
    #             continue
    #         if example.input_ids[start_index] < 1000:
    #             continue
    #         if example.input_ids[end_index] < 1000:
    #             continue
    #         length = end_index - start_index + 1
    #         if length > max_answer_length:
    #             continue
    #         #import pdb; pdb.set_trace()
    #         short_span_score = (
    #             start_logits[start_index] +
    #             end_logits[end_index])
    #         cls_token_score = (
    #             start_logits[0] + end_logits[0])
            
    #         # Span logits minus the cls logits seems to be close to the best.
    #         score = short_span_score - cls_token_score

    #         if score > best_score:
    #             best_score = score

    #             example.short_span_score = short_span_score
    #             example.cls_token_score = cls_token_score

    #             start_span = example.start_position - example.question_offset + start_index
    #             end_span =example.start_position - example.question_offset + end_index + 1
    #             short_text = example.tokens[start_index:end_index +1]

    #             example.short_text = short_text
    #             example.score = score
    #             example.doc_start = start_span
    #             example.doc_end = end_span
    # if example.score > threshold:
    #     return example
    # else:
    #     return 0

    # score, summary, start_span, end_span = sorted(predictions, reverse=True)[0]
    # short_span = Span(start_span, end_span)
    # long_span = Span(-1, -1)
    # for c in example.candidates:
    #     start = short_span.start_token_idx
    #     end = short_span.end_token_idx
    #     if c["top_level"] and c["start_token"] <= start and c["end_token"] >= end:
    #         long_span = Span(c["start_token"], c["end_token"])
    #         break

    # summary.predicted_label = {
    #     "example_id": example.example_id,
    #     "long_answer": {
    #         "start_token": long_span.start_token_idx,
    #         "end_token": long_span.end_token_idx,
    #         "start_byte": -1,
    #         "end_byte": -1
    #     },
    #     "long_answer_score": score,
    #     "short_answers": [{
    #         "start_token": short_span.start_token_idx,
    #         "end_token": short_span.end_token_idx,
    #         "start_byte": -1,
    #         "end_byte": -1
    #     }],
    #     "short_answers_score": score,
    #     "yes_no_answer": "NONE"
    # }
    #return summary


def decode(tokenizer, ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
    """Converts a sequence of ids in a string."""
    if ids:
        # tokens = tokenizer.convert_ids_to_tokens(ids)
        tokens = ids
    else:
        return "No Answer"
    out_string = ' '.join(tokens).replace('</w>', ' ').strip()
    if clean_up_tokenization_spaces:
        out_string = out_string.replace('<unk>', '')
        out_string = out_string.replace(' ##', '')
        out_string = out_string.replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' ,', ',').replace(' ,', ','
                ).replace(" ' ", "'").replace(" n't", "n't").replace(" 'm", "'m").replace(" do not", " don't"
                ).replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re")
    return out_string




# @do_cprofile
class QBert():


    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
        parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
        parser.add_argument("--model_checkpoint", type=str, default="logfiles/metamodel", help="Path, url or short name of the model")
        parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
        parser.add_argument("--batch_size", type=int, default=8, help="batch size for prediction")
        #parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
        parser.add_argument("--device", type=str, default="cpu", help="Device (cuda or cpu)")

        parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
        parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the output utterances")
        parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
        parser.add_argument("--seed", type=int, default=42, help="Seed")
        parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
        parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
        parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
        parser.add_argument("--more_than_one", action='store_true', help= "")
        self.args = parser.parse_args()
        

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__file__)
        self.logger.info(pformat(self.args))
        print(f"Only is : {self.args.more_than_one}")
        # if args.model_checkpoint == "":
        #     args.model_checkpoint = download_pretrained_model()

        random.seed(self.args.seed)
        torch.random.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        import pathlib
        main_path = pathlib.Path.cwd().parent
        self.args.model_checkpoint =  main_path / self.args.model_checkpoint
        self.logger.info("Get pretrained model and tokenizer")

        num_binary_labels = len(binary_labels)
        num_span_labels = len(span_labels)
        num_multi_labels = len(multi_labels)

        self.model = BertForMetaClassification.from_pretrained(self.args.model_checkpoint,  num_binary_labels=num_binary_labels, num_span_labels=num_span_labels, num_multi_labels=num_multi_labels)
        self.tokenizer = BertTokenizer.from_pretrained(self.args.model_checkpoint, never_split = stopper)
        self.model.to(self.args.device)
        self.model.eval()



        # logger.info("Sample a personality")
        # personalities = get_dataset_personalities(tokenizer, args.dataset_path, args.dataset_cache)
        # personality = random.choice(personalities)
        # logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))

        history = []

        #examplepara = "Evidence of prehistoric activity in the area comes from Ashton Moss – a 107-hectare (260-acre) peat bog – and is the only one of Tameside's 22 Mesolithic sites not located in the hilly uplands in the north east of the borough. A single Mesolithic flint tool has been discovered in the bog,[6][7] along with a collection of nine Neolithic flints.[8] There was further activity in or around the bog in the Bronze Age. In about 1911, an adult male skull was found in the moss; it was thought to belong to the Romano-British period – similar to the Lindow Man bog body – until radiocarbon dating revealed that it dated from 1,320–970 BC"
        #examplepara = tokenizer.encode(examplepara)

        self.search = Searcher(use_webscraper = True, use_api = True)
        self.threshold = 0.5872142
        #self.threshold = 0.8


    def get_answer(self, q = None, redo_calcs = True):
        if redo_calcs:
            if not q:

                raw_text = input(">>> ")
                start_time = time.time()
                while not raw_text:
                    print('Prompt should not be empty!')
                    raw_text = input(">>> ")
                    start_time = time.time()
            else:
                raw_text = q
                start_time = time.time()

            articlelist = self.search.searchandsplit(raw_text)

            pickle.dump(articlelist, open("intermediatearticles.p", "wb"))

        start_time = time.time()
        #raw_text = "chocolate cake recipe"
        articlelist = pickle.load(open("intermediatearticles.p", "rb"))

        query = raw_text
        toplist =[]
        topresults = 5
        # topmcs= [0.01] * topresults
        # threshold = 0.01
        prediction_list = []
        print("starting encoding")
        batch_list = build_input_batch(articlelist = articlelist, question= query, tokenizer = self.tokenizer, batch_size = self.args.batch_size, onlyone=self.args.more_than_one, webdata=True)
        
        print("finished encoding")
        if redo_calcs:
            with torch.no_grad():
                for batch in batch_list:
                    print("batch starts")
                    (input_batch, input_mask, input_segment, batch_article) = batch
                    # for ba in batch_article:
                        # print("next article \n\n")
                        # print(ba.article_id)
                        # print(ba.tokens)
                    tocudatime = time.time()
                    input_batch = input_batch.to(self.args.device)
                    input_mask = input_mask.to(self.args.device)
                    input_segment = input_segment.to(self.args.device)
                    tocudafin = time.time() - tocudatime
                    print(f"model to cuda after {tocudafin}")
                    print("model starts")
                    mts = time.time()
                    binary_logits, span_logits = self.model(input_ids = input_batch, token_type_ids = input_segment, attention_mask = input_mask)
                    
                    finaltime = time.time() - mts
                    print(f"model finished after {finaltime}")
                    print("model ends")
                    prediction_list.append([binary_logits, span_logits, batch_article])
                    print("appending ends")
                print("compute best predictions")

            pickle.dump(prediction_list, open("savebatches.p", "wb"))
        prediction_list = pickle.load(open("savebatches.p", "rb"))



        number_computed_paras = len(prediction_list) * self.args.batch_size
        cbs = time.time()
        top_results = compute_best_predictions(prediction_list, topk = topresults, stopper = stopper, threshold=self.threshold)
        finaltime = time.time() - cbs
        print(f"computing best preds finished after {finaltime}")
        if not q:
            for result_id, result in enumerate(top_results):
                
                print("\n\n\n")
                print(f"Top {result_id + 1}\n")
                print(f"Answer score: {result.score}\n")
                print(f"Answer propabilities: - Yes: {result.answer_type_logits[0]} - No: {result.answer_type_logits[1]} -No Answer {result.answer_type_logits[2]} -Short Answer {result.answer_type_logits[3]} -Only Long Answer: {result.answer_type_logits[4]}\n")
                
                if result.type_index == 0:
                    print("Answer: Yes\n")

                if result.type_index == 1:
                    print("Answer: No\n")
                
                if result.type_index == 2:
                    print("No Answer found here")


                if result.type_index == 3:
                    decoded_short_answer = decode(self.tokenizer, result.short_text)
                    print(f"Short Answer: {decoded_short_answer}  \n\n")

                if result.type_index == 4 or result.type_index == 0 or result.type_index == 1 or result.type_index == 3:
                    decoded_long_answer = decode(self.tokenizer, result.long_text)
                    print(f"Long Answer: \n{decoded_long_answer}")

        else:
            show_list = []
            for result_id, result in enumerate(top_results):
                
                answer_dict = {}            

                # if result.type_index == 2:
                #     continue


                if result.type_index == 0:
                    answer_dict["short"] = "Yes"

                if result.type_index == 1:
                    answer_dict["short"] = "No"
                



                if result.type_index == 3 or result.type_index == 2:
                    answer_dict["short"] = decode(self.tokenizer, result.short_text)
                

                if result.type_index == 4 or result.type_index == 0 or result.type_index == 1 or result.type_index == 3 or result.type_index == 2:
                    answer_dict["long"] = decode(self.tokenizer, result.long_text[1:])

                if result.type_index == 4:
                    answer_dict["short"] = ""
                answer_dict["url"] = result.url
                answer_dict["type"] = result.type_index
                if result.score > self.threshold:
                    show_list.append(answer_dict)
                else:
                    print("skipped, too low score")
            

            finaltime = time.time() - start_time
            print(f"Total Time: {finaltime}")

            return show_list




        #     for arti in articlelist:
        #         for para in arti:
        #             txtpara = para

        #             out_ids, mc = sample_sequence(query,para, tokenizer, model, args,threshold=threshold)

        #             out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        #             mcs = mc.item()
        #             if mcs > topmcs[0]:
        #                 toplist.append([mcs, out_text, txtpara])
        #                 print(f"Answer propability: {mcs}\n")
        #                 print(out_text)
        #                 topmcs.append(mcs)
        #                 topmcs.sort()
        #                 del topmcs[0]
        # sortedresults = sorted(toplist, key= lambda x: x[0], reverse=True)
        # toprange = min([topresults, len(sortedresults)])
        # for i in range(toprange):
        #     print("\n\n")
        #     print(f"Top {i}\n")
        #     print(f"Answer propability: {sortedresults[i][0]}\n")
        #     print("Answer: " + sortedresults[i][1] +"\n")
        #     print("Paragraph for this answer: " +sortedresults[i][2])

        # print("Number of paragraphs searched")
        # print(len(sortedresults))
        finaltime = time.time() - start_time
        print(f"Processing finished after {finaltime}")

if __name__ == "__main__":
        abc = QBert()
        abc.get_answer()
