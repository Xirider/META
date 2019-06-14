# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
from scrape import Searcher
import time
import torch
import torch.nn.functional as F
from collections import defaultdict
import pickle
import copy
import numpy as np
import cProfile

#from pytorch_pretrained_bert import OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer
from utils import SPECIAL_TOKENS, build_input_from_segments_ms
from nqdata import build_input_batch
from modeling import BertNQA
from pytorch_pretrained_bert import BertTokenizer

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



def compute_best_predictions(prediction_list, stopper, topk = 5,threshold = 0):
    funcstart = time.time()
    """ takes in list of predictions, creates list of prediction spans, returns the best span for the top spans """
    #articles = defaultdict(list)
    score_list = []
    for batch in prediction_list:
        loopstart = time.time()
        print("new batch")
        [start_logits, end_logits, answer_type_logits, batch_article] = batch
        batch_size = len(batch_article)
        start_logits = start_logits.tolist()
        end_logits = end_logits.tolist()
        answer_type_logits = answer_type_logits.tolist()
        # start_logits = start_logits.data.cpu().numpy()
        # end_logits = end_logits.data.cpu().numpy()
        # answer_type_logits = answer_type_logits.data.cpu().numpy()
        print("batch at cpu")

        for b in range(batch_size):
            example = batch_article[b]
            example.start_logits = start_logits[b]
            example.end_logits = end_logits[b]
            example.answer_type_logits = answer_type_logits[b]
            updated_example = score_short_spans(example, threshold=threshold)
            if updated_example != 0:
                score_list.append(updated_example)
        loopfin = time.time() - loopstart
        print(f"single batch best answer getting time {loopfin}")
    print("finished putting examples into lists")
    finaltime = time.time() - funcstart
    print(f"Processing finished of all batches before cutting them out {finaltime}")
    score_list.sort(reverse=True, key= lambda x: x.score)
    # for ex in score_list:
    #     print(ex.score)
    #     print(ex.doc_start)
    #     print(ex.doc_end)
    #     #print(ex.short_text)
    # import pdb; pdb.set_trace()
    lenlist = len(score_list)
    print("number of candidates")
    print(lenlist)
    top_results = []
    counter = 0
    while len(top_results) < topk:
        if lenlist == 0:
            break
        #import pdb; pdb.set_trace()
        current_example = score_list[counter]
        skip = False


        # decide which type of answer is necessary
        type_index = current_example.answer_type_logits.index(max(current_example.answer_type_logits))
        #type_index = np.argmax(current_example.answer_type_logits)

        current_example.type_index = type_index

        #check if article is already there
        for top in top_results:
            if current_example.article_id == top.article_id:
                #check for overlap
                if current_example.doc_start <= top.doc_end and top.doc_start <= current_example.doc_end:

                    counter += 1
                    skip = True
                elif current_example.start_position <= top.end_position and top.start_position <= current_example.end_position:
                    counter += 1
                    skip = True
                else:
                    pass
        if skip:
            if counter == lenlist - 1:
                break
            else:
                continue









        if type_index == 4 or type_index == 0 or type_index == 1 or type_index == 3:

            # add a long answer
            tok_counter = 0
            max_extra_tokens = 50
            token_map = []
            divided_article = defaultdict(list)

            for tok in current_example.all_doc_tokens:
                if tok in stopper:
                    tok_counter += 1
                token_map.append(tok_counter)
                divided_article[tok_counter].append(tok)

            para_number_start = token_map[current_example.doc_start]
            para_number_end = token_map[current_example.doc_start]

            if para_number_start == para_number_end:
                current_example.long_text = divided_article[para_number_start]
            else:
                current_example.long_text = []
        else:
            current_example.long_text = []



        top_results.append(current_example)
        counter += 1
        if counter == lenlist - 1:
            break


    return top_results


def score_short_spans(example, top_scores = 10, threshold = 0):
    	
    start_logits = example.start_logits
    end_logits = example.end_logits
    predictions = []
    n_best_size = top_scores
    max_answer_length = 30
    best_score = threshold
    example.score = -1000

    start_indexes = get_best_indexes(start_logits, n_best_size)
    end_indexes = get_best_indexes(end_logits, n_best_size)
    for start_index in start_indexes:
        for end_index in end_indexes:
            if end_index < start_index:
                continue
            if example.input_ids[start_index] < 1000:
                continue
            if example.input_ids[end_index] < 1000:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:
                continue
            #import pdb; pdb.set_trace()
            short_span_score = (
                start_logits[start_index] +
                end_logits[end_index])
            cls_token_score = (
                start_logits[0] + end_logits[0])
            
            # Span logits minus the cls logits seems to be close to the best.
            score = short_span_score - cls_token_score

            if score > best_score:
                best_score = score

                example.short_span_score = short_span_score
                example.cls_token_score = cls_token_score

                start_span = example.start_position - example.question_offset + start_index
                end_span =example.start_position - example.question_offset + end_index + 1
                short_text = example.tokens[start_index:end_index +1]

                example.short_text = short_text
                example.score = score
                example.doc_start = start_span
                example.doc_end = end_span
    if example.score > threshold:
        return example
    else:
        return 0

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
        parser.add_argument("--model_checkpoint", type=str, default="savedmodel", help="Path, url or short name of the model")
        parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
        parser.add_argument("--batch_size", type=int, default=64, help="batch size for prediction")
        parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

        parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
        parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the output utterances")
        parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
        parser.add_argument("--seed", type=int, default=42, help="Seed")
        parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
        parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
        parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
        self.args = parser.parse_args()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__file__)
        self.logger.info(pformat(self.args))

        # if args.model_checkpoint == "":
        #     args.model_checkpoint = download_pretrained_model()

        random.seed(self.args.seed)
        torch.random.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        stopper = ["[UNK]", "[SEP]", "[Q]", "[CLS]", "[ContextId=-1]", "[NoLongAnswer]"]
        for i in range(50):
            stopper.extend([f"[ContextId={i}]",f"[Paragraph={i}]",f"[Table={i}]",f"[List={i}]" ])
        self.stopper = tuple(stopper)
        self.logger.info("Get pretrained model and tokenizer")
        self.model = BertNQA.from_pretrained(self.args.model_checkpoint)
        self.tokenizer = BertTokenizer.from_pretrained(self.args.model_checkpoint, never_split = self.stopper)
        self.model.to(self.args.device)
        self.model.eval()

        # logger.info("Sample a personality")
        # personalities = get_dataset_personalities(tokenizer, args.dataset_path, args.dataset_cache)
        # personality = random.choice(personalities)
        # logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))

        history = []

        #examplepara = "Evidence of prehistoric activity in the area comes from Ashton Moss – a 107-hectare (260-acre) peat bog – and is the only one of Tameside's 22 Mesolithic sites not located in the hilly uplands in the north east of the borough. A single Mesolithic flint tool has been discovered in the bog,[6][7] along with a collection of nine Neolithic flints.[8] There was further activity in or around the bog in the Bronze Age. In about 1911, an adult male skull was found in the moss; it was thought to belong to the Romano-British period – similar to the Lindow Man bog body – until radiocarbon dating revealed that it dated from 1,320–970 BC"
        #examplepara = tokenizer.encode(examplepara)

        self.search = Searcher(use_nq_scraper = True, use_api = True)
        self.threshold = 0.5


    def get_answer(self, q = None):
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

        # raw_text = "Who is the current president"
        # #pickle.dump(articlelist, open("intermediatearticles.p", "wb"))
        # articlelist = pickle.load(open("intermediatearticles.p", "rb"))

        query = raw_text
        toplist =[]
        topresults = 5
        # topmcs= [0.01] * topresults
        # threshold = 0.01
        prediction_list = []
        print("starting encoding")
        batch_list = build_input_batch(articlelist = articlelist, question= query, tokenizer = self.tokenizer, batch_size = self.args.batch_size)
        print("finished encoding")
        with torch.no_grad():
            for batch in batch_list:
                print("batch starts")
                (input_batch, input_mask, input_segment, batch_article) = batch
                # for ba in batch_article:
                    # print("next article \n\n")
                    # print(ba.article_id)
                    # print(ba.tokens)
                input_batch = input_batch.to(self.args.device)
                input_mask = input_mask.to(self.args.device)
                input_segment = input_segment.to(self.args.device)
                print("model starts")
                mts = time.time()
                start_logits, end_logits, answer_type_logits = self.model(input_ids = input_batch, token_type_ids = input_segment, attention_mask = input_mask)
                finaltime = time.time() - mts
                print(f"model finished after {finaltime}")
                print("model ends")
                prediction_list.append([start_logits, end_logits, answer_type_logits, batch_article])
                print("appending ends")

            print("compute best predictions")
            number_computed_paras = len(prediction_list) * self.args.batch_size
            cbs = time.time()
            top_results = compute_best_predictions(prediction_list, topk = topresults, stopper = self.stopper, threshold=self.threshold)
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

                if result.type_index == 2:
                    continue


                if result.type_index == 0:
                    answer_dict["short"] = "Yes"

                if result.type_index == 1:
                    answer_dict["short"] = "No"
                



                if result.type_index == 3:
                    answer_dict["short"] = decode(self.tokenizer, result.short_text)
                

                if result.type_index == 4 or result.type_index == 0 or result.type_index == 1 or result.type_index == 3:
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
