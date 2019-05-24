# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
from scrape import searchandsplit

import torch
import torch.nn.functional as F

from pytorch_pretrained_bert import OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer
from utils import SPECIAL_TOKENS, build_input_from_segments_ms

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



def sample_sequence(query, para, tokenizer, model, args, current_output=None):
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

    return current_output, mc

def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=30, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        args.model_checkpoint = download_pretrained_model()

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer = OpenAIGPTTokenizer.from_pretrained(args.model_checkpoint)
    model = OpenAIGPTDoubleHeadsModel.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    model.eval()

    # logger.info("Sample a personality")
    # personalities = get_dataset_personalities(tokenizer, args.dataset_path, args.dataset_cache)
    # personality = random.choice(personalities)
    # logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))

    history = []

    #examplepara = "Evidence of prehistoric activity in the area comes from Ashton Moss – a 107-hectare (260-acre) peat bog – and is the only one of Tameside's 22 Mesolithic sites not located in the hilly uplands in the north east of the borough. A single Mesolithic flint tool has been discovered in the bog,[6][7] along with a collection of nine Neolithic flints.[8] There was further activity in or around the bog in the Bronze Age. In about 1911, an adult male skull was found in the moss; it was thought to belong to the Romano-British period – similar to the Lindow Man bog body – until radiocarbon dating revealed that it dated from 1,320–970 BC"
    #examplepara = tokenizer.encode(examplepara)

    
    raw_text = input(">>> ")
    while not raw_text:
        print('Prompt should not be empty!')
        raw_text = input(">>> ")

    paralist = searchandsplit(raw_text)
    query = tokenizer.encode(raw_text)
    toplist =[]

    with torch.no_grad():
        
        for para in paralist:
            txtpara = para
            para = tokenizer.encode(para)
            out_ids, mc = sample_sequence(query,para, tokenizer, model, args)

            out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            toplist.append([mc.item(), out_text, txtpara])
            print(f"Answer propability: {mc.item()}\n")
            print(out_text)

    sortedresults = sorted(toplist, key= lambda x: x[0])
    for i in range(10):
        print(f"Top {i}\n")
        print(f"Answer propability: {sortedresults[i][0]}\n")
        print(sortedresults[i][1])
        print("Paragraph for this answer: " +sortedresults[i][2])


if __name__ == "__main__":
    run()
