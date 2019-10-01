from prodigy.components.loaders import JSONL

from collections import defaultdict
import json
import random
import os
import copy
from labels import binary_labels, span_labels, multi_labels

all_labels = binary_labels + span_labels + multi_labels

def spans2newlinelabel(spans, labels, newline_pos, tok2newline, activelist, active_newlines):
    

    label_list = defaultdict(list)
    for label in labels:
        for span in spans:
            if span["label"] == label:
                tokenrange = tok2newline[span["token_start"]: span["token_end"] + 1]
                tokenrange = list(range(min(tokenrange), max(tokenrange) + 1))
                label_list[label].extend(tokenrange)

    
    
    final_labels = []
    # new_active = []
    # for active in active_newlines:
    #     new_active.append(tok2newline[active])
    new_active = active_newlines
    for newline, rlnewline in enumerate(newline_pos):
        added_value = False


        if rlnewline not in new_active:
            final_labels.append(-1)
            added_value = True

        if not added_value:
            for labelid, label in enumerate(labels):
                if newline in label_list[label]:
                    final_labels.append(labelid + 1)
                    added_value = True
                    
                    break



        if not added_value:
            final_labels.append(0)
            
    

    if labels[0] not in activelist:
        final_labels = [-1] * len(newline_pos)
    
    if sum(final_labels) * -1  != len(final_labels):
        is_active = True
    else:
        is_active = False

    return is_active,final_labels 


def spans2label(spans, label, input_ids, activelist, inactive_tokens):

    if label in activelist:


        final_labels = [0]*len(input_ids)
        for span in spans:
            if span["label"] == label:
                span_range = range(span["token_start"], span["token_end"] + 1)
                for positive in span_range:
                    final_labels[positive] = 1
                    
    

        active = list(range(inactive_tokens[0], inactive_tokens[1]))

        for tokid, tok in enumerate(final_labels):
            if tokid not in active:
                final_labels[tokid] = -1

    else:
        final_labels = [-1]*len(input_ids)
    
    if sum(final_labels) * -1  != len(final_labels):
        is_active = True
    else:
        is_active = False

    return is_active ,final_labels




def list_duplicates(seq, key):
    ddict = defaultdict(list)
    for i,item in enumerate(seq):
        
        ddict[item[key]].append(i)
    return ddict

if __name__ == "__main__":


    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--output_folder", type=str, default="", help="where to save the annotated example train and test files")
    parser.add_argument("--single_labels", action='store_true', help="One example per label")

    args = parser.parse_args()
    #db_prefix = "d3"
    db_name = args.output_folder
    db = db_name
    mainfolder = "traindata"
    test_prob = 0.20



    foldername = f"processed_{args.output_folder}"




    deduplication = False

    filename = f"outputsv1/{db}.jsonl"

    example_iterator = JSONL(filename)
    new_example_list = defaultdict(list)
    example_iterator = list(example_iterator)
    # modify examples to have only one example per text with all relevant spans, and a marker to show which label is active
    ddict = list_duplicates(example_iterator, "_input_hash")
    keylist = list(ddict.keys())
    deletelist = []
    print("generating keylist")
    for key in keylist:
        indexlist = ddict[key]
        spanlist = []
        activelist = []
        for index in indexlist:
            cur = example_iterator[index]


            # one time fix for incorrect naming
            if cur["_session_id"] == "d3-default":
                cur["_session_id"] = "d3_0-default"




            if cur["answer"] == "accept":
                if "spans" in cur:
                    spanlist.extend(cur["spans"])
                cur_session = cur["_session_id"]
                cur_session = cur_session.replace("-default" , "")
                cur_session = int(cur_session.split("_")[-1])
                accepted_label = all_labels[cur_session]
                if type(accepted_label) == list:
                    accepted_label = accepted_label[0]
                activelist.append(accepted_label)
                
                if not deduplication:
                    example_iterator[index]["activelist"] = [accepted_label]
                    if not "spans" in cur:
                        cur["spans"] = []
            else:
                if not deduplication:
                    deletelist.append(index)
        
        if deduplication:
            example_iterator[indexlist[0]]["spans"] = spanlist
            example_iterator[indexlist[0]]["activelist"] = activelist
        


        # mark either all or all but the first for deleting

        if deduplication:
            if len(activelist) == 0:
                deletelist.append(indexlist[0])
        
            for iid, index in enumerate(indexlist):
                if iid > 0:
                    deletelist.append(index)
                
        # else:
        #     if len(activelist) == 0:
        #         for iid, index in enumerate(indexlist):
        #             deletelist.append(index)


    deletelist.sort(reverse = True)
    for delitem in deletelist:
        del example_iterator[delitem]



    print("start converting labels")

    print(f"total of {len(example_iterator)} examples")
    for exampleid, example in enumerate(example_iterator):
        # convert spans into newline labels
        
        if exampleid % 50 == 0:
            print(f"Converting example {exampleid}")
        spanlist = example["spans"]

        non_minus_label = False
        for label in binary_labels:

            is_active,example[label] = spans2newlinelabel(spanlist, [label], example["[newline]"], example["tok2newline"], example["activelist"], example["active_newlines"])
            if is_active:
                non_minus_label = label
        
        for label in span_labels:
            is_active,example[label] =  spans2label(spanlist, label, example["input_ids"], example["activelist"], example["inactive_tokens"])
            if is_active:
                non_minus_label = label
        

        for label in multi_labels:
            is_active,example[label[0]] = spans2newlinelabel(spanlist, label, example["[newline]"], example["tok2newline"], example["activelist"], example["active_newlines"])
            if is_active:
                non_minus_label = label[0]

        if args.single_labels:
            # for each 

            if non_minus_label != False:
                cur_list = example[non_minus_label]
                cur_list_len = len(cur_list)
                for lid, l_number in enumerate(cur_list):
                    if l_number != -1:
                        cop_example = copy.deepcopy(example)
                        single_active_list = cur_list_len * [-1]
                        single_active_list[lid] = l_number
                        cop_example[non_minus_label] = list(single_active_list)
                        if exampleid == 0:
                            print(cop_example[non_minus_label])
                        new_example_list[non_minus_label].append(cop_example)

        else:
            if non_minus_label != False:
                new_example_list[non_minus_label].append(example)
            else:
                print("skipped example, as it had no non minus labels")

    if not os.path.exists(mainfolder):
        os.makedirs(mainfolder)

    subfoldername = f"{mainfolder}/{foldername}"
    if not os.path.exists(subfoldername):
        os.makedirs(subfoldername)

    full_filename = f"{subfoldername}/train.jsonl"
    test_filename = f"{subfoldername}/test.jsonl"

    if os.path.exists(full_filename):
        os.remove(full_filename)

    if os.path.exists(test_filename):
        os.remove(test_filename)


    for exkey in new_example_list:
        cur_active_list = new_example_list[exkey]
        writecounter = 0
        
        train_example_list = []
        test_example_list = []

        nel = len(cur_active_list)
        train_num = int(nel - (nel // (1/test_prob)))


        train_example_list = cur_active_list[:train_num]
        test_example_list = cur_active_list[train_num:]

        # for line in new_example_list:
        #     if random.random() > test_prob:
        #         train_example_list.append(line)
        #     else:
        #         test_example_list.append(line)

        with open(full_filename, 'a+') as outfile:
                for line in train_example_list:
                    json.dump(line, outfile)
                    outfile.write('\n')
                    writecounter += 1
        print(f"Wrote {writecounter} examples to train, ready for input, label: {exkey}")

        writecounter = 0
        with open(test_filename, 'a+') as outfile:
            for line in test_example_list:
                json.dump(line, outfile)
                outfile.write('\n')
                writecounter += 1
        print(f"Wrote {writecounter} examples to test, ready for input, label: {exkey}")


          

        