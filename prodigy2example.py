from prodigy.components.loaders import JSONL

from collections import defaultdict
import json
import random
import os

from labels import binary_labels, span_labels, multi_labels

all_labels = binary_labels + span_labels + multi_labels

def spans2newlinelabel(spans, labels, newline_pos, tok2newline, activelist):
    
    label_list = defaultdict(list)
    for label in labels:
        for span in spans:
            if span["label"] == label:
                tokenrange = tok2newline[span["token_start"]: span["token_end"] + 1]
                label_list[label].extend(tokenrange)

    
    
    final_labels = []
    
    for newline, _ in enumerate(newline_pos):
        added_value = False
        for labelid, label in enumerate(labels):
            if newline in label_list[label]:
                final_labels.append(labelid + 1)
                added_value = True
                break

        if not added_value:
            final_labels.append(0)
    
    if labels[0] not in activelist:
        final_labels = [-1] * len(newline_pos)
    return final_labels 


def spans2label(spans, label, input_ids, activelist):
    if label in activelist:

        final_labels = [0]*len(input_ids)
        for span in spans:
            if span["label"] == label:
                span_range = range(span["token_start"], span["token_end"] + 1)
                for positive in span_range:
                    final_labels[positive] = 1
    else:
        final_labels = [-1]*len(input_ids)
    return final_labels




def list_duplicates(seq, key):
    ddict = defaultdict(list)
    for i,item in enumerate(seq):
        
        ddict[item[key]].append(i)
    return ddict

if __name__ == "__main__":
    #db_prefix = "d3"
    db = "d3_merged"
    foldername = "processed_d3_merged"
    mainfolder = "traindata"
    test_prob = 0.20



    filename = f"outputsv1/{db}.jsonl"

    example_iterator = JSONL(filename)
    new_example_list = []
    example_iterator = list(example_iterator)
    # modify examples to have only one example per text with all relevant spans, and a marker to show which label is active
    ddict = list_duplicates(example_iterator, "_input_hash")
    keylist = list(ddict.keys())
    deletelist = []
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
        
        example_iterator[indexlist[0]]["spans"] = spanlist
        example_iterator[indexlist[0]]["activelist"] = activelist

        # mark either all or all but the first for deleting
        if len(activelist) == 0:
            deletelist.append(indexlist[0])
    
        for iid, index in enumerate(indexlist):
            if iid > 0:
                deletelist.append(index)

    deletelist.sort(reverse = True)
    for delitem in deletelist:
        del example_iterator[delitem]






    for example in example_iterator:
        # convert spans into newline labels
        

        spanlist = example["spans"]

        for label in binary_labels:
            example[label] = spans2newlinelabel(spanlist, [label], example["[Newline]"], example["tok2newline"], example["activelist"])
        
        for label in span_labels:
            example[label] =  spans2label(spanlist, label, example["input_ids"], example["activelist"])
        

        for label in multi_labels:
            example[label[0]] = spans2newlinelabel(spanlist, label, example["[Newline]"], example["tok2newline"], example["activelist"])

        new_example_list.append(example)

    if not os.path.exists(mainfolder):
        os.makedirs(mainfolder)

    subfoldername = f"{mainfolder}/{foldername}"
    if not os.path.exists(subfoldername):
        os.makedirs(subfoldername)

    full_filename = f"{subfoldername}/train.jsonl"
    test_filename = f"{subfoldername}/test.jsonl"
    writecounter = 0
    
    train_example_list = []
    test_example_list = []

    for line in new_example_list:
        if random.random() > test_prob:
            train_example_list.append(line)
        else:
            test_example_list.append(line)

    with open(full_filename, 'w') as outfile:
            for line in train_example_list:
                json.dump(line, outfile)
                outfile.write('\n')
                writecounter += 1
    print(f"Wrote {writecounter} examples to train, ready for input")

    writecounter = 0
    with open(test_filename, 'w') as outfile:
        for line in test_example_list:
            json.dump(line, outfile)
            outfile.write('\n')
            writecounter += 1
    print(f"Wrote {writecounter} examples to test, ready for input")


          

        