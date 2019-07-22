from prodigy.components.loaders import JSONL

from collections import defaultdict
import json
import random
import os


#binary_labels =["self_con_s", "new_real_para"  , "is_option" , "primary_relevance" , "secondary_relevance" , "is_summary" , "is_opinion" , "is_definition" , "is_navigation" ,  "is_non_content"]
binary_labels =["new_topic", "new_real_para"  , "is_option" , "primary_relevance" , "secondary_relevance" , "is_summary" , "is_opinion" , "is_definition" , "is_navigation" ,  "is_non_content"]

#binary_labels =["is_headline", "new_real_para"  , "is_option" , "primary_relevance" , "secondary_relevance" , "is_summary" , "is_opinion" , "is_definition" , "is_navigation" ,  "is_non_content"]
span_labels = ["identity_words", "topic_words"]
multi_labels = [["is_comment", "is_article", "is_wikipedia_level"], ["quality_low", "quality_medium", "quality_high"], ["detail_low", "detail_medium", "detail_high"]]



def spans2newlinelabel(spans, labels, newline_pos, tok2newline):
    
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
    return final_labels 


def spans2label(spans, label, input_ids):
    final_labels = [0]*len(input_ids)
    for span in spans:
        if span["label"] == label:
            span_range = range(span["token_start"], span["token_end"] + 1)
            for positive in span_range:
                final_labels[positive] = 1

    return final_labels


    

if __name__ == "__main__":

    db = "d3"
    foldername = "processed_3"




    filename = f"outputsv1/{db}.jsonl"

    example_iterator = JSONL(filename)
    new_example_list = []

    for example in example_iterator:
        # convert spans into newline labels
        if "answer" in example:
            if example["answer"] != "accept":
                print("skipped example as the answer field was either empty or doesnt have accept")
                continue

        if "spans" in example:
            spanlist = example["spans"]
        else:
            spanlist = []
        for label in binary_labels:
            example[label] = spans2newlinelabel(spanlist, [label], example["[Newline]"], example["tok2newline"])
        
        for label in span_labels:
            example[label] =  spans2label(spanlist, label, example["input_ids"])
        

        for label in multi_labels:
            example[label[0]] = spans2newlinelabel(spanlist, label, example["[Newline]"], example["tok2newline"])

        new_example_list.append(example)

    
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    full_filename = f"{foldername}/train.jsonl"
    test_filename = f"{foldername}/test.jsonl"
    writecounter = 0
    test_prob = 0.3
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


          

        