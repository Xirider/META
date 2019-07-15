
from scrape import Searcher
import copy

# query = "current president"



from pytorch_pretrained_bert import BertTokenizer
from nqdata import convert_single_example
import json


def create_single_para_examples(articlelist):
    # check for beginning para, then copy, modify and append
    
    updated_list = []

    stopper = ["[ContextId=over]"]
    for i in range(51):
        stopper.extend([f"[ContextId={i}]"])

    for article_id, article in enumerate(articlelist):
        input_object = convert_single_example(article, query, tokenizer, article_id=article_id)
        question_number = 0
        for feature in input_object:
            featuredict = feature.__dict__
            stoplist = []

            toklen = len(featuredict["tokens"])
            for tokid, token in enumerate(featuredict["tokens"]):
                if token in stopper:
                    if len(stoplist) == 0 and token != "[ContextId=0]":
                        for sepid, septok in enumerate(featuredict["tokens"]):
                            if septok == "[SEP]":
                                stoplist.append(sepid)
                                break

                    stoplist.append(tokid)
            
            stop_len = len(stoplist)

            if stop_len == 0:
                for sepid, token in enumerate(featuredict["tokens"]):
                    if token == "[SEP]":
                        stoplist = [sepid+1]
                        stop_len = 1
                        break
                        

            for stopid, tokid in enumerate(stoplist):
                curdict = copy.deepcopy(featuredict)
                # add start and end
                # if curdict["tokens"][tokid] == "ContextId=0":
                #     for sepid, token in enumerate(curdict["tokens"]):
                #         if token == "[SEP]":
                #             para_start = sepid
                #             para_end = stoplist[0]
                #             break
                if stopid == stop_len-1:
                    para_start = tokid
                    para_end = toklen - 1 # to account for last sep token
                else:
                    para_start = tokid
                    para_end = stoplist[stopid + 1]

                # add start and end special tokens in "text"
                curdict["para_start"] = para_start
                curdict["para_end"] = para_end

                copytokens = curdict["tokens"].copy()
                divider = " \n<<<<<<<<>>>>>>>>\n "
                # enddivider = ""

                copytokens.insert(para_start, divider)
                copytokens.insert(para_end + 1, divider)
                newtoks = []
                for copid, copytok in enumerate(copytokens):
                    if copytok in stopper:
                        newtoks.append("\n\n")
                    newtoks.append(copytok)
                curdict["tokens_with_br"] = newtoks
                textstring = " ".join(newtoks)
                textstring = textstring.replace(" ##", "")
                textstring = textstring + "\n\n[url]\n" + featuredict["url"]
                curdict["subpara_number"] = stopid
                curdict["text"] = textstring
                curdict["question_number"] = question_number

                if type(featuredict) is dict:
                    updated_list.append(curdict)
                else:
                    import pdb; pdb.set_trace()





            question_number += 1
    return updated_list


if __name__ == "__main__":

    searcher = Searcher(use_nq_scraper = True, use_api=True)
    
    article_object_list = []
    question_number = 0   

    stopper = ["[UNK]", "[SEP]", "[Q]", "[CLS]", "[ContextId=-1]", "[NoLongAnswer]"]
    for i in range(50):
        stopper.extend([f"[ContextId={i}]",f"[Paragraph={i}]",f"[Table={i}]",f"[List={i}]" ])

    tokenizer = BertTokenizer.from_pretrained("savedmodel", never_split =stopper)

    querylist = ["current president",
                "carrot cake recipes"]
                # "mark zuckerberg podcast",
                # "flixbus",
                # "linux search file",
                # "snorkel metal",
                # "india tourism visa",
                # "why is google so fast",
                # "flask get request example"]

    for query in querylist:
        ulist = searcher.searchandsplit(query, timeout=5.0)

        # test tokenizer ratio

        
        
        # tokenizer.max_len = 100000

        # wordcounter =  0
        # tokencounter =  0
        # for article in ulist:

        #     #import pdb; pdb.set_trace()
        #     splitcopy = article["text"].split()
        #     wordcounter += len(splitcopy)
        #     tokenized_article =  tokenizer.tokenize(article["text"])
        #     tokencounter += len(tokenized_article)






        # use ratio for q+p, and also the step size
        # max_len = 384
        # token_len = int(max_len / 1.35)
        
        # question

        # create sublists for each article


        # for article_id, article in enumerate(ulist):
        #     input_object = convert_single_example(article, query, tokenizer, article_id=article_id)

        #     for feature in input_object:
        #         featuredict = feature.__dict__
                

        #         textstring = " ".join(featuredict["tokens"])
        #         textstring = textstring + "[url] " + featuredict["url"]
        #         featuredict["text"] = textstring
        #         featuredict["question_number"] = question_number
        #         question_number += 1
        #         if type(featuredict) is dict:
        #             article_object_list.append(featuredict)
        #         else:
        #             import pdb; pdb.set_trace()
      



    # import json
    # with open('data.json', 'w') as outfile:
    # json.dump(data, outfile)




    example_list = create_single_para_examples(ulist)







    



    # with open('output.jsonl', 'w') as outfile:
    #     for entry in example_list:
    #         try:
    #             json.dump(entry, outfile)
    #         except:
    #             import pdb; pdb.set_trace()
    #         outfile.write('\n')
    #print(ulist)

    with open('justjson.json', 'w') as outfile:
        json.dump(example_list, outfile)


    # print(f"Wordcounter: {wordcounter}")
    # print(f"Tokencounter: {tokencounter}")
