
from scrape import Searcher
import copy
from newscrape import get_html
from inscriptis_p import get_text
import pickle
# query = "current president"



from pytorch_pretrained_bert import BertTokenizer, BertModel
from nqdata import convert_single_example
import json


from tqdm import tqdm
import time
import langid
import random
from scrape import Searcher
from prep_ft_text import create_text
from pytorch_pretrained_bert import BertTokenizer



def create_single_para_examples(articlelist, query, stopper, tokenizer, question_number, duplication_mode = False):
    # check for beginning para, then copy, modify and append
    
    updated_list = []
    doccano_id = 0

    for article_id, article in enumerate(articlelist):
        input_object = convert_single_example(article, query, tokenizer, article_id=article_id, webdata=True)

        for feature in input_object:

            #single segment

            featuredict = feature.__dict__
            stoplist = []

            # add one list for each special tag with the corresponding token ids

            for special_tag in stopper:
                if not special_tag.startswith("[Segment="):
                    special_list = []
                    for tokid, token in enumerate(featuredict["tokens"]):
                        if token == special_tag:
                            special_list.append(tokid)
                    featuredict[special_tag] = special_list


            # toklen = len(featuredict["tokens"])
            # for tokid, token in enumerate(featuredict["tokens"]):
            #     if token in stopper:
            #         if len(stoplist) == 0 and token != "[ContextId=0]":
            #             for sepid, septok in enumerate(featuredict["tokens"]):
            #                 if septok == "[SEP]":
            #                     stoplist.append(sepid)
            #                     break

            #         stoplist.append(tokid)
            
            # stop_len = len(stoplist)

            # if stop_len == 0:
            #     for sepid, token in enumerate(featuredict["tokens"]):
            #         if token == "[SEP]":
            #             stoplist = [sepid+1]
            #             stop_len = 1
            #             break
                        
            len_newlines = len(featuredict["[Newline]"])
            if duplication_mode:
                for stopid, tokid in enumerate(featuredict["[Newline]"]):
                    curdict = copy.deepcopy(featuredict)

                    para_start = tokid
                    if stopid == len_newlines - 1:
                        para_end = len(curdict["tokens"]) - 1
                    else:
                        para_end = curdict["[Newline]"][stopid + 1]


                    # add start and end special tokens in "text"
                    curdict["para_start"] = para_start
                    curdict["para_end"] = para_end
                    curdict["subpara_id"] = stopid


                    copytokens = curdict["tokens"].copy()
                    divider = " \n\n<<<<<<<<>>>>>>>>\n "
                    # enddivider = ""

                    for new_line_token_id in curdict["[Newline]"]:
                        copytokens[new_line_token_id] = "\n"

                    copytokens.insert(para_start, divider)
                    copytokens.insert(para_end, divider)
                    
                    char2tok = []

                    curdict["doccano_id"] = doccano_id


                    
                    # create map from displayed chars to the original token ids without divider
                    real_tok_counter = 0
                    for copytok in copytokens:
                        charcount = len(copytok)
                        char2tok.extend([real_tok_counter] * (charcount + 1))
                        if copytok != divider:
                            real_tok_counter += 1


                    curdict["char2tok"] = char2tok
                    curdict["tokens_with_br"] = copytokens
                    textstring = " ".join(copytokens)
                    #textstring = textstring.replace(" ##", "")
                    textstring = textstring + "\n\n[url]\n" + featuredict["url"]
                    
                    curdict["text"] = textstring

                    curdict["question_number"] = question_number

                    doccano_id += 1

                    if type(featuredict) is dict:
                        updated_list.append(curdict)
                    else:
                        import pdb; pdb.set_trace()

            else:
                curdict = featuredict

                copytokens = curdict["tokens"].copy()
                #divider = " \n\n<<<<<<<<>>>>>>>>\n "
                # enddivider = ""

                for new_line_token_id in curdict["[Newline]"]:
                    copytokens[new_line_token_id] = "\n"

                #copytokens.insert(para_start, divider)
                #copytokens.insert(para_end, divider)
                
                char2tok = []
                char2newline = []
                tok2newline = []

                curdict["doccano_id"] = doccano_id


                
                # create map from displayed chars to the original token ids without divider
                real_tok_counter = 0
                newline_counter = -1

                prodigy_tokens = []

                cur_start = 0
                cur_end = 0

                for copytokid, copytok in enumerate(copytokens):
                    if copytok == "\n":
                        newline_counter += 1
                    curnewline_counter = newline_counter
                    if copytok == "\n":
                        curnewline_counter = -1
                        disab = True
                    else:
                        disab = False
                    charcount = len(copytok)
                    char2tok.extend([real_tok_counter] * (charcount + 1))
                    char2newline.extend((charcount + 1)* [curnewline_counter])
                    tok2newline.append(curnewline_counter)
                    cur_end = cur_start + charcount
                    
                    prodigy_text = { "text": copytok, "start":cur_start , "end":cur_end , "id":copytokid, "disabled": disab }
                    prodigy_tokens.append(prodigy_text)
                    cur_start = cur_end + 1
                    

                curdict["char2newline"] = char2newline
                curdict["tok2newline"] = tok2newline


                curdict["char2tok"] = char2tok
                curdict["standard_tokens"] = curdict["tokens"]
                curdict["tokens_with_br"] = copytokens
                curdict["tokens"] = prodigy_tokens
                textstring = " ".join(copytokens)
                #textstring = textstring.replace(" ##", "")
                #textstring = textstring + "\n\n[url]\n" + featuredict["url"]
                
                curdict["text"] = textstring
                curdict["meta"] = {"url": featuredict["url"], "q_number":question_number}

                curdict["question_number"] = question_number

                doccano_id += 1

                if type(featuredict) is dict:
                    updated_list.append(curdict)


    return updated_list


def doccano_meta_format(list_of_dicts):
    """ moves all non-text columns into an extra meta dict for doccano input format """
    new_list_of_dicts = []
    for example_dict in list_of_dicts:
        new_dict = {"meta": {}}
        new_dict["text"] = example_dict["text"]
        for key in example_dict.keys():
            if key != "text":
                new_dict["meta"][key] = example_dict[key]
        new_list_of_dicts.append(new_dict)
    return new_list_of_dicts
        

def create_doccano_files(examples, label_list, clas_task, filename, foldername="doc_input/"):
    """ for each dict create one copy with the label included and save as file """
    updated_list = []
    for label in label_list:
        cur_label_list = []

        for exid, example in enumerate(examples):
            if exid == 0:
                formated_labels = []
                if clas_task == "binary" or clas_task == "span":
                    formated_labels.append([0 , 2, label])
                elif clas_task == "multi":
                    for slabel in label:
                        formated_labels.append([ 0 , 2, slabel])
                    
                example["labels"] = formated_labels
            
        
        full_filename = f'{foldername}doc_in-{clas_task}-{filename}-{label}.jsonl'
        writecounter = 0
        with open(full_filename, 'w') as outfile:
            for line in examples:
                json.dump(line, outfile)
                outfile.write('\n')
                writecounter += 1
        print(f"Wrote {writecounter} examples to the label file '''{label}''' for clas task: {clas_task}")



def create_prodigy_file(examples, label_list, clas_task, filename, foldername="doc_input/"):
    """ for each dict create one copy with the label included and save as file """
    # updated_list = []
    # for label in label_list:
    #     cur_label_list = []

    #     for exid, example in enumerate(examples):
    #         if exid == 0:
    #             formated_labels = []
    #             if clas_task == "binary" or clas_task == "span":
    #                 formated_labels.append([0 , 2, label])
    #             elif clas_task == "multi":
    #                 for slabel in label:
    #                     formated_labels.append([ 0 , 2, slabel])
                    
    #             example["labels"] = formated_labels
            
        
    #     full_filename = f'{foldername}doc_in-{clas_task}-{filename}-{label}.jsonl'
    full_filename = f'{foldername}{filename}prodfile.jsonl'
    writecounter = 0
    with open(full_filename, 'w') as outfile:
        for line in examples:
            json.dump(line, outfile)
            outfile.write('\n')
            writecounter += 1
    print(f"Wrote {writecounter} examples to the label file")









if __name__ == "__main__":

    searcher = Searcher(use_webscraper = True, use_api=True)
    
    article_object_list = []
    question_number = 0   

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



    binary_labels =["self_con_s", "new_real_para"  , "is_option" , "primary_relevance" , "secondary_relevance" , "is_summary" , "is_opinion" , "is_definition" , "is_navigation" ,  "is_non_content"]
    #binary_labels =["new_topic", "new_real_para"  , "is_option" , "primary_relevance" , "secondary_relevance" , "is_summary" , "is_opinion" , "is_definition" , "is_navigation" ,  "is_non_content"]

    #binary_labels =["is_headline", "new_real_para"  , "is_option" , "primary_relevance" , "secondary_relevance" , "is_summary" , "is_opinion" , "is_definition" , "is_navigation" ,  "is_non_content"]
    span_labels = ["identity_words", "topic_words"]
    multi_labels = [["is_comment", "is_article", "is_wikipedia_level"], ["quality_low", "quality_medium", "quality_high"], ["detail_low", "detail_medium", "detail_high"]]













    sfile = "savedhistory.html"

    saveto = "prodcreate_queries"

    filename = "examples_rdy_for_annotation"

    vocab = "savedmodel/vocab.txt"


    tokenizer = BertTokenizer(vocab, never_split = stopper)


    


    def create_data(extracting_queries= False, num_q = 200, download= False, samples = 200):
        
        if extracting_queries:
            with open(sfile, "r", encoding="utf-8") as f:
                samples = 10000
                maximum = 40000
                counter = 0
                contents = f.read()
                querylist = []
                langid.set_languages(["en", "de"])
                for id, item in enumerate(contents.split("Searched for")[1:]):
                    searchstring = item.split("\">")[1].split("</a")[0]
                    #print(searchstring)
                    language = langid.classify(searchstring)[0]
                    if counter > maximum:
                        break
                    if language == "en":
                        counter +=1
                        querylist.append(searchstring + "\n")
            querylist = list(dict.fromkeys(querylist))
            print(len(querylist))

            samples = min(samples, len(querylist))
            querylist = random.sample(querylist, k=samples)
            
            print("writing querys")
            with open(saveto, "w", encoding="utf-8")as f:
                f.writelines(querylist)


        with open(saveto, "r", encoding="utf-8") as f:
            
            qlist = f.readlines()
        
        
        searcher = Searcher(use_webscraper = True, use_api=True)
        print("downloading and saving text")
        qlist = qlist[0:num_q]

        example_list = []
        question_number = 0

        if download:
            for qid, query in enumerate(tqdm(qlist)):
                try:
                    article_list = searcher.searchandsplit(query, timeout = 10)
                except:
                    print("couldn't get articles, maybe because of api failure, skipping query now")
                    time.sleep(10)
                    continue

                example_result = create_single_para_examples(article_list, query, stopper, tokenizer, question_number)

                question_number += 1
                example_list.extend(example_result)
            
            pickle.dump( example_list, open( "example_list_for_annotations.p", "wb" ) )
        
        example_list = pickle.load( open( "example_list_for_annotations.p", "rb" ) )

        samples = min(samples, num_q)
        print("number of samples")
        print(samples)
        sampled_list = random.sample(example_list, k =samples)


        return sampled_list




    example_list = create_data(extracting_queries=True, num_q=210, download=True, samples=200)


    create_prodigy_file(example_list, multi_labels, "multi", filename)





    
    # querylist = [#"current president",
    #             "carrot cake recipes"
    #             #"mark zuckerberg podcast",
    #             # "flixbus",
    #             # "linux search file",
    #             # "snorkel metal",
    #             #"india tourism visa",
    #             # "why is google so fast",
    #             #"flask get request example"]
    #             ]
    # example_list = []
    # for query in querylist:
    #     #article_list = searcher.searchandsplit(query, timeout=5.0)

    #     #pickle.dump( article_list, open( "createnewexinterm.p", "wb" ) )
    #     article_list = pickle.load( open( "createnewexinterm.p", "rb" ) )



    #     example_result = create_single_para_examples(article_list, query, stopper, tokenizer, question_number)
    #     question_number += 1
    #     example_list.extend(example_result)
    
    #filename = "v1"

    # #print("converting to meta format for doccano")

    # #example_list = doccano_meta_format(example_list)

    # print("start with creation of label files")

    # # create_doccano_files(example_list,binary_labels, "binary", filename)
    # # create_doccano_files(example_list, span_labels, "span", filename)
    # # create_doccano_files(example_list, multi_labels, "multi", filename)



    


    










    # with open('output.jsonl', 'w') as outfile:
    #     for entry in example_list:
    #         try:
    #             json.dump(entry, outfile)
    #         except:
    #             import pdb; pdb.set_trace()
    #         outfile.write('\n')
    #print(ulist)

    # with open('justjson.json', 'w') as outfile:
    #     json.dump(example_list, outfile)


    # print(f"Wordcounter: {wordcounter}")
    # print(f"Tokencounter: {tokencounter}")

        # article_list.extend(article_result)

        # test tokenizer ratio
        # for ser in ulist:
        #     html = get_html(ser)
        
        #     text = get_text(html, display_images=True, display_links=True, deduplicate_captions=True)
        # # tokenizer.max_len = 100000

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

