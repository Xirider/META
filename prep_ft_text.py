
from scrape import Searcher
import copy
from newscrape import get_html
from inscriptis_p import get_text
import pickle
# query = "current president"



from pytorch_pretrained_bert import BertTokenizer, BertModel
from nqdata import convert_single_example
import json


def create_text(articlelist, tokenizer, filename="corpus.txt", minlen = 300):
    # check for beginning para, then copy, modify and append
    
    updated_list = []

    afile = open(filename, "a+", encoding="utf-8")

    for article_id, article in enumerate(articlelist):
        tokens = tokenizer.tokenize(article["text"])
        if len(tokens) < minlen:
            continue
        cur_sentence = []
        for tok in tokens:
            if tok == "[Newline]":
                cur_sentence.append("\n")
                

            cur_sentence.append(tok)
        print("added new article line")
        cur_sentence.append("\n")

    
        cur_text = " ".join(cur_sentence)
        afile.write(cur_text)
    
    afile.close()




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



    #bert_type = "bert-large-uncased-whole-word-masking"


    tokenizer = BertTokenizer("savedmodel/vocab.txt", never_split = stopper)

    #model = BertModel.from_pretrained(bert_type, cache_dir="savedmodel")

    
    querylist = [#"current president",
                "carrot cake recipes"
                #"mark zuckerberg podcast",
                # "flixbus",
                # "linux search file",
                # "snorkel metal",
                #"india tourism visa",
                # "why is google so fast",
                #"flask get request example"]
                ]
    number_articles = 0
    example_list = []
    for query in querylist:
        #article_list = searcher.searchandsplit(query, timeout=5.0)

        #pickle.dump( article_list, open( "createnewexinterm.p", "wb" ) )
        article_list = pickle.load( open( "createnewexinterm.p", "rb" ) )

        number_articles +=  len(article_list)

        create_text(article_list, tokenizer, filename="corpus.txt")

    print(f"{number_articles} Articles written to file")