

from tqdm import tqdm
import time
import langid
import random
from scrape import Searcher
from prep_ft_text import create_text
from pytorch_pretrained_bert import BertTokenizer
sfile = "savedhistory.html"

saveto = "listofqueries.txt"

textto = "corpus.txt"


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


    

def search_data(extracting_queries= False):
    
    if extracting_queries:
        with open(sfile, "r", encoding="utf-8") as f:
            samples = 10000
            max = 40000
            counter = 0
            contents = f.read()
            querylist = []
            langid.set_languages(["en", "de"])
            for id, item in enumerate(contents.split("Searched for")[1:]):
                searchstring = item.split("\">")[1].split("</a")[0]
                #print(searchstring)
                language = langid.classify(searchstring)[0]
                if counter > max:
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
    
    tokenizer = BertTokenizer("savedmodel/vocab.txt", never_split = stopper)
    searcher = Searcher(use_webscraper = True, use_api=True)
    print("downloading and saving text")
    qlist = qlist[0:200]
    for qid, query in enumerate(tqdm(qlist)):
        article_list = searcher.searchandsplit(query, timeout = 10)
        create_text(articlelist= article_list, tokenizer = tokenizer, filename=textto,  minlen = 300)
        # if qid > 15:
        #     print("something is wrong")
        #     break





            


if __name__ == "__main__":
    search_data()