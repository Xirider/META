

from tqdm import tqdm
import time
import langid
import random
from scrape import Searcher
from prep_ft_text import create_text
from pytorch_pretrained_bert import BertTokenizer
from tokenlist import stopper
sfile = "savedhistory.html"

saveto = "listofqueries.txt"

textto = "corpus.txt"



number_of_searches = 500

opentype = "a+"

extracting_queries_default = True

    #bert_type = "bert-large-uncased-whole-word-masking"


    

def search_data(extracting_queries= extracting_queries_default):
    
    if extracting_queries:
        with open(sfile, "r", encoding="utf-8") as f:
            samples = 10000
            max = 40000
            counter = 0
            contents = f.read()
            querylist = []
            
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
    

    searcher = Searcher(use_webscraper = True, use_api=True, a_number=10)
    print("downloading and saving text")
    qlist = qlist[0:number_of_searches]
    for qid, query in enumerate(tqdm(qlist)):
        article_list = searcher.searchandsplit(query, timeout = 15)
        create_text(articlelist= article_list,  filename=textto,  minlen = 300, opentype=opentype)
        # if qid > 15:
        #     print("something is wrong")
        #     break





            


if __name__ == "__main__":
    search_data()