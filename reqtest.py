




url = "https://flattrackcoffee.com/"


import time


from googlesearch import search
from nltk import sent_tokenize

from newspaper import Article, news_pool

from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
from nqdata import url_to_nq_inputlist

def artdownload(url):
    ltime = time.time()
    art = Article(url, fetch_images=False, memoize_articles=False)
    art.download()
    # downtime = time.time() - ltime
    maxchars = 10000
    minparalen = 40
    maxparalen = 700

    paralist = []
    try:
        art.parse()
        body = art.text
        body = body[:maxchars]

        splitbody = body.split("\n\n")
        
        for para in splitbody:
            sents = sent_tokenize(para)
            for i in range(len(sents)):
                string = ""
                for x in range(len(sents)-i):

                    if len(string) + len(sents[i + x]) > maxparalen:
                        break
                    string = "".join([string, sents[i + x]])
                if len(string) >= minparalen and string not in paralist:
                    paralist.append(string)
        #downtime = time.time() - ltime
        #print(f"Article downloaded and parsed after {downtime}")
        return paralist
    except:
        return []


    #print(f"Downloading finished after {downtime} seconds")
    return art



import time



start_time = time.time()

abc = artdownload(url)




finaltime = time.time() - start_time
print(f"Processing finished after {finaltime}")