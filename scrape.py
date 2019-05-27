import time


from googlesearch import search
from nltk import sent_tokenize

from newspaper import Article, news_pool

from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired


def artdownload(url):
    ltime = time.time()
    art = Article(url)
    art.download()
    downtime = time.time() - ltime

    print(f"Downloading finished after {downtime} seconds")
    return art

class Searcher():
    def __init__(self):
        self.pool = ProcessPool(max_workers=10)
        
    def searchandsplit(self, query, maxchars = 10000, minparalen = 40, maxparalen = 700):


        start_time = time.time()
        #query = "who is germans chancellor?"
        print("start searching")

        urllist = []
        #urllist = search(query, stop=10, pause = 31.0, only_standard = True)
        for urlr in search(query, stop= 10, pause = 0.0,only_standard = True):
            urllist.append(urlr)
            print("adding url ot url list")



        
        searchtime = time.time() - start_time
        print(f"search finished after {searchtime} seconds")

        #print("for search --- %s seconds ---" % (time.time() - start_time))


        articlelist = []

        timeout = 3.0

        lasttime = time.time()

        finishedmap = self.pool.map(artdownload, urllist, timeout=timeout)

        iterator = finishedmap.result()

        downloadtime = time.time() - lasttime
        print(f"Downloading finished after {downloadtime} seconds")

        timoutcounter = 0
        while True:
            try:
                result = next(iterator)
                articlelist.append(result)
            except StopIteration:
                break
            except TimeoutError as error:
                timeoutcounter += 1
            except ProcessExpired as error:
                print("%s. Exit code: %d" % (error, error.exitcode))
            except Exception as error:
                print("function raised %s" % error)
                print(error.traceback)  # Python's traceback of remote process            


        downloadtime = time.time() - lasttime
        print(f"List gathering finished after {downloadtime} seconds")
        # for url in urllist:
        #     #print(url)
        #     print("adding url to article object")
        #     article = Article(url)
        #     articlelist.append(article)

        # attime = time.time() - start_time
        # print(f"article adding to list after {attime} seconds")
        
        # for art in articlelist:
        #     lasttime = time.time()
        #     art.download()
        #     downtime = time.time() - lasttime

        #     print(f"Downloading finished after {downtime} seconds")


        # news_pool.set(articlelist, threads_per_source = 1)
        # news_pool.join()

        downloadtime = time.time() - start_time
        print(f"Downloading finished after {downloadtime} seconds")
        paralist = []
        for article in articlelist:
            try:
                article.parse()
                body = article.text
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
            except:
                continue
        processingtime = time.time() - start_time
        print(f"Processing finished after {processingtime}")
        return paralist

                    



            # print("NEW ARTICLE /n/n/n")
            # print(body)
            
        # article.download()
        # article.parse()
        # body = article.text
        # body = body[:maxchars]
        # print("NEW ARTICLE /n/n/n")
        # print(body)

        # print("download time--- %s seconds ---" % (downloadtime))

        # print("final--- %s seconds ---" % (time.time() - start_time))


















    # url = 'https://www.kurzweilai.net/essentials-of-general-intelligence-the-direct-path-to-agi'
    # article = Article(url)
    # article.download()
    # article.parse()

    # body = article.text
    # body = body[:maxchars]

    # paralist = body.split("\n\n")

    # for a in paralist: print(a + "\n")