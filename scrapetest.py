import time


from googlesearch import search
from nltk import sent_tokenize

from newspaper import Article, news_pool

from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired, ThreadPool
from nqdata import url_to_nq_inputlist


from lib.google_search_results import GoogleSearchResults

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

class Searcher():
    def __init__(self, use_nq_scraper = False, use_api=False):
        
        
        if use_nq_scraper:
            self.scrape_function = url_to_nq_inputlist
        else:
            self.scrape_function = artdownload
        self.use_api = use_api




    def searchandsplit(self, query):


        start_time = time.time()
        #query = "who is germans chancellor?"
        print("start searching")
        #self.pool = ProcessPool(max_workers=10)
        urllist = []
        #urllist = search(query, stop=10, pause = 31.0, only_standard = True)



        if self.use_api:
            
            params = {
                "q" : query,

                "hl" : "en",
                "gl" : "us",
                "google_domain" : "google.com",
                "api_key" : "d243cd857dad394fe6407afd0094bf9f05aaf775922193fe230b4ea415871576",
            }
            client = GoogleSearchResults(params)
            results = client.get_dict()

            urllist = [ x["link"] for x in results["organic_results"]]
            #extended_urllist = [ (x["link"], x["titel"]) for x in results["organic_results"]]
            #print(urllist)
        else:
            for urlr in search(query, stop= 10, pause = 0.0,only_standard = True):
                urllist.append(urlr)
                print("adding url or url list")



        
        searchtime = time.time() - start_time
        print(f"search finished after {searchtime} seconds")

        #print("for search --- %s seconds ---" % (time.time() - start_time))


        articlelist = []

        timeout = 1.5

        lasttime = time.time()

        with ThreadPool() as pool:
            future = pool.map(self.scrape_function, urllist, timeout=timeout)


        # finishedmap = self.pool.map(, urllist, timeout=timeout)


        iterator = future.result()



        downloadtime = time.time() - lasttime
        print(f"Map  finished after {downloadtime} seconds")

        timeoutcounter = 0
        for i in range(10):
            try:
                #shorttime = time.time()
                result = next(iterator)
                articlelist.append(result)
                timeoutcounter += 1
                #printtime = time.time() - shorttime
                #print(f"iter finished after {printtime} seconds")
            except:
                pass

        #articlelist = list(iterator)
            # except StopIteration:
            #     break
            # except TimeoutError as error:
            #     timeoutcounter += 1
            # except ProcessExpired as error:
            #     print("%s. Exit code: %d" % (error, error.exitcode))
            # except Exception as error:
            #     print("function raised %s" % error)
            #     print(error.traceback)  # Python's traceback of remote process 
        print("No timeouts in this many cases")           
        print(timeoutcounter)

        downloadtime = time.time() - lasttime
        print(f"map + List gathering finished after {downloadtime} seconds")
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
        print(f"Everything (search + download + parse) finished after {downloadtime} seconds")
        # paralist = []
        # for article in articlelist:
        #     try:
        #         article.parse()
        #         body = article.text
        #         body = body[:maxchars]

        #         splitbody = body.split("\n\n")
                
        #         for para in splitbody:
        #             sents = sent_tokenize(para)
        #             for i in range(len(sents)):
        #                 string = ""
        #                 for x in range(len(sents)-i):

        #                     if len(string) + len(sents[i + x]) > maxparalen:
        #                         break
        #                     string = "".join([string, sents[i + x]])
        #                 if len(string) >= minparalen and string not in paralist:
        #                     paralist.append(string)
        #     except:
        #         continue
        # processingtime = time.time() - start_time
        # print(f"Processing finished after {processingtime}")

        
        # self.pool.close()
        # self.pool.join()
        return articlelist

                    



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





if __name__ == "__main__":
    

    searcher = Searcher(use_nq_scraper = True, use_api=True)

    ulist = searcher.searchandsplit("what is the reason of my life")


    #print(ulist)







    # url = 'https://www.kurzweilai.net/essentials-of-general-intelligence-the-direct-path-to-agi'
    # article = Article(url)
    # article.download()
    # article.parse()

    # body = article.text
    # body = body[:maxchars]

    # paralist = body.split("\n\n")

    # for a in paralist: print(a + "\n")