import time
start_time = time.time()

from googlesearch import search
from nltk import sent_tokenize

from newspaper import Article, news_pool

def searchandsplit(query, maxchars = 10000, minparalen = 40, maxparalen = 700):



    #query = "who is germans chancellor?"


    urllist = search(query, stop=10, pause = 31.0, only_standard = True)



    #print("for search --- %s seconds ---" % (time.time() - start_time))


    articlelist = []


    for url in urllist:
        #print(url)
        article = Article(url, fetch_images = False, memoize_articles=False, request_timeout=0.5)
        articlelist.append(article)

    news_pool.set(articlelist, threads_per_source = 10)
    news_pool.join()

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