import time
start_time = time.time()

from googlesearch import search


from newspaper import Article, news_pool


maxchars = 50000


query = "who is germans chancellor?"


urllist = search(query, stop=10, pause = 2.0, only_standard = True)



print("for search --- %s seconds ---" % (time.time() - start_time))


articlelist = []


for url in urllist:
    print(url)
    article = Article(url)
    articlelist.append(article)

news_pool.set(articlelist, threads_per_source = 8)
news_pool.join()

downloadtime = time.time() - start_time


for article in articlelist:
    article.parse()
    body = article.text
    body = body[:maxchars]
    print("NEW ARTICLE /n/n/n")
    print(body)
    
# article.download()
# article.parse()
# body = article.text
# body = body[:maxchars]
# print("NEW ARTICLE /n/n/n")
# print(body)
print("download time--- %s seconds ---" % (downloadtime))

print("final--- %s seconds ---" % (time.time() - start_time))


















# url = 'https://www.kurzweilai.net/essentials-of-general-intelligence-the-direct-path-to-agi'
# article = Article(url)
# article.download()
# article.parse()

# body = article.text
# body = body[:maxchars]

# paralist = body.split("\n\n")

# for a in paralist: print(a + "\n")