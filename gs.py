from googlesearch import search
import time

tstart = time.time()
query = "fastest googlesearch"
urllist = []
for url in search(query, stop= 10, pause = 0.0,only_standard = True):
    urllist.append(url)

tend = tstart - time.time()
print(urllist)
print(tend)
print(tstart)
print(time.time())