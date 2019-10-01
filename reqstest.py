# from requests_threads import AsyncSession
# import time

# session = AsyncSession(n=5)


testurl = "https://dev.to/navonf/web-scraping-efficiently-1ba5"
binurl = 'http://httpbin.org/get'

urls = ['https://en.wikipedia.org/wiki/President_of_the_United_States', 'https://en.wikipedia.org/wiki/President_of_Germany', 'https://en.wikipedia.org/wiki/List_of_presidents_of_India', 'https://www.jagranjosh.com/general-knowledge/list-of-all-presidents-of-india-from1947-to-2017-with-tenure-1500293855-1', 'https://www.usa.gov/presidents', 'https://www.un.org/securitycouncil/content/presidency', 'https://www.president.gov.ua/en/', 'https://europa.eu/european-union/about-eu/presidents_en', 'https://www.fatf-gafi.org/about/fatfpresidency/']
# async def _main():
#     t0 = time.time()
#     rs = []
#     for _ in range(5):
#         rs.append(await session.get(testurl))
#     t1 = time.time()
    
#     total = t1-t0
#     print(rs)
#     # for rt in rs:
#     #     print(rt.text)
#     print(total)

# if __name__ == '__main__':

    
#     session.run(_main)
#     print("finished")

import grequests

import time
t0 = time.time()
# urls = [binurl] * 5

rs = (grequests.get(u, timeout = 1.0) for u in urls)

resp = grequests.map(rs)

for i in resp:
    print(i)

t1 = time.time()
total = t1-t0
print(total)



