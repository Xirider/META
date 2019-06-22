

import time







import requests

payload = {'api_key': "df9aa1d82559a991f5f9c63d2dc6b31f", 'url':
'https://www.google.com/search?q=us+president'}

# 'https://www.google.com/search?q=fastest+google+search'

tstart = time.time()
r = requests.get('http://api.scraperapi.com', params=payload)


tend = tstart - time.time()


#print (r.text)

print(tend)
print(tstart)
print(time.time())
