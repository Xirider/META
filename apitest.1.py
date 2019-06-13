
import time
import requests
# from urllib.request import urlopen

headers = {
    'apikey': 'bcfc0a80-87d0-11e9-a533-3901b97f6a9a',
}

params = (
    ('q', 'how fast can google be?'),
    ('location', 'United States'),
    ('search_engine', 'google.com'),
    ('hl', 'en'),
    ('gl', 'US')
)

start_time = time.time()
response = requests.get('https://app.zenserp.com/api/v2/search', headers=headers, params=params)

print(response.json())

finaltime = time.time() - start_time
# print(response_body)

print(f"Processing finished after {finaltime}")


# params = {
#     "q" : "where to fish in the summer",

#     "hl" : "en",
#     "gl" : "us",
#     "google_domain" : "google.com",
#     "api_key" : "d243cd857dad394fe6407afd0094bf9f05aaf775922193fe230b4ea415871576",
# # }

# client = GoogleSearchResults(params)
# results = client.get_dict()






#print(json.dumps(result, indent=2, sort_keys=True))