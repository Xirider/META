from lib.google_search_results import GoogleSearchResults

import time



start_time = time.time()


params = {
    "q" : "error: cannot open .git/FETCH_HEAD: Permission denied",

    "hl" : "en",
    "gl" : "de",
    "google_domain" : "google.com",
    "api_key" : "d243cd857dad394fe6407afd0094bf9f05aaf775922193fe230b4ea415871576",
}

client = GoogleSearchResults(params)
results = client.get_dict()




# from serpwow.google_search_results import GoogleSearchResults
# import json

# # create the serpwow object, passing in our API key
# serpwow = GoogleSearchResults("demo")

# # set up a dict for the search parameters

# start_time = time.time()
# params = {
#   "q" : "where to fish in the winter"
# }

# # retrieve the search results as JSON
# result = serpwow.get_json(params)

# #pretty-print the result




# import requests

# headers = {
#     'apikey': 'bcfc0a80-87d0-11e9-a533-3901b97f6a9a',
# }

# params = (
#     ('q', 'where to fish in the summer'),
#     ('location', 'United States'),
#     ('search_engine', 'google.com'),
#     ('language', 'English'),
# )

# response = requests.get('https://app.zenserp.com/api/search', headers=headers, params=params)







finaltime = time.time() - start_time
print(results)
print(f"Processing finished after {finaltime}")
#print(json.dumps(result, indent=2, sort_keys=True))