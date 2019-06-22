
import time
import requests
# from urllib.request import urlopen


def zenapi(query):
    headers = {
        'apikey': '1a9f9b10-94c4-11e9-a1b3-91ebf8eab926',
    }

    params = (
    ('q', query),
    ('location', 'United States'),
    ('search_engine', 'google.com'),
    ('hl', 'en'),
    ('gl', 'DE')
    )


    response = requests.get('https://app.zenserp.com/api/v2/search', headers=headers, params=params)
    print(response)
    results = response.json()
    #print(results)
    urllist = [ x["url"] for x in results["organic"] if "title" in x]
    
    return urllist


query = "how fast is an api"
start_time = time.time()

urllist = zenapi(query)


finaltime = time.time() - start_time
# print(response_body)

print(urllist)
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