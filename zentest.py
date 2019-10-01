
import requests
from scrape import fastdownload
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

    try:
        response = requests.get('https://app.zenserp.com/api/v2/search', headers=headers, params=params)
        results = response.json()
    except:
        raise Exception("Response error during request error")
    try:
        urllist = [ x["url"] for x in results["organic"] if "title" in x]
    except:
        
        print("Zenserp api returns error")
        urllist = []
    urllist = [url for url in urllist if not url.endswith(".pdf")]

    return urllist


urllist = zenapi("hello hello")
timeout = 1.0
htmltext = fastdownload(urllist, timeout)


print(htmltext[0])