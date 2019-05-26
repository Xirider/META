import time
start_time = time.time()

from newspaper import fulltext
import requests
url = "https://www.reddit.com/r/artificial/comments/6y0kl4/artificial_life_as_the_path_to_agi/"
html = requests.get(url).text
text = fulltext(html)
downloadtime = time.time() - start_time
print(text)
print(f"Downloading finished after {downloadtime} seconds")