import time
start_time = time.time()
import newspaper
#from newspaper import fulltext
import requests
import bs4
import unicodedata

url = "https://en.wikipedia.org/wiki/CRISPR"
#url = "https://en.wikipedia.org/wiki/Joona_Sotala"

# #url = "https://www.quora.com/Which-is-the-best-website-for-finance-related-freelancing"

import logging
from newspaper.cleaners import DocumentCleaner
from newspaper.configuration import Configuration
from newspaper.extractors import ContentExtractor
from newspaper.outputformatters import OutputFormatter
from newspaper.text import innerTrim
import lxml.html.clean
from html import unescape

class WithTagOutputFormatter(OutputFormatter):


    def convert_to_html(self):

        node = self.get_top_node()



        article_cleaner = lxml.html.clean.Cleaner()
        article_cleaner.javascript = True
        article_cleaner.style = True
        article_cleaner.allow_tags = [
            'a', 'span', 'p', 'br', 'strong', 'b',
            'em', 'i', 'tt', 'code', 'pre', 'blockquote', 'img', 'h1',
            'h2', 'h3', 'h4', 'h5', 'h6',
            'ul', 'ol', 'li', 'dl', 'dt', 'dd',
            'ta', 'table', 'tr', 'td']
        article_cleaner.remove_unknown_tags = False

        cleaned_node = article_cleaner.clean_html(node)

        #self.top_node = cleaned_node


        return self.parser.nodeToString(cleaned_node)

    def convert_to_text(self):
        topcounter = 0
        paracounter = 1
        txts = ["[ContextId=-1] [NoLongAnswer]"]
        for node in list(self.get_top_node()):
            try:
                txt = self.parser.getText(node)
            except ValueError as err:  # lxml error
                log.info('%s ignoring lxml node error: %s', __title__, err)
                txt = None
            
            if txt:
                #import pdb; pdb.set_trace()
                numberlis = 0
                txt = unescape(txt)

                if paracounter < 49:
                    paratoken = f"[Paragraph={paracounter}]"
                else:
                    paratoken = "[UNK]"

                if topcounter < 49:
                    contexttoken = f"[ContextId={topcounter}]"
                else:
                    contexttoken = "[UNK]"


                if "[Lis" in txt[:8] or "[Tab" in txt[:8]:
                    numberlis = int(txt[0:3]) -1

                    txt = txt[3:]

                else:
                    txt = " ".join((paratoken, txt ))
                    paracounter += 1
                txt = " ".join((contexttoken, txt ))
                txt_lis = innerTrim(txt).split(r'\n')
                txt_lis = [n.strip(' ') for n in txt_lis]
                txts.extend(txt_lis)
                
                topcounter += numberlis
                topcounter += 1
                # more than 500 contexts, then stop
                if topcounter > 500:
                    break
        return ' '.join(txts)


    def add_newline_to_br(self):
        for e in self.parser.getElementsByTag(self.top_node, tag='br'):
            e.text = ' '


    def add_newline_to_li(self):
        topcounter = 0
        for e in self.parser.getElementsByTag(self.top_node, tag='ul'):
            li_list = self.parser.getElementsByTag(e, tag='li')
            counter = 0
            topcounter += 1
            lenlis = len(li_list)
            if topcounter < 49:
                listtoken = f"[List={topcounter}] "
            else:
                listtoken = "[UNK] "
            for li in li_list[:-1]:
                counter += 1
                if counter == 1:
                    li.text ="{:03d}".format(lenlis) + listtoken  + self.parser.getText(li) #+ r'\n'
                else:
                    li.text = self.parser.getText(li) #+ r'\n'
                for c in self.parser.getChildren(li):
                    self.parser.remove(c)

    def add_newline_to_table(self):
        topcounter = 0
        for e in self.parser.getElementsByTag(self.top_node, tag='table'):
            li_list = self.parser.getElementsByTag(e, tag='tr')
            counter = 0
            topcounter += 1
            lenlis = len(li_list)
            if topcounter < 49:
                tabletoken = f"[Table={topcounter}] "
            else:
                tabletoken = "[UNK] "

            for li in li_list[:-1]:
                counter += 1
                if counter == 1:
                    if e.text:
                        etext = e.text
                    else:
                        etext = ""
                    li.text ="{:03d}".format(lenlis) + tabletoken + etext + self.parser.getText(li) #+ r'\n'
                    e.text = ""
                else:
                    li.text = self.parser.getText(li) #+ r'\n'
                for c in self.parser.getChildren(li):
                    self.parser.remove(c)





    def get_formatted(self, top_node):
            """Returns the body text of an article, and also the body article
            html if specified. Returns in (text, html) form
            """
            #import pdb; pdb.set_trace()
            self.top_node = top_node
            html, text = '', ''
            #import pdb; pdb.set_trace()
            self.remove_negativescores_nodes()

            self.config.keep_article_html = False

            self.links_to_text()
            #self.add_newline_to_br()# replace with space or nothing
            self.add_newline_to_li()
            self.add_newline_to_table()
            self.replace_with_text()
            self.remove_empty_tags()
            self.remove_trailing_media_div()

            if self.config.keep_article_html:
                html = self.convert_to_html()
            text = self.convert_to_text()
            # print(self.parser.nodeToString(self.get_top_node()))
            return (text, html)







def fulltext(html, language='en'):
    """Takes article HTML string input and outputs the fulltext
    Input string is decoded via UnicodeDammit if needed
    """
    
    config = Configuration()
    config.language = language

    extractor = ContentExtractor(config)
    document_cleaner = DocumentCleaner(config)
    output_formatter = WithTagOutputFormatter(config)

    doc = config.get_parser().fromstring(html)
    doc = document_cleaner.clean(doc)

    top_node = extractor.calculate_best_node(doc)


    
    top_node = extractor.post_cleanup(top_node)
    text, article_html = output_formatter.get_formatted(top_node)
    return text,article_html




start_time = time.time()
article = newspaper.Article(url, fetch_images=False, memoize_articles=False)
article.download()
html = article.html

#html = requests.get(url).text
afterdown = time.time()
downloadtime = afterdown - start_time
text, html = fulltext(html)
formattime = time.time() - afterdown
print(text)
#print(html)
print(f"Downloading finished after {downloadtime} seconds")
print(f"formatting and parsing finished after {formattime} seconds")

# # 


def find_and_filter_tag(tag, soup):
    """tag specific filter logic"""

    import pdb; pdb.set_trace()
    candidates = soup.find_all(tag)
    candidates = [
        unicodedata.normalize("NFKD", x)
        for x in candidates
        #if x.string is not None
    ]
    if tag == "p":
        candidates = [y.strip() for y in candidates if len(y.split(" ")) >= 4]
        count = sum(len(y.split(" ")) for y in candidates)
    else:
        raise NotImplementedError

    return (candidates, count)


def bs4_scraper(url, memoize):
    t1 = time.time()


    article = newspaper.Article(url, fetch_images=False, memoize_articles=memoize)
    article.download()
    html = article.html
    soup = bs4.BeautifulSoup(html, "lxml")
    text, count = find_and_filter_tag("p", soup)
    # DDB: keep text as a single string for consistency with
    # newspaper_scraper
    text = "\n".join(text)

    metadata = {
        "url": url,
        "word_count": count,
        "elapsed": time.time() - t1,
        "scraper": "bs4",
    }
    return text, metadata

# result, meta = bs4_scraper(url, False)

# print(result)
# def newspaper_scraper(url, memoize):
#     t1 = time.time()

#     try:
#         article = newspaper.Article(url, fetch_images=False, memoize_articles=memoize)
#         article.download()
#         article.parse()
#         text = article.text
#         count = len(text.split())
#     except:
#         return None, None

#     metadata = {
#         "url": url,
#         "word_count": count,
#         "elapsed": time.time() - t1,
#         "scraper": "newspaper",
#     }
#     return text, metadata


# result, _ = newspaper_scraper(url, False)

# print(result)

