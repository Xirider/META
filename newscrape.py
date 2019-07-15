
import newspaper

import pickle
import bs4

import html2text

def get_html(url):
    article = newspaper.Article(url, fetch_images=False, memoize_articles=False)
    article.download()
    html = article.html

    return html


def filter_tags(tag):
    wanted_tags = ["p", "table", "li", "td", "tr", "a"]
    unwanted_tags = ["aside", "nav"]
    #"ul"
    #if tag.has_attr("navigation") or tag.has_attr("site-header"):
    #    return False
    # if tag.
    if tag.name in unwanted_tags:
        return False

    else:
        if tag.name in wanted_tags:
            return True
        return False

def structured_parse(html):
    
    mainsoup = bs4.BeautifulSoup(html,'lxml')
    soup = mainsoup.find("body")

    wanted_tags = ["p", "table", "li", "td", "tr", "a" ,"h1", "h2", "h3", "h4", "h5", "h6"] #, "ul"
    unwanted_tags = ["aside", "nav"]

    # text = soup.find_all(filter_tags)

    # text = soup.find_all("p")
    # divtags = soup.find_all("div")
    # for divt in divtags:
    #     if divt.string != None:
    #         #import pdb; pdb.set_trace()
    #         div_text = divt.find(string=True, recursive=False)
    #         if div_text:
    #             div_text.wrap(mainsoup.new_tag("h1"))
    #             import pdb; pdb.set_trace()
    #         # import pdb; pdb.set_trace()
            # divt.string.wrap(mainsoup.new_tag("h1"))



    to_delete = soup.find_all(unwanted_tags)
    for del_element in to_delete:
        del_element.decompose()

    
    def check_for_tags(element):
        for parent in element.parents:
            if parent.name in wanted_tags:
                return False
        return True

    text_elements = soup.find_all(wanted_tags)
    new_tag_list = []
    for element in text_elements:
        contains_tag = check_for_tags(element)
        if contains_tag:
            new_tag_list.append(element)
        



    text = new_tag_list


    for t in text:
        print(t.name)
        print(t.get_text())

    # text = soup_obj.get_text()

    # text = soup_obj.find_all(string=True)


    # output = ''
    # blacklist = [
    #     '[document]',
    #     'noscript',
    #     'header',
    #     'html',
    #     'meta',
    #     'head', 
    #     'input',
    #     'script',
    #     # there may be more elements you don't want, such as "style", etc.
    # ]

    # for t in text:
    #     if t.parent.name not in blacklist:
    #         output += '{} '.format(t)

    # text = output
    return text



if __name__ == "__main__":

    # url = "https://sugarspunrun.com/best-carrot-cake-recipe/"

    url = "https://en.wikipedia.org/wiki/British_National_(Overseas)"
    #rl = "https://stackoverflow.com/questions/29721994/python-array-subtraction-loops-back-to-high-number-instead-of-giving-negative-va?noredirect=1&lq=1"
    ##url = "http://camendesign.com/code/video_for_everybody/test.html"
    html = get_html(url)

    #pickle.dump( html, open( "scrapeinterm.p", "wb" ) )

    #html = pickle.load( open( "scrapeinterm.p", "rb" ) )

    text = get_text(html, display_images=True, display_links=True, deduplicate_captions=True)

    #h = html2text.HTML2Text()

    #h.ignore_links = True

    #text = h.handle(html)

    # text = structured_parse(html)




    print(text)