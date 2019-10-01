from bs4 import BeautifulSoup, NavigableString
import newspaper

def get_html(url):
    article = newspaper.Article(url, fetch_images=False, memoize_articles=False)
    article.download()
    html = article.html

    return html

def recursivep(s, tlist, taglist):
    for t in s:
        if isinstance(t, NavigableString):
            pass
        else:
            tt = t.text
            if len(tt) > 0:
                activetag = False
                if t.name in taglist:
                    activetag = True
                    tlist.append(f"<{t.name}>")
                tlist.append(tt)
                tlist =recursivep(t, tlist, taglist)
                if activetag:
                    tlist.append(f"</{t.name}>")
                    activetag = False
    return tlist






def h2t(html, url):
    soup = BeautifulSoup(html)
    taglist = ["span", "h1", "h2", "h3", "h4", "h5", "h6", "p", "img", "a", "code"]
    startlist = []
    endlist = []
    for tag in taglist:
        startlist.append(f"<{tag}>")
        endlist.append(f"</{tag}>")
    doublelist = startlist + endlist

    for s in soup(['script', 'style']):
        s.decompose()

    soup = soup.body
    counter = 0
    textlist = []
    textlist = recursivep(soup, textlist, taglist)
    
    return " ".join(textlist)





if __name__ == "__main__":


    #url = "http://www.foodnetwork.co.uk/recipes/salted-caramel-cheesecake-squares.html"

    #url = "https://en.wikipedia.org/wiki/British_National_(Overseas)"
    #rl = "https://stackoverflow.com/questions/29721994/python-array-subtraction-loops-back-to-high-number-instead-of-giving-negative-va?noredirect=1&lq=1"
    #url = "http://camendesign.com/code/video_for_everybody/test.html"
    url = "https://en.wikipedia.org/wiki/Metacritic"
    html = get_html(url)
    import pathlib
    # main_path = pathlib.Path.cwd().parent
    # model_checkpoint = "logfiles/v1_3class"
    # model_checkpoint =  main_path / model_checkpoint
    # tokenizer = BertTokenizer.from_pretrained(model_checkpoint, never_split = stopper)

    #html = "<table> <tr> <th> Application deadlines for registration as a British National (Overseas)[37] </th></tr> <tr> <td> Year of birth </td> <td> Registration deadline </td></tr> <tr> <td> 1967 to 1971 </td> <td> 30 October 1993 </td></tr> <tr> <td> 1962 to 1966 </td> <td> 31 March 1994 </td></tr> <tr> <td> 1957 to 1961 </td> <td> 31 August 1994 </td></tr> <tr> <td> 1947 to 1956 </td> <td> 28 February 1995 </td></tr> <tr> <td> Prior to 1947 </td> <td> 30 June 1995 </td></tr> <tr> <td> 1972 to 1976 </td> <td> 31 October 1995 </td></tr> <tr> <td> 1977 to 1981 </td> <td> 30 March 1996 </td></tr> <tr> <td> 1982 to 1986 </td> <td> 29 June 1996 </td></tr> <tr> <td> 1987 to 1991 </td> <td> 30 September 1996 </td></tr> <tr> <td> 1992 to 1995 </td> <td> 31 December 1996 </td></tr> <tr> <td> 1996 </td> <td> 31 March 1997 </td></tr> <tr> <td> 1 January to 30 June 1997 </td> <td> 30 September 1997 </td></tr></table> "
    #from html2text import HTML2Text
    import time
    start = time.time()
    textdict = h2t(html, url)
    finished = time.time() - start
    
    text = textdict
    #import pdb; pdb.set_trace()
    #text = " [newline] ".join(text)
    #text = text.replace(" [newline] ", "\n")
    #text = text.replace(" [newline] ", "\n[newline] ")
    import pyperclip
    pyperclip.copy(text)

    print(text)
    print(finished)
