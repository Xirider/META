import newspaper

import pickle
import sys

sys.path.append(r"C:\Users\peter\Desktop\dl\qa")
sys.path.append(r"C:\Users\sophi\Desktop\peter\qa")

from html2text import HTML2Text

from pytorch_pretrained_bert import BertTokenizer


from tokenlist import stopper


def get_html(url):
    article = newspaper.Article(url, fetch_images=False, memoize_articles=False)
    article.download()
    html = article.html

    return html


def create_converter_settings(long_version, url):
    t = HTML2Text(baseurl= url)

    t.body_width = 0
    t.ignore_emphasis = True
    t.mark_code = True
    t.wrap_list_items = False
    t.wrap_links = True
    t.inline_links = True
    if long_version:
        t.bypass_tables = True
        t.wrap_links = True
        t.inline_links = True
        t.long_version = True

    else:
        t.ignore_links = True
    
    return t

def create_token_line_map(longtext, shorttext):

    longlistversion = longtext.split("[newline]")
    shortlist = shorttext.split("[newline]")
    
    longlist = [x.split(" ") for x in longlistversion]
    shortlist = [x.split(" ") for x in shortlist]

    longlen = len(longlist)

    longiter = iter(longlist)
    curll = -1

    line2line = []
    # print(len(shortlist))
    # print(len(longlist))
    activelonglines = []
    for sid, shortline in enumerate(shortlist):
        newset = set(shortline)
        
        if len(newset) == 1:
            #print("skipped")
            line2line.append(None)
            continue
        
        # print("shortid: " + str(sid) + " longid: "+ str(curll))
        # print(" ".join(shortline))

        
            

        while True:
            # if curll == 295:
            #     import pdb; pdb.set_trace()
            try:
                curiter = next(longiter)
            except:
                print("curiter error, as lines don't match up")
                import pdb; pdb.set_trace()

            curll += 1

            # print(" ".join(curiter))
            # if loops > 8:
            # # if curll == 90:
            #     import pdb; pdb.set_trace()
            # print(" ".join(curiter))
            # print(curll)

            if len(set(curiter)) > 1:
                check = newset.issubset(curiter)
                
                if check:
                    # print(curiter)
                    # print("\n\n")
                    line2line.append(curll)
                    activelonglines.append(curll)
                    break




    return line2line, longlistversion, activelonglines  #shorttoken2longline, line2line




def single_html2text(html, url):

    
    longconverter = create_converter_settings(long_version=True, url = url)
    shortconverter = create_converter_settings(long_version=False, url = url)
    longtext = longconverter.handle(html)
    shorttext = shortconverter.handle(html)

    shorttext = shorttext.replace("\n", " [newline] ")
    longtext = longtext.replace("\n", " [newline] ")

    line2line, longlist, activelonglines = create_token_line_map(longtext, shorttext)

    return { "text": shorttext , "url": url, "htmltext": longlist, 
            "line2line": line2line, "activelonglines": activelonglines }




if __name__ == "__main__":



    from gevent import monkey
    def stub(*args, **kwargs):  # pylint: disable=unused-argument
        pass
    monkey.patch_all = stub
    import grequests
    import requests
    def fastdownload(urls, timeout):
    
        def exception_handler(request, exception):
            pass
        rs = (grequests.get(u, timeout=timeout) for u in urls)
        resp = grequests.map(rs, exception_handler=exception_handler)
        return [(rt.text, rt.url)  for rt in resp if rt != None]


    #ll = ['https://addapinch.com/the-best-chocolate-cake-recipe-ever//', 'https://thestayathomechef.com/the-most-amazing-chocolate-cake/', 'https://www.allrecipes.com/recipe/17981/one-bowl-chocolate-cake-iii/', 'https://www.bbc.co.uk/food/recipes/easy_chocolate_cake_31070', 'https://www.bbcgoodfood.com/recipes/easy-chocolate-cake', 'https://www.lifeloveandsugar.com/best-chocolate-cake/', 'https://www.lifeloveandsugar.com/easy-moist-chocolate-cake/', 'https://foodess.com/moist-chocolate-cake/']


    #ll = ['https://stackoverflow.com/questions/1732438/how-do-i-run-all-python-unit-tests-in-a-directory']
    #ll = ['https://realpython.com/python-testing/', 'https://realpython.com/python-testing/', 'https://devguide.python.org/runtests/', 'https://docs.python.org/2/library/unittest.html', 'https://docs.python.org/3/library/unittest.html', 'https://docs.python-guide.org/writing/tests/', 'https://stackoverflow.com/questions/1732438/how-do-i-run-all-python-unit-tests-in-a-directory', 'https://pythontesting.net/framework/unittest/unittest-introduction/', 'https://jeffknupp.com/blog/2013/12/09/improve-your-python-understanding-unit-testing/', 'https://www.fullstackpython.com/testing.html']
    #ll = ['https://thestayathomechef.com/the-most-amazing-chocolate-cake/']


    ll = ['https://twistedsifter.com/2012/04/15-of-the-largest-animals-in-the-world/', 'https://en.wikipedia.org/wiki/Largest_organisms', 'https://awesomeocean.com/guest-columns/15-of-the-largest-sea-animals-in-the-world/', 'https://twistedsifter.com/2012/04/15-of-the-largest-animals-in-the-world/', 'https://www.thoughtco.com/largest-living-sea-creatures-2291904', 'https://www.thoughtco.com/what-is-the-biggest-animal-in-the-ocean-2291995', 'https://www.worldatlas.com/articles/the-biggest-animals-in-the-ocean.html', 'https://britishseafishing.co.uk/what-is-the-largest-animal-in-the-sea/', 'https://animals.howstuffworks.com/animal-facts/question687.htm']
    #ll = ["https://twistedsifter.com/2012/04/15-of-the-largest-animals-in-the-world/"]
    #url = "http://www.foodnetwork.co.uk/recipes/salted-caramel-cheesecake-squares.html"

    #url = "https://en.wikipedia.org/wiki/British_National_(Overseas)"
    #rl = "https://stackoverflow.com/questions/29721994/python-array-subtraction-loops-back-to-high-number-instead-of-giving-negative-va?noredirect=1&lq=1"
    #url = "http://camendesign.com/code/video_for_everybody/test.html"
    #url = "https://en.wikipedia.org/wiki/Metacritic"
    import pathlib
    htmllist = fastdownload(ll, timeout=10)
    print("downloading finished")
    for html, url in htmllist:

        # html = get_html(url)
        
        # main_path = pathlib.Path.cwd().parent
        # model_checkpoint = "logfiles/v1_3class"
        # model_checkpoint =  main_path / model_checkpoint
        # tokenizer = BertTokenizer.from_pretrained(model_checkpoint, never_split = stopper)

        #html = "<table> <tr> <th> Application deadlines for registration as a British National (Overseas)[37] </th></tr> <tr> <td> Year of birth </td> <td> Registration deadline </td></tr> <tr> <td> 1967 to 1971 </td> <td> 30 October 1993 </td></tr> <tr> <td> 1962 to 1966 </td> <td> 31 March 1994 </td></tr> <tr> <td> 1957 to 1961 </td> <td> 31 August 1994 </td></tr> <tr> <td> 1947 to 1956 </td> <td> 28 February 1995 </td></tr> <tr> <td> Prior to 1947 </td> <td> 30 June 1995 </td></tr> <tr> <td> 1972 to 1976 </td> <td> 31 October 1995 </td></tr> <tr> <td> 1977 to 1981 </td> <td> 30 March 1996 </td></tr> <tr> <td> 1982 to 1986 </td> <td> 29 June 1996 </td></tr> <tr> <td> 1987 to 1991 </td> <td> 30 September 1996 </td></tr> <tr> <td> 1992 to 1995 </td> <td> 31 December 1996 </td></tr> <tr> <td> 1996 </td> <td> 31 March 1997 </td></tr> <tr> <td> 1 January to 30 June 1997 </td> <td> 30 September 1997 </td></tr></table> "
        #from html2text import HTML2Text
        import time
        start = time.time()

        try:
            textdict = single_html2text(html, url)
            print(url)
            print("worked")
        except:

            print(url)
            print("didnt work")
            import pdb; pdb.set_trace()


        finished = time.time() - start
        
        text = textdict["text"]
        #import pdb; pdb.set_trace()
        #text = " [newline] ".join(text)
        #text = text.replace(" [newline] ", "\n")
        text = text.replace(" [newline] ", "\n[newline] ")
    import pyperclip
    pyperclip.copy(text)

    # print(text)
    # print(finished)













    #text = text.replace("\n", "<br>")
    #print(text)






#     good bert context:
# image and video tags with description start and end
# ignore links
# Tags for the start of different headlines
# nav start
# aside start
# code start
# code end
# remove random lines at tables
# table start and end

# bypass_tables
# single_line_break
# ignore_emphasis
# mark_code
# skip_internal_links
# ignore_links
# google_list_indent
