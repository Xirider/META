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
            curiter = next(longiter)
            curll += 1


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

    url = "https://sugarspunrun.com/best-carrot-cake-recipe/"

    #url = "https://en.wikipedia.org/wiki/British_National_(Overseas)"
    #rl = "https://stackoverflow.com/questions/29721994/python-array-subtraction-loops-back-to-high-number-instead-of-giving-negative-va?noredirect=1&lq=1"
    #url = "http://camendesign.com/code/video_for_everybody/test.html"
    #url = "https://en.wikipedia.org/wiki/Metacritic"
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
    textdict = single_html2text(html, url)
    finished = time.time() - start
    
    text = textdict["htmltext"]
    text = " [newline] ".join(text)
    text = text.replace(" [newline] ", "\n")
    import pyperclip
    pyperclip.copy(text)

    print(text)
    print(finished)













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
