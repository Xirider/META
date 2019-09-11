import newspaper

import pickle




def get_html(url):
    article = newspaper.Article(url, fetch_images=False, memoize_articles=False)
    article.download()
    html = article.html

    return html


if __name__ == "__main__":

    # url = "https://sugarspunrun.com/best-carrot-cake-recipe/"

    url = "https://en.wikipedia.org/wiki/British_National_(Overseas)"
    #rl = "https://stackoverflow.com/questions/29721994/python-array-subtraction-loops-back-to-high-number-instead-of-giving-negative-va?noredirect=1&lq=1"
    ##url = "http://camendesign.com/code/video_for_everybody/test.html"
    html = get_html(url)



    from inscriptis import get_text
    
    text = get_text(html)
    print(text)