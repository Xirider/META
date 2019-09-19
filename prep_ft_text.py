
from scrape import Searcher
import copy
from newscrape import get_html
from inscriptis_p import get_text
import pickle
import langid
# query = "current president"



from pytorch_pretrained_bert import BertTokenizer, BertModel
from nqdata import convert_single_example
import json


def create_text_with_tok(articlelist, tokenizer, filename="corpus.txt", minlen = 300):
    # check for beginning para, then copy, modify and append
    
    updated_list = []

    afile = open(filename, "a+", encoding="utf-8")

    for article_id, article in enumerate(articlelist):
        tokens = tokenizer.tokenize(article["text"])
        if len(tokens) < minlen:
            continue
        cur_sentence = []
        for tok in tokens:
            if tok == "[newline]":
                cur_sentence.append("\n")
                

            cur_sentence.append(tok)
        print("added new article line")
        cur_sentence.append("\n")

    
        cur_text = " ".join(cur_sentence)
        afile.write(cur_text)
    
    afile.close()

def insert_string(source_str, insert_str, pos):
    return source_str[:pos]+insert_str+source_str[pos:]

def add_linebreak_to_newline(string):

    newtok = "[newline]"
    lb = "\n"
    newlen = len(newtok)
    last_found = -1 -newlen  # Begin at -1 so the next position to search from is 0
    while True:
        # Find next index of substring, by starting after its last known position
        last_found = string.find(newtok, last_found + 1 + newlen)
        
        if last_found == -1:  
            break  # All occurrences have been found
        string = insert_string(string, lb, last_found )
    return string

def create_text(articlelist,  filename="corpus.txt", minlen = 300, opentype = "a+"):
    # check for beginning para, then copy, modify and append
    
    updated_list = []

    afile = open(filename, opentype, encoding="utf-8")

    for article_id, article in enumerate(articlelist):
        #tokens = tokenizer.tokenize(article["text"])
        text = article["text"]
        language = langid.classify(text)[0]
        print(language)
        if language != "en":
            continue
        if len(text.split()) < minlen:
            continue
        text = add_linebreak_to_newline(text)
        text = text + "\n\n"

        print("added new article line")
        afile.write(text)
    
    afile.close()




if __name__ == "__main__":

    searcher = Searcher(use_webscraper = True, use_api=True)
    
    article_object_list = []
    question_number = 0   

    #bert_type = "bert-large-uncased-whole-word-masking"
    from tokenlist import stopper

    tokenizer = BertTokenizer("savedmodel/vocab.txt", never_split = stopper)

    #model = BertModel.from_pretrained(bert_type, cache_dir="savedmodel")

    
    querylist = [#"current president",
                "carrot cake recipes"
                #"mark zuckerberg podcast",
                # "flixbus",
                # "linux search file",
                # "snorkel metal",
                #"india tourism visa",
                # "why is google so fast",
                #"flask get request example"]
                ]
    number_articles = 0
    example_list = []
    for query in querylist:
        #article_list = searcher.searchandsplit(query, timeout=5.0)

        #pickle.dump( article_list, open( "createnewexinterm.p", "wb" ) )
        article_list = pickle.load( open( "createnewexinterm.p", "rb" ) )

        number_articles +=  len(article_list)

        create_text(article_list, tokenizer, filename="corpus.txt")

    print(f"{number_articles} Articles written to file")