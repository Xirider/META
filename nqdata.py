# input is list of html articles
# change html to htmlmarker
# input to singleexample

# 

import time
start_time = time.time()
import newspaper
#from newspaper import fulltext
import requests
import unicodedata

import logging
from newspaper.cleaners import DocumentCleaner
from newspaper.configuration import Configuration
from newspaper.extractors import ContentExtractor
from newspaper.outputformatters import OutputFormatter
from newspaper.text import innerTrim
import lxml.html.clean
from html import unescape
import collections



def url_to_nq_inputlist(url):
  """ Converts single article url to list of input for the model (but no batching) """
  article = newspaper.Article(url, fetch_images=False, memoize_articles=False)
  article.download()
  html = article.html
  text , _ = html_to_marked_text(html)
  return text


def build_input_batch(articlelist, question):
  """ Take in list of prepared articles and a question , convert to batches of tensor inputs """

  for article in articlelist:





def convert_single_example(article, question, tokenizer, is_training, max_query_length=30, max_seq_length = 384):
  """Converts a single NqExample into a list of InputFeatures."""
  # tok_to_orig_index = []
  # orig_to_tok_index = []
  # all_doc_tokens = []
  features = []
  # for (i, token) in enumerate(example.doc_tokens):
  #   orig_to_tok_index.append(len(all_doc_tokens))
  #   sub_tokens = tokenize(tokenizer, token)
  #   tok_to_orig_index.extend([i] * len(sub_tokens))
  #   all_doc_tokens.extend(sub_tokens)

  # `tok_to_orig_index` maps wordpiece indices to indices of whitespace
  # tokenized word tokens in the contexts. The word tokens might themselves
  # correspond to word tokens in a larger document, with the mapping given
  # by `doc_tokens_map`.
  # if example.doc_tokens_map:
  #   tok_to_orig_index = [
  #       example.doc_tokens_map[index] for index in tok_to_orig_index
  #   ]
  all_doc_tokens = tokenizer.tokenize(article)



  # QUERY
  query_tokens = []
  query_tokens.append("[Q]")
  query_tokens.extend(tokenizer.tokenize(question))
  if len(query_tokens) > max_query_length:
    query_tokens = query_tokens[max_query_length:]

  # # ANSWER
  # tok_start_position = 0
  # tok_end_position = 0
  # if is_training:
  #   tok_start_position = orig_to_tok_index[example.start_position]
  #   if example.end_position < len(example.doc_tokens) - 1:
  #     tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
  #   else:
  #     tok_end_position = len(all_doc_tokens) - 1

  # The -3 accounts for [CLS], [SEP] and [SEP]
  max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

  # We can have documents that are longer than the maximum sequence length.
  # To deal with this we do a sliding window approach, where we take chunks
  # of up to our max length with a stride of `doc_stride`.
  _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
      "DocSpan", ["start", "length"])
  doc_spans = []
  start_offset = 0
  while start_offset < len(all_doc_tokens):
    length = len(all_doc_tokens) - start_offset
    length = min(length, max_tokens_for_doc)
    doc_spans.append(_DocSpan(start=start_offset, length=length))
    if start_offset + length == len(all_doc_tokens):
      break
    start_offset += min(length, FLAGS.doc_stride)

  for (doc_span_index, doc_span) in enumerate(doc_spans):
    tokens = []
    token_to_orig_map = {}
    token_is_max_context = {}
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    tokens.extend(query_tokens)
    segment_ids.extend([0] * len(query_tokens))
    tokens.append("[SEP]")
    segment_ids.append(0)

    for i in range(doc_span.length):
      split_token_index = doc_span.start + i
      token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

      is_max_context = check_is_max_context(doc_spans, doc_span_index,
                                            split_token_index)
      token_is_max_context[len(tokens)] = is_max_context
      tokens.append(all_doc_tokens[split_token_index])
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
    assert len(tokens) == len(segment_ids)

    

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (FLAGS.max_seq_length - len(input_ids))
    input_ids.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)

    assert len(input_ids) == FLAGS.max_seq_length
    assert len(input_mask) == FLAGS.max_seq_length
    assert len(segment_ids) == FLAGS.max_seq_length

    start_position = None
    end_position = None
    answer_type = None
    answer_text = ""



    if is_training:
      doc_start = doc_span.start
      doc_end = doc_span.start + doc_span.length - 1
      # For training, if our document chunk does not contain an annotation
      # we throw it out, since there is nothing to predict.
      contains_an_annotation = (
          tok_start_position >= doc_start and tok_end_position <= doc_end)
      if ((not contains_an_annotation) or
          example.answer.type == AnswerType.UNKNOWN):
        # If an example has unknown answer type or does not contain the answer
        # span, then we only include it with probability --include_unknowns.
        # When we include an example with unknown answer type, we set the first
        # token of the passage to be the annotated short span.
        if (FLAGS.include_unknowns < 0 or
            random.random() > FLAGS.include_unknowns):
          continue
        start_position = 0
        end_position = 0
        answer_type = AnswerType.UNKNOWN
      else:
        doc_offset = len(query_tokens) + 2
        start_position = tok_start_position - doc_start + doc_offset
        end_position = tok_end_position - doc_start + doc_offset
        answer_type = example.answer.type

      answer_text = " ".join(tokens[start_position:(end_position + 1)])

    feature = InputFeatures(
        unique_id=-1,
        example_index=-1,
        doc_span_index=doc_span_index,
        tokens=tokens,
        token_to_orig_map=token_to_orig_map,
        token_is_max_context=token_is_max_context,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        start_position=start_position,
        end_position=end_position,
        answer_text=answer_text,
        answer_type=answer_type)

    features.append(feature)

    # Pytorch test


  return features

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tokens,
               token_to_orig_map,
               token_is_max_context,
               input_ids,
               input_mask,
               segment_ids,
               start_position=None,
               end_position=None,
               answer_text="",
               answer_type=AnswerType.SHORT):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tokens = tokens
    self.token_to_orig_map = token_to_orig_map
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.start_position = start_position
    self.end_position = end_position
    self.answer_text = answer_text
    self.answer_type = answer_type





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







def html_to_marked_text(html, language='en'):
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