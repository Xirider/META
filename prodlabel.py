# coding: utf8
from __future__ import unicode_literals

import prodigy
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import add_tokens
from prodigy.util import split_string
import spacy

from labels import binary_labels, span_labels, multi_labels

# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe('manual',
    dataset=("The dataset to use", "positional", None, str),
    label_type=("the type of annotation", "positional", None, str),
    labelid=("One or more comma-separated labels", "positional", None,  int),
    exclude=("Names of datasets to exclude", "option", "e", split_string)
)
def manual(dataset, label_type, labelid, exclude=None):
    """
    Mark spans manually by token. Requires only a tokenizer and no entity
    recognizer, and doesn't do any active learning.
    """
    
    source = "doc_input/examples_rdy_for_annotationprodfile.jsonl"
    # Load the spaCy model for tokenization
    # nlp = spacy.load(spacy_model)


    if label_type == "binary":
        label = [binary_labels[labelid]]
    elif label_type == "span":
        label = [span_labels[labelid]]
    elif label_type == "multi":
        label = multi_labels[labelid]

    else:
        raise Exception("need label type")        





    # Loa   d the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    # Tokenize the incoming examples and add a "tokens" property to each
    # example. Also handles pre-defined selected spans. Tokenization allows
    # faster highlighting, because the selection can "snap" to token boundaries.
    # stream = add_tokens(nlp, stream)

    return {
        'view_id': 'ner_manual', # Annotation interface to use
        'dataset': dataset,      # Name of dataset to save annotations
        'stream': stream,        # Incoming stream of examples
        'exclude': exclude,      # List of dataset names to exclude
        'config': {              # Additional config settings, mostly for app UI
            
            'label': ', '.join(label),
            #'label': label,
            'labels':label
            #'labels': label      # Selectable label options
        }
    }