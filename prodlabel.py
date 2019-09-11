# coding: utf8
from __future__ import unicode_literals

import prodigy
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import add_tokens
from prodigy.util import split_string
import spacy

from labels import binary_labels, span_labels, multi_labels
from thresholds import binary_labels_threshold, span_labels_threshold, multi_labels_threshold

import torch
from modelingclassbert import BertForMetaClassification

MODELDIR = "logfiles/v1_3class"
DEVICE = "cuda"

class Model(object):
    # This is a dummy model to help illustrate how to use Prodigy with a model
    # in the loop. It currently "predicts" random numbers – but you can swap
    # it out for any model of your choice, for example a text classification
    # model implementation using PyTorch, TensorFlow or scikit-learn.

    def __init__(self, label_type_cond = False, label_id_cond= False,  modeldir = MODELDIR, device = DEVICE):
        # The model can keep arbitrary state – let's use a simple random float
        # to represent the current weights

        num_binary_labels = len(binary_labels)
        num_span_labels = len(span_labels)
        num_multi_labels = len(multi_labels)

        self.label_type_cond = label_type_cond.split(",")
        self.label_id_cond = label_id_cond.split(",")


        assert len(self.label_id_cond) == len(self.label_type_cond)



        self.device = device

        self.model = BertForMetaClassification.from_pretrained(modeldir,  num_binary_labels=num_binary_labels, num_span_labels=num_span_labels, num_multi_labels=num_multi_labels)
        self.model = self.model.to(self.device)


    
    
    def __call__(self, stream):
        with torch.no_grad():
            for example in stream:
                
                keep = 0
                
                input_batch = torch.tensor(example["input_ids"]).unsqueeze(0).to(self.device)
                input_mask = torch.tensor(example["input_mask"]).unsqueeze(0).to(self.device)
                input_segment= torch.tensor(example["segment_ids"]).unsqueeze(0).to(self.device)

                active_newlines = example["active_newlines"]

                binary_logits, span_logits = self.model(input_ids = input_batch, token_type_ids = input_segment, attention_mask = input_mask)

                binary_logits = binary_logits.cpu().numpy()
                span_logits = span_logits.cpu().numpy()

                bin_num = binary_logits.shape[1]
                seq_len = span_logits.shape[0]

                for ltcid, ltc in enumerate(self.label_type_cond):
                    
                    lic = int(self.label_id_cond[ltcid])

                    if ltc == "binary":
                        for a_newline in active_newlines:
                            if binary_logits[0][a_newline][lic] > binary_labels_threshold[lic]:
                                keep += 1
                                break
                            
                    elif ltc == "span":
                        for seq in range(seq_len):
                            if span_logits[0][seq][lic] > span_labels_threshold[lic]:
                                keep += 1
                                break

                if keep != len(self.label_type_cond):
                    continue

                yield example





# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe('manual',
    dataset=("The dataset to use", "positional", None, str),
    label_type=("the type of annotation", "positional", None, str),
    labelid=("One or more comma-separated labels", "positional", None,  int),
    label_type_cond=("the type of annotation", "option", "cond", str),
    label_id_cond=("the type of annotation", "option", "id_cond", str),


)
def manual(dataset, label_type, labelid,  label_type_cond, label_id_cond,  exclude=None):
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

    filtermodel = Model(label_type_cond=label_type_cond, label_id_cond= label_id_cond)

    filteredstream = filtermodel(stream)
    # Tokenize the incoming examples and add a "tokens" property to each
    # example. Also handles pre-defined selected spans. Tokenization allows
    # faster highlighting, because the selection can "snap" to token boundaries.
    # stream = add_tokens(nlp, stream)

    return {
        'view_id': 'ner_manual', # Annotation interface to use
        'dataset': dataset,      # Name of dataset to save annotations
        'stream': filteredstream,        # Incoming stream of examples
        'config': {              # Additional config settings, mostly for app UI
            
            'label': ', '.join(label),
            #'label': label,
            'labels':label
            #'labels': label      # Selectable label options
        }
    }