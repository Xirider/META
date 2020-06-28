# META search engine

Meta allows you to search on a paragraph level compared to current search engines that work on the web page level.


This repository contains scripts to collect search results, label them by relevance and finetune a BERT model from this data. It also allows you to host a Flask webpage with your own model.

Note that you need at least one V100 GPU for inference deployment (about $4 per hour), otherwise the ranking will take too long for each search.

## Setup

To run these scripts you need to install these libraries: Ignite, Prodigy (Labeling software, you also need to get a license for it), Pytorch, Tensorboard, tqdm

### Training data preparation

#### download examples and make them rdy for annotation
python create_ex_new.py --download --num_q 200

#### create new database
python -m prodigy dataset d3_11 "v1" --author meta

#### label a certain database with a type and a labelid
python -m prodigy manual d3_11 span 0 -F prodlabel.py

#### delete old merged database (after an error)
python -m prodigy drop d3_merged

#### merge labeled examples into one database (if you have multiple labels per example)
python -m prodigy db-merge d3,d3_1,d3_11 d3_merged

#### export merged database
python -m prodigy db-out d3_merged "outputsv1"

#### create train and testdata from exported examples
python prodigy2example.py --single_labels --output_folder newname


### Model Finetuning for domain adaption to web pages (increases model performance with few labeled examples)

#### download websites and create text file
python convertgoogle2file.py

#### convert text file to finetuning examples
python pregenerate_training_data.py --train_corpus corpus.txt --do_lower_case --do_whole_word_mask --max_seq_len 384 --output_dir bertfinetune

#### finetune on webdata
python finetune_on_pregenerated.py --pregenerated_data bertfinetune/ --bert_model bert-base-uncased --do_lower_case --output_dir finetuned_lm/ --epochs 3 --train_batch_size 30 --gradient_accumulation_steps 6
(check finetuned_lm empty, bertfinetune data with new data)

### Model Training/Finetuning with labeled examples

#### train classifier
python run_classifier.py --data_dir traindata/processed_d4_1 --output_dir logfiles/d4_1 --do_train --do_eval --bert_model  bert-base-uncased --gradient_accumulation_steps 16 --train_batch_size 32 --num_train_epochs 30 --overwrite_output_dir


### Inference without website for debugging:
python run_inf.py --no_inference --use_cached_calcs 


### Inference with website as flask app:
python app.py
