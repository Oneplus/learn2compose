#!/bin/bash

python ./scripts/yelp/extract_label_and_document.py ./data/yelp/train.csv
python ./scripts/yelp/extract_label_and_document.py ./data/yelp/test.csv
java -cp ./scripts/yelp/stanford-postagger-3.7.0.jar edu.stanford.nlp.process.PTBTokenizer -preserveLines < ./data/yelp/train.csv.doc > ./data/yelp/train.csv.doc.tokenized
java -cp ./scripts/yelp/stanford-postagger-3.7.0.jar edu.stanford.nlp.process.PTBTokenizer -preserveLines < ./data/yelp/test.csv.doc > ./data/yelp/test.csv.doc.tokenized
python ./scripts/yelp/lower_and_merge.py ./data/yelp/test.csv.lab ./data/yelp/test.csv.doc.tokenized > ./data/yelp/test.sent_tok_lower.dat
python ./scripts/yelp/lower_and_merge.py ./data/yelp/train.csv.lab ./data/yelp/train.csv.doc.tokenized > ./data/yelp/train.sent_tok_lower.dat
python ./scripts/yelp/train_dev_split.py ./data/yelp/train.sent_tok_lower.dat
python ./scripts/yelp/extract_vocab.py ./data/yelp/train.sent_tok_lower.dat.train 4 > ./data/yelp/train.sent_tok_lower_unk5.dat.train.vocab
python ./scripts/yelp/unk_dataset_by_vocab.py ./data/yelp/train.sent_tok_lower_unk5.dat.train.vocab ./data/yelp/train.sent_tok_lower.dat.train > ./data/yelp/train.sent_tok_lower_unk5.dat.train
python ./scripts/yelp/unk_dataset_by_vocab.py ./data/yelp/train.sent_tok_lower_unk5.dat.train.vocab ./data/yelp/train.sent_tok_lower.dat.dev > ./data/yelp/train.sent_tok_lower_unk5.dat.dev
python ./scripts/yelp/unk_dataset_by_vocab.py ./data/yelp/train.sent_tok_lower_unk5.dat.train.vocab ./data/yelp/test.sent_tok_lower.dat > ./data/yelp/test.sent_tok_lower_unk5.dat

DEBUG
head -10859 train.sent_tok_lower.dat.dev > train.sent_tok_lower.dat.dev.sample
python unk_dataset_by_vocab.py train.sent_tok_lower.dat.train.freq_5.vocab train.sent_tok_lower.dat.dev.sample > train.sent_tok_lower_unk.dat.dev.sample