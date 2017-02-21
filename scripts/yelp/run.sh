#!/bin/bash

python extract_label_and_document.py train.csv
python extract_label_and_document.py test.csv
java -cp stanford-postagger-3.7.0.jar edu.stanford.nlp.process.PTBTokenizer -preserveLines < train.csv.doc > train.csv.doc.tokenized
java -cp stanford-postagger-3.7.0.jar edu.stanford.nlp.process.PTBTokenizer -preserveLines < test.csv.doc > test.csv.doc.tokenized
python lower_and_merge.py test.csv.lab test.csv.doc.tokenized > test.sent_tok_lower.dat
python lower_and_merge.py train.csv.lab train.csv.doc.tokenized > train.sent_tok_lower.dat
python train_dev_split.py train.sent_tok_lower.dat
python extract_vocab.py train.sent_tok_lower.dat.train 4 > train.sent_tok_lower.dat.train.freq_5.vocab
python unk_dataset_by_vocab.py train.sent_tok_lower.dat.train.freq_5.vocab train.sent_tok_lower.dat.train > train.sent_tok_lower_unk.dat.train
python unk_dataset_by_vocab.py train.sent_tok_lower.dat.train.freq_5.vocab train.sent_tok_lower.dat.dev > train.sent_tok_lower_unk.dat.dev
python unk_dataset_by_vocab.py train.sent_tok_lower.dat.train.freq_5.vocab test.sent_tok_lower.dat > test.sent_tok_lower_unk.dat


DEBUG
head -10859 train.sent_tok_lower.dat.dev > train.sent_tok_lower.dat.dev.sample
python unk_dataset_by_vocab.py train.sent_tok_lower.dat.train.freq_5.vocab train.sent_tok_lower.dat.dev.sample > train.sent_tok_lower_unk.dat.dev.sample