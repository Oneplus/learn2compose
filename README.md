learn2compose
=============
An reproduction of [Learning to Compose Words into Sentences with Reinforcement Learning](https://arxiv.org/abs/1611.09100).

## Code Structure

* `alphabet.{cc|h}`: use to convert word type into index
* `layer.{cc|h}`: a high-level (layer-level) library based on dynet, support Merge, LSTM, and other network like Keras.
* `logging.{cc|h}`: logging library
* `system.{cc|h}`: the transition system, define State and the behaviour of SHIFT/REDUCE action.
* `trainer_utils.{cc|h}`: initialize different trainers
* `sst.cc`: main function for sst
* `sst_corpus.{cc|h} <- corpus.{cc|h}`: use to store and parse SST dataset
* `sst_model.{cc|h} <- reinforce.{cc|h}`:
    * `reinforce.{cc|h}`
        * `rollin`: roll-in/sampling a sequence of transition and return the final sentence representation.
        * `shift_function|reduce_function`: define perform Tree-LSTM composition
        * store parameters for Tree-LSTM
    * `sst_model.{cc|h}`
        * store parameters for policy network and classifier
        * `reinforce`: use the sentence representation from `rollin` to calculate the probability for classes.

##  Compile

First, get the dynet and eigen3 library:
```
git submodule init
git submodule update
```
Then compile
```
mkdir build
cmake -DEIGEN3_INCLUDE_DIR=../eigen/ ..
make
```
If success, you should found the executable `./bin/learn2compose_sst`.

## Stanford Sentiment Treebank (SST)

### Data Preparation

The structure of data folder
```
./data/glove/glove.840B.300d.txt
./data/sst/{train|dev|test}.txt
```

You can download the data by running:
```
python ./scripts/download.py
```
You can also cheat the downloader by putting the wordvec file and sst file in the corresponding directory.

Next, extract SST from tree structure.
```
python ./scripts/sst/extract_sst.py ./data/sst/train.txt > ./data/sst/train.5class.txt
python ./scripts/sst/extract_sst.py ./data/sst/dev.txt > ./data/sst/dev.5class.txt
python ./scripts/sst/extract_sst.py ./data/sst/test.txt > ./data/sst/test.5class.txt
```

Convert SST into binary classification data (format: `{label}\t{sentence}`) and filter the wordvec file (nltk is needed):
```
python ./scripts/sst/5to2.py ./data/sst/train.5class.txt > ./data/sst/train.2class.txt
python ./scripts/sst/5to2.py ./data/sst/dev.5class.txt > ./data/sst/dev.2class.txt
python ./scripts/sst/5to2.py ./data/sst/test.5class.txt > ./data/sst/test.2class.txt
```

Filter word vector.
```
python ./scripts/sst/filter_wordvec.py ./data/glove/glove.840B.300d.txt ./data/sst/train.5class.txt ./data/sst/dev.5class.txt ./data/sst/test.5class.txt > ./data/glove/glove.840B.300d.txt.sst_filtered
```

### Run
You can run the SST experiments with:
```
./bin/learn2compose_sst --dynet-mem 1024 \
    --dynet-seed 1234 \
    -T ./data/sst/train.2class.txt \
    -d ./data/sst/dev.2class.txt \
    -t ./data/sst/test.2class.txt \
    -w ./data/glove/glove.840B.300d.txt.sst_filtered \
    --word_dim 300 \
    --hidden_dim 200 \
    --system cons \
    --dropout 0.5 \
    --optimizer_enable_clipping true \
    --optimizer_enable_eta_decay true \
    --max_iter 50
```

### Results

| Model | Hyperparameters | Dev | Test |
|------|-----|-----|-----|
| system=constituent, policy=REINFORCE | dropout=0, seed=1234, l2=0 | 84.1 | 84.7 |
| system=constituent, policy=REINFORCE | dropout=.5, seed=1234, l2=0 | 82.9 | 83.9 |
| system=constituent, policy=REINFORCE, tune-embedding | dropout=0, seed=1234, l2=0 | 82.8 | 84.7 |
| system=dependency, policy=REINFORCE | dropout=0, seed=1234, l2=0 | 81.4 | 82.0 |
| system=dependency, policy=REINFORCE | dropout=.5, seed=1234, l2=0 | 82.0 | 82.7 |

## YELP

### Data Preparation
The structure of data folder
```
./data/glove/glove.6B.zip
./data/yelp/{train|test}.csv
```

First extract labels and documents (at the same time, split sentence) from the .csv data
```
python ./scripts/yelp/extract_label_and_document.py ./data/yelp/train.csv
python ./scripts/yelp/extract_label_and_document.py ./data/yelp/test.csv
```

Tokenize the data
```
java -cp ./scripts/yelp/stanford-postagger-3.7.0.jar edu.stanford.nlp.process.PTBTokenizer -preserveLines < ./data/yelp/train.csv.doc > ./data/yelp/train.csv.doc.tokenized
java -cp ./scripts/yelp/stanford-postagger-3.7.0.jar edu.stanford.nlp.process.PTBTokenizer -preserveLines < ./data/yelp/test.csv.doc > ./data/yelp/test.csv.doc.tokenized
```

Merge into new data.
```
python ./scripts/yelp/lower_and_merge.py ./data/yelp/test.csv.lab ./data/yelp/test.csv.doc.tokenized > ./data/yelp/test.sent_tok_lower.dat
python ./scripts/yelp/lower_and_merge.py ./data/yelp/train.csv.lab ./data/yelp/train.csv.doc.tokenized > ./data/yelp/train.sent_tok_lower.dat
```

Get development dataset from training data.
```
python ./scripts/yelp/train_dev_split.py ./data/yelp/train.sent_tok_lower.dat
```
Extract vocabulary and unk the dataset.
```
python ./scripts/yelp/extract_vocab.py ./data/yelp/train.sent_tok_lower.dat.train 4 > ./data/yelp/train.sent_tok_lower_unk5.dat.train.vocab
python ./scripts/yelp/unk_dataset_by_vocab.py train.sent_tok_lower.dat.train.freq_5.vocab ./data/yelp/train.sent_tok_lower.dat.train > ./data/yelp/train.sent_tok_lower_unk5.dat.train
python ./scripts/yelp/unk_dataset_by_vocab.py train.sent_tok_lower.dat.train.freq_5.vocab ./data/yelp/train.sent_tok_lower.dat.dev > ./data/yelp/train.sent_tok_lower_unk5.dat.dev
python ./scripts/yelp/unk_dataset_by_vocab.py train.sent_tok_lower.dat.train.freq_5.vocab ./data/yelp/test.sent_tok_lower.dat > ./data/yelp/test.sent_tok_lower_unk5.dat
```
Filter word embeddings
```
python ./scripts/yelp/filter_wordvec.py ./data/glove/glove.6B.100d.txt ./data/yelp/train.sent_tok_lower_unk5.dat.train ./data/yelp/train.sent_tok_lower_unk5.dat.dev ./data/yelp/test.sent_tok_lower_unk5.dat > ./data/glove/glove.6B.100d.txt.yelp_unk5_filtered
```

### Run
You can run the YELP experiments with:
```
./bin/learn2compose_sst --dynet-mem 1024 \
    --dynet-seed 1234 \
    -T ./data/yelp/train.sent_tok_lower_unk5.dat.train \
    -d ./data/yelp/train.sent_tok_lower_unk5.dat.dev \
    -t ./data/yelp/test.sent_tok_lower_unk5.dat \
    -w ./data/glove/glove.6B.100d.txt.yelp_unk5_filtered \
    --word_dim 100 \
    --hidden_dim 200 \
    --optimizer_enable_clipping true \
    --optimizer_enable_eta_decay true \
    --max_iter 300
```

### Result

On small dataset (1,000), self test:

| Model | Hyperparameters | Self |
|------|-----|-----|
| system=constituent, obj=(REINFORCE_joint_+), iter=1K | seed=1234, l2=0 | 82.0 |
| system=constituent, obj=(left_reward_+), iter=100 | seed=1234, l2=0 | 100. |
| system=constituent, obj=left_reward_10,(sample_policy_1,sample_reward_1), iter=100 | seed=1234, l2=0 | 99.4 |
