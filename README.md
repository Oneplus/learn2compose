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

## Stanford Sentiment Treebank (SST)

### Data Preparation

The structure of data folder

```
./data/glove/glove.6B.zip
./data/sst/{train|dev|test}.txt
```

You can download the data by running:
```
python ./scripts/download.py
```
You can also cheat the downloader by putting the wordvec file and sst file in the corresponding directory.

Next, convert SST into binary classification data (format: `{label}\t{sentence}`) and filter the wordvec file (nltk is needed):
```
python ./scripts/extract_sst.py ./data/sst/train.txt > ./data/sst/train.plain.5class.txt
python ./scripts/extract_sst.py ./data/sst/dev.txt > ./data/sst/dev.plain.5class.txt
python ./scripts/extract_sst.py ./data/sst/test.txt > ./data/sst/test.plain.5class.txt
python ./scripts/5to2.py ./data/sst/train.plain.5class.txt > ./data/sst/train.plain.2class.txt
python ./scripts/5to2.py ./data/sst/dev.plain.5class.txt > ./data/sst/dev.plain.2class.txt
python ./scripts/5to2.py ./data/sst/test.plain.5class.txt > ./data/sst/test.plain.2class.txt
python ./scripts/filter_wordvec.py ./data/glove/glove.6B.100d.txt ./data/sst/train.plain.5class.txt ./data/sst/dev.plain.5class.txt ./data/sst/test.plain.5class.txt > ./data/glove/glove.6B.100d.txt.sst_filtered
```

###  Compile

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
If success, you should found the executable `./bin/learn2compose_sst`. To run the SST experiments, you can run:
```
./bin/learn2compose_sst --dynet-mem 1024 \
    --dynet-seed 1234 \
    -T ./data/sst/train.plain.2class.txt \
    -d ./data/sst/dev.plain.2class.txt \
    -t ./data/sst/test.plain.2class.txt \
    -w ./data/glove/glove.6B.100d.txt.sst_filtered \
    --word_dim 100 \
    --hidden_dim 200 \
    --optimizer_enable_clipping true \
    --optimizer_enable_eta_decay true \
    --max_iter 200
```

### Results

| Hyperparameters | Dev | Test |
|-----|-----|-----|
|seed=1234, l2=0| 83.6 | 80.9 |
