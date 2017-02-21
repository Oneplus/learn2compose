#!/bin/bash

python ./scripts/sst/extract_lower_sst.py ./data/sst/train.txt > ./data/sst/train.lower.5class.txt
python ./scripts/sst/extract_lower_sst.py ./data/sst/dev.txt > ./data/sst/dev.lower.5class.txt
python ./scripts/sst/extract_lower_sst.py ./data/sst/test.txt > ./data/sst/test.lower.5class.txt
python ./scripts/sst/5to2.py ./data/sst/train.lower.5class.txt > ./data/sst/train.lower.2class.txt
python ./scripts/sst/5to2.py ./data/sst/dev.lower.5class.txt > ./data/sst/dev.lower.2class.txt
python ./scripts/sst/5to2.py ./data/sst/test.lower.5class.txt > ./data/sst/test.lower.2class.txt
python ./scripts/sst/extract_vocab.py ./data/sst/train.plain.2class.txt 1 > ./data/sst/train.unk_1.vocab
python ./scrtips/sst/unk_dataset_by_vocab.py ./data/sst/train.lower.2class.txt ./data/sst/train.unk_1.vocab > ./data/sst/train.lower_unk1.2class.txt
python ./scrtips/sst/unk_dataset_by_vocab.py ./data/sst/dev.lower.2class.txt ./data/sst/train.unk_1.vocab > ./data/sst/dev.lower_unk1.2class.txt
python ./scrtips/sst/unk_dataset_by_vocab.py ./data/sst/test.lower.2class.txt ./data/sst/train.unk_1.vocab > ./data/sst/test.lower_unk1.2class.txt
python ./scripts/sst/filter_wordvec.py ./data/glove/glove.6B.100d.txt ./data/sst/train.lower_unk1.2class.txt ./data/sst/dev.lower_unk1.2class.txt ./data/sst/test.lower_unk1.2class.txt > ./data/glove/glove.6B.100d.txt.sst_unk1_filtered
python ./scripts/sst/filter_wordvec.py ./data/glove/glove.6B.100d.txt ./data/sst/train.lower.5class.txt ./data/sst/dev.lower.5class.txt ./data/sst/test.lower.5class.txt > ./data/glove/glove.6B.100d.txt.sst_filtered
