#!/usr/bin/evn python
import sys

vocab = set([line.strip() for line in open(sys.argv[1], 'r')])

for line in open(sys.argv[2], 'r'):
    label, sentence = line.strip().split('\t')
    words = [word if word in vocab else '_UNK_' for word in sentence.split()]
    print '{0}\t{1}'.format(label, ' '.join(words))
