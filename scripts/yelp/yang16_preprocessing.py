#!/usr/bin/evn python
from __future__ import print_function
import sys

vocab = set([line.strip() for line in open(sys.argv[1], 'r')])
unk = set([line.strip() for line in open(sys.argv[2], 'r')])
data = ''
for line in open(sys.argv[3], 'r'):
    if len(line.strip()) == 0:
        lines = data.strip().split('\n')
        print(lines[0])
        for sentence in lines[1:]:
            words = []
            for word in sentence.split():
                if word in vocab:
                    words.append(word)
                elif word in unk:
                    words.append('UNK')
            if len(words) > 0:
                print(' '.join(words))
        print()
        data = ''
    else:
        data += line
