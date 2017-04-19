#!/usr/bin/env python
# convert fine grain classification to binary classification problem
# neutral is removed.
import sys

def convert(token):
    if token == '2':
        return 1
    elif token in ('3', '4'):
        return 2
    else:
        return 0

for line in open(sys.argv[1], 'r'):
    label, rest = line.strip().split('\t', 1)
    if '\t' in rest:
        sentence, rest = rest.split('\t', 1)
        tokens = rest.split()
        rest = ' '.join(['{0}:{1}'.format(token.split(':')[0], convert(token.split(':')[1])) for token in tokens])
        rest = '{0}\t{1}'.format(sentence, rest)
    if label == '2':
        continue
    elif label in ('3', '4'):
        label = '2'
    else:
        label = '0'
    print '{0}\t{1}'.format(label, rest)
