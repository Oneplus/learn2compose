#!/usr/bin/env python
# convert the data set released by tang 2015 into adaptable format.
from __future__ import print_function
import sys
for line in open(sys.argv[1], 'r'):
    tokens = line.strip().split('\t')
    label = tokens[4]
    text = tokens[6]
    output = [str(int(label) - 1)]
    for sentence in text.split('<sssss>'):
        sentence = sentence.strip()
        if len(sentence) > 2:
            output.append(sentence)
    if len(output) > 1:
        print('\n'.join(output), end='\n\n')
