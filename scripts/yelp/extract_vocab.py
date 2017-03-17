#!/usr/bin/env python
from __future__ import print_function
import sys

freq = int(sys.argv[2])
word_count = {}
data = ''
for line in open(sys.argv[1], 'r'):
    if len(line.strip()) == 0:
        # skip the label
        for word in data.split()[1:]:
            if word not in word_count:
                word_count[word] = 0
            word_count[word] += 1
        data = ''
    else:
        data += line

if len(data):
    for word in data.split():
        if word not in word_count:
            word_count[word] = 0
        word_count[word] += 1

for key in word_count:
    if word_count[key] > freq:
        print(key)
