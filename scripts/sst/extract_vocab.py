#!/usr/bin/env python
from __future__ import print_function
import sys

freq = int(sys.argv[2])
word_count = {}
for line in open(sys.argv[1], 'r'):
    tokens = line.strip().split()
    for word in tokens[1:]:
        if word not in word_count:
            word_count[word] = 0
        word_count[word] += 1

for key in word_count:
    if word_count[key] > freq:
        print(key)
