#!/usr/bin/env python
from __future__ import print_function
import sys

data = ''
vocab = set()

for line in open(sys.argv[2], 'r'):
    if len(line.strip()) == 0:
        for word in data.strip().split()[1:]:
            vocab.add(word)
        data = ''
    else:
        data += line

if len(data) > 0:
    for word in data.strip().split()[1:]:
        vocab.add(word)

print(0, 100)
for line in open(sys.argv[1], 'r'):
    word = line.strip().split()[0]
    if word in vocab:
        print(line.strip())
