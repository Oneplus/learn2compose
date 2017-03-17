#!/usr/bin/env python
from __future__ import print_function
import sys
import numpy
import codecs
numpy.random.seed(1234)

n_doc = 0
# first turn: get the number of documents
with codecs.open(sys.argv[1], "r", encoding='utf-8') as fi:
    for line in fi:
        line = line.strip()
        if len(line) == 0:
            n_doc += 1

n_doc_sampled = int(n_doc * 0.1)
sampled = numpy.random.choice(n_doc, n_doc_sampled)

doc = []
n_doc = 0
with codecs.open(sys.argv[1], "r", encoding='utf-8') as fi, codecs.open(sys.argv[2], "w", encoding='utf-8') as fo:
    for line in fi:
        line = line.strip()
        if len(line) == 0:
            if n_doc in sampled:
                docstr = u'\n'.join(doc)
                print(docstr, end='\n\n', file=fo)
            doc = []
            n_doc += 1
        else:
            doc.append(line)
