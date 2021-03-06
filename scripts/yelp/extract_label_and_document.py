#!/usr/bin/env python
from __future__ import unicode_literals
from __future__ import print_function
import nltk.data
import sys
import json
import codecs

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

did = 0
fpo_lab = open(sys.argv[1] + '.lab', 'w')
fpo_doc = open(sys.argv[1] + '.doc', 'w')

for line in codecs.open(sys.argv[1], 'r', encoding='utf-8'):
    # type(line) => 'unicode'
    label, raw_document = line.strip().split(',', 1)
    raw_document = raw_document[1:-1]
    raw_document = raw_document.strip()
    raw_document = raw_document.replace(u'\\""', u'\\"')
    raw_document = raw_document.decode('unicode_escape')
    raw_document = raw_document.replace(u'\r', u'\n')
    raw_paragraphs = raw_document.split(u'\n')
    for raw_paragraph in raw_paragraphs:
        if len(raw_paragraph) == 0:
            continue
        raw_paragraph = raw_paragraph.strip()
        raw_sentences = sentence_tokenizer.tokenize(raw_paragraph)
        for raw_sentence in raw_sentences:
            if len(raw_sentence) == 0:
                continue
            assert u'\n' not in raw_sentence, 'document: {0}'.format(did)
            print(raw_sentence.encode('utf-8'), file=fpo_doc)
    did += 1
    print(file=fpo_doc)
    label = int(label[1:-1]) - 1
    print(label, file=fpo_lab)
