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
	label, raw_document = line.strip().split(',', 1)
	label = int(label[1:-1]) - 1
	print(label, file=fpo_lab)
	raw_document = raw_document[1:-1]
	raw_document = raw_document.replace('\\""', '"')
	raw_paragraphs = raw_document.split('\\n')
	for raw_paragraph in raw_paragraphs:
		if len(raw_paragraph) == 0:
			continue
		raw_sentences = sentence_tokenizer.tokenize(raw_paragraph)
		sentence = []
		for raw_sentence in raw_sentences:
			if len(raw_sentence) == 0:
				continue
			print(raw_sentence.decode('unicode_escape').encode('utf-8'), file=fpo_doc)
	did += 1
	print(file=fpo_doc)