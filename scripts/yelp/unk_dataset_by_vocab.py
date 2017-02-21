#!/usr/bin/evn python
from __future__ import print_function
import sys

vocab = set([line.strip() for line in open(sys.argv[1], 'r')])
data = ''
for line in open(sys.argv[2], 'r'):
	if len(line.strip()) == 0:
		lines = data.strip().split('\n')
		print(lines[0])
		for sentence in lines[1:]:
			words = [word if word in vocab else '_UNK_' for word in sentence.split()]
			print(' '.join(words))
		print()
		data = ''
	else:
		data += line