#!/usr/bin/env python
import sys

vocab = set()
for f in sys.argv[2:]:
	for line in open(f, 'r'):
		words = line.strip().split()[1:]
		for word in words:
			vocab.add(word)

print 0, 100
for line in open(sys.argv[1], 'r'):
	word = line.strip().split()[0]
	if word in vocab:
		print line.strip()
