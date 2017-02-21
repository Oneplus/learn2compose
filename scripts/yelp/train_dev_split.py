#!/usr/bin/env python
from __future__ import print_function
import sys
import random

random.seed(1234)

n_lines = 0
fp = open(sys.argv[1], 'r')
for line in fp:
	if len(line.strip()) == 0:
		n_lines += 1
fp.close()

n_samples = int(n_lines * 0.1)
samples = set(random.sample(range(n_lines), n_samples))

n_lines = 0
fp_train = open(sys.argv[1] + '.train', 'w')
fp_dev = open(sys.argv[1] + '.dev', 'w')

data = ''
for line in open(sys.argv[1], 'r'):
	if len(line.strip()) == 0:
		if n_lines in samples:
			print(data, file=fp_dev)
		else:
			print(data, file=fp_train)
		n_lines += 1
		data = ''
	else:
		data += line

if len(data) > 0:
	if n_lines in samples:
		print(data, file=fp_dev)
	else:
		print(data, file=fp_train)
	n_lines += 1