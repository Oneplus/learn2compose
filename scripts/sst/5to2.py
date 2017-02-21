#!/usr/bin/env python
# convert fine grain classification to binary classification problem
# neutral is removed.
import sys
for line in open(sys.argv[1], 'r'):
	label, sentence = line.strip().split('\t')
	if label == '2':
		continue
	elif label in ('3', '4'):
		label = '1'
	else:
		label = '0'
	print '{0}\t{1}'.format(label, sentence)