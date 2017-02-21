#!/usr/bin/env python
from __future__ import print_function
import sys

data = ''
label_fpi = open(sys.argv[1], 'r')
document_fpi = open(sys.argv[2], 'r')

for line in document_fpi:
	if len(line.strip()) == 0:
		assert label_fpi
		label = label_fpi.readline()
		print(label.strip())
		data = data.lower().replace('-lrb-', '(').replace('-rrb-', ')')
		print(data)
		data = ''
	else:
		data += line

if len(data) > 0:
	assert label_fpi
	label = label_fpi.readline()
	print(label.strip())
	print(data.strip().lower())
	print()
