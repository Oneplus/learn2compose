#!/usr/bin/evn python
from __future__ import print_function
import sys
from nltk.tree import Tree

for line in open(sys.argv[1], 'r'):
	data = Tree.fromstring(line.strip())
	print('{0}\t{1}'.format(data.label(), ' '.join(data.leaves())))
