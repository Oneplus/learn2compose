#!/usr/bin/env python
from __future__ import print_function
import sys

data = ''
for line in open(sys.argv[1], 'r'):
    if len(line.strip()) == 0:
        # skip the label
        lines = data.splitlines()
        print('\n'.join(lines[1:]))
        data = ''
    else:
        data += line
