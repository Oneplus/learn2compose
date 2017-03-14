#!/usr/bin/env python
from __future__ import print_function
import sys

data = ''
label_fpi = open(sys.argv[1], 'r')
document_fpi = open(sys.argv[2], 'r')

for line in document_fpi:
    if len(line.strip()) == 0:
        label = label_fpi.readline()
        assert len(label) > 0
        print(label.strip())
        data = data.lower().replace('-lrb-', '(').replace('-rrb-', ')')
        if len(data.strip()) == 0:
            data = ''
            continue
        print(data)
        data = ''
    else:
        data += line

if len(data) > 0:
    label = label_fpi.readline()
    assert len(label) > 0
    print(label.strip())
    print(data.strip().lower())
