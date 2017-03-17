#!/usr/bin/evn python
from __future__ import print_function
import sys
import time

vocab = set([line.strip() for line in open(sys.argv[1], 'r')])
data = ''


def is_digit(word):
    try:
        float(word)
        return True
    except ValueError:
        return False


def is_time(word):
    if word[0].isdigit() and word.endswith('am') or word.endswith('pm'):
        return True
    try:
        time.strptime(word, '%H:%M')
        return True
    except ValueError:
        try:
            time.strptime(word, '-%H:%M')
            return True
        except ValueError:
            return False


def is_date(word):
    try:
        time.strptime(word, '%m/%d/%y')
        return True
    except ValueError:
        try:
            time.strptime(word, '%m/%d')
            return True
        except ValueError:
            return False


for line in open(sys.argv[2], 'r'):
    if len(line.strip()) == 0:
        lines = data.strip().split('\n')
        print(lines[0])
        for sentence in lines[1:]:
            words = []
            for word in sentence.split():
                if word in vocab:
                    words.append(word)
                elif is_digit(word):
                    words.append('_NUM_')
                elif is_time(word):
                    words.append('_TIME_')
                elif is_date(word):
                    words.append('_DATE_')
                else:
                    words.append('_UNK_')
            print(' '.join(words))
        print()
        data = ''
    else:
        data += line
