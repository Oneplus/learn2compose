#!/usr/bin/env python
from __future__ import print_function
import sys
import argparse


def main():
    cmd = argparse.ArgumentParser('')
    cmd.add_argument('--freq', default=0, type=int, required=True, help='the minimal frequency.')
    cmd.add_argument('--lowercase', default=False, action='store_true', help='lowercase')
    cmd.add_argument('files', nargs='*', help='the path to the files')
    args = cmd.parse_args()
    word_count = {}

    for f in args.files:
        for line in open(f, 'r'):
            tokens = line.strip().split()
            for word in tokens[1:]:
                if args.lowercase:
                    word = word.lower()
                if word not in word_count:
                    word_count[word] = 0
                word_count[word] += 1

    for key, freq in sorted(word_count.items()):
        if freq > args.freq:
            print(key)

if __name__ == "__main__":
    main()
