#! /bin/bash

cut -f 1 -d$'\t' ../../data/popular-names.txt | LANG=C sort | uniq
