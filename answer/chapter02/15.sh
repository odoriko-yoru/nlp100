#! /bin/bash

n=$1

mkdir output

split -l $n ../../data/popular-names.txt output/15_split_
