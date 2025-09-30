#! /bin/bash

n=$1

mkdir output

split -n $n ../../data/popular-names.txt ./output/
