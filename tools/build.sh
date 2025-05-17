#!/bin/sh
rsync -avh data _build/html/
jupyter-book build --config ja/_config.yml --toc ja/_toc.yml .
cp ja/images/nlp100-2025-ja.jpg _build/html/_images
python ./tools/modify.py
