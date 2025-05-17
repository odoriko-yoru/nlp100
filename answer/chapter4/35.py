"""係り受け解析."""

import spacy
from spacy import displacy

# model load
nlp = spacy.load("ja_ginza")

text = "メロスは激怒した。"

# 解析
doc = nlp(text)

# 可視化
# https://spacy.io/usage/visualizers
displacy.serve(doc, style="dep", port=8888)
