"""係り受け解析."""

import spacy

# model load
nlp = spacy.load("ja_ginza")
# テキストの読み込み
with open("sample.txt", "rt", encoding="utf-8") as f:
    text = f.read()

# 解析
doc = nlp(text)

for token in doc:
    if token.dep_ != "ROOT":
        print(f"{token.text}\t{token.head.text}")
