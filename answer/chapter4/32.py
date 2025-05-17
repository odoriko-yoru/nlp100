"""MeCabを使った形態素解析."""

import MeCab
import unidic_lite
from mecab_models import MeCabToken


# テキストの読み込み
with open("sample.txt", "rt", encoding="utf-8") as f:
    text = f.read()

# MeCabの初期化
# https://www.kikagaku.co.jp/kikagaku-blog/practice-morphological-analysis/
# 辞書はunidic_liteを利用する
mecab = MeCab.Tagger(f"-d {unidic_lite.DICDIR}")
parsed = mecab.parse(text)

# 前処理
blocks = parsed.split("\n")

# token化したblockをdataclassに保存する
tokens = [
    MeCabToken(block.split("\t")) for block in blocks if block != "EOS" or not block
]

for i, t in enumerate(tokens):
    if (
        t.surface == "の"
        and "名詞" == tokens[i - 1].split_pos()[0]
        and "名詞" == tokens[i + 1].split_pos()[0]
    ):
        n1 = tokens[i - 1].surface
        n3 = tokens[i + 1].surface
        print(f"{n1}{t.surface}{n3}")
