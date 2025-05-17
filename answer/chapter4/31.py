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
blocks = list(filter(lambda x: x != "", blocks))

# token化したblockをdataclassに保存する
tokens = [MeCabToken(block.split("\t")) for block in blocks if block != "EOS"]

for t in tokens:
    # 品詞が動詞の場合だけ表示
    if "動詞" == t.split_pos()[0]:
        print(f"動詞：{t.surface}\t原型：{t.base}")
