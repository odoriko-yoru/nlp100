"""名詞の出現頻度."""

from collections import Counter
import os
import json
import gzip
from pathlib import Path
import MeCab
import unidic_lite
from mecab_models import MeCabToken
from utils import remove_markup


path = os.environ.get("DATA_DIR", "")
path = Path(path)
filename = Path("jawiki-country.json.gz")

filepath = path / filename

# MeCabの初期化
# https://www.kikagaku.co.jp/kikagaku-blog/practice-morphological-analysis/
# 辞書はunidic_liteを利用する
mecab = MeCab.Tagger(f"-d {unidic_lite.DICDIR}")
word = []
with gzip.open(filepath, "rt", encoding="utf-8") as f:
    for line in f:
        article = json.loads(line)
        text = article["text"].strip()

        # マークアップを除去
        text = remove_markup(text)

        # パーサーインスタンスの作成
        parsed = mecab.parse(text)

        # token化したblockをdataclassに保存する
        blocks = parsed.split("\n")
        blocks = list(filter(lambda x: x != "", blocks))

        tokens = [
            MeCabToken(block.split("\t"))
            for block in blocks
            if block != "EOS" or not block
        ]

        for t in tokens:
            if t.split_pos()[0] == "名詞":
                word.append(t.surface)

    count = Counter(word)
    for word, word_count in count.most_common(20):
        print(f"{word}:{word_count}")
