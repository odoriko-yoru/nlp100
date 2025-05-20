"""TD-IDFの算出."""

from collections import Counter, defaultdict
import math
import os
import json
import gzip
from pathlib import Path
import MeCab
import unidic_lite
from mecab_models import MeCabToken
from utils import remove_markup


# TD-IDFの定式化
# N件の文書からなるコーパスにおいて
# ある文書dにおける単語xの出現頻度をTF(x, d)、単語xが出現する文書の数をDF(x)で書くことにすると
# 文書dにおける単語xのTF-TDFスコアは次式で定義される
# TF-IDF(x, d) = TF(x, d) x log(N/DF(x))

# References
# ==========
# 1. 岡崎直観 他, 自然言語処理の基礎(2023) オーム社
# 2. https://github.com/upura/nlp100v2025/blob/update-v2025/ch04/ans38.py


path = os.environ.get("DATA_DIR", "")
path = Path(path)

filename = Path("jawiki-country.json.gz")
filepath = path / filename

# MeCabの初期化
# https://www.kikagaku.co.jp/kikagaku-blog/practice-morphological-analysis/
# 辞書はunidic_liteを利用する
mecab = MeCab.Tagger(f"-d {unidic_lite.DICDIR}")

# 文書数をカウントするための変数(N)
total_docs = 0
# 各名詞が出現する文書数をカウント(DF(x))
doc_freq = defaultdict(int)
# 日本に関する記事の名詞の出現頻度をカウント(x)
japan_noun_freq = Counter()

with gzip.open(filepath, "rt", encoding="utf-8") as f:
    for line in f:
        # Nの更新
        total_docs += 1
        article = json.loads(line)
        text = article["text"].strip()

        # マークアップを除去
        text = remove_markup(text)

        # パーサーインスタンスの作成
        parsed = mecab.parse(text)

        # token化したblockをdataclassに保存
        blocks = parsed.split("\n")
        blocks = list(filter(lambda x: x != "", blocks))

        tokens = [
            MeCabToken(block.split("\t"))
            for block in blocks
            if block != "EOS" or not block
        ]

        # 文書で出現した単語を記録 -> DF(x)の更新のため
        doc_nouns = set()
        for t in tokens:
            if t.split_pos()[0] == "名詞":
                noun = t.surface
                doc_nouns.add(t.surface)

                # "日本"の文書内の単語出現頻度を記録
                # TF(x, d)
                if article["title"] == "日本":
                    japan_noun_freq[noun] += 1

        # DF(x)の更新
        for noun in doc_nouns:
            doc_freq[noun] += 1

# TF-IDF scoreの計算
# "日本"文書の各単語についてTF, IDF, TF-IDFを計算
tfidf = {}
for noun, tf in japan_noun_freq.items():
    # TF-IDF(x, d) = TF(x, d) x log(N/DF(x))
    idf = math.log(tf / doc_freq[noun])
    tf_idf = tf * idf
    tfidf[noun] = {"TF": tf, "IDF": idf, "TF-IDF": tf_idf}

# TF-IDFがTOP20の単語を表示
for noun, score in sorted(tfidf.items(), key=lambda x: x[1]["TF-IDF"], reverse=True)[
    :20
]:
    print(
        f"{noun}\tTF:{score['TF']}\tIDF:{score['IDF']:.4f}\tTF-IDF:{score['TF-IDF']:.4f}"
    )
