"""加法構成性によるアナロジー."""

import os
from pathlib import Path

from gensim.models import KeyedVectors

data_dir = os.environ.get("DATA_DIR", "")
data_dir = Path(data_dir)
file = Path("GoogleNews-vectors-negative300.bin.gz")

data_path = data_dir / file

# Ref: https://radimrehurek.com/gensim/models/keyedvectors.html#how-to-obtain-word-vectors
wv_from_bin = KeyedVectors.load_word2vec_format(data_path, binary=True)

# top10 cosin similarity
print(wv_from_bin.most_similar(positive=["Spain", "Athens"], negative=["Madrid"], topn=10))

# note
# 文章の指示通り "Spain_vec - Madrid_vec + Athens_vec" を計算して
# most_similarメソッドに渡しても類似度の高い単語を取得できるが、計算対象に入力単語(Spain, Madrid, Athens)も含まれる。
# そこで、positive, negative引数に単語を渡すと結果からinputの単語を除いてくれる。
# Ref : https://github.com/piskvorky/gensim/blob/develop/gensim/models/keyedvectors.py#L853
