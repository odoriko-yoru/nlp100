"""類似度の高い単語10件."""

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
print(wv_from_bin.most_similar(["United_States"], topn=10))
