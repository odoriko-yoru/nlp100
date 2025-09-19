"""単語埋め込みの読み込み."""

import os
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors

data_dir = os.environ.get("DATA_DIR", "")
data_dir = Path(data_dir)
file = Path("GoogleNews-vectors-negative300.bin.gz")

data_path = data_dir / file

# Ref: https://radimrehurek.com/gensim/models/keyedvectors.html#how-to-obtain-word-vectors
wv_from_bin = KeyedVectors.load_word2vec_format(data_path, binary=True)

# Ref : https://github.com/piskvorky/gensim/blob/develop/gensim/models/keyedvectors.py#L241
idx_to_key = {0: "<PAD>"}
key_to_idx = {"<PAD>": 0}

# Ref: https://github.com/piskvorky/gensim/blob/develop/gensim/models/keyedvectors.py#L245
V, d_emb = wv_from_bin.vectors.shape

E = np.zeros((V + 1, d_emb))

for key, idx in wv_from_bin.key_to_index.items():
    idx += 1
    idx_to_key[idx] = key
    key_to_idx[key] = idx
    E[idx, :] = wv_from_bin[key]

# 最初の例を確認
print(f"idx_to_key[1]:\n{idx_to_key[1]}")
print(f"The shape of the embedding matrix: {E.shape}")
