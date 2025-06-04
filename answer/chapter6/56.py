"""WordSimilarity-353での評価."""

import os
from pathlib import Path

import pandas as pd
from gensim.models import KeyedVectors

data_dir = os.environ.get("DATA_DIR", "")
data_dir = Path(data_dir)
model_file = Path("GoogleNews-vectors-negative300.bin.gz")

data_path = data_dir / model_file

# Ref: https://radimrehurek.com/gensim/models/keyedvectors.html#how-to-obtain-word-vectors
wv_from_bin = KeyedVectors.load_word2vec_format(data_path, binary=True)

# WordSimilarity-353
file_path = data_dir / "wordsim353" / "combined.csv"
df = pd.read_csv(file_path)

# calc cosine similarity
df["word2vec_sim"] = df.apply(lambda row: wv_from_bin.similarity(row["Word 1"], row["Word 2"]), axis=1)

# calc spearman correlation
corr_spearman = df[["Human (mean)", "word2vec_sim"]].corr(method="spearman")
print(corr_spearman.iloc[0, 1])
