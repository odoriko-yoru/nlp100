"""t-SNEによる可視化."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from utils import load_analogy_section

data_dir = os.environ.get("DATA_DIR", "")
data_dir = Path(data_dir)
wv = Path("GoogleNews-vectors-negative300.bin.gz")

dataset = data_dir / wv
model = KeyedVectors.load_word2vec_format(dataset, binary=True)

file = Path("questions-words.txt")

target_section = ["capital-common-countries", "capital-world"]

df = []
for target in target_section:
    df.append(load_analogy_section(data_dir / file, target))

df = pd.concat(df, axis=0)

# 国名の入った列を抽出
countries = pd.unique(df["word4"])

X = []
for i in countries:
    X.append(model[i])
X = np.array(X)

# t-SNE
tsne = TSNE(n_components=2, metric="cosine", random_state=20250604)
reduced_coor = tsne.fit_transform(X)

# plot
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

ax = sns.scatterplot(x=reduced_coor[:, 0], y=reduced_coor[:, 1])

# 各点に国名を追加
for i, name in enumerate(countries):
    ax.annotate(name, (reduced_coor[i, 0], reduced_coor[i, 1]), xytext=(3, 3), textcoords="offset points", fontsize=8)

# 保存
plt.title("t-SNE")
plt.tight_layout()
fig.savefig("t-SNE-countries.jpg")
plt.clf()
plt.close()
