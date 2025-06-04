"""k-meansクラスタリング."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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

# k-means clustering
X = []
for i in countries:
    X.append(model[i])
X = np.array(X)

label = KMeans(n_clusters=5, random_state=20250604).fit_predict(X, y=countries)

# 可視化
reduced_coor = PCA(n_components=2).fit_transform(X)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

ax = sns.scatterplot(x=reduced_coor[:, 0], y=reduced_coor[:, 1], hue=label, palette="Set1")

# 各点に国名を追加
for i, name in enumerate(countries):
    ax.annotate(name, (reduced_coor[i, 0], reduced_coor[i, 1]), xytext=(5, 5), textcoords="offset points")

# 保存
fig.savefig("k-means-countries.jpg")
plt.clf()
plt.close()
