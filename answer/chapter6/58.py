"""Ward法によるクラスタリング."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from utils import load_analogy_section


# Ref: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
def plot_dendrogram(model, labels=None, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, labels=labels, **kwargs)


def main() -> None:
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

    model = AgglomerativeClustering(linkage="ward", distance_threshold=0, n_clusters=None)
    model = model.fit(X)

    # plot
    plt.figure(figsize=(22, 12))
    plt.title("Hierarchical Clustering Dendrogram", fontsize=16)
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, labels=countries, truncate_mode="level")
    plt.xlabel("Country", fontsize=14)
    plt.xticks(rotation=90, ha="right", fontsize=12)
    plt.title("Hierarchical clustering")
    plt.tight_layout()

    # 保存
    plt.savefig("hierarchical-clustering-countries.jpg")
    plt.clf()
    plt.close()


if __name__ == "__main__":
    main()
