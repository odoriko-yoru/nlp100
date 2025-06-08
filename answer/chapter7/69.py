"""正則化パラメータの変更."""

import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

with open("sst2_train_feature.pkl", "rb") as f:
    train_feat = pkl.load(f)

with open("sst2_val_feature.pkl", "rb") as f:
    val_feat = pkl.load(f)

# one-hot vector特徴量を作成する
tokenizer = DictVectorizer(sparse=False)

# train
train_X = [f["feature"] for f in train_feat]
train_X = tokenizer.fit_transform(train_X)

train_y = [f["label"] for f in train_feat]

# validation
val_X = [f["feature"] for f in val_feat]
val_X = tokenizer.transform(val_X)

val_y = [f["label"] for f in val_feat]

# [0.001, 1000]の範囲
C = np.logspace(-3, 3, 100)

acc = []

for c in tqdm(C):
    # 学習
    # L2正則化を追加
    clf = LogisticRegression(penalty="l2", C=c, max_iter=5000, random_state=20250606)

    clf.fit(train_X, train_y)

    # val_Xで予測
    val_pred = clf.predict(val_X)

    # 正解率を計算
    accuracy = (val_pred == val_y).mean()
    acc.append(accuracy)

df = pd.DataFrame({"C": C, "accuracy": acc})

# plot
plt.figure(figsize=(10, 6))

sns.lineplot(data=df, x="C", y="accuracy", marker="o", markersize=4)

plt.xscale("log")
plt.xlabel("regularization parameter C")
plt.ylabel("Accuracy")
plt.title("Changes regularization parameter")
plt.tight_layout()
plt.savefig("regularization_parameter_accuracy.png")
plt.clf()
plt.close()
