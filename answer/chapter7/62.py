"""学習."""

import pickle as pkl

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

with open("sst2_train_feature.pkl", "rb") as f:
    train_feat = pkl.load(f)

with open("sst2_val_feature.pkl", "rb") as f:
    val_feat = pkl.load(f)


# one-hot vector特徴量を作成する
tokenizer = DictVectorizer(sparse=False)

train_X = [f["feature"] for f in train_feat]
train_X = tokenizer.fit_transform(train_X)

train_y = [f["label"] for f in train_feat]

# 学習
clf = LogisticRegression(penalty=None, max_iter=5000, random_state=20250606)

clf.fit(train_X, train_y)

# 学習済モデルの保存
with open("logistic_regression_sst2.pkl", "wb") as f:
    pkl.dump(clf, f)

# tokenizerの保存
with open("sst2_tokenizer.pkl", "wb") as f:
    pkl.dump(tokenizer, f)
