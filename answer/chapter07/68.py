"""特徴量の重みの確認."""

import pickle as pkl

# 学習済モデル
with open("logistic_regression_sst2.pkl", "rb") as f:
    clf = pkl.load(f)

# tokenizer
with open("sst2_tokenizer.pkl", "rb") as f:
    tokenizer = pkl.load(f)


feat_name = tokenizer.get_feature_names_out()
coef = clf.coef_[0]

weight = list(zip(coef, feat_name, strict=False))
weight_sorted = sorted(weight, key=lambda x: x[0])

# Hight weight - top20
print(weight_sorted[-20:][::-1])

# Low weight - top20
print(weight_sorted[:20])
