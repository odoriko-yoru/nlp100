"""予測."""

import pickle as pkl

# tokenizer
with open("sst2_tokenizer.pkl", "rb") as f:
    tokenizer = pkl.load(f)

# 学習済モデル
with open("logistic_regression_sst2.pkl", "rb") as f:
    clf = pkl.load(f)

# 検証用データ
with open("sst2_val_feature.pkl", "rb") as f:
    val_feat = pkl.load(f)

val_X = [f["feature"] for f in val_feat]
val_X = tokenizer.transform(val_X)

val_y = [f["label"] for f in val_feat]

# 予測
val_pred = clf.predict(val_X)

# 正解率
print(
    f"""
    先頭の検証データのラベル
    Predict:       {val_pred[0]}
    GroundTruth:   {val_y[0]}
    """
)
