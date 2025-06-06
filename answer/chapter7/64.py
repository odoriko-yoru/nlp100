"""条件付き確率."""

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

# 予測
val_pred_label = clf.predict(val_X)
val_pred_proba = clf.predict_proba(val_X)

# 正解率
print(
    f"""
    先頭の検証データの条件付き確率
    label 0: {val_pred_proba[0, 0]: .4f}
    label 1: {val_pred_proba[0, 1]: .4f}
    """
)
