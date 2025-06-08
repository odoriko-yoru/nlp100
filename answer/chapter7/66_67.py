"""混同行列の作成."""

import pickle as pkl
from collections import defaultdict

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# tokenizer
with open("sst2_tokenizer.pkl", "rb") as f:
    tokenizer = pkl.load(f)

# 学習済モデル
with open("logistic_regression_sst2.pkl", "rb") as f:
    clf = pkl.load(f)

# 学習用データ
with open("sst2_train_feature.pkl", "rb") as f:
    train_feat = pkl.load(f)

# 検証用データ
with open("sst2_val_feature.pkl", "rb") as f:
    val_feat = pkl.load(f)


# No.66 混同行列の作成
dataset = {"train": train_feat, "validation": val_feat}

result = defaultdict(dict)

for key, feat in dataset.items():
    X = [f["feature"] for f in feat]
    X = tokenizer.transform(X)

    y = [f["label"] for f in feat]

    pred = clf.predict(X)
    cm = confusion_matrix(y, pred, labels=["0", "1"])
    result[key] = {"y_true": y, "y_pred": pred, "confusion_matrix": cm}

# No.67 精度の測定
# accuracy = (tn + tp) / cm.sum()
# precision = tp / (tp + fp)
# recall = tp / (tp + fn)
# f1_score = (2 * precision * recall) / (precision + recall)

for key, r in result.items():
    cm = r["confusion_matrix"]
    y_true = r["y_true"]
    y_pred = r["y_pred"]

    # 評価指標の計算
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="1")
    recall = recall_score(y_true, y_pred, pos_label="1")
    f1 = f1_score(y_true, y_pred, pos_label="1")

    print(
        f"""
{"学習データ" if key == "train" else "検証データ"}
{r["confusion_matrix"]}

accuracy   : {acc: .4f}
precision  : {precision: .4f}
recall     : {recall: .4f}
f1_score   : {f1: .4f}
"""
    )

print("(positiveを正例、negativeを負例とした二値分類問題と仮定したときのmetrics)")
