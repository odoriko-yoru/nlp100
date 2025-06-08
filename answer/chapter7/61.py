"""特徴ベクトル."""

import os
import pickle as pkl
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

import pandas as pd


class TextFeat(TypedDict):
    text: str
    label: str
    feature: defaultdict[str, int]


def create_feature_vector(sentence: str, label: str) -> TextFeat:
    """Create feture vector.

    Parameters
    ----------
    sentence : str
        Sentence.
    label : str
        Label.

    Returns
    -------
    TextFeat
    """
    feature = defaultdict(int)
    for word in sentence.split():
        feature[word] += 1
    return {"text": sentence, "label": label, "feature": feature}


data_dir = os.environ.get("DATA_DIR", "")
data_dir = Path(data_dir)

train_file = "SST-2/train.tsv"
val_file = "SST-2/dev.tsv"

train_df = pd.read_csv(data_dir / train_file, sep="\t")
val_df = pd.read_csv(data_dir / val_file, sep="\t")

# 特徴量ベクトルの作成
train_feat = []
for _, row in train_df.iterrows():
    train_feat.append(create_feature_vector(row["sentence"], str(row["label"])))

val_feat = []
for _, row in val_df.iterrows():
    val_feat.append(create_feature_vector(row["sentence"], str(row["label"])))

# 62で利用するために保存
with open("sst2_train_feature.pkl", "wb") as f:
    pkl.dump(train_feat, f)

with open("sst2_val_feature.pkl", "wb") as f:
    pkl.dump(val_feat, f)

print(train_feat[0])
