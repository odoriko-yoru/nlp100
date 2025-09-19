"""データの入手・整形."""

import os
from pathlib import Path

import pandas as pd

data_dir = os.environ.get("DATA_DIR", "")
data_dir = Path(data_dir)

train_file = "SST-2/train.tsv"
val_file = "SST-2/dev.tsv"

train_df = pd.read_csv(data_dir / train_file, sep="\t")
val_df = pd.read_csv(data_dir / val_file, sep="\t")

print(
    f"""
学習データ
ポジティブ:\t{(train_df["label"] == 1).sum()}
ネガディブ:\t{(train_df["label"] == 0).sum()}

検証データ
ポジティブ:\t{(val_df["label"] == 1).sum()}
ネガディブ:\t{(val_df["label"] == 0).sum()}
"""
)
