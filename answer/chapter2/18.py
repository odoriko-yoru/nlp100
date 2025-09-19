"""各行の1列目の文字列の出現頻度を求め、出現頻度の高い順に並べる."""

import os
from pathlib import Path

import pandas as pd

path = os.environ.get("DATA_DIR")
filename = Path("popular-names.txt")

df = pd.read_csv(path / filename, sep="\t", header=None)

print(df.iloc[:, 0].value_counts())
