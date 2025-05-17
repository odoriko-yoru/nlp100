import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

path = os.environ.get("DATA_DIR")
filename = Path("popular-names.txt")

df = pd.read_csv(path / filename, sep="\t", header=None)

n_split = int(sys.argv[1])

idx = np.linspace(0, len(df), n_split, dtype=int)

for i in range(len(idx) - 1):
    print(df.iloc[idx[i] : idx[i + 1]].to_csv(header=None))
