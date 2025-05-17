import os
from pathlib import Path
import pandas as pd

path = os.environ.get("DATA_DIR")
filename = Path("popular-names.txt")

df = pd.read_csv(path / filename, sep="\t", header=None)
print(df.head(10).to_csv(sep=" ", index=False, header=None))
