import os
from pathlib import Path
import pandas as pd
from utils import read_uk_jsonlines_gz


path = os.environ.get("DATA_DIR")
filename = Path("jawiki-country.json.gz")

filepath = path / filename

# json
uk_json = read_uk_jsonlines_gz(filepath)
print(uk_json)

# pandas
df = pd.read_json(filepath, lines=True)
uk_df = df.query('title=="イギリス"')["text"].values[0]
print(uk_df)
