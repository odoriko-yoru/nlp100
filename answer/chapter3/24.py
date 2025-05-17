import re
import os
import json
from pathlib import Path
import pandas as pd
from utils import read_uk_jsonlines_gz


path = os.environ.get("DATA_DIR")
filename = Path("jawiki-country.json.gz")

filepath = path / filename

# json
uk_json = json.loads(read_uk_jsonlines_gz(filepath))
uk_text = uk_json.get("text")

pattern = re.compile("\[\[ファイル:(.*?)\|")
for i in re.findall(pattern, uk_text):
    print(i)

# pandas
df = pd.read_json(filepath, lines=True)
uk_df = df.query('title=="イギリス"')
uk_text = uk_df["text"].values[0]

for i in re.findall(pattern, uk_text):
    print(i)
