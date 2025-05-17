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
uk_text = uk_json.get("text").split("\n")
section = list(filter(lambda x: "==" in x, uk_text))

# name, level
ans = [[t.strip("="), t.count("=") // 2 - 1] for t in section]
print(ans)


# pandas
df = pd.read_json(filepath, lines=True)
uk_df = df.query('title=="イギリス"')
uk_text = uk_df["text"].values[0].split("\n")
section = list(filter(lambda x: "==" in x, uk_text))

# name, level
ans_ = [[t.strip("="), t.count("=") // 2 - 1] for t in section]
print(ans_)
