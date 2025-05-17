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
category = list(filter(lambda x: "[Category:" in x, uk_text))
ans = [
    text.replace("[[Category:", "").replace("|*", "").replace("]]", "")
    for text in category
]
print(ans)


# pandas
df = pd.read_json(filepath, lines=True)
uk_df = df.query('title=="イギリス"')
uk_text = uk_df["text"].values[0].split("\n")
category = list(filter(lambda x: "[Category:" in x, uk_text))
ans = [
    text.replace("[[Category:", "").replace("|*", "").replace("]]", "")
    for text in category
]
print(ans)
