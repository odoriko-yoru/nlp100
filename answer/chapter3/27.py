import re
import os
import json
from pathlib import Path
import pandas as pd
from utils import read_uk_jsonlines_gz


path = os.environ.get("DATA_DIR")
filename = Path("jawiki-country.json.gz")

filepath = path / filename

p1 = re.compile("\|(.+?)\s=\s*(.+)")

# [[aaa]]
# [[aaa|bbb]]
# [[aaa#xxx|bbb]]
p2 = re.compile("\[\[(.+\||)(.+?)\]\]")

# json
uk_json = json.loads(read_uk_jsonlines_gz(filepath))
uk_text = uk_json.get("text").split("\n")
ans = {}
for line in uk_text:
    r = re.search(p1, line)
    if r:
        stripped = r[2].replace("'", "")
        stripped = p2.sub(r"\2", stripped)
        ans[r[1]] = stripped
print(ans)

# pandas
df = pd.read_json(filepath, lines=True)
uk_df = df.query('title=="イギリス"')
uk_text = uk_df["text"].values[0].split("\n")
ans = {}
for line in uk_text:
    r = re.search(p1, line)
    if r:
        stripped = r[2].replace("'", "")
        stripped = p2.sub(r"\2", stripped)
        ans[r[1]] = stripped
print(ans)
