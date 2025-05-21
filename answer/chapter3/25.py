import re
import os
import json
from pathlib import Path
import pandas as pd
from utils import read_uk_jsonlines_gz


path = os.environ.get("DATA_DIR")
filename = Path("jawiki-country.json.gz")

filepath = path / filename

# 基礎情報の構造
# {{基礎情報 国
# |略名  =イギリス
# |日本語国名 = グレートブリテン及び北アイルランド連合王国
# |公式国名 = {{lang|en|United Kingdom of Great Britain and Northern Ireland}}<ref>英語以外での正式国名:<br />
# *{{lang|gd|An Rìoghachd Aonaichte na Breatainn Mhòr agus Eirinn mu Thuath}}（[[スコットランド・ゲール語]]）
# |xxx = bbb
# }}
pattern = re.compile(r"\|(.+?)\s=\s*(.+)")

# json
uk_json = json.loads(read_uk_jsonlines_gz(filepath))
uk_text = uk_json.get("text").split("\n")
ans = {}
for line in uk_text:
    r = re.search(pattern, line)
    if r:
        ans[r[1]] = r[2]
print(ans)

# pandas
df = pd.read_json(filepath, lines=True)
uk_df = df.query('title=="イギリス"')
uk_text = uk_df["text"].values[0].split("\n")
ans = {}
for line in uk_text:
    r = re.search(pattern, line)
    if r:
        ans[r[1]] = r[2]
print(ans)
