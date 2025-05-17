import re
import os
import json
from pathlib import Path
import pandas as pd
from utils import read_uk_jsonlines_gz


path = os.environ.get("DATA_DIR")
filename = Path("jawiki-country.json.gz")

filepath = path / filename


def rm_mediawiki_markup(text: str) -> str:
    """Remove MediaWiki markup from text.

    Parameters
    ----------
    text : str
        Inputted text.

    Returns
    -------
    str
        Text removed markup.
    """
    p0 = re.compile("'+")  # 強調 -> '''aaa'''
    p1 = re.compile("\[\[(.*?)(?:\|.*?)?\]\]")  # [[aaa]]
    p2 = re.compile("\[\[(.+\||)(.+?)\]\]")  # [[aaa|bbb]], # [[aaa#xxx|bbb]]
    p3 = re.compile("\{\{(.+\||)(.+?)\}\}")  # {{aaa|bbb}}, # {{aaa#xxx|bbb}}
    p4 = re.compile("<\s*?/*?\s*?br\s*?/*?\s*>")  # <br>, <br />
    p5 = re.compile("<\s*?/*?\s*?ref\s*?(.*?)/*?\s*>")  # <ref>, <ref name="aaa" />

    text = p0.sub("", text)
    text = p1.sub(r"\1", text)
    text = p2.sub(r"\2", text)
    text = p3.sub(r"\2", text)
    text = p4.sub("", text)
    text = p5.sub("", text)

    return text


# 基礎情報
p = re.compile("\|(.+?)\s=\s*(.+)")

# # json
uk_json = json.loads(read_uk_jsonlines_gz(filepath))
uk_text = uk_json.get("text").split("\n")
ans = {}
for line in uk_text:
    r = re.search(p, line)
    if r:
        stripped = rm_mediawiki_markup(r[2])
        ans[r[1]] = stripped
print(ans)

# pandas
df = pd.read_json(filepath, lines=True)
uk_df = df.query('title=="イギリス"')
uk_text = uk_df["text"].values[0].split("\n")
ans = {}
for line in uk_text:
    r = re.search(p, line)
    if r:
        stripped = rm_mediawiki_markup(r[2])
        ans[r[1]] = stripped
print(ans)
