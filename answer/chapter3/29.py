"""国旗画像のURLを取得する."""

import json
import os
import re
from pathlib import Path

import pandas as pd
import requests
from utils import read_uk_jsonlines_gz


def get_url(dc: dict) -> str:
    """Get URL via MediaWiki API.

    Parameters
    ----------
    dc : dict
        Dict.

    Returns
    -------
    str
        URL.
    """
    url_file = dc["国旗画像"].replace(" ", "_")
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query&titles=File:"
        + url_file
        + "&prop=imageinfo&iiprop=url&format=json"
    )
    data = requests.get(url)
    return re.search(r'"url":"(.+?)"', data.text).group(1)


def main() -> None:
    path = os.environ.get("DATA_DIR")
    filename = Path("jawiki-country.json.gz")

    filepath = path / filename

    # 基礎情報
    p = re.compile(r"\|(.+?)\s=\s*(.+)")

    # json
    uk_json = json.loads(read_uk_jsonlines_gz(filepath))
    uk_text = uk_json.get("text").split("\n")
    ans = {}
    for line in uk_text:
        r = re.search(p, line)
        if r:
            ans[r[1]] = r[2]
    print(get_url(ans))

    # pandas
    df = pd.read_json(filepath, lines=True)
    uk_df = df.query('title=="イギリス"')
    uk_text = uk_df["text"].values[0].split("\n")
    ans = {}
    for line in uk_text:
        r = re.search(p, line)
        if r:
            ans[r[1]] = r[2]

    print(get_url(ans))


if __name__ == "__main__":
    main()
