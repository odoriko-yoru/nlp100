"""Wikipedia記事の「イギリス」に関するJSONファイルを読み込む."""

from typing import Union
from pathlib import Path
import gzip
import json


def read_uk_jsonlines_gz(file_path: Union[str, Path]) -> str:
    """Read a gzipped JSON lines file about UK.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the gzipped JSON linnes file.

    Returns
    -------
    str
        The text of the article with title "イギリス".
    """
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 空行をスキップ
                article = json.loads(line)

                if article.get("title") == "イギリス":
                    return line

                else:
                    continue
