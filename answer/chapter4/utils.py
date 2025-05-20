"""Loading method of json.gz file."""

from typing import Union, List
from pathlib import Path
import gzip
import json


def read_jsonlines_gz(file_path: Union[str, Path]) -> List[str]:
    """Read a gzipped JSON lines file.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the gzipped JSON linnes file.

    Returns
    -------
    str
        The text of the article.
    """
    text = []
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 空行をスキップ
                article = json.loads(line)
                text.append(article.get("text"))

    return text
