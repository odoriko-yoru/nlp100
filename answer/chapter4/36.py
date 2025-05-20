"""Frequency of words."""

from typing import Union
import os
from pathlib import Path
import json
import gzip
import re
import MeCab
from collections import Counter


def remove_markup(text):
    # | の除去
    text = re.sub(r"\|\s?", "", text)
    # indentの除去
    text = re.sub(r"\*+\s?", "", text)
    # 強調マークアップの除去
    text = re.sub(r"\'{2,5}", "", text)
    text = re.sub(r"^(\:|\;)", "", text)
    text = re.sub(r"^\*{2,5}", "", text)
    text = re.sub(r"=+\s?(.*?)\s?=+", r"\1", text)
    # 内部リンクの除去
    text = re.sub(r"\[\[(?:[^|\]]*?\|)??([^|\]]+?)\]\]", r"\1", text)
    # 外部リンクの除去
    text = re.sub(r"\[http://[^\]]+\]", "", text)
    # HTMLタグの除去
    text = re.sub(r"<[^>]+>", "", text)
    # テンプレートの除去
    text = re.sub(r"\{\{.*?\}\}", "", text)
    return text


def analyze_word_frequency(file: Union[str, Path]) -> None:
    # MeCabの初期化
    mecab = MeCab.Tagger("-Owakati")

    # 単語の出現頻度をカウントするためのCounter
    word_counter = Counter()

    # gzipファイルを読み込む
    with gzip.open(file, "rt", encoding="utf-8") as f:
        for line in f:
            article = json.loads(line)
            text = article["text"]

            # マークアップを除去
            text = remove_markup(text)

            # 形態素解析を行い、単語をカウント
            words = mecab.parse(text).strip().split()
            word_counter.update(words)

    # 出現頻度の高い20語を表示
    for word, count in word_counter.most_common(20):
        print(f"{word}: {count}")


if __name__ == "__main__":
    path = os.environ.get("DATA_DIR", "")
    path = Path(path)
    filename = Path("jawiki-country.json.gz")

    filepath = path / filename

    analyze_word_frequency(filepath)
