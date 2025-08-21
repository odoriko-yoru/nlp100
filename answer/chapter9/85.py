"""データセットの準備."""

import os
from pathlib import Path
from typing import List

import pandas as pd
from tokenizers import Encoding
from transformers import AutoTokenizer

# 環境変数の読み込み
DATA_DIR = os.environ.get("DATA_DIR", "")


def tokenize(tokenizer, sentences: List[str]) -> List[Encoding]:
    """Tokenize inuptted sentences.

    Parameters
    ----------
    tokenizer :
        Pretrained BERT model tokenizer.
    sentences : List[str]
        Sentences to tokenize.

    Returns
    -------
    List[Encoding] :
        Result of tokenize.
    """
    # https://huggingface.co/docs/transformers/ja/pad_truncation
    return tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)


def main() -> None:
    data_dir = Path(DATA_DIR)

    # Datasetの読み込み
    train_df = pd.read_csv(data_dir / "SST-2/train.tsv", sep="\t")
    dev_df = pd.read_csv(data_dir / "SST-2/dev.tsv", sep="\t")

    # 多言語事前学習済モデル
    # https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment
    model_id = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 文章、ラベルを取得
    train_sentence = train_df["sentence"].tolist()
    train_labels = train_df["label"].tolist()  # noqa: F841

    dev_sentence = dev_df["sentence"].tolist()
    dev_labels = dev_df["label"].tolist()  # noqa: F841

    # tokenize
    train_tokens = tokenize(tokenizer, train_sentence)
    dev_tokens = tokenize(tokenizer, dev_sentence)  # noqa: F841

    # 1例のみ表示
    print(train_tokens[0].tokens)


if __name__ == "__main__":
    main()
