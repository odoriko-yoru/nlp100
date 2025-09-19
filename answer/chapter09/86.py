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

    # 冒頭4例をミニバッチとしてtokenize
    batch_size = 4
    train_tokens = tokenize(tokenizer, train_sentence[:batch_size])
    dev_tokens = tokenize(tokenizer, dev_sentence[:batch_size])  # noqa: F841

    # tokenの表示、すべて同じtoken長になっているはず
    for i in range(batch_size):
        token = train_tokens[i].tokens
        print(f"{i + 1}: トークン長 {len(token)}")


if __name__ == "__main__":
    main()
