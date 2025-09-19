"""データセットの読み込み."""

import os
from pathlib import Path

import pandas as pd
import torch
from gensim.models import KeyedVectors


# 前処理用関数
def convert_to_token(row: pd.Series, key_to_idx: dict) -> pd.Series:
    """Convert inputted text and label to dict object.

    Parameters
    ----------
    row : pd.Series
        Row of the dataset.
    key_to_idx : dict
        Dictionary of word to index.
    Returns
    -------
    pd.Series
    """
    sentence = row["sentence"]
    label = row["label"]
    input_ids = []

    for w in sentence.lower().split():
        try:
            idx = key_to_idx[w]
            input_ids.append(idx)

        except KeyError:
            continue

    return pd.Series(
        [{"text": sentence, "label": torch.tensor(label), "input_ids": torch.tensor(input_ids)}, len(input_ids)]
    )


def main() -> None:
    data_dir = os.environ.get("DATA_DIR", "")
    data_dir = Path(data_dir)

    bin_file = Path("GoogleNews-vectors-negative300.bin.gz")

    train_file = "SST-2/train.tsv"
    val_file = "SST-2/dev.tsv"

    wv_from_bin = KeyedVectors.load_word2vec_format(data_dir / bin_file, binary=True)

    key_to_idx = {"<PAD>": 0}

    for key, idx in wv_from_bin.key_to_index.items():
        idx += 1
        key_to_idx[key] = idx

    train_df = pd.read_csv(data_dir / train_file, sep="\t")
    val_df = pd.read_csv(data_dir / val_file, sep="\t")

    # sentenceのindex化
    train_df[["token_dict", "len_of_token"]] = train_df.apply(convert_to_token, args=(key_to_idx,), axis=1)
    val_df[["token_dict", "len_of_token"]] = val_df.apply(convert_to_token, args=(key_to_idx,), axis=1)

    # token数が0の行を除いた辞書オブジェクトの抽出
    train_data = train_df.query("len_of_token != 0")["token_dict"]

    # 最初の例を確認
    print(train_data[0])


if __name__ == "__main__":
    main()
