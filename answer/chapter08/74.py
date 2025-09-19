"""モデルの評価."""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader, Dataset

# 環境変数の読み込み
DATA_DIR = os.environ.get("DATA_DIR", "")


class SSTDataset(Dataset):
    """
    Dataset Class for the SST-2.
    """

    def __init__(self, data: List[Dict[str, torch.Tensor]], embedding_matrix: torch.Tensor) -> None:
        super().__init__()
        self.data = data
        self.embedding_matrix = embedding_matrix

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        object = self.data[index]
        input_ids = object["input_ids"]
        embeddings = self.embedding_matrix[input_ids]

        # 平均化ベクトルの取得
        mean_embedding = torch.mean(embeddings, dim=0)
        return mean_embedding, object["label"]


class SemanticClassifier(nn.Module):
    """
    Bag of words.
    """

    def __init__(self, in_dimension: int, n_classes: int, device=None) -> None:
        super(SemanticClassifier, self).__init__()
        self.in_dimension = in_dimension
        self.n_classes = n_classes
        self.linear1 = nn.Linear(in_features=in_dimension, out_features=1, bias=False, device=device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.linear1(x)).squeeze(1)


def fix_seeds(seed: int) -> None:
    """Fix seeds, Pytorch, random, numpy.

    Parameters
    ----------
    seed : int
        Number of a seed.
    """
    random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_embedding_matrix(
    word_embedding_model_path: Union[str, Path],
    vocabulary: Set[str],
) -> Tuple[Dict[str, int], torch.Tensor]:
    """Extract a matrix from the pre-trained word embedding vector.

    Parameters
    ----------
    word_embedding_model_path : Union[str, Path]
        Path to the pre-trained word embedding model
    start_index : int, optional
        Starting index for the vocabulary, by default 0

    Returns
    -------
    Tuple[torch.Tensor, Dict[str, int]]
        Embedding matrix and word to index mapping

    Reference
    ---------
    https://github.com/upura/nlp100v2025/blob/main/ch08/ans73.py#L88C1-L91C57
    """
    wv_from_bin = KeyedVectors.load_word2vec_format(word_embedding_model_path, binary=True)

    # "<PAD>"は予約語
    key_to_idx = {"<PAD>": 0}

    # 単語埋め込み行列の取得
    # 最初の行は<PAD>用
    _, d_emb = wv_from_bin.vectors.shape
    E = [torch.zeros(d_emb, dtype=torch.float32)]

    # 単語が学習済み単語ベクトルに含まれているときのみ、ベクトルを取得
    for word in vocabulary:
        if word in wv_from_bin.key_to_index:
            key_to_idx[word] = len(key_to_idx)
            E.append(torch.tensor(wv_from_bin[word]))

    embedding_matrix = torch.stack(E)

    return key_to_idx, embedding_matrix


def tokenize(row: pd.Series, key_to_idx: Dict[str, int]) -> Tuple[Dict[str, Any], int]:
    """Convert inputted text and label to dict object.

    Parameters
    ----------
    row : pd.Series
        Row of the dataset.
    key_to_idx : Dict[str, int]
        Dictionary of word to index.
    Returns
    -------
    Tuple[Dict[str, Any], int]
        Tokenized data dictionary and token count
    """
    sentence = row["sentence"]
    label = row["label"]
    input_ids = []

    for word in sentence.lower().split():
        if word in key_to_idx:
            input_ids.append(key_to_idx[word])

    token_dict = {
        "text": sentence,
        "label": torch.tensor(label, dtype=torch.long),
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
    }

    return token_dict, len(input_ids)


def convert_to_token(df: pd.DataFrame, key_to_idx: Dict[str, int]) -> List[Dict[str, torch.Tensor]]:
    """Apply tokenize function to each row of the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset dataframe.
    key_to_idx : Dict[str, int]
        Dictionary of word to index.

    Returns
    -------
    List[Dict[str, torch.Tensor]]
        List of tokenized data dictionaries
    """
    # sentenceのindex化
    tokenized_data = df.apply(tokenize, args=(key_to_idx,), axis=1)

    # token数が0の行を除く
    result = [token_dict for token_dict, token_count in tokenized_data if token_count > 0]

    return result


def get_vocabulary(sentence: List[str]) -> Set[str]:
    """Get the set of vocabulary in the dataset.

    Parameters
    ----------
    sentence : List[str]
        List of texts.

    Returns
    -------
    Set[str]
    """
    result = set()

    for item in sentence:
        result.update(item.lower().split())

    return result


def evaluate(
    model: SemanticClassifier,
    devloader: DataLoader,
    device: Union[str, torch.device] = "cpu",
) -> float:
    """Evaluate the model on dev dataset.

    Parameters
    ----------
    model : SemanticClassifier
        Model to evaluate.
    devloader : DataLoader
        DataLoader for evaluation.
    device : Union[str, torch.device], optional
        Device to use for evaluation.

    Returns
    -------
    flaot
        Accuracy on dev dataset.
    """
    model.eval()
    correct = 0
    total = 0  # batch_size == len(DataLoader)にすれば不要

    with torch.no_grad():
        for mean_embedding, label in devloader:
            mean_embedding = mean_embedding.to(device)
            label = label.to(device).to(torch.float32)

            pred = model(mean_embedding)

            pred_binary = (pred.squeeze() >= 0.5).float()
            correct += (pred_binary == label).sum().item()
            total += label.size(0)

    accuracy = correct / total

    return accuracy


def main(args) -> None:
    fix_seeds(args.seed)
    data_dir = Path(DATA_DIR)

    # Datasetの読み込み
    dev_df = pd.read_csv(data_dir / "SST-2/dev.tsv", sep="\t")

    # Datasetに含まれる語彙の取得
    vocabulary = get_vocabulary(dev_df["sentence"].tolist())

    # 単語埋め込み行列, key-index辞書の作成
    key_to_idx, embedding_matrix = create_embedding_matrix(
        data_dir / "GoogleNews-vectors-negative300.bin.gz", vocabulary
    )

    # Datasetの前処理(token->idに変換)
    dev_data = convert_to_token(dev_df, key_to_idx)
    dev_dataset = SSTDataset(dev_data, embedding_matrix)

    # DataLoaderの作成
    dev_loader = DataLoader(dev_dataset, batch_size=len(dev_dataset), shuffle=False)

    device = torch.device("cpu")

    # モデルの作成と学習済みパラメータの読み込み
    model = SemanticClassifier(in_dimension=embedding_matrix.size(1), n_classes=2, device=device)
    model.load_state_dict(torch.load("73_model.pth"))

    accuracy = evaluate(
        model=model,
        devloader=dev_loader,
        device=device,
    )

    print(f"Dev Dataset Accuracy : {accuracy: .4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=29)
    args = parser.parse_args()
    main(args)
