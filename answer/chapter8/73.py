"""モデルの学習.

CPU上で学習を行う。

1. Dataset(SST-2, train/dev)の読み込み
2. Datasetに含まれる語彙の取得
3. 単語埋め込み行列, key-index辞書の作成
4. Datasetの前処理(token->idに変換)
5. DataLoaderの作成
6. 学習

===================================
Reference
1. Optimizerの解説記事
【決定版】スーパーわかりやすい最適化アルゴリズム -損失関数からAdamとニュートン法-
https://qiita.com/omiita/items/1735c1d048fe5f611f80
"""

import os
import random
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

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


def train(
    model: SemanticClassifier,
    trainloader: DataLoader,
    devloader: DataLoader,
    optimizer: optim.Adam,
    criterion: nn.BCELoss,
    epoch: int,
    epochs: int,
    device: Union[str, torch.device] = "cpu",
) -> None:
    """Train the model.

    Parameters
    ----------
    model : SemanticClassifier
        Model to train.
    trainloader : DataLoader
        DataLoader for training.
    optimizer : optim.Adam
        Optimizer for training.
    criterion : nn.BCELoss
        Loss function for training.
    epoch : int
        Current epoch.
    epochs : int
        Total number of epochs.
    device : Union[str, torch.device], optional
        Device to use for training.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    with tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}") as t:
        for mean_embedding, label in t:
            mean_embedding = mean_embedding.to(device)
            label = label.to(device).to(torch.float32)

            # optimizerの初期化
            optimizer.zero_grad()

            # 推論
            pred = model(mean_embedding)

            # 損失値の算出
            loss = criterion(pred, label)

            # 損失値を基にした勾配の計算
            # model.linear1.weight.gradに勾配が格納
            loss.backward()

            # 勾配を基にAdamアルゴリズムを用いて重み更新
            optimizer.step()

            # 損失値の記録
            total_loss += loss.item()
            num_batches += 1
            t.set_postfix(train_loss=f"{loss.item():.4f}")

        # dev datasetでの評価
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0
        dev_loss, dev_accuracy = evaluate(model, devloader, criterion, device)
        t.set_postfix(train_loss=f"{avg_train_loss:.4f}", dev_loss=f"{dev_loss:.4f}", dev_acc=f"{dev_accuracy:.4f}")

    # エポック終了時の詳細表示
    print(f"\nEpoch {epoch + 1}/{epochs} Summary:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Dev Loss: {dev_loss:.4f}")
    print(f"  Dev Accuracy: {dev_accuracy:.4f} ({dev_accuracy * 100:.2f}%)")


def evaluate(
    model: SemanticClassifier,
    devloader: DataLoader,
    criterion: nn.BCELoss,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[float, float]:
    """Evaluate the model on dev dataset.

    Parameters
    ----------
    model : SemanticClassifier
        Model to evaluate.
    devloader : DataLoader
        DataLoader for evaluation.
    criterion : nn.BCELoss
        Loss function for evaluation.
    device : Union[str, torch.device], optional
        Device to use for evaluation.

    Returns
    -------
    Tuple[float, float]
        Average loss and accuracy on dev dataset.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for mean_embedding, label in devloader:
            mean_embedding = mean_embedding.to(device)
            label = label.to(device).to(torch.float32)

            pred = model(mean_embedding)
            loss = criterion(pred, label)

            pred_binary = (pred.squeeze() >= 0.5).float()
            correct += (pred_binary == label).sum().item()
            total += label.size(0)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def main(args) -> None:
    fix_seeds(args.seed)
    data_dir = Path(DATA_DIR)

    # 1. Datasetの読み込み
    train_df = pd.read_csv(data_dir / "SST-2/train.tsv", sep="\t")
    dev_df = pd.read_csv(data_dir / "SST-2/dev.tsv", sep="\t")

    # 2. Datasetに含まれる語彙の取得
    vocabulary = get_vocabulary(train_df["sentence"].tolist())
    vocabulary.update(get_vocabulary(dev_df["sentence"].tolist()))

    # 3. 単語埋め込み行列, key-index辞書の作成
    key_to_idx, embedding_matrix = create_embedding_matrix(
        data_dir / "GoogleNews-vectors-negative300.bin.gz", vocabulary
    )

    # 4. Datasetの前処理(token->idに変換)
    train_data = convert_to_token(train_df, key_to_idx)
    dev_data = convert_to_token(dev_df, key_to_idx)

    train_dataset = SSTDataset(train_data, embedding_matrix)
    dev_dataset = SSTDataset(dev_data, embedding_matrix)

    if args.dryrun:
        print("dryrun. only 1 epoch.")
        epochs = 1
    else:
        epochs = args.epochs

    # 5. DataLoaderの作成
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    # 6. 学習
    device = torch.device("cpu")  # 本問題ではCPU上で学習する
    model = SemanticClassifier(in_dimension=embedding_matrix.size(1), n_classes=2, device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()

    for epoch in range(epochs):
        train(
            model=model,
            trainloader=train_loader,
            devloader=dev_loader,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            epochs=epochs,
            device=device,
        )

    # 学習済モデルの保存
    torch.save(model.state_dict(), "73_model.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=29)
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("-b", "--batch_size", default=32, type=int)
    parser.add_argument("-p", "--postfix", type=str)
    parser.add_argument("--dryrun", action="store_true")
    args = parser.parse_args()
    main(args)
