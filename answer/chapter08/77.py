"""GPU上での学習."""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 環境変数の読み込み
DATA_DIR = os.environ.get("DATA_DIR", "")


class SSTDataset(Dataset):
    """
    Dataset Class for the SST-2.
    """

    def __init__(self, data: List[Dict[str, torch.Tensor]]) -> None:
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # 修正点: Paddingトークンの導入により、平均化ベクトルの変化が生じるため
        # 平均化ベクトルの取得はデータセット取得後に行う
        return self.data[index]


class SemanticClassifier(nn.Module):
    """
    Bag of words.
    """

    def __init__(self, in_dimension: int, n_classes: int) -> None:
        super(SemanticClassifier, self).__init__()
        self.in_dimension = in_dimension
        self.n_classes = n_classes
        self.linear1 = nn.Linear(in_features=in_dimension, out_features=1, bias=False)
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


def padding(tensor: torch.Tensor, target_len: int, pad_value: int = 0) -> torch.Tensor:
    """Padding tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Inputted tensor.
    target_len : int
        Target length.
    pad_value : int, optional
        Constant value, by default 0

    Returns
    -------
    torch.Tensor
        Padded tensor.
    """
    # inputされたtensorのサイズ
    current_len = tensor.size(0)

    # paddingする要素数
    pad_size = target_len - current_len

    dtype = tensor.dtype
    device = tensor.device

    # 乱数配列の取得
    result = torch.empty(target_len, dtype=dtype, device=device)

    # inputしたtensorの代入
    result[:current_len] = tensor

    # padding
    if pad_size > 0:
        result[current_len:] = pad_value

    return result


def collate(batch: List[Dict[str, torch.Tensor]], pad_value: int = 0) -> Dict[str, torch.Tensor]:
    """Collate for padding tensors.

    Preprocessing for mini-batches taken from the dataset.

    Parameters
    ----------
    batch : List[Dict[str, torch.Tensor]]
        Inputted batch.
    pad_value : int, optional
        Constant value, by default 0

    Returns
    -------
    Dict[str, torch.Tensor]
    """
    # input_idsの要素数でbatch内のitemを降順sort
    lengths = [len(item["input_ids"]) for item in batch]
    sorted_indices = sorted(range(len(batch)), key=lambda x: lengths[x], reverse=True)
    sorted_batch = [batch[i] for i in sorted_indices]

    # padding
    max_length = max(lengths)
    padded_input_ids = torch.stack([padding(item["input_ids"], max_length, pad_value) for item in sorted_batch])
    labels = torch.stack([item["label"] for item in sorted_batch])

    return {
        "input_ids": padded_input_ids,
        "label": labels,
    }


def train(
    model: SemanticClassifier,
    trainloader: DataLoader,
    devloader: DataLoader,
    embedding_matrix: torch.Tensor,
    optimizer: optim.Adam,
    criterion: nn.BCELoss,
    epoch: int,
    epochs: int,
    device: Union[str, torch.device],
) -> None:
    """Train the model.

    Parameters
    ----------
    model : SemanticClassifier
        Model to train.
    trainloader : DataLoader
        DataLoader for training.
    devloader : DataLoader
        DataLoader for evaluation.
    embedding_matrix : torch.Tensor
        Embedding matrix.
    optimizer : optim.Adam
        Optimizer for training.
    criterion : nn.BCELoss
        Loss function for training.
    epoch : int
        Current epoch.
    epochs : int
        Total number of epochs.
    device : Union[str, torch.device]
        Device to use for training.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    with tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}") as t:
        for batch_data in t:
            # データをGPUに移動
            input_ids = batch_data["input_ids"].to(device, non_blocking=True)
            label = batch_data["label"].to(device, non_blocking=True).to(torch.float32)

            # 平均化ベクトルの取得
            embeddings = embedding_matrix[input_ids]
            mean_embedding = torch.mean(embeddings, dim=1)

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
        dev_loss, dev_accuracy = evaluate(model, devloader, embedding_matrix, criterion, device)
        t.set_postfix(train_loss=f"{avg_train_loss:.4f}", dev_loss=f"{dev_loss:.4f}", dev_acc=f"{dev_accuracy:.4f}")

    # エポック終了時の詳細表示
    print(f"\nEpoch {epoch + 1}/{epochs} Summary:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Dev Loss: {dev_loss:.4f}")
    print(f"  Dev Accuracy: {dev_accuracy:.4f} ({dev_accuracy * 100:.2f}%)")


def evaluate(
    model: SemanticClassifier,
    devloader: DataLoader,
    embedding_matrix: torch.Tensor,
    criterion: nn.BCELoss,
    device: Union[str, torch.device],
) -> Tuple[float, float]:
    """Evaluate the model on dev dataset.

    Parameters
    ----------
    model : SemanticClassifier
        Model to evaluate.
    devloader : DataLoader
        DataLoader for evaluation.
    embedding_matrix : torch.Tensor
        Embedding matrix.
    criterion : nn.BCELoss
        Loss function for evaluation.
    device : Union[str, torch.device]
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
        for batch_data in devloader:
            # データをGPUに移動
            input_ids = batch_data["input_ids"].to(device, non_blocking=True)
            label = batch_data["label"].to(device, non_blocking=True).to(torch.float32)

            # 平均化ベクトルの取得
            embeddings = embedding_matrix[input_ids]
            mean_embedding = torch.mean(embeddings, dim=1)

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

    train_dataset = SSTDataset(train_data)
    dev_dataset = SSTDataset(dev_data)

    if args.dryrun:
        print("dryrun. only 1 epoch.")
        epochs = 1
    else:
        epochs = args.epochs

    # 5. DataLoaderの作成
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collate, shuffle=False)

    # 6. デバイスの設定（GPUが利用可能な場合はGPUを使用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 埋め込み行列をGPUに移動
    embedding_matrix = embedding_matrix.to(device)

    # モデルをGPUに移動
    model = SemanticClassifier(in_dimension=embedding_matrix.size(1), n_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    for epoch in range(epochs):
        train(
            model=model,
            trainloader=train_loader,
            devloader=dev_loader,
            embedding_matrix=embedding_matrix,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            epochs=epochs,
            device=device,
        )

    # 学習済モデルの保存
    torch.save(model.state_dict(), "77_model.pth")

    # 7. 評価
    accuracy = evaluate(
        model=model,
        devloader=dev_loader,
        embedding_matrix=embedding_matrix,
        criterion=criterion,
        device=device,
    )

    print(f"Dev Dataset Accuracy : {accuracy[1]: .4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=29)
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("-b", "--batch_size", default=64, type=int)
    parser.add_argument("--dryrun", action="store_true")
    args = parser.parse_args()
    main(args)
