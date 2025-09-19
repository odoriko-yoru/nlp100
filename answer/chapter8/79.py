"""アーキテクチャの変更."""

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
        return self.data[index]


class SemanticRNNClassifier(nn.Module):
    """
    RNN for semantic text binary classification.
    """

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        num_layers: int = 2,
        hidden_size: int = 64,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_matrix = nn.Parameter(embedding_matrix)
        self.batch_first = batch_first
        input_dim = embedding_matrix.size(1)
        self.rnn = nn.RNN(
            input_dim, hidden_size, num_layers, batch_first=True, bidirectional=False
        )  # 単方向 num_layers層 RNN
        self.h2o = nn.Linear(hidden_size, out_features=1)  # hidden to logits

    def forward(self, x, hx=None):
        # [batch, seq_len, embedding_dim]
        x = self.embedding_matrix[x]

        # RNNの初期隠れ状態を適切に初期化
        if hx is None:
            batch_size = x.size(0)
            hx = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=x.device, dtype=x.dtype)

        # hidden: [num_layers, batch, hidden_size]
        _, hidden = self.rnn(x, hx)

        # 最後の時刻の隠れ状態を取得
        # [batch, hidden_size]
        last_hidden = hidden[-1]

        # [batch, 1]
        logits = self.h2o(last_hidden)

        return logits


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
    sentence = str(row["sentence"])
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
    model: SemanticRNNClassifier,
    trainloader: DataLoader,
    devloader: DataLoader,
    optimizer,
    criterion,
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
            batch = batch_data["input_ids"].to(device, non_blocking=True)
            label = batch_data["label"].to(device, non_blocking=True).to(torch.float32)

            # optimizerの初期化
            optimizer.zero_grad()

            # 推論
            logits = model(batch)

            # ラベルの形状を予測に合わせる
            label = label.unsqueeze(1)

            # 損失値の算出
            loss = criterion(logits, label)

            # 損失値を基にした勾配の計算
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
    model: SemanticRNNClassifier,
    devloader: DataLoader,
    criterion,
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
            batch = batch_data["input_ids"].to(device, non_blocking=True)
            label = batch_data["label"].to(device, non_blocking=True).to(torch.float32)

            logits = model(batch)

            label = label.unsqueeze(1)

            loss = criterion(logits, label)

            # 予測はsigmoidを適用して0.5と比較
            pred_label = torch.sigmoid(logits.squeeze()) > 0.5
            label_squeezed = label.squeeze()

            # デバッグ情報（最初のバッチのみ）
            if num_batches == 0:
                print(f"Debug - pred_label shape: {pred_label.shape}, label shape: {label_squeezed.shape}")
                print(f"Debug - pred_label: {pred_label[:5]}, label: {label_squeezed[:5]}")
                print(f"Debug - correct count: {(pred_label == label_squeezed).sum().item()}")

            correct += (pred_label == label_squeezed).sum().item()
            total += label.size(0)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def main(args):
    fix_seeds(args.seed)
    data_dir = Path(DATA_DIR)

    # Datasetの読み込み
    train_df = pd.read_csv(data_dir / "SST-2/train.tsv", sep="\t")
    dev_df = pd.read_csv(data_dir / "SST-2/dev.tsv", sep="\t")

    # Datasetに含まれる語彙の取得
    vocabulary = get_vocabulary(train_df["sentence"].tolist())
    vocabulary.update(get_vocabulary(dev_df["sentence"].tolist()))

    # 単語埋め込み行列, key-index辞書の作成
    key_to_idx, embedding_matrix = create_embedding_matrix(
        data_dir / "GoogleNews-vectors-negative300.bin.gz", vocabulary
    )

    # Datasetの前処理(token->idに変換)
    train_data = convert_to_token(train_df, key_to_idx)
    dev_data = convert_to_token(dev_df, key_to_idx)

    # Datasetインスタンスの作成
    train_dataset = SSTDataset(train_data)
    dev_dataset = SSTDataset(dev_data)

    # dryrun
    if args.dryrun:
        print("dryrun. only 1 epoch.")
        epochs = 1
    else:
        epochs = args.epochs

    # デバイスの設定（GPUが利用可能な場合はGPUを使用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DataLoaderの作成
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, collate_fn=collate, shuffle=True, num_workers=2
    )
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collate, shuffle=False, num_workers=2)

    # 埋め込み行列をdeviceに移動
    embedding_matrix = embedding_matrix.to(device)

    # モデルをdeviceに移動
    model = SemanticRNNClassifier(embedding_matrix=embedding_matrix).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 学習率を下げる
    criterion = nn.BCEWithLogitsLoss()

    # 学習
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
    torch.save(model.state_dict(), "79_model.pth")

    # 評価
    accuracy = evaluate(
        model=model,
        devloader=dev_loader,
        criterion=criterion,
        device=device,
    )

    print(f"Dev Dataset Accuracy : {accuracy[1]: .4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=29)
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("-b", "--batch_size", default=32, type=int)
    parser.add_argument("--dryrun", action="store_true")
    args = parser.parse_args()
    main(args)
