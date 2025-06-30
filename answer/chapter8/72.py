"""Bag of wordsモデルの構築."""

from typing import Dict
from typing import List
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset


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
        return self.sigmoid(self.linear1(x))


class SSTDataset(Dataset):
    """Dataset Class for the SST-2."""

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
