"""パディング."""

import os
from typing import Dict, List

import torch

# 環境変数の読み込み
DATA_DIR = os.environ.get("DATA_DIR", "")


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
    current_len = tensor.size(0)

    pad_size = target_len - current_len

    dtype = tensor.dtype
    device = tensor.device

    result = torch.empty(target_len, dtype=dtype, device=device)

    result[:current_len] = tensor

    if pad_size > 0:
        result[current_len:] = pad_value

    return result


def collate(batch: List[Dict[str, torch.Tensor]], pad_value: int = 0) -> Dict[str, torch.Tensor]:
    """Collate for padding tensors.

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

    return {"input_ids": padded_input_ids, "label": labels}


def main() -> None:
    """Main function."""

    # 問題の例で確認
    input = [
        {
            "text": "hide new secretions from the parental units",
            "label": torch.Tensor([0.0]),
            "input_ids": torch.Tensor([5785, 66, 113845, 18, 12, 15095, 1594]),
        },
        {
            "text": "contains no wit , only labored gags",
            "label": torch.Tensor([0.0]),
            "input_ids": torch.Tensor([3475, 87, 15888, 90, 27695, 42637]),
        },
        {
            "text": "that loves its characters and communicates something rather beautiful about human nature",
            "label": torch.Tensor([1.0]),
            "input_ids": torch.Tensor([4, 5053, 45, 3305, 31647, 348, 904, 2815, 47, 1276, 1964]),
        },
        {
            "text": "remains utterly satisfied to remain the same throughout",
            "label": torch.Tensor([0.0]),
            "input_ids": torch.Tensor([987, 14528, 4941, 873, 12, 208, 898]),
        },
    ]

    result = collate(input)

    print(result)


if __name__ == "__main__":
    main()
