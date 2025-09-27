"""ファイルをN分割する."""

import os
from pathlib import Path

import numpy as np
import pandas as pd


def main(args) -> None:
    path = os.environ.get("DATA_DIR")
    filename = Path("popular-names.txt")

    df = pd.read_csv(path / filename, sep="\t", header=None)

    n_split = args.n_split

    idx = np.linspace(0, len(df), n_split + 1, dtype=int)

    for i in range(len(idx) - 1):
        print("=" * 20 + f"{i + 1} splits" + "=" * 20)
        print(df.iloc[idx[i] : idx[i + 1]].to_csv(header=None))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_split", type=int, default=10)
    args = parser.parse_args()

    main(args)
