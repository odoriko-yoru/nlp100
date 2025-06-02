"""アナロジーデータでの実験."""

import os
from pathlib import Path
from typing import Union

import pandas as pd
from gensim.models import KeyedVectors
from tqdm import tqdm

data_dir = os.environ.get("DATA_DIR", "")
data_dir = Path(data_dir)
wv = Path("GoogleNews-vectors-negative300.bin.gz")

dataset = data_dir / wv
model = KeyedVectors.load_word2vec_format(dataset, binary=True)

file = Path("questions-words.txt")


def load_analogy_section(filepath: Union[str, Path], target_section: str) -> pd.DataFrame:
    """Load specified section from text file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the text tile
    target_section : str
        Target section

    Returns
    -------
    pd.DataFrame
    """
    data = []
    current_section = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(":"):
                current_section = line[1:].strip()
            elif line and current_section == target_section:
                words = line.split()
                if len(words) == 4:
                    data.append(words)

    return pd.DataFrame(data, columns=["word1", "word2", "word3", "word4"])


def calc_similarity(row: pd.Series, model: KeyedVectors) -> tuple[str, float]:
    """Calculate cosine similarity and return a most similar word and similarity.

    Parameters
    ----------
    row : pd.Series
        Row
    model : KeyedVectors
        gensim's KeyedVectors

    Returns
    -------
    tuple[str, float]
        A most similar word and that similarity
    """
    most_similar_key, similarity = model.most_similar(positive=[row["word1"], row["word3"]], negative=[row["word2"]])[0]
    return most_similar_key, similarity


tqdm.pandas()

# load text file
target_section = "capital-common-countries"
df_capital = load_analogy_section(data_dir / file, target_section)

# calc simitarity and a extract most similar word
df_capital[["most_similar_word", "similarity"]] = df_capital.progress_apply(
    calc_similarity, axis=1, args=(model,), result_type="expand"
)  # type: ignore

# save
df_capital.to_csv(
    "capital_common_countries.csv",
    sep=" ",
    index=False,
    header=False,
)
