"""アナロジーデータでの実験."""

import os
from pathlib import Path

import pandas as pd
from gensim.models import KeyedVectors
from tqdm import tqdm
from utils import load_analogy_section


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
    most_similar_key, similarity = model.most_similar(positive=[row["word2"], row["word3"]], negative=[row["word1"]])[0]
    return most_similar_key, similarity


def main() -> None:
    data_dir = os.environ.get("DATA_DIR", "")
    data_dir = Path(data_dir)
    wv = Path("GoogleNews-vectors-negative300.bin.gz")

    dataset = data_dir / wv
    model = KeyedVectors.load_word2vec_format(dataset, binary=True)

    file = Path("questions-words.txt")

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


if __name__ == "__main__":
    main()
