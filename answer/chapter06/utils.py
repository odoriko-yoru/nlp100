"""utils."""

from pathlib import Path
from typing import Union

import pandas as pd


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
