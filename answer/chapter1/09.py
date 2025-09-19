"""Typoglycemia."""

import random


def shuffle_middle_char(word: str) -> str:
    """Shuffle middle characters of the words randomly.

    Parameters
    ----------
    word : str
        Inputted word.

    Returns
    -------
    str
        Shuffled word.
    """
    if len(word) <= 4:
        return word

    else:
        word = list(word)
        middle = word[1:-1]

        random.shuffle(middle)

        return f"{word[0]}{''.join(middle)}{word[-1]}"


def main() -> None:
    seed = 2025
    random.seed(seed)

    sentence = """
    I couldn't believe that I could actually understand what I was reading :
     the phenomenal power of the human mind ."""

    sentence = sentence.split()
    print(" ".join([shuffle_middle_char(word) for word in sentence]))


if __name__ == "__main__":
    main()
