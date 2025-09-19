"""n-gram."""

from typing import List, Union


def create_n_gram(n: int, sequence: Union[str, List]) -> Union[list[str], List[List[str]]]:
    """Separate inputted sentence and output a char/word n-gram.

    Parameters
    ----------
    n : int
        Number of gram

    sequence : str or list
        Sentences to split

    Returns
    -------
    Union[list[str], List[List[str]]]
    """
    return [sequence[i : i + n] for i in range(len(sequence) - n + 1)]


def main() -> None:
    sentence = "I am an NLPer"

    # tri-gram of characters
    print(create_n_gram(3, sentence))

    # bi-gram of words
    print(create_n_gram(2, sentence.split()))


if __name__ == "__main__":
    main()
