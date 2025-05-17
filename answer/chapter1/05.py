from typing import Union


def create_n_gram(n: int, sequence: Union[str, list]) -> list[str]:
    """Separate inputted sentence and output a char/word n-gram.

    Parameters
    ----------
    n : int
        Number of gram.

    sequence : str or list
        inputted sentence

    Returns
    -------
    tuple[str]
    """
    return [sequence[i : i + n] for i in range(len(sequence) - n + 1)]


sentence = "I am an NLPer"

for i in range(1, 4):
    # n-gram of the character
    print(create_n_gram(i, sentence))

    # n-gram of words
    print(create_n_gram(i, sentence.split()))
