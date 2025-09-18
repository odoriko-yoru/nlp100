"""テンプレートによる文生成."""


def gen_sentence(x: str, y: str, z: str) -> str:
    """Generate a sentence from the template and three args.

    Parameters
    ----------
    x : str
        Arg1
    y : str
        Arg2
    z : str
        Arg3

    Returns
    -------
    str
        generated sentence
    """
    return f"{x}時の{y}は{z}"


def main() -> None:
    x = "12"
    y = "気温"
    z = "22.4"

    print(gen_sentence(x, y, z))


if __name__ == "__main__":
    main()
