def cipher(sequence: str) -> str:
    """Replace lowercase letters to '219 - <ASCII's code>'.

    Other characters are unchanged.

    Parameters
    ----------
    sequence : str
        Inputted sentence

    Returns
    -------
    str
        Ciphered sentence
    """
    ans = [
        chr(219 - ord(char)) if 97 <= ord(char) <= 122 else char for char in sequence
    ]
    return "".join(ans)


input = "aA0/!"

ans = cipher(input)
print(ans)
print(cipher(ans))
