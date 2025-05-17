sentence = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."

table = {
    ".": None,
    ",": None,
}

num = [1, 5, 6, 7, 8, 9, 15, 16, 19]

ans = []

for i, word in enumerate(sentence.split()):
    word = word.translate(word.maketrans(table))
    if i in num:
        ans.append((word[0], i))

    else:
        ans.append((word[:2], i))

print(dict(ans))
