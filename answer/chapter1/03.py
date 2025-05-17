sentence = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

table = {
    ".": None,
    ",": None,
}

print([len(word.translate(word.maketrans(table))) for word in sentence.split()])
