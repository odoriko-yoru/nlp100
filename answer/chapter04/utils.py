"""Remvoe Wiki markup."""

import re


def remove_markup(text) -> str:
    """Remove Wiki markup.

    Parameters
    ----------
    text : _type_
        Inputted raw text

    Returns
    -------
    str
        Processed text
    """
    # | の除去
    text = re.sub(r"\|\s?", "", text)
    # indentの除去
    text = re.sub(r"\*+\s?", "", text)
    # 強調マークアップの除去
    text = re.sub(r"\'{2,5}", "", text)
    text = re.sub(r"^(\:|\;)", "", text)
    text = re.sub(r"^\*{2,5}", "", text)
    text = re.sub(r"=+\s?(.*?)\s?=+", r"\1", text)
    # 内部リンクの除去
    text = re.sub(r"\[\[(?:[^|\]]*?\|)??([^|\]]+?)\]\]", r"\1", text)
    # 外部リンクの除去
    text = re.sub(r"\[http://[^\]]+\]", "", text)
    # HTMLタグの除去
    text = re.sub(r"<[^>]+>", "", text)
    # テンプレートの除去
    text = re.sub(r"\{\{.*?\}\}", "", text)
    return text
