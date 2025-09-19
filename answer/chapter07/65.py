"""テキストのポジネガの予測."""

import pickle as pkl
from collections import defaultdict


def create_feature_vector(sentence: str) -> defaultdict[str, int]:
    """Create feture vector beased on BoW.

    Parameters
    ----------
    sentence : str
        Sentence.

    Returns
    -------
    defaultdict[str, int]
    """
    feature = defaultdict(int)
    for word in sentence.split():
        feature[word] += 1
    return feature


# Memo:
# NLPにおいて「テキストのポジネガの予測」タスクは「sentiment analysis - 感情分析」と呼ぶ
def pred_sentiment(sentence: str) -> str:
    """Predict the sentiment of inputted text.

    Parameters
    ----------
    sentence : str
        Text to classify.

    Returns
    -------
    str
        predicted label.
    """
    # tokenizer
    with open("sst2_tokenizer.pkl", "rb") as f:
        tokenizer = pkl.load(f)

    # 学習済モデル
    with open("logistic_regression_sst2.pkl", "rb") as f:
        clf = pkl.load(f)

    # textの特徴ベクトル作成
    feature = create_feature_vector(sentence)
    X = tokenizer.transform(feature)

    # predict
    label = clf.predict(X)

    return "positive" if label == "1" else "negative"


# 推論実行用関数
def main(sentence: str) -> str:
    """Predict the sentiment."""
    return pred_sentiment(sentence)


if __name__ == "__main__":
    # 入力
    sentence = input("判定したいテキスト：")

    # 推論
    result = pred_sentiment(sentence)

    # 出力
    print(f'テキスト "{sentence}" は "{result}" な文章です。')
