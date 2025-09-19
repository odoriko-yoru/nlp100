"""LLMによる評価の頑健性."""

import anthropic
import numpy as np
from numpy.typing import NDArray

# Hyperparameter
MODEL = "claude-3-7-sonnet-latest"
MAX_TOKENS = 1024
TEMPERATURE = 0.1  # 分散の計算をするため、モデルの回答が揺らぐように0 -> 0.1に変更
N_EVALUATIONS = 10
SUFFIX = " 100本ノック48本目"


def load_senryu() -> str:
    """川柳ファイルを読み込む."""
    with open("senryu.txt", "r", encoding="utf-8") as f:
        return f.read().strip()


def add_suffix_to_senryu(senryu: str, suffix: str) -> str:
    """各川柳の文末に特定の文字列を追加する."""
    return "\n".join(line + suffix for line in senryu.split("\n"))


def create_prompt(senryu: str) -> str:
    """評価用のプロンプトを生成する."""
    return f"""
以下に10句の川柳を提示します。各川柳の面白さを10段階(1-10)で評価してください。
各句の後ろに採点結果を提示してください。また、評価の際は点数のみを出力してください。

【例】
1. あなたより　長く生きるつもりなど　ないとは言えないのに 点数:3
2. じゃあね、うんまた　眼にぷかぷか　ブラックホール　点数:5

【作品】
{senryu}
"""


def evaluate_senryu(client: anthropic.Anthropic, prompt: str, n_times: int) -> NDArray:
    """川柳を指定回数評価する."""
    scores = []
    for _ in range(n_times):
        message = client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, messages=[{"role": "user", "content": prompt}]
        )
        # 点数取得
        score = [int(t[-1]) for t in message.content[0].text.split("\n")]  # type: ignore
        scores.append(score)
    return np.array(scores)


def print_statistics(description: str, result: NDArray) -> None:
    """評価結果の統計を出力する."""
    print(
        f"""
    {description}に対して評価を繰り返した場合、
    10句の平均はそれぞれ
    {result.mean(axis=0)}

    標準偏差はそれぞれ
    {result.std(axis=0)}
    """
    )


def main() -> None:
    """メイン処理."""
    # クライアントの初期化
    client = anthropic.Anthropic()

    # 川柳の読み込みと加工
    senryu = load_senryu()
    senryu_with_suffix = add_suffix_to_senryu(senryu, SUFFIX)

    # オリジナルの川柳の評価
    prompt1 = create_prompt(senryu)
    result1 = evaluate_senryu(client, prompt1, N_EVALUATIONS)
    print_statistics("同じ句", result1)

    # suffix付きの川柳の評価
    prompt2 = create_prompt(senryu_with_suffix)
    result2 = evaluate_senryu(client, prompt2, N_EVALUATIONS)
    print_statistics("特定のsuffixをつけた句", result2)


if __name__ == "__main__":
    main()
