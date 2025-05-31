"""Zero-Shot推論."""

import anthropic

client = anthropic.Anthropic()

# Hyperparameter
model = "claude-3-5-haiku-latest"
max_tokens = 1024
temperature = 0

# prompt
prompt = """
9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。

ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。
イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。
"""

message = client.messages.create(
    model=model,
    max_tokens=max_tokens,
    temperature=temperature,
    messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
)

# 回答を出力
print(message.content[0].text)  # type: ignore
