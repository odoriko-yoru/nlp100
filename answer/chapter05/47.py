"""LLMによる評価."""

import anthropic

with open("senryu.txt", "r", encoding="utf-8") as f:
    output = f.read()

client = anthropic.Anthropic()

# Hyperparameter
model = "claude-3-7-sonnet-latest"
max_tokens = 1024
temperature = 0

prompt = f"""
以下に10句の川柳を提示します。各川柳の面白さを10段階(1-10)で評価してください。
各句の後ろに採点結果を提示してください。

【例】
1. あなたより　長く生きるつもりなど　ないとは言えないのに (1)

【作品】
{output}
"""

message = client.messages.create(
    model=model, max_tokens=max_tokens, temperature=temperature, messages=[{"role": "user", "content": prompt}]
)

# 解答を出力
print(message.content[0].text)  # type: ignore
