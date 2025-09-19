"""川柳の作成."""

import anthropic

client = anthropic.Anthropic()

# Hyperparameter
model = "claude-3-7-sonnet-latest"
max_tokens = 1024
temperature = 0

# prompt
prompt = "川柳を10句作成してください。出力は1-10の番号と句のみ出力してください。"

message = client.messages.create(
    model=model, max_tokens=max_tokens, temperature=temperature, messages=[{"role": "user", "content": prompt}]
)

output = message.content[0].text  # type: ignore

# 解答を出力
print(output)

with open("senryu.txt", "w", encoding="utf-8") as f:
    f.write(output)
