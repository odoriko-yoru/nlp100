"""対話."""
# ruff: noqa: E501

import anthropic

client = anthropic.Anthropic()

# Hyperparameter
model = "claude-3-7-sonnet-latest"
max_tokens = 1024
temperature = 0

# prompt
prompt1 = """
つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。
自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。

東急東横線の駅は以下です。急行停車駅を****で強調しています。
**渋谷** → 代官山 → **中目黒** → 祐天寺 → 学芸大学 → 都立大学 → **自由が丘** → 田園調布 → 多摩川 → 新丸子 → **武蔵小杉** → 元住吉 → **日吉** → 綱島 → 大倉山 → **菊名** → 妙蓮寺 → 白楽 → 東白楽 → 反町 → **横浜**

東急大井町線の駅は以下です。急行停車駅を****で強調しています。
大井町方面
**自由が丘** → 奥沢 → 緑が丘 → **大岡山** → 北千束 → **旗の台** → 中延 → 荏原町 → 下神明 → **大井町**

二子玉川方面
**自由が丘** → 九品仏 → 尾山台 → **等々力** → 上野毛 → **二子玉川**
"""

message = client.messages.create(
    model=model,
    max_tokens=max_tokens,
    temperature=temperature,
    messages=[{"role": "user", "content": [{"type": "text", "text": prompt1}]}],
)

# 回答を出力
print(message.content[0].text)  # type: ignore
