"""パタトクカシーー."""

patrol_car = "パトカー"
taxi = "タクシー"

# strictをFalseにすると第1引数と第2引数の要素数が異なってもエラーが生じない
ans = [str1 + str2 for str1, str2 in zip(patrol_car, taxi, strict=True)]

# リストの要素を""(空白なし)で結合
print("".join(ans))
