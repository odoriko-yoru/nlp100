"""パタトクカシーー."""

patrol_car = "パトカー"
taxi = "タクシー"

ans = [str1 + str2 for str1, str2 in zip(patrol_car, taxi, strict=False)]

print("".join(ans))
