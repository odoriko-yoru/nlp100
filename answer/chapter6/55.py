"""アナロジータスクでの正解率."""

import pandas as pd

df_capital = pd.read_csv("capital_common_countries.csv", sep=" ", header=None)

# 意味的アナロジーの正解率
# 文法的アナロジーの算出は不要(詳細はREADMEに記載)
accuracy = (df_capital[3] == df_capital[4]).sum() / len(df_capital)
print(f"意味的アナロジー正解率: {accuracy: .4f}")
