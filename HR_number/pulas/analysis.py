#HR数と関係がある項目を見つけるために，相関係数を調べる．

#ライブラリのインポート
import pandas as pd 

#CSVデータの読み込み
df = pd.read_csv('./Data/PlayerData.csv', sep = ',')

#DeNA以外の11球団のデータを取り出す
df11 = df[df["チーム"] != "DeNA"]
#DeNaAのデータ
df_De = df[df["チーム"] == "DeNA"]

#HR(本塁打)との相関係数をDataFrameとして変数に保存
df_HR = pd.DataFrame(df11.corr()["本塁打"])

#本塁打と各項目の相関係数のみ取り出す.
print(df11.corr()["本塁打"])
