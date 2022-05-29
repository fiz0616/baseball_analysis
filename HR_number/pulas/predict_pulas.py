'''
analysis_1.pyの結果を用いて，使用する相関係数の値を決めたら，
その項目のデータを抜き出して，RSMEを計算する．

RSME 0に近いほど，正確な予測．
'''

#ライブラリのインポート
import pandas as pd
# 機械学習
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as MSE
# 行列計算ライブラリ(機械学習の結果がnumpyで返却される)
import numpy as np

#CSVデータの読み込み
df = pd.read_csv('../Data/PlayerData.csv', sep = ',')

#DeNA以外の11球団のデータを取り出す
df11 = df[df["チーム"] != "DeNA"]
#DeNaAのデータ
df_De = df[df["チーム"] == "DeNA"]

#HR(本塁打)との相関係数をDataFrameとして変数に保存
df_HR = pd.DataFrame(df11.corr()["本塁打"])

#相関係数が"0.7"以上の項目を取り出す． RSMEの値によって変更する．
X_colums = df_HR[df_HR["本塁打"] >= 0.7].index.tolist()
#目的変数の本塁打を取り除く．
X_colums.remove("本塁打")

#説明変数と目的変数を決める．
X = df11[X_colums]
y = df11["本塁打"]

#データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

#モデルの箱を用意，学習
lr = LR()
lr.fit(X_train, y_train)

#予測値の算出
y_pred_test = lr.predict(X_test)

#MSEの算出
mse_test = MSE(y_test, y_pred_test)

#RMSEの算出
rmse_test = np.sqrt(mse_test)

#RSMEの表示 0に近いほど，正しい予測ができている．
print(rmse_test)

# DeNAデータ(説明変数)
X_De = df_De[X_colums]

# 予測ホームラン数の算出
y_pred_De = lr.predict(X_De)

# 予測本塁打列の追加
df_De['予測本塁打'] = np.round(y_pred_De).astype(np.int32)

# 予測値と実際の値との比較結果
print(df_De[['選手名','本塁打','予測本塁打']])
