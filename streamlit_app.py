import numpy as np
import pandas as pd
import datetime
#import scipy
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import r_regression
from sklearn.linear_model import LinearRegression, Lasso, Ridge

# RandomForestRegressor를 임포트합니다.
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pickle
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

if platform.system() == "Darwin":  #
    plt.rc("font", family="AppleGothic")
else:
    plt.rc("font", family="NanumGothic")

fe = fm.FontEntry(
    fname=r"/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # ttf 파일이 저장되어 있는 경로
    name="NanumGothic",
)  # 원하는 폰트 설정
fm.fontManager.ttflist.insert(0, fe)  # Matplotlib에 폰트 추가

plt.rcParams.update({"font.size": 18, "font.family": "NanumGothic"})  # 폰트 설정

plt.rcParams["axes.unicode_minus"] = False

load_df = pd.read_csv("SN_total.csv")
load_df = load_df.set_index("시간")
load_df.index = pd.DatetimeIndex(load_df.index)
raw_df = load_df.copy()
raw_df.head(5)

df = raw_df.copy()

df["로그 원수 탁도"] = np.log10(df["원수 탁도"])
df["로그 응집제 주입률"] = np.log10(df["3단계 1계열 응집제 주입률"])
df

X = df[
    [
        "로그 원수 탁도",
        "원수 pH",
        "원수 알칼리도",
        "원수 전기전도도",
        "원수 수온",
        "3단계 원수 유입 유량",
        "3단계 침전지 체류시간",
    ]
]
y = df["로그 응집제 주입률"]
Xt, Xts, yt, yts = train_test_split(X, y, test_size=0.2, shuffle=False)


from xgboost import XGBRegressor

regressor = XGBRegressor(random_state=2, n_jobs=-1)

params = {
    "max_depth": [2],
    "n_estimators": [100],
    "eta": [0.02],
    "subsample": [0.8],
    "min_child_weight": [1],
}

rscv = RandomizedSearchCV(regressor, params, n_iter=30)
rscv.fit(Xt, yt)
model = rscv.best_estimator_
model

yt_pred = model.predict(Xt)
yts_pred = model.predict(Xts)

mse_train = mean_squared_error(10**yt, 10**yt_pred)
mse_test = mean_squared_error(10**yts, 10**yts_pred)
print(f"학습 데이터 MSE: {mse_train}")
print(f"테스트 데이터 MSE: {mse_test}")

r2_train = r2_score(10**yt, 10**yt_pred)
r2_test = r2_score(10**yts, 10**yts_pred)
print(f"학습 데이터 R2: {r2_train}")
print(f"테스트 데이터 R2: {r2_test}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.scatter(Xt["로그 원수 탁도"], yt, s=3, label="학습 데이터 (실제)")
ax.scatter(Xt["로그 원수 탁도"], yt_pred, s=3, label="학습 데이터 (예측)", c="r")
ax.grid()
ax.legend(fontsize=13)
ax.set_xlabel("로그 원수 탁도")
ax.set_ylabel("로그 응집제 주입률")
ax.set_title(
    rf"학습 데이터  MSE: {round(mse_train, 4)}, $R^2$: {round(r2_train, 2)}",
    fontsize=18,
)

ax = axes[1]
ax.scatter(Xts["로그 원수 탁도"], yts, s=3, label="테스트 데이터 (실제)")
ax.scatter(Xts["로그 원수 탁도"], yts_pred, s=3, label="테스트 데이터 (예측)", c="r")
ax.grid()
ax.legend(fontsize=13)
ax.set_xlabel("로그 원수 탁도")
ax.set_ylabel("로그 응집제 주입률")
ax.set_title(
    rf"테스트 데이터  MSE: {round(mse_test, 4)}, $R^2$: {round(r2_test, 2)}",
    fontsize=18,
)
