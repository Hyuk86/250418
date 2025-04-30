import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit에서 사용할 폰트 설정
if platform.system() == "Darwin":  # MacOS 환경에서의 폰트 설정
    plt.rc("font", family="AppleGothic")
else:
    plt.rc("font", family="NanumGothic")

fe = fm.FontEntry(
    fname=r"/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # ttf 파일 경로
    name="NanumGothic",
)
fm.fontManager.ttflist.insert(0, fe)
plt.rcParams.update({"font.size": 18, "font.family": "NanumGothic"})
plt.rcParams["axes.unicode_minus"] = False

st.markdown("## XGBoost Example")

# 데이터 로딩
@st.cache_data
def load_data():
    data = pd.read_csv("data.csv", encoding='euc-kr')
    nfm = pd.read_csv("nfm.csv", encoding='euc-kr')

    # 데이터 전처리 및 병합 과정
    data_melted = data.melt(id_vars=['소블록명', '연도', '구분'], var_name='월', value_name='값')
    data_melted['연월'] = data_melted['연도'].astype(str) + '-' + data_melted['월'].str.replace('월', '').str.zfill(2)
    
    supply = data_melted[data_melted['구분'] == '일평균공급량'].copy()
    customer = data_melted[data_melted['구분'] == '수용가수'].copy()
    wrr = data_melted[data_melted['구분'] == '유수율'].copy()

    supply['값'] = supply['값'].str.replace(',', '').astype(float)
    customer['값'] = customer['값'].str.replace(',', '').astype(float)
    wrr['값'] = wrr['값'].str.replace('%', '').astype(float)

    supply = supply[['소블록명', '연월', '값']].rename(columns={'값': '일평균공급량'})
    customer = customer[['소블록명', '연월', '값']].rename(columns={'값': '수용가수'})
    wrr = wrr[['소블록명', '연월', '값']].rename(columns={'값': 'wrr'})

    data_pivot = supply.merge(customer, on=['소블록명', '연월'], how='left') \
                       .merge(wrr, on=['소블록명', '연월'], how='left')

    nfm['날짜'] = pd.to_datetime(nfm['날짜'], errors='coerce')
    nfm['연월'] = nfm['날짜'].dt.to_period('M').astype(str)
    nfm['야간최소유량'] = pd.to_numeric(nfm['야간최소유량'], errors='coerce')

    nfm_grouped = nfm.groupby(['소블록명', '연월'], as_index=False)['야간최소유량'].mean()

    final_data = data_pivot.merge(nfm_grouped, on=['소블록명', '연월'], how='left')
    final_data = final_data.dropna(subset=['야간최소유량'])

    # 스케일링
    scale_cols = ['일평균공급량', '수용가수', 'wrr', '야간최소유량']
    final_data_scaled = final_data.copy()
    final_data_scaled[scale_cols] = (final_data[scale_cols] - final_data[scale_cols].min()) / (final_data[scale_cols].max() - final_data[scale_cols].min())

    return final_data, final_data_scaled

# 데이터 로딩
final_data, final_data_scaled = load_data()

# 데이터 미리보기
st.write("## 데이터 미리보기")
st.write(final_data.head())

# Target 변수 선택
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### Target 변수를 선택하세요")
target_column = st.selectbox("Target 변수를 선택하세요", final_data.columns[1:])

# Input 변수 선택
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### Input 변수를 선택하세요")
input_columns = st.multiselect("복수의 컬럼을 선택하세요", final_data.columns[1:])

# 데이터 분리
Xt, Xts, yt, yts = train_test_split(final_data[input_columns], final_data[target_column], test_size=0.2, shuffle=False)

# 하이퍼파라미터 설정
max_depth = st.slider("max_depth", 1, 20, 3)
n_estimators = st.slider("n_estimators", 50, 500, 100)
learning_rate = st.slider("learning_rate", 0.0, 1.0, 0.1)
subsample = st.slider("subsample", 0.1, 1.0, 0.8)

# XGBoost 모델 학습
xgb = XGBRegressor(
    max_depth=max_depth,
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    subsample=subsample,
    random_state=2,
    n_jobs=-1,
)

xgb.fit(Xt, yt)

# 예측
yt_pred = xgb.predict(Xt)
yts_pred = xgb.predict(Xts)

# 평가 결과
mse_train = mean_squared_error(yt, yt_pred)
mse_test = mean_squared_error(yts, yts_pred)
r2_train = r2_score(yt, yt_pred)
r2_test = r2_score(yts, yts_pred)

st.write(f"학습 데이터 MSE: {mse_train}")
st.write(f"테스트 데이터 MSE: {mse_test}")
st.write(f"학습 데이터 R²: {r2_train}")
st.write(f"테스트 데이터 R²: {r2_test}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.scatter(Xt[input_columns[0]], yt, s=3, label="실제 데이터")
ax.scatter(Xt[input_columns[0]], yt_pred, s=3, label="예측 데이터", c="r")
ax.grid()
ax.legend(fontsize=13)
ax.set_xlabel(input_columns[0])
ax.set_ylabel(target_column)
ax.set_title(f"학습 데이터: MSE = {mse_train:.4f}, $R^2$ = {r2_train:.2f}")

ax = axes[1]
ax.scatter(Xts[input_columns[0]], yts, s=3, label="실제 데이터")
ax.scatter(Xts[input_columns[0]], yts_pred, s=3, label="예측 데이터", c="r")
ax.grid()
ax.legend(fontsize=13)
ax.set_xlabel(input_columns[0])
ax.set_ylabel(target_column)
ax.set_title(f"테스트 데이터: MSE = {mse_test:.4f}, $R^2$ = {r2_test:.2f}")

st.pyplot(fig)
