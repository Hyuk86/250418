import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as MSE, r2_score as R2

# 데이터 로딩 함수 (CSV 파일 두 개)
@st.cache_data
def load_data():
    # 첫 번째 CSV 파일 로드
    data = pd.read_csv('data.csv', encoding='euc-kr')  # 'data.csv' 파일을 로드
    
    # 두 번째 CSV 파일 로드
    nfm = pd.read_csv('nfm.csv', encoding='euc-kr')  # 'nfm.csv' 파일을 로드
    
    # 두 데이터프레임을 하나로 합치기 (예시로 merge 사용)
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
    scaler = MinMaxScaler()
    final_data_scaled = final_data.copy()
    final_data_scaled[scale_cols] = scaler.fit_transform(final_data[scale_cols])

    return final_data, final_data_scaled

# 데이터 로딩
final_data, final_data_scaled = load_data()

# 데이터 미리보기
st.write("## 원본 데이터")
st.write(final_data.head())

# 히스토그램 그리기
st.write("## 수치형 변수 히스토그램")
numeric_cols = ['일평균공급량', '수용가수', 'wrr', '야간최소유량']
final_data_numeric = final_data[numeric_cols]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for ax, col in zip(axes, final_data_numeric.columns):
    final_data_numeric[col].hist(bins=20, ax=ax)
    ax.set_title(f"{col} 분포")
plt.tight_layout()
st.pyplot(fig)

# 상관관계 히트맵
st.write("## 상관관계 히트맵")
plt.figure(figsize=(8, 6))
sns.heatmap(final_data_numeric.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
st.pyplot()

# 모델 학습 및 평가
X = final_data[['일평균공급량', '수용가수', '야간최소유량']]
y = final_data['wrr']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤포레스트 모델 학습
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 예측
rf_pred = rf_model.predict(X_test)

# 평가
mse_rf = MSE(y_test, rf_pred)
rmse_rf = np.sqrt(mse_rf)
r2_rf = R2(y_test, rf_pred)

# XGBoost 모델 학습
xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)

# 예측
xgb_pred = xgb_model.predict(X_test)

# 평가
mse_xgb = MSE(y_test, xgb_pred)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = R2(y_test, xgb_pred)

# 평가 결과 출력
st.write("## 모델 성능")
st.write(f"### Random Forest")
st.write(f"MSE: {mse_rf:.2f}, RMSE: {rmse_rf:.2f}, R²: {r2_rf:.2f}")
st.write(f"### XGBoost")
st.write(f"MSE: {mse_xgb:.2f}, RMSE: {rmse_xgb:.2f}, R²: {r2_xgb:.2f}")

# RF vs XGBoost 비교 그래프
st.write("## 모델 비교")
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(y_test, rf_pred, alpha=0.7, label='Random Forest')
ax.scatter(y_test, xgb_pred, alpha=0.7, label='XGBoost', marker='x')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax.set_xlabel('True wrr')
ax.set_ylabel('Predicted wrr')
ax.legend()
ax.set_title('Random Forest vs XGBoost 비교')
st.pyplot(fig)
