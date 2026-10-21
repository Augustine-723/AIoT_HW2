"""
Script to build and execute problem2/appliances_energy_prediction.ipynb
"""

import nbformat as nbf
import os
import subprocess

nb = nbf.v4.new_notebook()

cells = []

# Title & Metadata
cells.append(nbf.v4.new_markdown_cell("""# AIoT HW2 — Problem 2: Appliances Energy Prediction Using Multiple Linear Regression, RFE & Conformal Prediction

- **學生學號 (Student ID):** 7115093001
- **研究題目 (Research Title):** Multiple Linear Regression & Feature Selection for Residential Appliances Energy Prediction
- **資料來源 (Data Sources):**
  - UCI Machine Learning Repository: [Appliances Energy Prediction (ID: 374)](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
  - Kaggle Competition Reference: [Machine Learning SB1](https://www.kaggle.com/competitions/machine-learning-sb1)
  - Kaggle Public Dataset: [Appliances Energy Prediction](https://www.kaggle.com/datasets/loveall/appliances-energy-prediction)
  - Original Author Repository: [Luis Candanedo GitHub](https://github.com/LuisM78/Appliances-energy-prediction-data)
- **方法論架構 (Methodology):** CRISP-DM (Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment)
- **驗證設定 (Validation):** 70% / 15% / 15% 時序切分 (Full Date Aligned) + Training 擴展視窗交叉驗證 (Expanding-Window CV) + 獨立校準集 95% 分裂共形預測區間 (Split Conformal Prediction Interval)
"""))

# 1. Business Understanding
cells.append(nbf.v4.new_markdown_cell("""## 1. Business Understanding (業務與問題理解)

### 1.1 研究背景與目標
在低耗能建築與智慧家庭 (Smart Home / AIoT) 環境中，精準預測每 10 分鐘之家用電器耗電量 (Appliances Energy Consumption, 單位 Wh) 對於智慧電網調度、需量反應 (Demand Response)、尖峰用電調節與儲能系統充放電策略具備關鍵價值。

本研究採用比利時低耗能住宅由 ZigBee 無線感測器網路收集之室內溫濕度數據、Chièvres 氣象站觀測數據與燈光耗能，建立統計多元線性迴歸 (Multiple Linear Regression, MLR) 與特徵篩選管線。

### 1.2 同步觀測與預測設定定義
- **預測標的 (Target):** `Appliances` (連續數值，單位 Wh，每 10 分鐘耗電量)。
- **同步耗能估計 (Synchronous Telemetry):** 模型輸入包含同步的室內溫濕度感測器讀數、燈光能耗 (`lights`) 與氣象觀測。此設定定義為「基於全戶 IoT 感測器狀態之即時能耗診斷與同步推估」。
- **非負物理邊界 (Physical Non-negativity Constraint):** 家用電器耗能具有實體下限 ($Wh \\ge 0$)，在預測區間建構時，區間下界依物理原則截斷至 0，並與點預測嚴格區分。
"""))

# 2. Data Understanding
cells.append(nbf.v4.new_markdown_cell("""## 2. Data Understanding (資料理解與探索性資料分析)

### 2.1 原始資料審計與特徵定義
原始資料集包含 19,735 筆 10 分鐘解析度觀測，涵蓋 2016-01-11 17:00:00 至 2016-05-27 18:00:00，共 29 個欄位：
1. `date`: 時間戳記 (格式 `YYYY-MM-DD HH:MM:SS`)。
2. `Appliances`: 標的變數 (連續數值，Wh)。
3. `lights`: 燈光耗能 (Wh)。
4. `T1` ~ `T9`: 9 個室內溫濕度監測區域溫度 (°C)。
5. `RH_1` ~ `RH_9`: 9 個區域相對濕度 (%)。
6. `T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`: Chièvres 氣象站 6 項大氣指標。
7. `rv1`, `rv2`: 隨機對照變數 (Uniform[0, 50])，文獻中用於特徵選擇健全性驗證。
"""))

cells.append(nbf.v4.new_code_cell("""import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
np.random.seed(42)

# Load data audit
audit_path = 'results/data_audit.json'
if os.path.exists(audit_path):
    with open(audit_path, 'r', encoding='utf-8') as f:
        data_audit = json.load(f)
    print(f"Dataset SHA-256: {data_audit['sha256']}")
    print(f"Observations: {data_audit['observations']} (Matches expected: {data_audit['observations_match']})")
    print(f"Total columns: {data_audit['total_columns']}")
    print(f"Total nulls: {data_audit['data_cleanliness']['total_null_values']}")
    print(f"Total duplicates: {data_audit['data_cleanliness']['total_duplicates']}")

# Load CSV
df = pd.read_csv('data/energydata_complete.csv')
print(f"Data shape: {df.shape}")
df.head(3)
"""))

cells.append(nbf.v4.new_code_cell("""# Target distribution and summary statistics
target = df['Appliances']
print("Target (Appliances Wh) Statistics:")
print(target.describe())
print(f"Skewness: {target.skew():.4f} (Heavily right-skewed)")
print(f"Kurtosis: {target.kurt():.4f}")

# Sanity check with random variables
print(f"Correlation with rv1 (Random Variable 1): {df['rv1'].corr(target):.5f}")
print(f"Correlation with rv2 (Random Variable 2): {df['rv2'].corr(target):.5f}")
"""))

# 3. Data Preparation
cells.append(nbf.v4.new_markdown_cell("""## 3. Data Preparation (資料準備與時序防洩漏切分)

### 3.1 嚴格時序切分原則 (Full-Date Aligned Split)
為避免時序資料的前後洩漏，本研究不採用隨機洗牌，而採依日曆日邊界對齊之時序切分：
- **訓練集 (Training Set, 70.26%):** 2016-01-11 17:00:00 ~ 2016-04-16 23:50:00 (共 97 天，13,866 筆)
- **校準集 (Calibration Set, 15.32%):** 2016-04-17 00:00:00 ~ 2016-05-07 23:50:00 (共 21 天，3,024 筆，專供 Split Conformal 分位数估算)
- **獨立測試集 (Test Set, 14.42%):** 2016-05-08 00:00:00 ~ 2016-05-27 18:00:00 (共 20 天，2,845 筆，凍結最終泛化評估)

所有特徵標準化 (`StandardScaler`) 僅在 Training 資料上 `fit`，再 `transform` 至 Calibration 與 Test，杜絕 Data Leakage。
"""))

cells.append(nbf.v4.new_code_cell("""# Feature engineering from timestamp
df['date_dt'] = pd.to_datetime(df['date'])
df['hour'] = df['date_dt'].dt.hour
df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
df['day_of_week'] = df['date_dt'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Distinguish feature sets
physical_features = [
    'lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5',
    'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9',
    'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint'
]
formal_26_features = physical_features + ['hour']
candidate_features = physical_features + ['hour', 'sin_hour', 'cos_hour', 'is_weekend']

print(f"Physical predictors from CSV: {len(physical_features)}")
print(f"Formal raw predictors (including primary hour): {len(formal_26_features)}")
print(f"Total candidate features (including cyclical time): {len(candidate_features)}")

# Date aligned split
unique_days = sorted(df['date_dt'].dt.date.unique())
train_days = unique_days[:97]
cal_days = unique_days[97:118]
test_days = unique_days[118:]

train_df = df[df['date_dt'].dt.date.isin(train_days)].copy()
cal_df = df[df['date_dt'].dt.date.isin(cal_days)].copy()
test_df = df[df['date_dt'].dt.date.isin(test_days)].copy()

print(f"Train obs: {len(train_df)}, Cal obs: {len(cal_df)}, Test obs: {len(test_df)}")
assert len(train_df) + len(cal_df) + len(test_df) == len(df)
"""))

# 4. Modeling
cells.append(nbf.v4.new_markdown_cell("""## 4. Modeling (多元線性迴歸、RFE 特徵淘汰與正規化模型)

### 4.1 候選模型庫與擴展視窗交叉驗證
僅於訓練集執行 5-Fold Expanding-Window Cross Validation (`TimeSeriesSplit(n_splits=5)`)，計算無偏樣本標準差 ($ddof=1$)：
1. **M0:** DummyRegressor (預測訓練均值，基準線)
2. **M1:** Full Multiple Linear Regression (OLS, 全特徵 $p=29$)
3. **M2:** RFE + OLS (遞迴特徵淘汰，搜尋 $k=5 \\sim 29$ 最佳特徵數)
4. **M3:** Ridge Regression (L2 正規化，網格搜尋最佳 $\\alpha$)
5. **M4:** Lasso Regression (L1 稀疏正規化，網格搜尋最佳 $\\alpha$)
"""))

cells.append(nbf.v4.new_code_cell("""# Display Model Comparison Results from results/model_comparison.csv
model_comp_df = pd.read_csv('results/model_comparison.csv')
model_comp_df[['model_id', 'name', 'features_count', 'cv_rmse_mean', 'cv_rmse_std']]
"""))

cells.append(nbf.v4.new_code_cell("""# Display Selected Features from results/selected_features.csv
selected_f_df = pd.read_csv('results/selected_features.csv')
print("RFE Selected Features:")
print(selected_f_df)
"""))

# 5. Evaluation
cells.append(nbf.v4.new_markdown_cell("""## 5. Evaluation (獨立測試集嚴格評估與分裂共形預測區間)

### 5.1 最終模型凍結與測試指標
依據訓練集 CV 表現，Ridge Regression (M3, $\\alpha=1000.0$) 獲最佳 CV RMSE ($95.8454 \\pm 6.2597$)；同時 RFE+OLS (M2, $k=6$) 達成最簡特徵組合 ($96.2026 \\pm 5.8893$)。

在完全未參與訓練與超參數挑選之獨立測試集 (2,845 筆) 上進行評估：
"""))

cells.append(nbf.v4.new_code_cell("""# Load final test metrics
with open('results/final_test_metrics.json', 'r', encoding='utf-8') as f:
    final_metrics = json.load(f)

print(f"Selected Model: {final_metrics['model_name']}")
print("Test Metrics:")
for k, v in final_metrics['metrics'].items():
    print(f"  {k}: {v}")

print("\\nDummy Baseline Metrics:")
for k, v in final_metrics['dummy_baseline'].items():
    print(f"  {k}: {v}")

print("\\nConformal Prediction Interval Metrics:")
conf = final_metrics['conformal_prediction_interval']
print(f"  Calibration sample count (n): {conf['calibration_sample_count']}")
print(f"  Quantile rank: {conf['quantile_rank']}")
print(f"  Calibration conformal quantile (q): {conf['calibration_quantile_q']} Wh")
print(f"  Nominal Target Coverage: {conf['nominal_coverage_pct']}%")
print(f"  Empirical Test Coverage (Raw): {conf['empirical_test_coverage_raw_pct']}%")
print(f"  Empirical Test Coverage (Clipped): {conf['empirical_test_coverage_clipped_pct']}%")
print(f"  Mean Interval Width (Clipped): {conf['mean_interval_width_clipped']} Wh")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 5.2 Split Conformal 可交換性前提與時間序列討論
標準 Split Conformal 預測區間依賴觀測值之「可交換性 (Exchangeability)」前提（即 $y_1, \\dots, y_{n+1}$ 為 i.i.d. 或可自由置換）。
在實際物聯網建築能耗時間序列中，環境溫度與使用者行為具備明顯的自相關性 (Autocorrelation) 與晝夜季節性，可交換性前提理論上嚴格不成立。因此 95% 為名義設定目標，實際覆蓋率必須以完全獨立之測試集實證檢驗。本實驗實證覆蓋率達 **95.25%**，成功覆蓋名義 95% 目標。
"""))

# 6. Deployment
cells.append(nbf.v4.new_markdown_cell("""## 6. Deployment (模型規格匯出與前端推論規格)

### 6.1 Option B 純客戶端確定性推論規格
為達成極致響應速度並避免伺服器冷啟動延遲，推論規格匯出為 `results/model_spec.json`。
包含：
- 截距與各特徵之係數 (`coefficient`)、標準化均值 (`mean`) 與標準差 (`scale`)。
- 分裂共形預測量化值 ($q = 143.8311$ Wh)。
- 各特徵預設數值與物理邊界。
"""))

cells.append(nbf.v4.new_code_cell("""with open('results/model_spec.json', 'r', encoding='utf-8') as f:
    model_spec = json.load(f)

print(f"Model ID: {model_spec['best_model_id']}")
print(f"Intercept: {model_spec['intercept']}")
print(f"Calibration Quantile q: {model_spec['calibration_q']}")
print(f"Feature specs count: {len(model_spec['feature_specs'])}")
print("Sample feature specs:")
for spec in model_spec['feature_specs'][:5]:
    print(f"  {spec['feature']}: coef={spec['coefficient']:.4f}, mean={spec['mean']:.4f}, scale={spec['scale']:.4f}")
"""))

nb.cells = cells

# Save notebook
notebook_path = os.path.join(os.path.dirname(__file__), 'appliances_energy_prediction.ipynb')
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Created notebook at: {notebook_path}")
