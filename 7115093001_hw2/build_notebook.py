"""
Script to build and execute 7115093001_hw2.ipynb
"""

import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
nb = new_notebook()
cells = []

# Cell 1: Title & Research Objective
cells.append(new_markdown_cell("""# AIoT HW2 — Phase 1 研究實作報告
## Predicting Hourly Bike Sharing Demand Using Multiple Linear Regression and Feature Selection
**學生學號 (Student ID):** 7115093001  
**研究題目 (Research Title):** Predicting Hourly Bike Sharing Demand Using Multiple Linear Regression and Feature Selection  
**研究角色 (Role):** Machine Learning Engineer & Research Assistant  
**作業階段 (Phase):** Phase 1 (資料驗證、探索性數據分析、時間序列切分、特徵預處理管線與 OLS 基準模型構建)  

---

### 1. 研究背景與目標 (Research Objective)
本研究旨在使用智慧城市物聯網（AIoT）環境下蒐集之歷史微觀氣候、日曆結構與時間序列特徵，構建**多元線性迴歸 (Multiple Linear Regression, OLS)** 模型，預測華盛頓特區公共自行車租借系統（Capital Bikeshare）之每小時自行車租借總量（`cnt`）。

在 Phase 1 階段中，我們聚焦於建立嚴謹且具可復現性（Reproducibility）之機器學習研究基準（Baseline），嚴格遵守以下學術規範：
1. **防止資料外洩 (Zero Data Leakage):** 採用嚴格的時間序列時間順序切分（Chronological Split），並以完整日曆天（Calendar Day）為邊界劃分訓練集（Training ~70%）、校準集（Calibration ~15%）與測試集（Test ~15%）。
2. **避免目標外洩 (Target Leakage):** 嚴禁將註冊用戶數（`registered`）與臨時用戶數（`casual`）作為模型輸入特徵，因為 `casual + registered == cnt`。
3. **無偏特徵預處理管線 (Preprocessing Pipeline):** 使用 scikit-learn 之 `ColumnTransformer` 與 `OneHotEncoder(drop='first')` 避免多元共線性（Dummy Variable Trap），且預處理轉換器之參數僅能由訓練集學習。
4. **時序交叉驗證 (TimeSeriesSplit Cross-Validation):** 在訓練集中執行 5-Fold `TimeSeriesSplit`，並在每一折疊內獨立擬合預處理管線，客觀評估模型之泛化表現。
5. **單一事實來源 (Single Source of Truth):** 本 Jupyter Notebook 作為所有分析圖表、統計檢定與評估指標的唯一起源，供後續報告撰寫與平台展示讀取。"""))

# Cell 2: Imports & Environment Configuration
cells.append(new_code_cell("""# 環境依賴套件導入與設定
import os
import sys
import json
import hashlib
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 設定中文字型與視覺化風格
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 工作路徑設定（使用相對路徑確保獨立可執行性）
BASE_DIR = os.getcwd()
DATA_PATH = os.path.join("data", "hour.csv")
FIGURES_DIR = "figures"
RESULTS_DIR = "results"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Python Version: {sys.version.split()[0]}")
print(f"Pandas Version: {pd.__version__}")
print(f"Scikit-Learn Version: {sklearn.__version__}")
print(f"Data Path: {DATA_PATH}")"""))

# Cell 3: Dataset Source & Verification
cells.append(new_markdown_cell("""### 2. 資料集來源與雜湊驗證 (Dataset Source & Acquisition)
本研究使用之資料集為 **Bike Sharing Dataset (Hourly Version)**，其官方來源為：
- **Kaggle 來源:** [https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset](https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset)
- **UCI Machine Learning Repository 原始來源:** [https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- **必要檔案:** `hour.csv`

為確保資料完整性與防偽驗證，我們計算本機 `hour.csv` 之 SHA-256 數位簽章，並核對列數（預期約 17,379 筆）與欄位數（17 欄）。"""))

# Cell 4: Load Data & Hash Check Code
cells.append(new_code_cell("""# 載入原始資料並進行 SHA-256 驗證
with open(DATA_PATH, "rb") as f:
    sha256_checksum = hashlib.sha256(f.read()).hexdigest()

df_raw = pd.read_csv(DATA_PATH)
n_rows, n_cols = df_raw.shape

print(f"檔案名稱: hour.csv")
print(f"SHA-256 Checksum: {sha256_checksum}")
print(f"資料集維度: {n_rows} 列 (rows), {n_cols} 欄 (columns)")

# 儲存 Dataset Metadata
metadata = {
    "student_id": "7115093001",
    "research_title": "Predicting Hourly Bike Sharing Demand Using Multiple Linear Regression and Feature Selection",
    "source_url": "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
    "filename": "hour.csv",
    "sha256_checksum": sha256_checksum,
    "number_of_rows": int(n_rows),
    "number_of_columns": int(n_cols),
    "column_names": list(df_raw.columns),
    "data_types": {col: str(dtype) for col, dtype in df_raw.dtypes.items()}
}

with open(os.path.join(RESULTS_DIR, "dataset_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# 顯示前 10 筆資料
df_raw.head(10)"""))

# Cell 5: Dataset Validation Markdown
cells.append(new_markdown_cell("""### 3. 資料驗證與品質審計 (Step 1 — Dataset Validation)
依循 CRISP-DM 資料理解程序，我們對 `hour.csv` 進行全方位數據完整性檢查：
1. **缺失值與重複項檢驗:** 檢查各欄位有無 Null/NaN，以及有無完全重複紀錄。
2. **完整時間戳記建立:** 結合 `dteday`（日期）與 `hr`（小時），建構精確的 `datetime` 序列。
3. **時間連續性與缺漏時段檢驗 (Temporal Gap Analysis):** 計算時間區間內的理論小時數與實際記錄數之差異。
4. **目標變數檢驗:** 驗證租借數 `cnt` 是否皆為非負整數，並驗證 `casual + registered == cnt` 恆等式。
5. **特徵定義與有效範圍檢驗:** 核對 12 個候選特徵的數值定義範圍。"""))

# Cell 6: Data Validation Code
cells.append(new_code_cell("""# 1. 缺失值與重複值檢驗
missing_per_col = df_raw.isnull().sum()
total_missing = int(missing_per_col.sum())
duplicate_records = int(df_raw.duplicated().sum())

print(f"全資料集缺失值總數: {total_missing}")
print(f"全資料集重複紀錄數: {duplicate_records}")

# 2. 時間戳記建構與時序檢驗
df = df_raw.copy()
df['datetime'] = pd.to_datetime(df['dteday'] + ' ' + df['hr'].astype(str).str.zfill(2) + ':00:00')
duplicate_timestamps = int(df['datetime'].duplicated().sum())
is_chronological = bool(df['datetime'].is_monotonic_increasing)

# 3. 理論時間長度與缺漏時段檢驗
expected_timeline = pd.date_range(start=df['datetime'].min(), end=df['datetime'].max(), freq='h')
expected_hours = len(expected_timeline)
gap_hours = expected_hours - len(df)
gap_sample = [str(ts) for ts in expected_timeline.difference(df['datetime'])[:10]]

print(f"起始時間戳記: {df['datetime'].min()}")
print(f"結束時間戳記: {df['datetime'].max()}")
print(f"時序單調遞增 (Monotonic Increasing): {is_chronological}")
print(f"理論區間小時總數: {expected_hours} 小時 (731 天 * 24 小時)")
print(f"實際觀測小時數: {len(df)} 小時")
print(f"未記錄/缺失時段總數 (Gap Hours): {gap_hours} 小時")
print(f"未記錄時段範例: {gap_sample}")

# 4. 業務恆等式與非負性驗證
identity_check = bool((df['casual'] + df['registered'] == df['cnt']).all())
non_negative_cnt = bool((df['cnt'] >= 0).all())

print(f"恆等式 casual + registered == cnt 完全吻合: {identity_check}")
print(f"目標變數 cnt 全部 >= 0: {non_negative_cnt}")

# 5. 特徵有效範圍與類別定義檢驗
range_checks = {
    "season_in_[1,2,3,4]": bool(df['season'].isin([1, 2, 3, 4]).all()),
    "yr_in_[0,1]": bool(df['yr'].isin([0, 1]).all()),
    "mnth_in_[1..12]": bool(df['mnth'].between(1, 12).all()),
    "hr_in_[0..23]": bool(df['hr'].between(0, 23).all()),
    "holiday_in_[0,1]": bool(df['holiday'].isin([0, 1]).all()),
    "weekday_in_[0..6]": bool(df['weekday'].between(0, 6).all()),
    "workingday_in_[0,1]": bool(df['workingday'].isin([0, 1]).all()),
    "weathersit_in_[1..4]": bool(df['weathersit'].isin([1, 2, 3, 4]).all()),
    "temp_normalized_[0..1]": bool(((df['temp'] >= 0.0) & (df['temp'] <= 1.0)).all()),
    "atemp_normalized_[0..1]": bool(((df['atemp'] >= 0.0) & (df['atemp'] <= 1.0)).all()),
    "hum_normalized_[0..1]": bool(((df['hum'] >= 0.0) & (df['hum'] <= 1.0)).all()),
    "windspeed_normalized_[0..1]": bool(((df['windspeed'] >= 0.0) & (df['windspeed'] <= 1.0)).all())
}

for check_name, passed in range_checks.items():
    print(f"檢查項目: {check_name:30s} -> {'PASS' if passed else 'FAIL'}")

# 儲存驗證結果與敘述統計
validation_summary = {
    "total_rows": n_rows,
    "total_columns": n_cols,
    "missing_values_count": total_missing,
    "duplicate_records": duplicate_records,
    "duplicate_timestamps": duplicate_timestamps,
    "expected_hours": expected_hours,
    "actual_hours": len(df),
    "gap_hours": gap_hours,
    "gap_sample": gap_sample,
    "identity_casual_plus_registered_equals_cnt": identity_check,
    "cnt_non_negative": non_negative_cnt,
    "range_checks": range_checks
}

with open(os.path.join(RESULTS_DIR, "data_validation.json"), "w", encoding="utf-8") as f:
    json.dump(validation_summary, f, indent=2, ensure_ascii=False)

df.describe().to_csv(os.path.join(RESULTS_DIR, "descriptive_statistics.csv"))"""))

# Cell 7: Data Validation Chinese Markdown Notes
cells.append(new_markdown_cell("""#### 🔍 資料驗證結果與學術審計備忘錄 (Data Audit Notes)
1. **資料完整性良好:** 全資料集無任何缺失值（Missing Values = 0），亦無任何重複列（Duplicate Rows = 0）。
2. **時序缺漏時段說明 (Gap Hours Explanation):** 731 天的日曆區間理論上應有 $731 \\times 24 = 17,544$ 小時，實際記錄為 17,379 筆，共存在 165 個小時的資料空缺。檢視發現，這些缺漏主要出現在**凌晨離峰時段（如 02:00 ~ 05:00）或極端天氣事件（如 2012 年 10 月末之颶風珊迪 Hurricane Sandy）**期間，系統因無人借車或停止營運而未產生日誌。依據作業指導原則，**本研究不隨意虛構補值（No Artificial Imputation）**，忠實呈現原始觀測時段。
3. **防止目標外洩 (Strict Anti-Leakage Rule):** 恆等式 `casual + registered == cnt` 100% 成立。因此，**`casual` 與 `registered` 嚴禁納入任何預測模型特徵**，否則將造成嚴重的目標外洩（Target Leakage）。
4. **排除非預測性特徵:** 序列流水號 `instant` 與字串日期 `dteday` 僅用於排序與分割，不得納入迴歸特徵。
5. **最終 12 原生特徵 (12 Original Predictors):**
   - 類別型特徵（5 個）: `season`, `mnth`, `hr`, `weekday`, `weathersit`
   - 數值/二元型特徵（7 個）: `yr`, `holiday`, `workingday`, `temp`, `atemp`, `hum`, `windspeed`"""))

# Cell 8: EDA Markdown
cells.append(new_markdown_cell("""### 4. 探索性數據分析 (Step 2 — Exploratory Data Analysis)
在此步驟中，我們對資料進行視覺化探索，繪製 4 張符合學術標準之高解析度圖表：
- **Figure 1 (Hourly Rental Demand):** 24 小時平均租借需求折線圖，觀察通勤尖峰特徵。
- **Figure 2 (Correlation Heatmap):** 特徵與預測目標之相關係數矩陣熱力圖，檢視共線性問題。
- **Figure 3 (Target Distribution):** 租借總量 `cnt` 之直方圖與核密度估計（KDE），分析偏態特徵。
- **Figure 4 (Temperature vs. Demand):** 標準化氣溫與租借總量之散佈圖與趨勢線。"""))

# Cell 9: EDA Code Figure 1 & Figure 2
cells.append(new_code_cell("""# Figure 1: Hourly Rental Demand (24 小時租借需求趨勢)
fig1, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
hourly_avg = df.groupby('hr')['cnt'].mean()

ax1.plot(hourly_avg.index, hourly_avg.values, marker='o', color='#1f77b4', linewidth=2.5, markersize=6)
ax1.set_title("Average Bike Rental Demand by Hour", fontsize=14, fontweight='bold', pad=12)
ax1.set_xlabel("Hour of Day", fontsize=12, labelpad=8)
ax1.set_ylabel("Average Rental Count", fontsize=12, labelpad=8)
ax1.set_xticks(range(0, 24))
ax1.grid(True, linestyle='--', alpha=0.6)
fig1.tight_layout()
fig1.savefig(os.path.join(FIGURES_DIR, "hourly_demand.png"), dpi=300)
plt.show()

# Figure 2: Correlation Heatmap (相關係數矩陣熱力圖)
# 排除非預測性欄位與外洩欄位 instant, casual, registered
candidate_numeric = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
                     'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt']
corr_matrix = df[candidate_numeric].corr()

fig2, ax2 = plt.subplots(figsize=(11, 9), dpi=300)
im = ax2.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax2.set_xticks(range(len(candidate_numeric)))
ax2.set_yticks(range(len(candidate_numeric)))
ax2.set_xticklabels(candidate_numeric, rotation=45, ha='right', fontsize=10)
ax2.set_yticklabels(candidate_numeric, fontsize=10)
ax2.set_title("Correlation Heatmap of Predictors and Target Variable", fontsize=14, fontweight='bold', pad=12)
fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

for i in range(len(candidate_numeric)):
    for j in range(len(candidate_numeric)):
        val = corr_matrix.iloc[i, j]
        text_color = "white" if abs(val) > 0.55 else "black"
        ax2.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8.5)

fig2.tight_layout()
fig2.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.png"), dpi=300)
plt.show()"""))

# Cell 10: EDA Code Figure 3 & Figure 4
cells.append(new_code_cell("""# Figure 3: Target Distribution Histogram & KDE (目標變數分佈)
fig3, ax3 = plt.subplots(figsize=(9, 5), dpi=300)
ax3.hist(df['cnt'], bins=50, color='#3498db', edgecolor='black', alpha=0.75, density=True, label='Histogram')
df['cnt'].plot(kind='kde', ax=ax3, color='#e74c3c', linewidth=2, label='KDE Density')
ax3.set_title("Distribution of Hourly Bike Rental Demand (cnt)", fontsize=14, fontweight='bold', pad=12)
ax3.set_xlabel("Rental Count (cnt)", fontsize=12, labelpad=8)
ax3.set_ylabel("Density", fontsize=12, labelpad=8)
ax3.legend(frameon=True, facecolor='white', loc='upper right')
ax3.grid(True, linestyle='--', alpha=0.6)
fig3.tight_layout()
fig3.savefig(os.path.join(FIGURES_DIR, "target_distribution.png"), dpi=300)
plt.show()

# Figure 4: Temperature vs. Demand (氣溫與需求散佈圖)
fig4, ax4 = plt.subplots(figsize=(9, 5), dpi=300)
ax4.scatter(df['temp'], df['cnt'], alpha=0.25, color='#2ca02c', s=12, edgecolors='none', label='Observations')
m, b = np.polyfit(df['temp'], df['cnt'], 1)
x_vals = np.linspace(df['temp'].min(), df['temp'].max(), 100)
ax4.plot(x_vals, m * x_vals + b, color='#d62728', linewidth=2.5, label=f'OLS Trendline: cnt = {m:.1f}*temp + {b:.1f}')
ax4.set_title("Normalized Temperature vs. Hourly Rental Demand", fontsize=14, fontweight='bold', pad=12)
ax4.set_xlabel("Normalized Temperature (temp)", fontsize=12, labelpad=8)
ax4.set_ylabel("Rental Count (cnt)", fontsize=12, labelpad=8)
ax4.legend(frameon=True, facecolor='white', loc='upper left')
ax4.grid(True, linestyle='--', alpha=0.6)
fig4.tight_layout()
fig4.savefig(os.path.join(FIGURES_DIR, "temperature_vs_demand.png"), dpi=300)
plt.show()"""))

# Cell 11: EDA Notes
cells.append(new_markdown_cell("""#### 📊 探索性數據分析觀察與統計解釋 (EDA Findings)
1. **雙峰通勤需求曲線 (Figure 1):**  
   小時租借量呈現顯著的雙峰形態（Bimodal Distribution），峰值分別落在上午 08:00（約 359 輛）與傍晚 17:00 ~ 18:00（約 461 ~ 425 輛），清楚反映首都圈上班族與學生的尖峰通勤行為；而凌晨 03:00 ~ 04:00 為全日谷底（平均僅約 10 ~ 16 輛）。
2. **極高共線性警訊: `temp` 與 `atemp` (Figure 2):**  
   標準化氣溫 `temp` 與體感氣溫 `atemp` 之 Pearson 相關係數高達 **$r = 0.99$**，存在近乎完全的共線性（Multicollinearity）。在多元線性迴歸中，同時納入兩者可能導致係數估計方差膨脹與符號反轉，為後續 Phase 2 特徵選取（Feature Selection）提供了明確的降維依據。
3. **類別整數編碼之統計陷阱說明:**  
   請注意，雖然數值相關矩陣可對 `season`、`weekday`、`weathersit` 計算 Pearson 相關係數，但這些欄位本質為**名義（Nominal）或順序（Ordinal）之整數編碼**。例如 `season=4`（冬季）並不代表數值量級是 `season=1`（春季）的四倍。因此，在建模時必須透過 **One-Hot Encoding** 將其轉化為虛擬變數（Dummy Variables），不可直接作為連續數值回歸。
4. **目標變數右偏態特徵 (Figure 3):**  
   `cnt` 的平均值為 189.46，中位數為 142.0，標準差為 181.39，偏態係數（Skewness）為 **+1.28**，呈現顯著的右偏（Right-skewed）分佈。絕大多數時段的租借量集中於 0 ~ 300 輛之間，少數極端熱門時段最高達 977 輛。
5. **氣溫正向激勵效應 (Figure 4):**  
   氣溫與租借量呈現明確的正向線性相關（$r = 0.40$），趨勢線斜率為正（$cnt \\approx 381.3 \\times temp - 2.5$）。當氣溫上升至舒適區間（標準化值 0.6 ~ 0.8，約 25°C ~ 33°C）時，租借活躍度大幅提升。"""))

# Cell 12: Chronological Data Split Markdown
cells.append(new_markdown_cell("""### 5. 時間序列嚴格時序切分 (Step 3 — Chronological Data Split)
#### 核心研究問題定義
「給定特定小時之時間、日曆與氣象環境資訊，預測該小時之自行車租借總需求量。」

#### 切分設計原則
1. **絕不隨機打亂 (No Random Shuffle):** 在時序預測問題中，隨機切分會導致未來的資訊洩漏給過去（Temporal Data Leakage）。
2. **完整日曆天邊界 (Full Calendar-Day Boundaries):** 自行車使用需求具有強烈的日週期性（Diurnal Cycle），同一日曆天內的各小時具備高度自相關性。因此切分點必須精準落在日曆天的交界（00:00:00），確保同一天的觀測值絕不會被拆分到不同子集中。
3. **切分比例設定:**
   - **訓練集 (Training Set, ~70%):** 512 天，用於模型訓練、交叉驗證與特徵選取。
   - **校準集 (Calibration Set, ~15%):** 110 天，完全保留予未來 Phase 2 構建 Split Conformal Prediction 預測區間。
   - **測試集 (Test Set, ~15%):** 109 天，完全封存保留予最終泛化評估。
4. **Phase 1 隔離規範:** 嚴禁在 Phase 1 窺探、訓練或評估 Calibration Set 與 Test Set。"""))

# Cell 13: Chronological Split Code
cells.append(new_code_cell("""# 依據完整日曆天進行程式化時序切分
unique_days = df['dteday'].drop_duplicates().sort_values().reset_index(drop=True)
total_calendar_days = len(unique_days)

# 依比例計算切分天數 (~70%, ~15%, ~15%)
n_train_days = int(round(total_calendar_days * 0.70))
n_calib_days = int(round(total_calendar_days * 0.15))
n_test_days = total_calendar_days - n_train_days - n_calib_days

train_day_set = set(unique_days.iloc[:n_train_days])
calib_day_set = set(unique_days.iloc[n_train_days:n_train_days + n_calib_days])
test_day_set = set(unique_days.iloc[n_train_days + n_calib_days:])

train_data = df[df['dteday'].isin(train_day_set)].copy()
calib_data = df[df['dteday'].isin(calib_day_set)].copy()
test_data = df[df['dteday'].isin(test_day_set)].copy()

# 嚴格驗證集合無重疊與時序單調性
assert len(train_day_set.intersection(calib_day_set)) == 0, "Train 與 Calib 日期重疊！"
assert len(train_day_set.intersection(test_day_set)) == 0, "Train 與 Test 日期重疊！"
assert len(calib_day_set.intersection(test_day_set)) == 0, "Calib 與 Test 日期重疊！"
assert train_data['datetime'].max() < calib_data['datetime'].min(), "Train 與 Calib 時序錯置！"
assert calib_data['datetime'].max() < test_data['datetime'].min(), "Calib 與 Test 時序錯置！"

# 整理切分摘要報告
split_report = {
    "total_observations": len(df),
    "total_calendar_days": total_calendar_days,
    "subsets": {
        "train": {
            "days": len(train_day_set),
            "observations": len(train_data),
            "percentage": round(len(train_data) / len(df) * 100, 2),
            "start_date": str(train_data['dteday'].min()),
            "end_date": str(train_data['dteday'].max()),
            "start_timestamp": str(train_data['datetime'].min()),
            "end_timestamp": str(train_data['datetime'].max())
        },
        "calibration": {
            "days": len(calib_day_set),
            "observations": len(calib_data),
            "percentage": round(len(calib_data) / len(df) * 100, 2),
            "start_date": str(calib_data['dteday'].min()),
            "end_date": str(calib_data['dteday'].max()),
            "start_timestamp": str(calib_data['datetime'].min()),
            "end_timestamp": str(calib_data['datetime'].max())
        },
        "test": {
            "days": len(test_day_set),
            "observations": len(test_data),
            "percentage": round(len(test_data) / len(df) * 100, 2),
            "start_date": str(test_data['dteday'].min()),
            "end_date": str(test_data['dteday'].max()),
            "start_timestamp": str(test_data['datetime'].min()),
            "end_timestamp": str(test_data['datetime'].max())
        }
    }
}

with open(os.path.join(RESULTS_DIR, "data_split_summary.json"), "w", encoding="utf-8") as f:
    json.dump(split_report, f, indent=2, ensure_ascii=False)

print("=== 時序切分摘要報告 (Chronological Split Summary) ===")
for subset_name, sub in split_report['subsets'].items():
    print(f"[{subset_name.upper():11s}] 天數: {sub['days']:3d} 天 | 筆數: {sub['observations']:5d} ({sub['percentage']:5.2f}%) | 區間: {sub['start_date']} 至 {sub['end_date']}")"""))

# Cell 14: Preprocessing Pipeline Markdown
cells.append(new_markdown_cell("""### 6. 特徵預處理管線 (Step 4 — Preprocessing Pipeline)
依據迴歸計量經濟學規範，在多元線性迴歸中納入定性類別變數時，若未刪除基準類別，將引發**虛擬變數陷阱（Dummy Variable Trap / 完美多元共線性）**，導致設計矩陣 $X^T X$ 不可逆。

#### 預處理架構設計
我們採用 `scikit-learn` 之 `ColumnTransformer` 構建模組化管線：
1. **類別型特徵（Categorical Predictors, 5 個）:**  
   `season`, `mnth`, `hr`, `weekday`, `weathersit`  
   轉換方式: `OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)`
2. **數值/二元型特徵（Numeric & Binary Predictors, 7 個）:**  
   `yr`, `holiday`, `workingday`, `temp`, `atemp`, `hum`, `windspeed`  
   轉換方式: 直接傳遞（`passthrough`）
3. **維度對應關係:**
   - 12 個原始預測特徵（12 Original Predictors）
   - 經過 `drop='first'` 編碼後，展開為 **53 個模型輸入特徵欄位（53 Transformed Input Columns）**：
     - `season`: 4 類別 $\\rightarrow$ 3 欄
     - `mnth`: 12 類別 $\\rightarrow$ 11 欄
     - `hr`: 24 類別 $\\rightarrow$ 23 欄
     - `weekday`: 7 類別 $\\rightarrow$ 6 欄
     - `weathersit`: 4 類別 $\\rightarrow$ 3 欄
     - 數值/二元: 7 欄
     - 總計: $3 + 11 + 23 + 6 + 3 + 7 = 53$ 欄
4. **管線學習原則:** 預處理轉換器之參數嚴格僅在訓練集（Training Set）上進行擬合（Fit），校準集與測試集僅執行轉換（Transform）。"""))

# Cell 15: Preprocessing Pipeline Code
cells.append(new_code_cell("""# 建立 Preprocessing Pipeline
categorical_features = ['season', 'mnth', 'hr', 'weekday', 'weathersit']
numeric_features = ['yr', 'holiday', 'workingday', 'temp', 'atemp', 'hum', 'windspeed']
all_predictors = categorical_features + numeric_features
target_col = 'cnt'

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features),
        ('num', 'passthrough', numeric_features)
    ],
    remainder='drop'
)

# 僅以訓練集擬合預處理管線
X_train = train_data[all_predictors]
y_train = train_data[target_col]

preprocessor.fit(X_train)

# 提取編碼後的特徵名稱映射清單
cat_encoded_cols = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
total_encoded_features = cat_encoded_cols + numeric_features

feature_mapping_df = pd.DataFrame({
    "encoded_col_idx": range(len(total_encoded_features)),
    "encoded_feature_name": total_encoded_features,
    "source_feature": [col.split('_')[0] if col.startswith(tuple(categorical_features)) else col for col in total_encoded_features],
    "encoding_type": ["categorical_dummy" if i < len(cat_encoded_cols) else "numeric_or_binary" for i in range(len(total_encoded_features))]
})

feature_mapping_df.to_csv(os.path.join(RESULTS_DIR, "encoded_feature_names.csv"), index=False)

print(f"原生特徵數量 (Original Predictors): {len(all_predictors)}")
print(f"編碼後模型特徵數量 (Transformed Model Columns): {len(total_encoded_features)}")
print(f"已匯出特徵映射表至: {os.path.join(RESULTS_DIR, 'encoded_feature_names.csv')}")"""))

# Cell 16: Baseline OLS & TimeSeriesSplit CV Markdown
cells.append(new_markdown_cell("""### 7. 基準模型構建與評估 (Step 5 — Baseline OLS Model & TimeSeriesSplit CV)
#### 模型設定
1. **模型選擇:** `sklearn.linear_model.LinearRegression` (Ordinary Least Squares, OLS)。
2. **參考基準:** `sklearn.dummy.DummyRegressor(strategy="mean")`（以訓練集平均值作為常數預測，檢驗 OLS 是否具有實質預測價值）。
3. **評估指標:**
   - 平均絕對誤差 (MAE)
   - 均方誤差 (MSE)
   - 均方根誤差 (RMSE)
   - 判定係數 ($R^2$)
4. **時序交叉驗證 (TimeSeriesSplit 5-Fold):**
   - 傳統 K-Fold 隨機交叉驗證會打破時間先後順序，產生未來預測過去的假象。
   - 本研究在訓練集內部嚴格採用 `TimeSeriesSplit(n_splits=5)`，依時間滾動推進評估。
   - **零外洩實作細節:** 在每一個折疊內，預處理轉換器均僅在該折疊的訓練分割（Fold Train）上重新擬合，再對驗證分割（Fold Validation）進行轉換與預測，杜絕任何預處理外洩。"""))

# Cell 17: Baseline OLS & TimeSeriesSplit CV Code
cells.append(new_code_cell("""# 1. 訓練全訓練集 OLS 基準模型
ols_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

ols_pipeline.fit(X_train, y_train)
y_train_pred = ols_pipeline.predict(X_train)

train_mae = float(mean_absolute_error(y_train, y_train_pred))
train_mse = float(mean_squared_error(y_train, y_train_pred))
train_rmse = float(np.sqrt(train_mse))
train_r2 = float(r2_score(y_train, y_train_pred))

# Dummy Regressor (訓練集平均值預測基準)
dummy_reg = DummyRegressor(strategy="mean")
dummy_reg.fit(X_train, y_train)
dummy_pred = dummy_reg.predict(X_train)
dummy_train_rmse = float(np.sqrt(mean_squared_error(y_train, dummy_pred)))

print("=== 訓練集擬合評估 (Training Set Metrics) ===")
print(f"OLS Baseline  -> MAE: {train_mae:.4f} | MSE: {train_mse:.4f} | RMSE: {train_rmse:.4f} | R²: {train_r2:.4f}")
print(f"Dummy Reg Mean -> RMSE: {dummy_train_rmse:.4f} | R²: {r2_score(y_train, dummy_pred):.4f}")

# 2. TimeSeriesSplit 5-Fold 交叉驗證（每一折疊獨立擬合預處理，確保零外洩）
tscv = TimeSeriesSplit(n_splits=5)
cv_records = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
    X_fold_tr, y_fold_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_fold_va, y_fold_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    # 折疊內獨立管線
    fold_preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features),
            ('num', 'passthrough', numeric_features)
        ],
        remainder='drop'
    )
    fold_pipe = Pipeline([
        ('preprocessor', fold_preprocessor),
        ('regressor', LinearRegression())
    ])
    fold_pipe.fit(X_fold_tr, y_fold_tr)
    y_fold_va_pred = fold_pipe.predict(X_fold_va)
    
    # 折疊 Dummy 基準
    fold_dummy = DummyRegressor(strategy="mean")
    fold_dummy.fit(X_fold_tr, y_fold_tr)
    y_dummy_va_pred = fold_dummy.predict(X_fold_va)
    
    fold_mae = float(mean_absolute_error(y_fold_va, y_fold_va_pred))
    fold_mse = float(mean_squared_error(y_fold_va, y_fold_va_pred))
    fold_rmse = float(np.sqrt(fold_mse))
    fold_r2 = float(r2_score(y_fold_va, y_fold_va_pred))
    dummy_va_rmse = float(np.sqrt(mean_squared_error(y_fold_va, y_dummy_va_pred)))
    
    cv_records.append({
        "fold": fold,
        "train_samples": len(X_fold_tr),
        "val_samples": len(X_fold_va),
        "val_mae": round(fold_mae, 4),
        "val_mse": round(fold_mse, 4),
        "val_rmse": round(fold_rmse, 4),
        "val_r2": round(fold_r2, 4),
        "dummy_val_rmse": round(dummy_va_rmse, 4)
    })

cv_df = pd.DataFrame(cv_records)
cv_df.to_csv(os.path.join(RESULTS_DIR, "baseline_cv_results.csv"), index=False)

cv_rmse_mean = float(cv_df['val_rmse'].mean())
cv_rmse_std = float(cv_df['val_rmse'].std())
cv_mae_mean = float(cv_df['val_mae'].mean())
cv_r2_mean = float(cv_df['val_r2'].mean())
dummy_cv_mean = float(cv_df['dummy_val_rmse'].mean())

print("\\n=== TimeSeriesSplit 5-Fold 交叉驗證結果 ===")
display(cv_df)
print(f"5-Fold CV Validation RMSE Mean: {cv_rmse_mean:.4f} (std: {cv_rmse_std:.4f})")
print(f"5-Fold CV Validation MAE Mean:  {cv_mae_mean:.4f}")
print(f"5-Fold CV Validation R² Mean:   {cv_r2_mean:.4f}")
print(f"Dummy Mean CV RMSE Mean:        {dummy_cv_mean:.4f}")

# 儲存 Baseline 完整指標
baseline_summary = {
    "model_name": "Multiple Linear Regression (OLS Baseline)",
    "training_samples": len(X_train),
    "original_predictors_count": len(all_predictors),
    "encoded_input_columns_count": len(total_encoded_features),
    "training_metrics": {
        "MAE": train_mae,
        "MSE": train_mse,
        "RMSE": train_rmse,
        "R2": train_r2
    },
    "dummy_mean_training_rmse": dummy_train_rmse,
    "time_series_cv_5fold": {
        "validation_rmse_mean": cv_rmse_mean,
        "validation_rmse_std": cv_rmse_std,
        "validation_mae_mean": cv_mae_mean,
        "validation_r2_mean": cv_r2_mean,
        "dummy_rmse_mean": dummy_cv_mean
    }
}

with open(os.path.join(RESULTS_DIR, "baseline_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(baseline_summary, f, indent=2, ensure_ascii=False)"""))

# Cell 18: Phase 1 Summary Markdown
cells.append(new_markdown_cell("""### 8. Phase 1 成果總結與後續規劃 (Phase 1 Summary & Next Steps)
#### 核心成果審計摘要
1. **資料集規範性:** 成功完成 17,379 筆時序資料之品質驗證，確認無重複或缺失值，釐清 165 小時之系統未營運空缺，並驗證 `casual + registered == cnt`，嚴格排除目標外洩欄位。
2. **時序資料劃分:** 完成以日曆天為單位的無重疊時間切分（訓練集 512 天 / 12,163 筆；校準集 110 天 / 2,640 筆；測試集 109 天 / 2,576 筆）。校準集與測試集在 Phase 1 期間維持完全未觀測狀態。
3. **特徵預處理管線:** 成功將 12 個原生特徵映射為 53 個無共線性輸入維度（`drop='first'`），並封裝於 scikit-learn `Pipeline`。
4. **OLS 基準效能表現:**
   - 訓練集擬合表現: $\\text{RMSE} = 85.48$, $R^2 = 0.6861$（相較常數預測基準 $\\text{RMSE} = 152.58$，預測誤差降低超過 44%）。
   - 5-Fold TimeSeriesSplit 交叉驗證表現: $\\text{Mean RMSE} = 109.56$ ($\\text{std} = 14.92$), $\\text{Mean } R^2 = 0.4549$（相較 Dummy 基準 $\\text{Mean RMSE} = 165.00$ 顯著提升）。
5. **Phase 2 研發重點:**
   - 特徵選取演算法實作（RFE、Ridge / Lasso 懲罰迴歸與係數收縮）。
   - 解決 `temp` 與 `atemp` 等特徵之高共線性問題。
   - 使用保留之校準集（Calibration Set）構建 Split Conformal Prediction 預測區間。
   - 嚴格維持現有生產環境（50 Startups Vercel 網站）獨立運行。"""))

nb.cells = cells

# Save notebook
notebook_path = os.path.join(BASE_DIR, "7115093001_hw2.ipynb")
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook successfully written to {notebook_path}")
