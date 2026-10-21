# AIoT HW2 — Problem 2: Appliances Energy Prediction

**學生學號 (Student ID):** 7115093001  
**研究題目 (Research Title):** Multiple Linear Regression & Feature Selection for Residential Appliances Energy Prediction  
**儲存庫 (Repository):** [https://github.com/Augustine-723/AIoT_HW2](https://github.com/Augustine-723/AIoT_HW2)  
**線上展示網站 (Website):** [https://a-io-t-hw-2.vercel.app/](https://a-io-t-hw-2.vercel.app/)  
**實驗識別碼 (Experiment ID):** `AIoT_HW2_P2_EXP01`  

---

## 摘要與資料集來源 (Dataset Acquisition & Sources)

本研究為 **AIoT HW2 第二個獨立研究問題**，聚焦於低耗能建築 (Low-energy House) 之家用電器每 10 分鐘耗電量 (Appliances Energy Consumption, 單位 Wh) 的統計多元線性迴歸建模、遞迴特徵淘汰 (RFE)、正則化比較與分裂共形預測區間 (Split Conformal Prediction Interval)。

### 資料來源驗證狀態 (Source Verification)
1. **UCI 原始發佈庫 (Official Data Source):**  
   - [UCI Machine Learning Repository: Appliances Energy Prediction (Dataset ID: 374)](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)  
   - 官方下載連結: `https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip`
2. **原始作者開源庫 (Original Author Repository):**  
   - Luis M. Candanedo (Université de Mons, Belgium)  
   - [GitHub: Appliances-energy-prediction-data](https://github.com/LuisM78/Appliances-energy-prediction-data)  
   - 原始完整檔案: `energydata_complete.csv` (11,979,363 bytes)
3. **Kaggle 來源比對 (Kaggle Status):**  
   - Kaggle 競賽參考來源: [Machine Learning SB1 Competition](https://www.kaggle.com/competitions/machine-learning-sb1) (競賽頁面公開可訪；原生檔案下載受 Kaggle 帳號認證與競賽條款接受限制)。  
   - Kaggle 公開同源資料集: [Appliances Energy Prediction on Kaggle](https://www.kaggle.com/datasets/loveall/appliances-energy-prediction)  
   - **位元全等校驗:** UCI 官方壓縮檔解壓之 `energydata_complete.csv` 與作者 GitHub 原始檔之 SHA-256 均為 `2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d`，100% 位元完全相同。

---

## 原始欄位審計與特徵分層定義 (Data Audit & Feature Layers)

依據實際 CSV 程式審計結果 (`problem2/results/data_audit.json`)：
- **總觀測數 (Observations):** 19,735 筆 (2016-01-11 17:00:00 至 2016-05-27 18:00:00，嚴格 10 分鐘連續間隔，0 缺漏、0 重複、0 缺失值)。
- **特徵層次結構嚴格統一說明:**
  1. **29 個原始資料欄位:** CSV 全部欄位（`date` 時間戳記 1 個 + `Appliances` 標的 1 個 + `rv1`, `rv2` 隨機對照 2 個 + 實體感測及氣象欄位 25 個）。
  2. **25 個原始感測及氣象預測變數:**
     - 燈光能耗 (1): `lights` (室內燈具耗電量，Wh)。
     - 室內溫度 (9): `T1` (廚房), `T2` (客廳), `T3` (洗衣間), `T4` (辦公室), `T5` (浴室), `T6` (外牆), `T7` (燙衣間), `T8` (次臥), `T9` (主臥)，單位 °C。
     - 室內相對濕度 (9): `RH_1` ~ `RH_9`，單位 %。
     - 氣象站戶外觀測 (6): `T_out` (氣象站溫度), `Press_mm_hg` (氣壓), `RH_out` (戶外濕度), `Windspeed` (風速), `Visibility` (能見度), `Tdewpoint` (露點溫度)。
  3. **26 個基本模型輸入特徵（包含衍生 hour）:** 25 個原始感測及氣象預測變數 + 主時間特徵 `hour` (時段，0~23)。嚴格排除 `rv1` 與 `rv2` 兩個隨機欄位。
  4. **29 個擴充候選特徵:** 26 個基本模型輸入特徵 + 日週期轉換特徵 `sin_hour`, `cos_hour` 與行事曆平日/週末標記 `is_weekend`。

---

## CRISP-DM 六大階段完整實作

### 1. Business Understanding (業務與問題理解)
- **應用場景:** 智慧家庭物聯網 (AIoT) 電器能耗即時診斷、尖峰電力負載預警與動態需量反應管理。
- **預測本質定義:** 模型輸入包含同步之全戶 ZigBee 感測節點遙測與氣象站資訊，本質為**同步能耗狀態估計 (Synchronous Telemetry Energy Estimation)**。
- **物理非負約束:** 電器耗電量具備不可為負 ($Wh \\ge 0$) 之物理邊界。

### 2. Data Understanding (資料探索與審計)
- 探索性資料分析 (EDA) 視覺化圖表已生成至 `problem2/figures/`：
  - `eda_appliances_distribution.png`: 標的變數分布與右偏特性 (偏度 3.386)。
  - `eda_time_series_profile.png`: 138 天時間序列歷程與切分邊界。
  - `eda_correlation_matrix.png`: 各區域溫濕度與氣象變數之 Pearson 相關係數矩陣。
- **隨機變數健全性驗證:** `rv1` 與 `rv2` 與標的相關性僅 $-0.01114$，於特徵選擇中皆名列最先被淘汰群體 (淘汰名次 30/31 與 31/31)。

### 3. Data Preparation (資料準備與時序防洩漏切分)
- **時序日曆對齊切分 (Full-Date Aligned Split):**
  - **Training Set (70.26%):** 2016-01-11 ~ 2016-04-16 (97 天，13,866 筆)
  - **Calibration Set (15.32%):** 2016-04-17 ~ 2016-05-07 (21 天，3,024 筆，專供共形區間分位數計算)
  - **Test Set (14.42%):** 2016-05-08 ~ 2016-05-27 (20 天，2,845 筆，凍結獨立評估)
- **Pipeline 防洩漏機制:** 所有特徵標準化 (`StandardScaler`) 封裝於 `sklearn.pipeline.Pipeline` 內部，僅在 Training 期擬合。

### 4. Modeling (多元線性迴歸、特徵淘汰與正則化)
於 Training Set 執行 5-Fold Expanding-Window Cross Validation (`TimeSeriesSplit(n_splits=5)`)，報告無偏樣本標準差 ($ddof=1$)：

| 模型代號 | 模型名稱 | 特徵數 | 超參數設定 | 5-Fold CV RMSE (Wh) | CV 標準差 ($ddof=1$) |
|:---:|:---|:---:|:---|:---:|:---:|
| **M0** | DummyRegressor (Mean Baseline) | 0 | `strategy='mean'` | 102.7427 | $\pm$ 8.0952 |
| **M1** | Full Multiple Linear Regression (OLS) | 29 | `fit_intercept=True` | 98.0718 | $\pm$ 7.3897 |
| **M2** | RFE + OLS (Optimal Subset) | 6 | $k=6$, `step=1` | 96.2026 | $\pm$ 5.8893 |
| **M3** | Ridge Regression (L2 Regularized) | 29 | $\alpha=1000.0$ | **95.8454** | $\pm$ 6.2597 |
| **M4** | Lasso Regression (L1 Regularized) | 29 | $\alpha=2.0$ | 95.9075 | $\pm$ 6.3640 |

> **模型決策:**  
> 依據嚴格 Training CV 結果，**Ridge Regression (M3, $\alpha=1000.0$)** 達到最低 CV RMSE (95.8454 Wh)，獲選為最終部署模型；同時 **RFE + OLS (M2, $k=6$)** 篩選出具備高度可解釋性之 6 大核心特徵 (`RH_1`, `RH_2`, `T3`, `T2`, `sin_hour`, `cos_hour`)。

### 5. Evaluation (獨立測試集嚴格評估與共形區間)
以 Training 期最佳模型 M3 (Ridge) 於獨立測試集 (2,845 筆) 進行盲測評估：

- **測試集點預測表現:**
  - **Test MAE:** **49.2955 Wh** (基準線 Dummy: 53.2476 Wh)
  - **Test RMSE:** **83.9152 Wh** (基準線 Dummy: 91.0182 Wh)
  - **Test $R^2$:** **0.1494** (基準線 Dummy: -0.0008)
- **名義 95% 分裂共形預測區間 (Split Conformal Prediction Interval):**
  - 校準集樣本數 ($n$): 3,024 筆
  - 分位數秩次: $\lceil (n+1) \times 0.95 \rceil = 2874$ ($p = 0.9504$)
  - **校準共形分位數 ($q$):** **143.8311 Wh**
  - **名義目標覆蓋率:** 95.0%
  - **測試集實證覆蓋率 (未截斷):** **95.25%** (平均區間寬度: 287.66 Wh)
  - **測試集實證覆蓋率 (非負截斷):** **95.25%** (平均區間寬度: 245.98 Wh)
- **可交換性 (Exchangeability) 討論:**  
  標準 Split Conformal 預測依賴樣本可交換性 (i.i.d.) 假設。在時間序列下，環境感測與生活用電具備顯著時序自相關性，可交換性前提理論上未嚴格滿足。因此 95% 僅作為名義設定目標，實際可靠度必須以獨立測試集之經驗覆蓋率 (95.25%) 為準。

### 6. Deployment (推論規格匯出)
- 序列化管線: `problem2/models/best_model.joblib`
- 確定性推論規格: `problem2/results/model_spec.json` 與 `problem2/results/m2_model_spec.json`
- 前端推論採 Option B 純客戶端確定性架構，在 Next.js 中執行標準化與矩陣點乘。

---

## 驗收確認清單 (Verification Log)

| 階段與項目 | 驗收狀態 | 驗證依據與輸出產物 |
|:---|:---:|:---|
| **Stage 1: Dataset Acquisition** | **PASS** | CSV SHA-256 位元完全校驗；`problem2/results/data_audit.json` 完整記錄 19,735 筆、29 欄位。 |
| **Stage 2: CRISP-DM Pipeline** | **PASS** | 完整實作 6 大階段；70/15/15 時序日曆對齊切分；Pipeline 內標準化無洩漏。 |
| **Stage 3: Feature Selection** | **PASS** | RFE 遍歷 $k=5 \sim 29$；`selected_features.csv` 與 `model_comparison.csv` 產出；隨機變數健全性通過。 |
| **Stage 4: Evaluation & Conformal** | **PASS** | 獨立測試集評估 (MAE 49.30 Wh, RMSE 83.92 Wh)；95% 共形區間覆蓋率達 95.25%；非負截斷寬度 245.98 Wh。 |
| **Stage 5: Website Integration** | **PASS** | 下拉選單雙 Problem 切換；Option B 確定性推論；Parity 測試誤差達機器精度極限。 |
| **Stage 6: Custom Dataset Bonus** | **PASS** | `/custom-dataset` 支援 URL/上傳、動態多元線性迴歸、圖表與 SSRF 安全防護。 |
| **Stage 7: Delivery & README** | **PASS** | 雙問題完整文檔、Notebook 可重現執行、CI 檢查全部通過。 |
