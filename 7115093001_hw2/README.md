# AIoT HW2 — Phase 1 專案研究報告 (Research Workspace)

**學號 (Student ID):** 7115093001  
**研究題目 (Research Title):** Predicting Hourly Bike Sharing Demand Using Multiple Linear Regression and Feature Selection  
**研究階段 (Phase):** Phase 1 — 資料驗證、探索性數據分析、嚴格時序切分、預處理管線與 OLS 基準模型構建  
**正式部署網站 (Existing Production Website):** [https://50-starup-i9za.vercel.app/](https://50-starup-i9za.vercel.app/)  

---

## 1. 專案目錄結構 (Project Structure)

本研究工作區獨立構建於既有儲存庫中之 `7115093001_hw2/` 目錄，完全隔離於生產環境前端與後端，確保現有 50 Startups Vercel 應用程式不受任何影響：

```text
7115093001_hw2/
├── 7115093001_hw2.ipynb        # 研究核心 Notebook（單一事實來源，包含完整圖表與執行輸出）
├── README.md                   # Phase 1 完整研究技術報告與規範備忘錄
├── requirements.txt            # 獨立環境依賴清單
├── run_phase1.py               # Phase 1 自動化驗證、運算與成果產出腳本
├── build_notebook.py           # Notebook 建構與生成腳本
├── data/
│   └── hour.csv                # 官方 Bike Sharing 每小時資料集 (17,379 筆)
├── figures/                    # 學術級高解析度圖表 (300 DPI)
│   ├── hourly_demand.png       # Figure 1: 24 小時平均租借需求趨勢圖
│   ├── correlation_heatmap.png # Figure 2: 候選特徵與目標相關係數熱力圖
│   ├── target_distribution.png # Figure 3: 目標變數 cnt 之分佈直方圖與 KDE
│   └── temperature_vs_demand.png # Figure 4: 標準化氣溫與租借總量散佈圖及趨勢線
└── results/                    # 機器可讀之實驗結果資料庫 (JSON & CSV)
    ├── dataset_metadata.json   # 資料集來源網址、維度、欄位型態與環境版本
    ├── data_validation.json    # 資料品質審計、缺失值、時序連續性與業務檢定
    ├── descriptive_statistics.csv # 全欄位敘述統計數據
    ├── data_split_summary.json # 完整日曆天時序切分區間與樣本統計
    ├── encoded_feature_names.csv # 53 個編碼特徵對應原生特徵映射表
    ├── baseline_metrics.json   # OLS 基準模型訓練指標與 5-Fold CV 總結
    └── baseline_cv_results.csv # 5-Fold TimeSeriesSplit 各折疊驗證指標
```

---

## 2. 資料集來源與雜湊驗證 (Dataset Verification)

| 項目 | 規格與檢驗值 |
| :--- | :--- |
| **資料集名稱** | Bike Sharing Dataset (Hourly Version) |
| **官方 Kaggle 來源** | [https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset](https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset) |
| **UCI 原始來源** | [https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) |
| **檔案名稱** | `hour.csv` |
| **SHA-256 Checksum** | `e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f` |
| **樣本維度** | 17,379 列 (rows) $\times$ 17 欄 (columns) |
| **預測目標 (Target)** | `cnt` (每小時自行車租借總量，非負整數) |
| **目標外洩排除欄位** | `casual`, `registered`（業務恆等式 `casual + registered == cnt` 100% 吻合，為杜絕 Target Leakage，嚴禁輸入模型） |
| **非預測性排除欄位** | `instant` (流水號), `dteday` (字串日期) |

---

## 3. 資料驗證與品質審計報告 (Step 1 — Data Validation)

1. **缺失值與重複項:**
   - 全欄位缺失值總數: `0` (No missing values)。
   - 重複列與重複時間戳記數: `0` (No duplicated rows or timestamps)。
2. **時序連續性與缺漏時段分析 (Temporal Gap Audit):**
   - 起始時間: `2011-01-01 00:00:00`，結束時間: `2012-12-31 23:00:00`。
   - 731 天理論總小時數: $731 \times 24 = 17,544$ 小時。
   - 實際記錄小時數: `17,379` 小時，共存在 **165 個小時之資料空缺**。
   - **學術審計說明:** 抽樣檢驗顯示，空缺集中於**凌晨無人租借時段（如 03:00、04:00）或極端天氣事件（如 2012 年 10 月末 Hurricane Sandy 颶風期間暫停營運）**。依據規範，研究**嚴禁自行捏造插值補全（No Artificial Imputation）**，忠實維持真實觀測日誌。
3. **特徵範圍驗證:**
   - 季節（`season` $\in [1..4]$）、年份（`yr` $\in [0, 1]$）、月份（`mnth` $\in [1..12]$）、小時（`hr` $\in [0..23]$）、假日（`holiday` $\in [0, 1]$）、星期（`weekday` $\in [0..6]$）、工作日（`workingday` $\in [0, 1]$）、天氣（`weathersit` $\in [1..4]$）、氣溫（`temp` $\in [0, 1]$）、體感氣溫（`atemp` $\in [0, 1]$）、濕度（`hum` $\in [0, 1]$）、風速（`windspeed` $\in [0, 1]$）全數通過範圍與定義檢驗。

---

## 4. 探索性數據分析摘要 (Step 2 — EDA Findings)

- **Figure 1 (Average Bike Rental Demand by Hour):** 呈現鮮明的雙峰通勤模式，峰值位於 08:00（359 輛）與 17:00 ~ 18:00（461 ~ 425 輛），深具通勤尖峰特徵。
- **Figure 2 (Correlation Heatmap):** 標準化氣溫 `temp` 與體感氣溫 `atemp` 之相關係數高達 **$r = 0.99$**，存在近乎完全之多元共線性；此外，明確指認 `season`、`weekday`、`weathersit` 為名義/順序編碼，不可逕行當作連續測量數值進行線性迴歸。
- **Figure 3 (Target Distribution):** 租借總量 `cnt` 之平均值為 189.46、中位數 142.0，偏態係數為 **+1.28**，呈現顯著右偏態分佈。
- **Figure 4 (Temperature vs. Demand):** 氣溫與租借量具有明確正向關係（$r = 0.40$），趨勢線斜率為正（$cnt \approx 381.3 \times temp - 2.5$）。

---

## 5. 嚴格時序資料切分 (Step 3 — Chronological Split)

為防止未來資訊外洩至歷史資料（Temporal Data Leakage），且考量自行車租借行為具有強烈之日週期自相關性，切分點精準鎖定在**完整日曆天邊界（Full Calendar-Day Boundaries）**：

| 資料子集 (Subset) | 涵蓋日曆天數 | 樣本筆數 (Observations) | 佔比 (%) | 起始日期與時間 | 結束日期與時間 | 研究階段職責 (Role) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Training (訓練集)** | **512 天** | **12,163 筆** | **69.99%** | 2011-01-01 00:00 | 2012-05-26 23:00 | 模型擬合、特徵選取與時序交叉驗證 |
| **Calibration (校準集)** | **110 天** | **2,640 筆** | **15.19%** | 2012-05-27 00:00 | 2012-09-13 23:00 | **保留**予 Phase 2 Conformal Prediction 區間校準 |
| **Test (測試集)** | **109 天** | **2,576 筆** | **14.82%** | 2012-09-14 00:00 | 2012-12-31 23:00 | **完全封存**保留予最終泛化能力評估 |

> [!IMPORTANT]
> 嚴格遵守方法論規範：Phase 1 期間 **Calibration Set 與 Test Set 保持完全未觀測（Unseen）狀態**，絕不用於挑選模型或調整超參數。

---

## 6. 特徵預處理管線 (Step 4 — Preprocessing Pipeline)

使用 `scikit-learn` 之 `ColumnTransformer` 與 `Pipeline`：
1. **5 個類別特徵:** `['season', 'mnth', 'hr', 'weekday', 'weathersit']`  
   轉換器: `OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)`  
   採用 `drop='first'` 消除虛擬變數陷阱（Dummy Variable Trap），避免 OLS 設計矩陣 $X^T X$ 奇異不可逆。
2. **7 個數值/二元特徵:** `['yr', 'holiday', 'workingday', 'temp', 'atemp', 'hum', 'windspeed']`  
   轉換器: `passthrough` 直接傳遞。
3. **維度轉換對應:**
   $$\text{12 原生特徵} \xrightarrow[\text{drop='first'}]{OHE} \text{53 個模型輸入維度}$$
   - `season` (3) + `mnth` (11) + `hr` (23) + `weekday` (6) + `weathersit` (3) + 數值二元 (7) = **53 個特徵**。
4. **零資料外洩原則:** 預處理參數僅在訓練集擬合（Fit），並映射記錄於 `results/encoded_feature_names.csv`。

---

## 7. OLS 基準模型評估 (Step 5 — Baseline Evaluation)

### 訓練集擬合表現 (Training Set Metrics)
- **MAE:** 63.20 輛
- **MSE:** 7,307.19
- **RMSE:** **85.48 輛**
- **$R^2$ 判定係數:** **0.6861 (68.61%)**
- *對比參考 DummyRegressor (均值常數基準):* $\text{RMSE} = 152.58$, $R^2 = 0.0$（OLS 誤差降低達 43.98%）。

### TimeSeriesSplit 5-Fold 時序交叉驗證表現
在訓練集內部嚴格執行時序推進交叉驗證，且在**每個折疊內部獨立擬合預處理轉換器**（Zero Preprocessing Leakage）：

| 折疊 (Fold) | 訓練樣本數 | 驗證樣本數 | Validation MAE | Validation MSE | Validation RMSE | Validation $R^2$ | Dummy Val RMSE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 2,028 | 2,027 | 80.26 | 13,423.11 | 115.86 | 0.3348 | 169.76 |
| **Fold 2** | 4,055 | 2,027 | 70.74 | 9,093.25 | 95.36 | 0.5924 | 164.16 |
| **Fold 3** | 6,082 | 2,027 | 84.07 | 11,456.50 | 107.04 | 0.3914 | 138.04 |
| **Fold 4** | 8,109 | 2,027 | 75.20 | 9,533.28 | 97.64 | 0.3625 | 122.98 |
| **Fold 5** | 10,136 | 2,027 | 96.29 | 17,403.86 | 131.92 | 0.5937 | 230.05 |
| **平均 (Mean)**| — | — | **81.31** | **12,182.00** | **109.56** | **0.4549** | **165.00** |
| **標準差 (Std)**| — | — | 9.77 | 3,361.37 | **14.92** | 0.1246 | 39.46 |

> [!NOTE]
> 5-Fold CV 平均 RMSE 為 **109.56**，大幅優於常數基準的 **165.00**。此 OLS 基準模型作為 Phase 1 的起始對照點，尚未進行特徵篩選或正則化收縮。

---

## 8. 獨立重現與執行指南 (Independent Execution)

在安裝依賴後，本工作區完全支援獨立執行：

```bash
# 1. 進入研究工作區
cd 7115093001_hw2

# 2. 安裝必要套件
pip install -r requirements.txt

# 3. 執行自動化驗證與運算
python run_phase1.py

# 4. 或啟動 Jupyter Notebook 檢閱完整研究流程
jupyter notebook 7115093001_hw2.ipynb
```

---

## 9. 現有線上生產環境相容性確認 (Compatibility Confirmation)
- 本研究工作區位於 `7115093001_hw2/`，既有 Next.js 前端（`frontend/`）與 FastAPI 後端（`backend/`）完全未更動。
- 現有 50 Startups Vercel 部署設定檔（`vercel.json`）完全保留。
- 線上展示網站 [https://50-starup-i9za.vercel.app/](https://50-starup-i9za.vercel.app/) 運作穩定不受任何干擾。
