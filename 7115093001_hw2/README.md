# AIoT HW2 — Phase 1 專案研究報告 (Research Workspace)

**學號 (Student ID):** 7115093001  
**研究題目 (Research Title):** Predicting Hourly Bike Sharing Demand Using Multiple Linear Regression and Feature Selection  
**研究階段 (Phase):** Phase 1 — 資料驗證、探索性數據分析、完整日期邊界時序切分、預處理管線與 OLS 基準模型構建  
**正式 GitHub Repository:** [https://github.com/Augustine-723/AIoT_HW2.git](https://github.com/Augustine-723/AIoT_HW2.git)  

---

## 1. 專案目錄結構 (Project Structure)

本研究工作區獨立構建於 `AIoT_HW2` 儲存庫之 `7115093001_hw2/` 目錄：

```text
AIoT_HW2/
├── .gitignore                          # Git 忽略檔 (.venv, __pycache__, .ipynb_checkpoints)
├── README.md                           # 儲存庫根目錄說明文件
└── 7115093001_hw2/
    ├── 7115093001_hw2.ipynb            # 研究核心 Notebook（單一事實來源，包含完整圖表與執行輸出）
    ├── README.md                       # Phase 1 完整研究技術報告與規範備忘錄
    ├── requirements.txt                # 獨立環境套件依賴清單
    ├── run_phase1.py                   # Phase 1 自動化驗證、特徵轉換與 OLS 運算腳本
    ├── build_notebook.py               # Notebook 建構與生成腳本
    ├── data/
    │   └── hour.csv                    # 官方 Bike Sharing 每小時觀測資料集 (17,379 筆)
    ├── figures/                        # 學術級高解析度圖表 (300 DPI)
    │   ├── hourly_demand.png           # Figure 1: 24 小時平均租借需求趨勢圖
    │   ├── correlation_heatmap.png     # Figure 2: 候選特徵與目標相關係數熱力圖
    │   ├── target_distribution.png     # Figure 3: 目標變數 cnt 之分佈直方圖與 KDE
    │   └── temperature_vs_demand.png   # Figure 4: 標準化氣溫與租借總量散佈圖及趨勢線
    └── results/                        # 機器可讀之實驗結果資料庫 (JSON & CSV)
        ├── dataset_metadata.json       # 資料集來源網址、維度、欄位型態與環境版本
        ├── data_validation.json        # 資料品質審計、缺失值、時序連續性與業務檢定
        ├── missing_timestamps.json     # 全量 165 個缺漏小時清單、時段分佈與連續缺漏區塊分析
        ├── descriptive_statistics.csv  # 全欄位敘述統計數據
        ├── data_split_summary.json     # 完整日曆天時序切分區間與樣本統計
        ├── encoded_feature_names.csv   # 53 個編碼特徵對應原生特徵映射表
        ├── baseline_metrics.json       # OLS 基準模型訓練指標與 Date-Bounded CV 總結
        └── baseline_cv_results.csv     # 5-Fold Expanding Window 各折疊日期與驗證指標
```

---

## 2. 資料集來源與雜湊驗證 (Dataset Verification)

| 項目 | 規格與檢驗值 | 狀態 |
| :--- | :--- | :---: |
| **資料集名稱** | Bike Sharing Dataset (Hourly Version) | **PASS** |
| **UCI 官方原始來源** | [https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) | **PASS** |
| **Kaggle 來源** | [https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset](https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset) | **待確認 (Pending)** |
| **目前使用檔案** | `data/hour.csv` (1,156,736 Bytes) | **PASS** |
| **SHA-256 Checksum** | `e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f` | **PASS** |
| **樣本維度** | 17,379 列 (rows) $\times$ 17 欄 (columns) | **PASS** |
| **預測目標 (Target)** | `cnt` (每小時自行車租借總量，非負整數) | **PASS** |
| **目標外洩排除欄位** | `casual`, `registered`（業務恆等式 `casual + registered == cnt` 100% 吻合，為杜絕 Target Leakage，嚴禁輸入模型） | **PASS** |
| **非預測性排除欄位** | `instant` (流水號), `dteday` (字串日期) | **PASS** |

> [!NOTE]
> **Kaggle 與 UCI 來源差異記錄:**  
> 本專案使用之 `hour.csv` 經由 UCI 官方壓縮檔直接下載解壓縮取得（17,379 列，17 欄，SHA-256 驗證通過）。Kaggle `marklvl/bike-sharing-dataset` 為其公開轉載鏡像。因 Kaggle 下載需個人 API 憑證（`kaggle.json`），標註為待比對；若手動下載 Kaggle 檔案置於 `data/` 目錄，可比對 SHA-256 雜湊碼。

---

## 3. 資料驗證與缺漏時段客觀審計 (Step 1 — Data Validation)

1. **缺失值與重複項:**
   - 全欄位缺失值總數: `0` (No missing values)。
   - 重複列與重複時間戳記數: `0` (No duplicated rows or timestamps)。
2. **缺漏時段客觀統計分析 (Empirical Gap Audit):**
   - 起始時間: `2011-01-01 00:00:00`，結束時間: `2012-12-31 23:00:00`。
   - 731 天理論總小時數: $731 \times 24 = 17,544$ 小時。
   - 實際記錄小時數: `17,379` 小時，共存在 **165 個小時之資料空缺**（完整清單詳見 `results/missing_timestamps.json`）。
   - **時段分佈:** 165 個缺漏小時中，有 **105 個小時（63.6%）集中在凌晨離峰時段（00:00 至 05:00）**，其中以 03:00 與 04:00 各 34 次最高；白天時段（06:00 至 23:00）各小時缺漏次數僅介於 1 至 6 次之間。
   - **連續性分佈:** 包含 64 次單小時缺漏，以及少數多小時連續缺漏區塊（例如 2012 年 10 月 29 日至 30 日出現 36 小時連續未記錄）。
   - **學術規範說明:** 本研究嚴格維持數據客觀性，**不進行任何主觀原因猜測，亦不對缺漏資料進行人工補值（No Artificial Imputation）**。
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

切分點精準鎖定在**完整日曆天邊界（Full Calendar-Day Boundaries）**：

| 資料子集 (Subset) | 涵蓋日曆天數 | 樣本筆數 (Observations) | 佔比 (%) | 起始日期與時間 | 結束日期與時間 | 研究階段職責 (Role) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Training (訓練集)** | **512 天** | **12,163 筆** | **69.99%** | 2011-01-01 00:00 | 2012-05-26 23:00 | 模型擬合、特徵選取與時序交叉驗證 |
| **Calibration (校準集)** | **110 天** | **2,640 筆** | **15.19%** | 2012-05-27 00:00 | 2012-09-13 23:00 | **保留**予 Phase 2 Conformal Prediction 區間校準 |
| **Test (測試集)** | **109 天** | **2,576 筆** | **14.82%** | 2012-09-14 00:00 | 2012-12-31 23:00 | **完全封存**保留予最終泛化能力評估 |

- **日期無交集驗證:** $\text{Train} \cap \text{Calib} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Calib} \cap \text{Test} = \emptyset$（100% 通過）。
- **Phase 1 隔離守則:** Calibration Set 與 Test Set 保持完全未觀測（Unseen）狀態。

---

## 6. 特徵預處理管線 (Step 4 — Preprocessing Pipeline)

使用 `scikit-learn` 之 `ColumnTransformer` 與 `Pipeline`：
1. **5 個類別特徵:** `['season', 'mnth', 'hr', 'weekday', 'weathersit']`  
   轉換器: `OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)`  
   採用 `drop='first'` 消除虛擬變數陷阱（Dummy Variable Trap）。
2. **7 個數值/二元特徵:** `['yr', 'holiday', 'workingday', 'temp', 'atemp', 'hum', 'windspeed']`  
   轉換器: `passthrough` 直接傳遞。
3. **維度轉換對應:**
   $$\text{12 原生特徵} \xrightarrow[\text{drop='first'}]{OHE} \text{53 個模型輸入維度}$$
4. **零資料外洩原則:** 預處理參數僅在訓練集擬合（Fit），並記錄於 `results/encoded_feature_names.csv`。

---

## 7. OLS 基準模型評估 (Step 5 — Baseline Evaluation)

### 訓練集擬合表現 (Training Set Metrics)
- **MAE:** 63.20 輛
- **MSE:** 7,307.19
- **RMSE:** **85.48 輛**
- **$R^2$ 判定係數:** **0.6861 (68.61%)**
- *對比參考 DummyRegressor (均值常數基準):* $\text{RMSE} = 152.58$, $R^2 = 0.0$（OLS 誤差降低達 43.98%）。

### 完整日期邊界擴展視窗 5-Fold 交叉驗證 (Date-Bounded Expanding Window CV)
依據程式碼審查改進建議，切分單元改為完整唯一日曆天（Unique Calendar Days），確保同一天的資料絕不跨越 Fold：

| 折疊 (Fold) | 訓練日數 | 訓練樣本數 | 驗證日數 | 驗證樣本數 | 訓練日期區間 | 驗證日期區間 | Val MAE | Val RMSE | Val $R^2$ | Dummy RMSE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 87 天 | 1,995 | 85 天 | 2,039 | 2011-01-01 ~ 2011-03-28 | 2011-03-29 ~ 2011-06-21 | 79.53 | **114.92** | 0.3406 | 168.63 |
| **Fold 2** | 172 天 | 4,034 | 85 天 | 2,024 | 2011-01-01 ~ 2011-06-21 | 2011-06-22 ~ 2011-09-14 | 72.00 | **92.89** | 0.6158 | 164.96 |
| **Fold 3** | 257 天 | 6,058 | 85 天 | 2,038 | 2011-01-01 ~ 2011-09-14 | 2011-09-15 ~ 2011-12-08 | 83.54 | **106.60** | 0.3940 | 137.81 |
| **Fold 4** | 342 天 | 8,096 | 85 天 | 2,030 | 2011-01-01 ~ 2011-12-08 | 2011-12-09 ~ 2012-03-02 | 74.86 | **97.43** | 0.3655 | 122.99 |
| **Fold 5** | 427 天 | 10,126 | 85 天 | 2,037 | 2011-01-01 ~ 2012-03-02 | 2012-03-03 ~ 2012-05-26 | 96.11 | **131.59** | 0.5959 | 229.59 |
| **平均 (Mean)**| — | — | **85 天** | — | — | — | **81.21** | **108.69** | **0.4624** | **164.79** |
| **標準差 (Std)**| — | — | 0 天 | — | — | — | 8.87 | **15.36** | 0.1264 | 39.56 |

#### 程式碼審查修正前後 CV 對比 (Code Review Comparison)
- **修正前 (依資料列數切分):** 5-Fold CV Mean RMSE = **109.5628 輛** (std: 14.9163), Mean $R^2 = 0.4549$
- **修正後 (完整日期邊界擴展視窗):** 5-Fold CV Mean RMSE = **108.6860 輛** (std: 15.3604), Mean $R^2 = 0.4624$
- **技術改進亮點:** 修正後各 Fold 驗證集均嚴格包含 85 個完整日曆天，杜絕了同一天跨越 Fold 之時間洩漏隱患，且預處理轉換器在每個折疊內部完全獨立擬合。

---

## 8. 獨立重現與執行指南 (Independent Execution)

```bash
# 1. 進入研究工作區 (或由儲存庫根目錄執行)
cd 7115093001_hw2

# 2. 安裝必要套件
pip install -r requirements.txt

# 3. 執行自動化驗證與運算
python run_phase1.py

# 4. 或啟動 Jupyter Notebook 檢閱完整研究流程
jupyter notebook 7115093001_hw2.ipynb
```
