# AIoT HW2 — Phase 1 & Phase 2 專案研究報告 (Research Workspace)

**學號 (Student ID):** 7115093001  
**研究題目 (Research Title):** Predicting Hourly Bike Sharing Demand Using Multiple Linear Regression and Feature Selection  
**研究階段 (Phase):** Phase 1 (資料驗證、EDA、時序切分、預處理管線、OLS 基準) & Phase 2 / Phase 2.1 (特徵選取、正則化收縮、群組消融、模型評選與嚴謹審計修訂)  
**正式 GitHub Repository:** [https://github.com/Augustine-723/AIoT_HW2.git](https://github.com/Augustine-723/AIoT_HW2.git)  

---

## 1. 專案目錄結構 (Project Structure)

本研究工作區獨立構建於 `AIoT_HW2` 儲存庫之 `7115093001_hw2/` 目錄，嚴格與 50 Startups 分離：

```text
AIoT_HW2/
├── .gitignore                          # Git 忽略檔 (.venv, __pycache__, .ipynb_checkpoints)
├── README.md                           # 儲存庫根目錄說明文件
└── 7115093001_hw2/
    ├── 7115093001_hw2.ipynb            # Phase 1 & 2 核心研究 Notebook（單一事實來源，包含完整圖表與執行輸出）
    ├── README.md                       # Phase 1 & 2 完整技術報告與重現指南
    ├── requirements.txt                # 獨立環境套件依賴清單
    ├── run_phase1.py                   # Phase 1 自動化驗證、特徵轉換與 OLS 運算腳本
    ├── run_phase2.py                   # Phase 2 模型調優、RFE、Ridge/Lasso 與消融分析腳本
    ├── build_notebook.py               # Notebook 自動化構建腳本（單一維護源）
    ├── data/
    │   └── hour.csv                    # 官方 Bike Sharing 每小時觀測資料集 (17,379 筆)
    ├── figures/                        # 300 DPI 學術級視覺化圖表
    │   ├── hourly_demand.png           # Figure 1: 24 小時平均租借需求趨勢圖
    │   ├── correlation_heatmap.png     # Figure 2: 候選特徵與目標相關係數熱力圖
    │   ├── target_distribution.png     # Figure 3: 目標變數 cnt 之分佈直方圖與 KDE
    │   ├── temperature_vs_demand.png   # Figure 4: 標準化氣溫與租借總量散佈圖及趨勢線
    │   ├── feature_selection_rmse.png  # Figure 5: RFE 標準化網格特徵維度 k 與 5-Fold CV RMSE 變化曲線
    │   ├── model_comparison.png        # Figure 6: 4 大候選模型 5-Fold CV RMSE 對比柱狀圖 (Sample Std ddof=1)
    │   ├── regularization_alpha_tuning.png # Figure 7: Ridge 與 Lasso 正則化強度 alpha 調優曲線
    │   └── feature_group_ablation.png  # Figure 8: 12 原生特徵群組消融 ΔRMSE 影響圖
    ├── models/                         # 最佳模型序列化權重儲存目錄
    │   └── best_model_pipeline.joblib  # 程式自動評選之最佳模型 M2 (RFE + OLS, k=30) 完整 Pipeline 權重
    └── results/                        # 機器可讀之實驗結果資料庫 (JSON & CSV)
        ├── dataset_metadata.json       # 資料集來源網址、維度、欄位型態與環境版本
        ├── data_validation.json        # 資料品質審計、缺失值、時序連續性與業務檢定
        ├── missing_timestamps.json     # 全量 165 個缺漏小時清單、時段分佈與連續缺漏區塊分析
        ├── descriptive_statistics.csv  # 全欄位敘述統計數據
        ├── data_split_summary.json     # 完整日曆天時序切分區間與樣本統計
        ├── encoded_feature_names.csv   # 53 個編碼特徵對應原生特徵映射表
        ├── baseline_metrics.json       # OLS 基準模型訓練指標與 Date-Bounded CV 總結
        ├── baseline_cv_results.csv     # 5-Fold Expanding Window 各折疊日期與驗證指標
        ├── rfe_search_results.csv      # RFE 標準化候選網格在 5 折中的驗證表現與各折可用特徵數
        ├── ridge_search_results.csv    # Ridge 各 alpha 在 5 折中的驗證表現
        ├── lasso_search_results.csv    # Lasso 各 alpha 在 5 折中的驗證表現與非零特徵數
        ├── model_comparison.csv        # 4 大模型訓練集與 5-Fold CV 綜合評比總表 (ddof=1)
        ├── selected_features.csv       # 各模型選定特徵清單、嚴格原生變數映射、淘汰標記 (NA) 與正確對齊係數
        ├── grouped_feature_ablation.csv # 12 原生特徵群組消融 ΔRMSE 排序表
        └── best_model_metadata.json    # 最佳模型配置、超參數、訓練與 CV 完整 Metadata (含收斂檢查)
```

---

## 2. Phase 2 模型評比總表 (Model Comparison)

所有模型評比均使用**相同 5-Fold Date-Bounded Expanding Window 時序交叉驗證**，每一折疊皆以完整日曆天為界，且預處理、特徵選取與標準化縮放均在折疊內部**獨立擬合（Zero Leakage）**。交叉驗證標準差全面統一採用**樣本標準差 (Sample Standard Deviation, ddof=1)**：

| 模型 ID | 模型名稱 (Model Name) | 最佳超參數 | 入選特徵數 | 訓練集 RMSE | CV RMSE Mean | CV RMSE Std (ddof=1) | CV MAE Mean | CV $R^2$ Mean |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1** | OLS Baseline | 全特徵 (53) | 53 / 53 | **85.4821 輛** | 108.6860 輛 | **15.3604 輛** | 81.2065 輛 | 0.4623 |
| **M2** | **RFE + OLS (勝出最佳)** | **k = 30** | **30 / 53** | 86.1871 輛 | **107.4394 輛** | 16.5259 輛 | 79.9859 輛 | **0.4767** |
| **M3** | Ridge Regression | $\alpha = 1.0$ | 53 / 53 | 85.4888 輛 | 108.5437 輛 | 15.5112 輛 | 81.0700 輛 | 0.4645 |
| **M4** | Lasso Regression | $\alpha = 0.1$ | 47 / 53 | 85.7621 輛 | 107.7984 輛 | 17.0164 輛 | **79.7394 輛** | 0.4757 |
| *Ref* | *DummyRegressor (均值)*| *常數預測* | *0* | *152.58 輛* | *164.7931 輛* | *36.5757 輛* | *122.98 輛* | *0.0000* |

### 客觀評比指標解析與平局決策 (Nuanced Metric Findings & Selection Rule)
1. **多面向評比指標客觀呈現 (避免過度宣稱單一模型全勝):**
   - **平均驗證誤差 (CV RMSE):** **M2 (RFE + OLS, k=30)** 最優（**107.4394 輛**），相較 M1 OLS 降低 1.2466 輛（改善約 1.15%），屬於穩健但溫和之效能提升。
   - **平均絕對誤差 (CV MAE):** **M4 (Lasso Regression, $\alpha=0.1$)** 最優（**79.7394 輛**），相較 M2（79.9859 輛）具微幅絕對誤差優勢。
   - **跨折疊穩定度 (CV RMSE Std, ddof=1):** **M1 (OLS Baseline)** 最優（**15.3604 輛**），展現最小之時序折疊波動度。
2. **自動化模型選取演算法 (Automated Decision Policy):**
   - **主要準則:** 最低平均 CV RMSE。
   - **平手容忍度判定 (Tie-breaking Tolerance):** 預設容許差距 $\Delta \text{RMSE} \le 0.5$ 輛。在本次實驗中，M2 (107.4394) 與 M4 (107.7984) 差距為 $0.3590 \le 0.5$，構成實質平局。
   - **奧卡姆剃刀決定性破局:** 優先選擇模型結構更簡約、參數更少之模型。M2 僅需 **30 個輸入特徵**，顯著少於 M4 Lasso 的 47 個特徵與 OLS 的 53 個特徵（節省 17 個維度），因此程式自動化評選確定由 **M2** 勝出。
   - **模型管線自動序列化:** 系統自動根據程式評選結果導出勝出模型 Pipeline 至 `models/best_model_pipeline.joblib`，杜絕手動硬編碼。

---

## 3. RFE 遞迴特徵消除與跨折疊維度審計 (RFE Findings)

- **跨折疊可用特徵維度審計:**
  - Fold 1: 42 維（2011-01 至 2011-03 尚無後半年代碼）
  - Fold 2: 46 維
  - Fold 3: 49 維
  - Fold 4 & 5: 53 維
  - **共同支援維度上限:** 42 維。
- **標準化候選搜尋網格:**
  為防杜在 Fold 1 發生請求 $k=50$ 卻被靜默截斷為 $k=42$ 之不對等評估，將標準化網格嚴格設定為：  
  $$k \in [5, 10, 15, 20, 25, 30, 35, 40]$$
  保證所有折疊均嚴格評估相同數量之特徵。
- **最佳特徵數:** **$k=30$**（Mean CV RMSE = 107.4394 輛）。
- **特徵篩選結果:**
  - **保留特徵 (30 個):** 白天活動時段（`hr_6` 至 `hr_23` 共 18 小時）、年份趨勢 `yr`、季節 `season`（秋夏冬）、降雨天氣 `weathersit_3, weathersit_4`，以及 `temp`, `atemp`, `hum`, `windspeed`。
  - **淘汰特徵 (23 個):** 凌晨離峰（`hr_1` 至 `hr_5`）、多數月份虛擬變數、星期幾（`weekday_1` 至 `weekday_6`）與假日（`holiday`）。
- **學術嚴謹性宣告:** 特徵被 RFE 淘汰僅代表其在此線性模型與目前特徵組合下的增量配適貢獻較小，**絕不可視為其在統計學上「無顯著性」（Statistically Insignificant），本研究並未實施正式統計顯著性檢定**。

---

## 4. 正則化迴歸分析與收斂監控 (Ridge & Lasso Findings)

### 嶺迴歸 (Ridge Regression, L2)
- **搜尋網格:** $\alpha \in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]$。
- **最佳超參數:** $\alpha = 1.0$（Mean CV RMSE = **108.5437 輛**）。
- **作用機制:** 透過收縮係數平方和，有效控制了 `temp` 與 `atemp` 間的高共線性引起的估計方差。

### 套索迴歸 (Lasso Regression, L1)
- **搜尋網格:** $\alpha \in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]$。
- **最佳超參數:** $\alpha = 0.1$（Mean CV RMSE = **107.7984 輛**，Mean CV MAE = **79.7394 輛**）。
- **數值收斂性主動監控:** 移除了全域警告靜默抑制，使用 `warnings.catch_warnings` 主動監控 `ConvergenceWarning`。在 `max_iter=10000, tol=1e-4` 配置下，所有折疊與全訓練集共截獲 **0 個 ConvergenceWarning**，保證完全數值收斂。
- **特徵稀疏化能力:** 在 $\alpha=0.1$ 時，模型將 6 個特徵係數精確收縮至 0，保留 47 個有效特徵。

---

## 5. 原生特徵群組消融分析 (Grouped Feature Ablation)

對 12 個原生特徵群組依序進行**單一群組剔除實驗 (Drop-One)**，量化其對 5-Fold CV RMSE 之邊際衝擊（$\Delta \text{RMSE}$）：

| 剔除之特徵群組 | 剔除後 CV RMSE | $\Delta \text{RMSE}$ 影響 | 線性模型敏感度解釋 |
| :--- | :---: | :---: | :--- |
| **小時 (`hr`)** | **142.88 輛** | **+34.19 輛** | **最關鍵預測因子**（雙峰通勤規律，剔除後誤差暴增 31%） |
| **年份 (`yr`)** | **114.01 輛** | **+5.32 輛** | 系統逐年成長趨勢指標 |
| **季節 (`season`)** | **111.86 輛** | **+3.17 輛** | 宏觀氣候週期與活躍度 |
| **天氣狀況 (`weathersit`)** | **110.23 輛** | **+1.54 輛** | 降雨降雪抑制戶外騎乘 |
| **氣溫 (`temp`)** | **110.13 輛** | **+1.44 輛** | 舒適度指標（因有 atemp，單獨剔除衝擊較小） |
| **風速 (`windspeed`)** | **109.60 輛** | **+0.91 輛** | 逆風阻力對租借需求之輕微抑制 |
| **星期 (`weekday`)** | **108.99 輛** | **+0.30 輛** | 平日與週末活動模式差異 |
| **體感氣溫 (`atemp`)** | **108.95 輛** | **+0.26 輛** | 單獨剔除幾乎無影響（因 temp 承載近 100% 相同資訊） |
| **假日 / 工作日** | **108.69 輛** | **±0.00 輛** | 邊際影響可被小時與星期吸收 |
| **濕度 (`hum`)** | **107.84 輛** | **-0.85 輛** | 剔除後誤差微降，反映濕度與降雨存在些微雜訊干擾 |
| **月份 (`mnth`)** | **105.59 輛** | **-3.09 輛** | 剔除後誤差下降，顯示月份細項與季節存在一定過度配適 |

> [!IMPORTANT]
> **氣溫與體感氣溫共線性實證:**
> - 單獨剔除 `atemp`: $\Delta \text{RMSE} = +0.26$ 輛
> - 單獨剔除 `temp`: $\Delta \text{RMSE} = +1.44$ 輛
> - **同時剔除 `temp` 與 `atemp`:** CV RMSE 暴增至 **131.45 輛 ($\Delta \text{RMSE} = +22.77$ 輛)**！
> - **方法論警示:** 群組消融與迴歸係數分析屬於**線性模型的敏感度剖析，絕不可推論為因果關係或統計顯著性證明（Correlation is NOT Causation）**。

---

## 6. 特徵映射與係數導出驗證 (Feature Mapping Audit)

在 `results/selected_features.csv` 中：
1. **嚴格特徵名稱映射:** 修正了舊版子字串匹配之瑕疵，透過字首與分隔符精準解析，徹底杜絕 `num__atemp` 被誤配為 `temp` 之問題（`num__atemp -> atemp`，`num__temp -> temp`），全部 53 個欄位均唯一對應至 12 個原生特徵。
2. **正確對齊係數:** M2 RFE 選定之 30 個特徵正確對齊最終估計器 `coef_`；淘汰之 23 個特徵標記為 `eliminated` 且係數為 `NaN`，絕不偽稱為估計係數為 0。
3. **單元測試斷言:** 程式碼與 Notebook 均包含完整斷言，驗證入選數與係數維度完全吻合。

---

## 7. 時序資料劃分與隔離守則 (Chronological Split Guarantee)

| 資料子集 (Subset) | 涵蓋日曆天數 | 樣本筆數 (Observations) | 佔比 (%) | 起始日期與時間 | 結束日期與時間 | 研究階段職責 (Role) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Training (訓練集)** | **512 天** | **12,163 筆** | **69.99%** | 2011-01-01 00:00 | 2012-05-26 23:00 | Phase 1 & 2 模型訓練、RFE、超參數調優與時序交叉驗證 |
| **Calibration (校準集)** | **110 天** | **2,640 筆** | **15.19%** | 2012-05-27 00:00 | 2012-09-13 23:00 | **嚴格保留封存**，保留予 Phase 3 Split Conformal Prediction 區間校準 |
| **Test (測試集)** | **109 天** | **2,576 筆** | **14.82%** | 2012-09-14 00:00 | 2012-12-31 23:00 | **嚴格封存未觀測**，保留予 Phase 3 最終泛化評估 |

---

## 8. 獨立執行指南 (Independent Execution)

```bash
# 1. 進入研究工作區 (或由儲存庫根目錄執行)
cd 7115093001_hw2

# 2. 安裝必要套件
pip install -r requirements.txt

# 3. 執行 Phase 1 驗證與運算
python run_phase1.py

# 4. 執行 Phase 2 特徵選取、正則化與消融運算
python run_phase2.py

# 5. 重新構建並端到端執行 Jupyter Notebook
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace 7115093001_hw2.ipynb
```
