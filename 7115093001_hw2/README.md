# AIoT HW2 — Phase 1, Phase 2 & Phase 3 專案研究報告 (Research Workspace)

**學號 (Student ID):** 7115093001  
**研究題目 (Research Title):** Predicting Hourly Bike Sharing Demand Using Multiple Linear Regression and Feature Selection  
**研究階段 (Phase):**  
- **Phase 1:** 資料驗證、EDA、時序切分、預處理管線、OLS 基準模型與 Expanding Window CV  
- **Phase 2 / Phase 2.1:** 特徵選取 (RFE)、正則化收縮 (Ridge/Lasso)、原生特徵群組消融、模型客觀評選與嚴謹審計修訂  
- **Phase 3:** 最佳模型完整性審核、Split Conformal Prediction 預測區間校準、嚴格一次性測試集評估與學術研究成果  
**正式 GitHub Repository:** [https://github.com/Augustine-723/AIoT_HW2.git](https://github.com/Augustine-723/AIoT_HW2.git)  

---

## 1. 專案目錄結構 (Project Structure)

本研究工作區獨立構建於 `AIoT_HW2` 儲存庫之 `7115093001_hw2/` 目錄，嚴格與 50 Startups 隔離：

```text
AIoT_HW2/
├── .gitignore                          # Git 忽略檔 (.venv, __pycache__, .ipynb_checkpoints)
├── README.md                           # 儲存庫根目錄說明文件
└── 7115093001_hw2/
    ├── 7115093001_hw2.ipynb            # Phase 1, 2, 3 核心研究 Notebook（單一事實來源，包含完整圖表與輸出）
    ├── README.md                       # Phase 1, 2, 3 完整技術報告與重現指南
    ├── requirements.txt                # 獨立環境套件依賴清單
    ├── run_phase1.py                   # Phase 1 自動化驗證、特徵轉換與 OLS 運算腳本
    ├── run_phase2.py                   # Phase 2 模型調優、RFE、Ridge/Lasso 與消融分析腳本
    ├── run_phase3.py                   # Phase 3 測試集最終評估、Conformal 區間校準與診斷腳本
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
    │   ├── feature_group_ablation.png  # Figure 8: 12 原生特徵群組消融 ΔRMSE 影響圖
    │   ├── actual_vs_predicted.png     # Figure A: 測試集真實值 vs 預測值散佈圖 (含 y=x 基準線)
    │   ├── prediction_interval.png     # Figure B: 測試集代表性連續時段預測區間時間序列圖 (前 7 日曆天)
    │   ├── residual_analysis.png       # Figure C: 殘差 vs 預測值散佈圖與殘差分佈直方圖
    │   └── interval_coverage_by_hour.png # Figure D: 各小時 (0-23) 經驗預測區間覆蓋率診斷圖
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
        ├── selected_features.csv       # 各模型選定特徵清單、原生變數映射、淘汰標記 (NA) 與對齊係數
        ├── grouped_feature_ablation.csv # 12 原生特徵群組消融 ΔRMSE 排序表
        ├── best_model_metadata.json    # 最佳模型配置、超參數、訓練與 CV 完整 Metadata (含收斂檢查)
        ├── calibration_scores.csv      # 校準集殘差分數表 (2,640 筆，絕對殘差排序)
        ├── test_predictions.csv        # 測試集全量預測、原始與截斷區間、單點覆蓋判斷表 (2,576 筆)
        ├── final_test_metrics.json     # 測試集嚴格一次性評估指標 (MAE, MSE, RMSE, R2, 基準對比)
        ├── prediction_interval_metrics.json # 預測區間核心指標 (名義/經驗覆蓋率、區間寬度、截斷率)
        └── phase3_experiment_metadata.json  # Phase 3 實驗環境、切分邊界、方法論宣告與完整 Metadata
```

---

## 2. Phase 3 測試集最終評估與預測區間核心結果 (Phase 3 Final Results)

本研究遵循統計機器學習最嚴謹之方法論：在 Phase 1 與 Phase 2 期間，測試集（Test Set, 2012-09-14 至 2012-12-31，共 2,576 筆觀測值）與校準集（Calibration Set, 2012-05-27 至 2012-09-13，共 2,640 筆觀測值）均處於**完全密封隔離狀態**。所有特徵選取（RFE）與超參數評選均僅使用訓練集（Training Set）上的 5-Fold Expanding Window CV 完成。

Phase 3 啟動後，使用定案之最佳模型 **M2 (RFE + OLS, k=30)** 進行**嚴格一次性（Exactly Once）測試集最終評估**，絕不根據測試集表現回頭調整模型或特徵。

### 2.1 測試集最終評估指標對比 (Final Evaluation vs Benchmarks)

| 資料子集 / 評估情境 | 評估指標 | MAE (輛) | MSE (輛$^2$) | RMSE (輛) | $R^2$ 判定係數 | 樣本筆數 (N) | 涵蓋時間跨度 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Training (訓練集擬合)** | M2 擬合值 | 61.6496 | 7428.2223 | **86.1871** | **0.6806** | 12,163 | 2011-01-01 至 2012-05-26 |
| **5-Fold CV (時序交叉驗證)** | M2 折疊平均 | 79.9859 | 11816.1485 | **107.4394** | **0.4767** | 12,163 (時序擴展) | 2011-01-01 至 2012-05-26 |
| **Test (未觀測測試集 - 最終)** | **M2 泛化評估** | **97.5211** | **17744.5877** | **133.2088** | **0.6112** | **2,576** | **2012-09-14 至 2012-12-31** |
| *Test (常數基準 - 均值)* | *DummyRegressor* | *178.6917* | *51130.1292* | *226.1197* | *-0.1198* | *2,576* | *2012-09-14 至 2012-12-31* |

#### 關鍵發現與學術討論：
1. **強大之基準超額表現:** M2 模型在未觀測測試集上的 RMSE 為 **133.2088 輛**，較 DummyRegressor 常數基準（226.1197 輛）大幅降低了 **92.9109 輛 (誤差縮減 41.09%)**，且 $R^2 = 0.6112$ 展現顯著之實質解釋力（DummyRegressor 之 $R^2 = -0.1198$ 顯示時序總量偏移時常數預測之嚴重劣勢）。
2. **測試集 RMSE 上升之客觀時序成因:**
   - 測試集 RMSE (133.21 輛) 高於 CV RMSE (107.44 輛) 與訓練集 RMSE (86.19 輛)。
   - 此現象符合現實世界時序資料之**概念漂移 (Concept Drift) 與規模增長 (Scale Growth)**：Bike Sharing 系統在 2012 年秋冬之整體使用規模、會員基數與峰值租借量均顯著高於 2011 年初。由於目標變數 $y$ 的絕對數值與方差增大，在維持相同相對配適力（$R^2 = 0.6112$ 甚至高於 CV 平均之 $0.4767$）下，絕對誤差 RMSE 必然呈現規模性放大。
   - 本研究據實報告此時序泛化現象，絕不事後重選模型或過度擬合測試集。

---

### 2.2 Split Conformal Prediction 預測區間建構 (Conformal Prediction Intervals)

為解決傳統線性迴歸高斯殘差假設（常態性與同質變異數）在計數資料中極易失效之問題，本研究採用分拆共形預測（Split Conformal Prediction）架構，在獨立校準集（Calibration Set, $n=2,640$）上建構具備**有限樣本無分佈保證（Distribution-Free Finite-Sample Guarantee）**之非對稱預測區間。

#### 1. 共形校準統計程序 (Calibration Procedure)
- **非整合度分數 (Conformity Score):** 採用絕對殘差 $s_i = |y_i - \hat{y}_i|$。
- **目標名義覆蓋率 (Nominal Target Coverage):** $1 - \alpha = 95.0\%$ ($\alpha = 0.05$)。
- **有限樣本修正分位數 (Finite-Sample Quantile):**
  $$k^* = \lceil (n + 1)(1 - \alpha) \rceil = \lceil (2640 + 1) \times 0.95 \rceil = \lceil 2508.95 \rceil = 2509$$
- **校準分位數值:** 對校準集 2,640 筆殘差排序後，$s_{(2509)} = \mathbf{302.6015 \text{ 輛}}$。
- **預測區間形式:**
  - 原始未截斷區間 (Raw Interval): $[\hat{y}_t - 302.6015, \; \hat{y}_t + 302.6015]$
  - 非負截斷區間 (Clipped Interval): $[\max(0, \hat{y}_t - 302.6015), \; \hat{y}_t + 302.6015]$

#### 2. 測試集預測區間驗證表現 (Empirical Performance on Test)

| 預測區間類型 (Interval Type) | 名義覆蓋率 (Nominal) | 經驗覆蓋率 (Empirical) | 涵蓋樣本數 (Covered) | 平均區間寬度 (Mean Width) | 中位數區間寬度 (Median Width) | 區間下界截斷率 (Lower Truncated) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **原始對稱區間 (Raw Interval)** | 95.0% | **94.84%** | 2,443 / 2,576 | 605.2015 輛 | 605.2015 輛 | 0.00% (無截斷) |
| **非負截斷區間 (Clipped Interval)**| 95.0% | **94.84%** | 2,443 / 2,576 | **513.1870 輛** | **539.8648 輛** | **72.17% (1,859 筆)** |

#### 關鍵發現與統計推論：
1. **極高之經驗覆蓋可靠度:** 在完全未觀測之 2012 年末測試集上，經驗覆蓋率達到 **94.84% (2,443 / 2,576)**，與 95.0% 之名義目標僅有 **0.16%** 之極微差距，充分證實了分拆共形預測方法論在時序外推情境下依然具備高度穩健的區間校準能力。
2. **非負領域截斷之顯著效益 (15.2% 區間緊湊度改善):**
   - 由於自行車租借量本質為非負計數（$y \ge 0$），原始區間有高達 **72.17% (1,859 筆)** 的下界小於 0。
   - 將下界截斷至 0 後，**覆蓋率完全維持 94.84% 不變**（因為真實觀測值從不小於 0），但平均區間寬度由 605.20 輛劇降至 **513.19 輛 (節省 92.01 輛，寬度縮減 15.20%)**！
   - 此實證證明了**結合統計無分佈保證與領域業務約束（Domain Knowledge Truncation）**之巨大優勢。

---

### 2.3 點預測與區間下界負值診斷 (Negative Value Diagnostics)

1. **點預測負值率 (Negative Point Prediction Rate):**
   - 測試集全量 2,576 筆預測中，共有 **20 筆 (0.78%)** 出現負值預測，最小預測值為 **-21.46 輛**。
   - 診斷特徵：全部發生在**凌晨極端離峰（01:00 至 05:00）且氣溫極低之冬季工作日**。
   - 統計意涵：普通最小平方法（OLS）作為未受約束之連續線性模型，在接近 0 的邊界自然存在向下越界之微小機率。本研究遵循學術規範，**在計算 MSE/RMSE 評估指標時保持原始實數預測，絕不事後改動評估基準**。
2. **區間下界負值率:** 原始區間下界負值率為 **72.17%**，其成因為校準分位數 $q=302.60$ 顯著大於多數離峰與常態時段之點預測值，經非負截斷後完美解決。

---

### 2.4 學術級成果圖表 (Phase 3 Visualizations)

所有圖表均以學術發表規格（300 DPI, 緊湊邊界, 專業色盤）輸出至 `figures/` 目錄：

1. **Figure A: 測試集真實值 vs 預測值散佈圖 (`figures/actual_vs_predicted.png`)**
   - 橫軸為真實租借需求（Actual cnt），縱軸為模型預測值（Predicted cnt），並疊加 $y=x$ 理想基準對角線。
   - 標註 Test RMSE = 133.2088 輛與 $R^2 = 0.6112$。清單呈現中低需求區間配適緊密，超高峰區間（$cnt > 600$）模型受線性極限壓制而有些微低估之實證特徵。
2. **Figure B: 測試集代表性連續時段預測區間時間序列圖 (`figures/prediction_interval.png`)**
   - 遵循預先設定之規則，選取測試集起始之**前 7 個完整日曆天 (2012-09-14 至 2012-09-20，共 168 個連續小時)**。
   - 繪製真實觀測值（深灰實線點）、M2 點預測（藍色虛線）與 95% 截斷共形預測區間（半透明藍色陰影帶）。
   - 清楚展現模型成功捕捉平日雙峰通勤特徵（08:00 與 17:00-18:00）與週末平緩單峰型態，且 95% 預測區間完美包絡真實波動。
3. **Figure C: 殘差診斷與誤差分佈圖 (`figures/residual_analysis.png`)**
   - 左圖：殘差 vs 預測值散佈圖（Residual vs Predicted），揭示高預測值時殘差變異數略微擴大之輕微異質變異（Heteroscedasticity）。
   - 右圖：殘差直方圖與 KDE 分佈曲線，相較於常態假設，呈現稍微右偏且具肥尾（Heavy-tailed）之特性，強烈佐證了採用無分佈 Conformal Prediction 之必要性。
4. **Figure D: 各小時 (0-23) 經驗預測區間覆蓋率診斷圖 (`figures/interval_coverage_by_hour.png`)**
   - 呈現 24 個小時獨立計算之經驗覆蓋率柱狀圖，並標示 95% 名義參考線。
   - 作為條件性診斷（Diagnostic Only），揭示離峰時段（02:00-06:00）覆蓋率接近 98-99%，而極端通勤高峰（08:00, 17:00）覆蓋率略微下探至 90-92%，誠實反映出邊際保證（Marginal Guarantee）與條件保證（Conditional Guarantee）之間的實證差距。

---

## 3. Phase 2 模型評比總表 (Model Comparison Reference)

所有模型評比均使用**相同 5-Fold Date-Bounded Expanding Window 時序交叉驗證**，每一折疊皆以完整日曆天為界，且預處理、特徵選取與標準化縮放均在折疊內部**獨立擬合（Zero Leakage）**。交叉驗證標準差全面統一採用**樣本標準差 (Sample Standard Deviation, ddof=1)**：

| 模型 ID | 模型名稱 (Model Name) | 最佳超參數 | 入選特徵數 | 訓練集 RMSE | CV RMSE Mean | CV RMSE Std (ddof=1) | CV MAE Mean | CV $R^2$ Mean |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1** | OLS Baseline | 全特徵 (53) | 53 / 53 | **85.4821 輛** | 108.6860 輛 | **15.3604 輛** | 81.2065 輛 | 0.4623 |
| **M2** | **RFE + OLS (勝出最佳)** | **k = 30** | **30 / 53** | 86.1871 輛 | **107.4394 輛** | 16.5259 輛 | 79.9859 輛 | **0.4767** |
| **M3** | Ridge Regression | $\alpha = 1.0$ | 53 / 53 | 85.4888 輛 | 108.5437 輛 | 15.5112 輛 | 81.0700 輛 | 0.4645 |
| **M4** | Lasso Regression | $\alpha = 0.1$ | 47 / 53 | 85.7621 輛 | 107.7984 輛 | 17.0164 輛 | **79.7394 輛** | 0.4757 |
| *Ref* | *DummyRegressor (均值)*| *常數預測* | *0* | *152.58 輛* | *164.7931 輛* | *36.5757 輛* | *122.98 輛* | *0.0000* |

### 客觀評比指標解析與平局決策 (Nuanced Metric Findings & Selection Rule)
1. **多面向評比指標客觀呈現:**
   - **平均驗證誤差 (CV RMSE):** **M2 (RFE + OLS, k=30)** 最優（**107.4394 輛**），相較 M1 OLS 降低 1.2466 輛（改善約 1.15%）。
   - **平均絕對誤差 (CV MAE):** **M4 (Lasso Regression, $\alpha=0.1$)** 最優（**79.7394 輛**）。
   - **跨折疊穩定度 (CV RMSE Std, ddof=1):** **M1 (OLS Baseline)** 最優（**15.3604 輛**）。
2. **自動化模型選取演算法 (Automated Decision Policy):**
   - **平手容忍度判定:** 預設容許差距 $\Delta \text{RMSE} \le 0.5$ 輛。M2 (107.4394) 與 M4 (107.7984) 差距為 $0.3590 \le 0.5$，構成實質平局。
   - **奧卡姆剃刀破局:** 優先選擇模型結構更簡約、參數更少之模型。M2 僅需 **30 個輸入特徵**，顯著少於 M4 Lasso 的 47 個特徵與 OLS 的 53 個特徵（節省 17 個維度），因此程式自動化評選確定由 **M2** 勝出。

---

## 4. RFE 遞迴特徵消除與跨折疊維度審計 (RFE Findings)

- **跨折疊可用特徵維度審計:**
  - Fold 1: 42 維（2011-01 至 2011-03 尚無後半年代碼）
  - Fold 2: 46 維
  - Fold 3: 49 維
  - Fold 4 & 5: 53 維
  - **共同支援維度上限:** 42 維。
- **標準化候選搜尋網格:**
  為防杜在 Fold 1 發生請求 $k=50$ 卻被靜默截斷為 $k=42$ 之不對等評估，將標準化網格嚴格設定為：  
  $$k \in [5, 10, 15, 20, 25, 30, 35, 40]$$
- **最佳特徵數:** **$k=30$**（Mean CV RMSE = 107.4394 輛）。
- **保留特徵 (30 個):** 白天活動時段（`hr_6` 至 `hr_23` 共 18 小時）、年份趨勢 `yr`、季節 `season`（秋夏冬）、降雨天氣 `weathersit_3, weathersit_4`，以及 `temp`, `atemp`, `hum`, `windspeed`。
- **淘汰特徵 (23 個):** 凌晨離峰（`hr_1` 至 `hr_5`）、多數月份虛擬變數、星期幾（`weekday_1` 至 `weekday_6`）與假日（`holiday`）。
- **學術嚴謹性宣告:** 特徵被 RFE 淘汰僅代表其在此線性模型與目前特徵組合下的增量配適貢獻較小，**絕不可視為其在統計學上「無顯著性」（Statistically Insignificant），本研究並未實施正式統計顯著性檢定**。

---

## 5. 正則化迴歸分析與收斂監控 (Ridge & Lasso Findings)

- **嶺迴歸 (Ridge Regression, L2):**
  - 最佳超參數: $\alpha = 1.0$（Mean CV RMSE = **108.5437 輛**）。
  - 有效控制了 `temp` 與 `atemp` 高共線性之方差膨脹。
- **套索迴歸 (Lasso Regression, L1):**
  - 最佳超參數: $\alpha = 0.1$（Mean CV RMSE = **107.7984 輛**，Mean CV MAE = **79.7394 輛**）。
  - 全域截獲 **0 個 ConvergenceWarning**，保證完全數值收斂。將 6 個特徵精確收縮至 0。

---

## 6. 原生特徵群組消融分析 (Grouped Feature Ablation)

對 12 個原生特徵群組依序進行**單一群組剔除實驗 (Drop-One)**：

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

---

## 7. 時序資料劃分與隔離守則 (Chronological Split Guarantee)

本研究嚴格落實**完整日曆天邊界時序切分**，絕對不打亂（No Shuffle），徹底杜絕時序資料洩漏：

| 資料子集 (Subset) | 涵蓋日曆天數 | 樣本筆數 (Observations) | 佔比 (%) | 起始日期與時間 | 結束日期與時間 | 研究階段職責 (Role) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Training (訓練集)** | **512 天** | **12,163 筆** | **69.99%** | 2011-01-01 00:00 | 2012-05-26 23:00 | Phase 1 & 2 模型訓練、RFE、超參數調優與時序交叉驗證 |
| **Calibration (校準集)** | **110 天** | **2,640 筆** | **15.19%** | 2012-05-27 00:00 | 2012-09-13 23:00 | Phase 3 Split Conformal Prediction 區間校準（非整合度分數） |
| **Test (測試集)** | **109 天** | **2,576 筆** | **14.82%** | 2012-09-14 00:00 | 2012-12-31 23:00 | Phase 3 嚴格一次性未觀測最終泛化評估 |

---

## 8. 統計學嚴謹詮釋與研究局限性宣告 (Statistical Interpretation & Limitations)

1. **可交換性假設與時序相依性 (Exchangeability Assumption):**
   - 分拆共形預測之名義覆蓋率（95%）在數學上依賴樣本之間具備可交換性（Exchangeability）或獨立同分佈（i.i.d.）。
   - 本研究資料集為每小時連續時間序列，存在自相關性（Autocorrelation）與季節性概念漂移（Concept Drift）。因此，95% 應視為**名義目標覆蓋率（Nominal Target Coverage）**，而非絕對無條件之數學承諾。
   - 實證測試集覆蓋率達到 **94.84%**，顯示在宏觀趨勢下共形校準仍具備優異強健性。
2. **邊際覆蓋與條件覆蓋 (Marginal vs Conditional Coverage):**
   - 95% 保證為邊際覆蓋率（Marginal Coverage，即跨越所有測試時段之整體期望值）。
   - 由 Figure D 可見，特定極端尖峰通勤小時（如 08:00）的局部經驗覆蓋率略低於 95%，屬於邊際共形預測的已知特性。
3. **相關性不等於因果性 (Correlation is NOT Causation):**
   - 本研究所分析之迴歸係數與群組消融衝擊，僅代表在特定特徵集與多元線性模型結構下的敏感度與統計關聯性，**嚴禁解讀為現實世界中改變氣溫或風速能直接因果導致租借量變動之結論**。

---

## 9. 獨立執行重現指南 (Independent Reproducibility)

所有實驗與圖表皆具備 100% 確定性可重現性：

```bash
# 1. 進入研究工作區 (或由儲存庫根目錄執行)
cd 7115093001_hw2

# 2. 安裝必要套件
pip install -r requirements.txt

# 3. 執行 Phase 1 驗證與運算
python run_phase1.py

# 4. 執行 Phase 2 特徵選取、正則化與消融運算
python run_phase2.py

# 5. 執行 Phase 3 最終測試評估與 Conformal 區間校準
python run_phase3.py

# 6. 重新構建並端到端執行 Jupyter Notebook (產生完整圖表與輸出)
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace 7115093001_hw2.ipynb
```
