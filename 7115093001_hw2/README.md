# AIoT HW2 — Phase 1 & Phase 2 專案研究報告 (Research Workspace)

**學號 (Student ID):** 7115093001  
**研究題目 (Research Title):** Predicting Hourly Bike Sharing Demand Using Multiple Linear Regression and Feature Selection  
**研究階段 (Phase):** Phase 1 (資料驗證、EDA、時序切分、預處理管線、OLS 基準) & Phase 2 (特徵選取、正則化收縮、群組消融與模型評選)  
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
    │   ├── feature_selection_rmse.png  # Figure 5: RFE 特徵維度 k 與 5-Fold CV RMSE 變化曲線
    │   ├── model_comparison.png        # Figure 6: 4 大候選模型 5-Fold CV RMSE 對比柱狀圖
    │   ├── regularization_alpha_tuning.png # Figure 7: Ridge 與 Lasso 正則化強度 alpha 調優曲線
    │   └── feature_group_ablation.png  # Figure 8: 12 原生特徵群組消融 ΔRMSE 影響圖
    ├── models/                         # 最佳模型序列化權重儲存目錄
    │   └── best_model_pipeline.joblib  # 最佳模型 M2 (RFE + OLS, k=30) 完整 Pipeline 權重
    └── results/                        # 機器可讀之實驗結果資料庫 (JSON & CSV)
        ├── dataset_metadata.json       # 資料集來源網址、維度、欄位型態與環境版本
        ├── data_validation.json        # 資料品質審計、缺失值、時序連續性與業務檢定
        ├── missing_timestamps.json     # 全量 165 個缺漏小時清單、時段分佈與連續缺漏區塊分析
        ├── descriptive_statistics.csv  # 全欄位敘述統計數據
        ├── data_split_summary.json     # 完整日曆天時序切分區間與樣本統計
        ├── encoded_feature_names.csv   # 53 個編碼特徵對應原生特徵映射表
        ├── baseline_metrics.json       # OLS 基準模型訓練指標與 Date-Bounded CV 總結
        ├── baseline_cv_results.csv     # 5-Fold Expanding Window 各折疊日期與驗證指標
        ├── rfe_search_results.csv      # RFE 各特徵維度 k 在 5 折中的驗證表現
        ├── ridge_search_results.csv    # Ridge 各 alpha 在 5 折中的驗證表現
        ├── lasso_search_results.csv    # Lasso 各 alpha 在 5 折中的驗證表現與非零特徵數
        ├── model_comparison.csv        # 4 大模型訓練集與 5-Fold CV 綜合評比總表
        ├── selected_features.csv       # 各模型選定特徵清單、原生變數對應與係數
        ├── grouped_feature_ablation.csv # 12 原生特徵群組消融 ΔRMSE 排序表
        └── best_model_metadata.json    # 最佳模型配置、超參數、訓練與 CV 完整 Metadata
```

---

## 2. Phase 2 模型評比總表 (Model Comparison)

所有模型評比均使用**相同 5-Fold Date-Bounded Expanding Window 時序交叉驗證**，每一折疊皆以完整日曆天為界，且預處理、特徵選取與標準化縮放均在折疊內部**獨立擬合（Zero Leakage）**：

| 模型 ID | 模型名稱 (Model Name) | 最佳超參數 | 入選特徵數 | 訓練集 RMSE | CV RMSE Mean | CV RMSE Std | CV MAE Mean | CV $R^2$ Mean |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1** | OLS Baseline | 全特徵 (53) | 53 / 53 | **85.48 輛** | 108.6860 輛 | 13.7388 輛 | 81.2065 輛 | 0.4623 |
| **M2** | **RFE + OLS (勝出最佳)** | **k = 30** | **30 / 53** | 86.1871 輛 | **107.4394 輛** | **14.7812 輛** | **79.9859 輛** | **0.4767** |
| **M3** | Ridge Regression | $\alpha = 1.0$ | 53 / 53 | 85.4888 輛 | 108.5437 輛 | 13.8737 輛 | 81.0700 輛 | 0.4645 |
| **M4** | Lasso Regression | $\alpha = 0.1$ | 47 / 53 | 85.7621 輛 | 107.7984 輛 | 15.2200 輛 | 79.7394 輛 | 0.4757 |
| *Ref* | *DummyRegressor (均值)*| *常數預測* | *0* | *152.58 輛* | *164.7931 輛* | *36.5757 輛* | *122.98 輛* | *0.0000* |

### 模型選取與決策依據 (Model Selection Decision)
1. **主要準則:** **M2 (RFE + OLS, k=30)** 取得全場最低之平均驗證誤差（**Mean CV RMSE = 107.44 輛**），相較基準模型降低超過 1.25 輛，驗證判定係數 $R^2$ 達 0.4767 為 4 大模型最高。
2. **簡約性原則 (Parsimony & Occam's Razor):** M2 成功自 53 個輸入特徵中剔除 23 個不顯著之弱特徵，**僅保留 30 個最核心維度**，大幅降低參數過度配適風險與模型複雜度。
3. **穩定性準則:** M2 之 CV RMSE 標準差僅為 14.78 輛，低於 Lasso 的 15.22 輛，各時序折疊預測表現最為均衡。
4. **選定模型宣告:** 本研究正式選定 **M2 (RFE + OLS, k=30)** 為 Phase 2 勝出模型，並將完整 Pipeline 儲存至 `models/best_model_pipeline.joblib`。

---

## 3. RFE 遞迴特徵消除實驗分析 (RFE Tuning Findings)

- **候選特徵維度搜尋網格:** $k \in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 53]$。
- **曲線特徵 (Figure 5):**
  - 當 $k=5$ 時，CV RMSE 高達 128.56 輛（欠擬合）；
  - 隨著 $k$ 增加至 20 ~ 30，誤差急遽下降，於 **$k=30$ 達到最低點 107.44 輛**；
  - 超過 30 個特徵後，誤差緩步回升，在全特徵 $k=53$ 時回升至 108.69 輛。
- **特徵淘汰觀察:**
  - 淘汰特徵（23 個）: 凌晨離峰時段（`hr_1` 至 `hr_5`）、大部分月份虛擬變數、星期幾虛擬變數（`weekday_1` 至 `weekday_6`）以及假日（`holiday`）。
  - 保留特徵（30 個）: 白天通勤熱門時段（`hr_6` 至 `hr_23`）、年份趨勢 `yr`、季節 `season`、降雨天氣 `weathersit_3`, `weathersit_4` 以及關鍵溫濕度特徵。

---

## 4. 正則化迴歸分析 (Ridge & Lasso Findings)

### 嶺迴歸 (Ridge Regression, L2)
- **搜尋網格:** $\alpha \in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]$。
- **最佳超參數:** $\alpha = 1.0$（CV RMSE = **108.54 輛**）。
- **作用機制:** 透過收縮係數平方和，有效控制了 `temp` 與 `atemp` 間的高共線性引起的估計方差。

### 套索迴歸 (Lasso Regression, L1)
- **搜尋網格:** $\alpha \in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]$。
- **最佳超參數:** $\alpha = 0.1$（CV RMSE = **107.80 輛**）。
- **特徵稀疏化能力:** 在 $\alpha=0.1$ 時，模型自動將 6 個不顯著特徵之係數嚴格壓縮為 0（包含 `mnth_2`, `mnth_6`, `mnth_11`, `hr_23`, `weekday_2`, `weathersit_4`），保留 47 個有效特徵。

---

## 5. 原生特徵群組消融分析 (Grouped Feature Ablation)

對 12 個原生特徵群組依序進行**單一群組剔除實驗 (Drop-One)**，量化其對 5-Fold CV RMSE 之邊際衝擊（$\Delta \text{RMSE}$）：

| 剔除之特徵群組 | 剔除後 CV RMSE | $\Delta \text{RMSE}$ 影響 | 統計顯著性與經濟學意義 |
| :--- | :---: | :---: | :--- |
| **小時 (`hr`)** | **142.88 輛** | **+34.19 輛** | **最關鍵預測因子**（日週期雙峰通勤規律，剔除後誤差暴增 31%） |
| **年份 (`yr`)** | **114.01 輛** | **+5.32 輛** | 系統逐年成長趨勢重要指標 |
| **季節 (`season`)** | **111.86 輛** | **+3.17 輛** | 宏觀氣候週期與旅遊活躍度 |
| **天氣狀況 (`weathersit`)** | **110.23 輛** | **+1.54 輛** | 降雨降雪直接抑制戶外騎乘意願 |
| **氣溫 (`temp`)** | **110.13 輛** | **+1.44 輛** | 舒適度指標（因有 atemp 存在，單獨剔除衝擊較小） |
| **風速 (`windspeed`)** | **109.60 輛** | **+0.91 輛** | 逆風阻力對租借需求之輕微負面影響 |
| **星期 (`weekday`)** | **108.99 輛** | **+0.30 輛** | 平日與週末活動模式差異 |
| **體感氣溫 (`atemp`)** | **108.95 輛** | **+0.26 輛** | 單獨剔除幾乎無影響（因 temp 承載近 100% 相同資訊） |
| **假日 / 工作日** | **108.69 輛** | **±0.00 輛** | 邊際影響可被小時與星期虛擬變數吸收 |
| **濕度 (`hum`)** | **107.84 輛** | **-0.85 輛** | 剔除後誤差微降，反映濕度與降雨存在些微雜訊干擾 |
| **月份 (`mnth`)** | **105.59 輛** | **-3.09 輛** | 剔除後誤差下降，顯示月份細項與季節存在一定過度配適 |

> [!IMPORTANT]
> **氣溫與體感氣溫共線性實證:**
> - 單獨剔除 `atemp`: $\Delta \text{RMSE} = +0.26$ 輛
> - 單獨剔除 `temp`: $\Delta \text{RMSE} = +1.44$ 輛
> - **同時剔除 `temp` 與 `atemp`:** CV RMSE 暴增至 **131.45 輛 ($\Delta \text{RMSE} = +22.77$ 輛)**！
> - **計量經濟結論:** 氣溫指標對租借需求具備決定性影響，但 `temp` 與 `atemp` 互為高度替代品（$r=0.99$），兩者並存引發共線性，為特徵選取提供了堅實依據。
> - **因果警示:** 統計相關性與迴歸權重不等於因果推論（Correlation $\neq$ Causation）。

---

## 6. 時序資料劃分與隔離守則 (Chronological Split Guarantee)

| 資料子集 (Subset) | 涵蓋日曆天數 | 樣本筆數 (Observations) | 佔比 (%) | 起始日期與時間 | 結束日期與時間 | 研究階段職責 (Role) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Training (訓練集)** | **512 天** | **12,163 筆** | **69.99%** | 2011-01-01 00:00 | 2012-05-26 23:00 | Phase 1 & 2 模型訓練、RFE、超參數調優與時序交叉驗證 |
| **Calibration (校準集)** | **110 天** | **2,640 筆** | **15.19%** | 2012-05-27 00:00 | 2012-09-13 23:00 | **嚴格保留封存**，保留予 Phase 3 Split Conformal Prediction 區間校準 |
| **Test (測試集)** | **109 天** | **2,576 筆** | **14.82%** | 2012-09-14 00:00 | 2012-12-31 23:00 | **嚴格封存未觀測**，保留予 Phase 3 最終泛化評估 |

---

## 7. 獨立執行指南 (Independent Execution)

```bash
# 1. 進入研究工作區 (或由儲存庫根目錄執行)
cd 7115093001_hw2

# 2. 安裝必要套件
pip install -r requirements.txt

# 3. 執行 Phase 1 驗證與運算
python run_phase1.py

# 4. 執行 Phase 2 特徵選取、正則化與消融運算
python run_phase2.py

# 5. 啟動 Jupyter Notebook 檢閱全流程圖表與學術報告
jupyter notebook 7115093001_hw2.ipynb
```
