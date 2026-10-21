# AIoT HW2 — Multiple Linear Regression & Machine Learning Research

**學生學號 (Student ID):** 7115093001  
**課程名稱:** 人工智慧與物聯網應用 (AIoT)  
**GitHub 專案儲存庫:** [https://github.com/Augustine-723/AIoT_HW2](https://github.com/Augustine-723/AIoT_HW2)  
**線上研究與互動展示網站 (Vercel Live App):** [https://a-io-t-hw-2.vercel.app/](https://a-io-t-hw-2.vercel.app/)  
**最新交付架構:** 支援 Problem 1、Problem 2 雙核心獨立多元線性迴歸研究問題切換，與 Custom Dataset 通用迴歸加分功能。

---

## 目錄 (Table of Contents)
1. [專案架構與兩大獨立研究問題](#1-專案架構與兩大獨立研究問題)
2. [資料集來源與規格對照](#2-資料集來源與規格對照)
3. [CRISP-DM 六大研究階段落實](#3-crisp-dm-六大研究階段落實)
4. [特徵選取與正則化收縮深度分析](#4-特徵選取與正則化收縮深度分析)
5. [模型評估與 Split Conformal 預測區間實證](#5-模型評估與-split-conformal-預測區間實證)
6. [網站導覽、下拉選單與互動推論指引](#6-網站導覽下拉選單與互動推論指引)
7. [加分功能：通用資料集多元線性迴歸 Studio (Bonus)](#7-加分功能通用資料集多元線性迴歸-studio-bonus)
8. [可重現性驗證、單元測試與 Parity 測試](#8-可重現性驗證單元測試與-parity-測試)
9. [學術誠信、NotebookLM 與 AI 輔助開發聲明](#9-學術誠信notebooklm-與-ai-輔助開發聲明)
10. [研究限制與未來展望](#10-研究限制與未來展望)
11. [驗收記錄表 (PASS / FAIL / NOT VERIFIED Checklist)](#11-驗收記錄表-pass--fail--not-verified-checklist)

---

## 1. 專案架構與兩大獨立研究問題

本儲存庫遵循嚴格之工程與學術隔離規範，將兩大研究問題與加分模組獨立劃分，確保原始實驗數據、模型權重、圖表與前端展示互不干擾、零程式碼耦合：

```text
AIoT_HW2/
├── .gitignore
├── README.md                                  # 本全域總覽與交付說明文件
├── FINAL_REPRODUCIBILITY_REPORT.md            # Problem 1 複現驗證報告
│
├── 7115093001_hw2/                            # 【Problem 1: Bike Sharing 研究工作區 (完整保留)】
│   ├── 7115093001_hw2.ipynb                   # Phase 1 ~ 3 完整研究實驗單一事實來源 Notebook
│   ├── README.md                              # Problem 1 專屬技術報告
│   ├── requirements.txt                       # 獨立環境套件依賴
│   ├── run_phase1.py, run_phase2.py, ...      # 自動化運算腳本
│   ├── data/hour.csv                          # Bike Sharing 官方每小時資料集 (17,379 筆)
│   ├── figures/                               # 12 張 300 DPI 學術級研究圖表
│   ├── results/                               # 機器可讀實驗結果 (含 m2_model_spec.json)
│   └── models/best_model_pipeline.joblib      # M2 (RFE+OLS, k=30) 序列化模型
│
├── problem2/                                  # 【Problem 2: Appliances Energy 研究工作區 (新獨立問題)】
│   ├── appliances_energy_prediction.ipynb     # CRISP-DM 六階段完整執行 Notebook
│   ├── run_experiment.py                      # 端到端自動化研究管線腳本
│   ├── build_notebook.py                      # Notebook 自動化構建器
│   ├── export_website_data.py                 # 前端靜態展示資料同步器
│   ├── README.md                              # Problem 2 專屬技術報告與驗證說明
│   ├── requirements.txt                       # 獨立 Python 套件依賴
│   ├── data/energydata_complete.csv           # 官方完整資料集 (19,735 筆, 11.98 MB)
│   ├── figures/                               # 8 張 300 DPI 實驗圖表
│   ├── results/                               # data_audit.json, model_spec.json, 評估數據
│   └── models/best_model.joblib               # M3 Ridge 序列化模型
│
└── website/                                   # 【次世代學術儀表板與確定性推論前端】
    ├── app/
    │   ├── page.tsx, dataset/, models/, ...   # Problem 1 原有頁面 (完整保留無覆蓋)
    │   ├── problem2/page.tsx                  # Problem 2 完整儀表板與互動推論
    │   ├── custom-dataset/page.tsx            # Stage 6 加分功能：通用資料集 MLR Studio
    │   └── api/fetch-csv/route.ts             # SSRF 防護之通用 CSV 安全擷取 API
    ├── components/Navbar.tsx                  # 原生 HTML <select> 下拉式切換器
    ├── lib/
    │   ├── problem2-model.ts                  # Problem 2 純客戶端 Option B 確定性推論引擎
    │   └── custom-dataset-solver.ts           # 動態矩陣求逆、One-Hot 與 Conformal 求解器
    ├── tests/
    │   ├── reproducible_parity_test.ts        # Problem 1 Python–TS Parity (< 5.7e-14)
    │   ├── problem2_parity_test.ts            # Problem 2 Python–TS Parity (< 1.0e-12)
    │   └── test_custom_solver.ts              # 50 Startups 基準自動化驗收測試
    └── data/                                  # 前端靜態凍結資料庫 (分別獨立存放 P1 與 P2)
```

---

## 2. 資料集來源與規格對照

本專案兩大問題皆採用權威學術公開資料集，並經過嚴格的 SHA-256 數位簽章與程式化審計驗證：

| 評比項目 | Problem 1: Hourly Bike Sharing Demand | Problem 2: Appliances Energy Prediction |
|:---|:---|:---|
| **研究目標** | 首都自行車共享系統每小時租借總量 (`cnt`) 預測 | 低耗能智慧建築每 10 分鐘電器能耗 (`Appliances`) 估計 |
| **總觀測數 (N)** | **17,379 筆** (2011-01-01 至 2012-12-31) | **19,735 筆** (2016-01-11 至 2016-05-27) |
| **取樣週期** | 1 小時連續時序 (存在 165 個缺漏小時) | 10 分鐘嚴格均勻連續時序 (0 缺漏、0 重複、0 缺失值) |
| **原始資料欄位** | **16 欄** (若含列序號則為 17 欄) | **29 個原始資料欄位** |
| **排除變數** | `instant` (流水號), `dteday` (日期), `casual`, `registered` (資訊洩漏) | `date` (時間戳記), `rv1`, `rv2` (隨機對照欄位，防干擾) |
| **原始感測及氣象預測變數** | **12 項** (時段、季節、年份、月份、假日、星期、工作日、天氣、氣溫、體感溫度、濕度、風速) | **25 個原始感測及氣象預測變數** (燈光能耗 1 項 + 9 間居室溫度 + 9 間居室濕度 + 氣象站觀測 6 項) |
| **基本模型輸入特徵** | **12 個** (編碼展開後候選池為 53 維) | **26 個基本模型輸入特徵（包含衍生 hour）** (25 個原始感測變數 + 主時間特徵 `hour`) |
| **擴充候選特徵** | 53 維 (類別 One-Hot + 數值標準化) | **29 個擴充候選特徵** (26 個基本特徵 + `sin_hour`, `cos_hour`, `is_weekend`) |
| **資料來源網址** | - [UCI Bike Sharing Dataset (ID: 275)](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)<br>- [Kaggle Bike Sharing Demand Competition](https://www.kaggle.com/competitions/bike-sharing-demand) | - [UCI Appliances Energy Prediction (ID: 374)](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)<br>- [Original Author GitHub (Luis Candanedo)](https://github.com/LuisM78/Appliances-energy-prediction-data)<br>- [Kaggle SB1 Competition (Reference)](https://www.kaggle.com/competitions/machine-learning-sb1)<br>- [Kaggle Public Dataset (loveall)](https://www.kaggle.com/datasets/loveall/appliances-energy-prediction) |
| **數位雜湊 (SHA-256)**| `e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f` | `2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d` |
| **Kaggle 來源說明** | Kaggle 競賽公開且已對照一致。 | Kaggle SB1 競賽原檔需帳號登入與競賽規則同意；本專案比對 Kaggle 公開同源資料集，並透過 UCI 官方壓縮檔與作者 GitHub 原始檔驗證，**SHA-256 達到 100% 位元完全相同**。 |

---

## 3. CRISP-DM 六大研究階段落實

兩大問題均端到端完整貫徹 CRISP-DM 跨產業資料探勘標準流程：

```mermaid
flowchart LR
    A["1. Business Understanding"] --> B["2. Data Understanding"]
    B --> C["3. Data Preparation"]
    C --> D["4. Modeling"]
    D --> E["5. Evaluation"]
    E --> F["6. Deployment"]
```

### 1. Business Understanding (商業理解)
- **Problem 1 (Bike Sharing):** 聚焦公共自行車系統因潮汐人流導致的站點車輛枯竭或滿溢問題，將物聯網感測預測任務定義為連續數值迴歸，確立以 MAE、RMSE 為主之精度評估準則，並透過預測區間提供維運調度之風險緩衝估算。
- **Problem 2 (Appliances Energy):** 低耗能住宅 (Low-energy House) 電器耗能監測與動態負載反應管理。依據感測器採集架構，嚴謹定義為**同步全戶遙測能耗估計 (Synchronous Telemetry Energy Estimation)**。
- **物理邊界:** 兩者標的皆具備非負性 ($y \ge 0$)，在預測區間推論時必須精確界定下界截斷。

### 2. Data Understanding (資料理解與審計)
- **探索性資料分析 (EDA):** 涵蓋標的變數右偏常態性檢驗、時序趨勢探索、特徵相關係數熱力圖。
- **時序完整性審計:**
  - Problem 1: 審計 2011–2012 年曆法跨度中 165 個缺漏小時（主要分佈於凌晨時段，確認無重複記錄，且資料時間序列嚴格單調遞增）。
  - Problem 2: 審計 19,735 筆 10 分鐘記錄，0 缺漏、0 重複、0 缺失值，連續時間戳記完全單調遞增。
- **Problem 2 隨機對照驗證:** 原始資料包含之均勻隨機欄位 `rv1` 與 `rv2`，與標的相關係數僅 $-0.011$，在特徵選取過程中被成功識別為無效特徵並第一時間遭到淘汰（排名 30/31 與 31/31），確認特徵工程演算法具備高度抗噪健全性。

### 3. Data Preparation (資料準備與時序防洩漏切分)
- 嚴格實施**完整日曆天邊界時序切分 (Chronological Full-Date Aligned Split)**，嚴禁任何資料洗牌 (No Shuffle)：
  - **Problem 1:** Training 69.99% (12,163 筆, 512 天) $\to$ Calibration 15.19% (2,640 筆, 110 天) $\to$ Test 14.82% (2,576 筆, 109 天)。
  - **Problem 2:** Training 70.26% (13,866 筆, 97 天) $\to$ Calibration 15.32% (3,024 筆, 21 天) $\to$ Test 14.42% (2,845 筆, 20 天)。
- **Pipeline 防洩漏機制:** 數值標準化 (`StandardScaler`) 與 One-Hot 編碼均嚴格封裝於 `sklearn.pipeline.Pipeline` 中，僅使用 Training 資料集擬合參數。

### 4. Modeling (模型建構與 Expanding-Window CV)
- 僅在 Training 資料集上使用 **5-Fold Expanding Window Cross Validation**，評估 OLS、RFE+OLS、Ridge 與 Lasso。
- 報告無偏樣本標準差 ($ddof=1$)，禁止使用測試集選擇模型或超參數。

### 5. Evaluation (盲測評估與 Split Conformal 區間)
- 最佳模型凍結後，對未觀測測試集執行一次性嚴格盲測評估。
- 採用獨立校準集計算無分佈假設之 95% Split Conformal 預測區間，檢驗經驗覆蓋率與平均區間寬度。

### 6. Deployment (確定性前端部署)
- 匯出數值標準化均值、標準差、回歸係數與截距至 `model_spec.json`。
- 採純客戶端 Option B 架構，於 Next.js 進行毫秒級確定性推論，無伺服器延遲或冷啟動問題。

---

## 4. 特徵選取與正則化收縮深度分析

### Problem 1: Bike Sharing 特徵選取
- **RFE 搜尋歷程:** 於 53 維編碼特徵中以 5-Fold Expanding-Window CV 尋找最優維度，最終在 **$k=30$** 取得最低 CV RMSE (**107.4394 輛**)。
- **原生群組消融分析 (Grouped Feature Ablation):**
  - **最關鍵特徵:** 時段 (`hr`)，剔除後 CV RMSE 暴增 **+34.19 輛** (誤差上升 31%)，展現都市通勤之決定性雙峰特徵。
  - **共線性實證:** 氣溫 (`temp`) 與體感氣溫 (`atemp`) 相關係數高達 0.985。單獨剔除其一影響極微 ($\Delta \text{RMSE} \le +1.44$)，但**兩者同時剔除時 CV RMSE 暴增 +22.77 輛**。

### Problem 2: Appliances Energy 特徵選取與正則化
- **特徵淘汰與正則化對比 (5-Fold CV 實證數據):**

| 模型代號 | 演算法模型 | 選定特徵數 | 超參數設定 | 5-Fold CV RMSE (Wh) | CV 標準差 ($ddof=1$) | 演算法評選結果 |
|:---:|:---|:---:|:---|:---:|:---:|:---|
| **M0** | DummyRegressor (常數均值) | 0 | `strategy='mean'` | 102.7427 | $\pm$ 8.0952 | 基準對照 |
| **M1** | Full OLS (全特徵多元線性迴歸) | 29 | `fit_intercept=True` | 98.0718 | $\pm$ 7.3897 | 欠缺正則化抑制 |
| **M2** | RFE + OLS (最優稀疏特徵子集) | **6** | $k=6$, `step=1` | 96.2026 | $\pm$ 5.8893 | **極致可解釋稀疏解** |
| **M3** | **Ridge Regression (L2 正則化)** | **29** | $\mathbf{\alpha=1000.0}$ | **95.8454** | $\mathbf{\pm 6.2597}$ | **★ 實證最佳模型 (部署主選)** |
| **M4** | Lasso Regression (L1 正則化) | 29 | $\alpha=2.0$ | 95.9075 | $\pm$ 6.3640 | 表現接近 Ridge |

- **客觀模型決策:**  
  依據真實 CV 實驗，**Ridge Regression (M3, $\alpha=1000.0$)** 成功平抑全戶 9 間房間溫濕度高度共線性所引發之係數方差膨脹，以 **95.8454 Wh** 奪得全場最低 CV RMSE，依規範選定為首選部署模型。  
- **RFE 精選 6 大實體因子 (M2):**  
  廚房濕度 (`RH_1`)、客廳濕度 (`RH_2`)、洗衣間溫度 (`T3`)、客廳溫度 (`T2`) 以及日週期時段轉換特徵 (`sin_hour`, `cos_hour`)。

---

## 5. 模型評估與 Split Conformal 預測區間實證

### 5.1 獨立測試集最終盲測評估結果

各模型在未曾參與訓練與調參之獨立測試集上的客觀表現如下：

| 研究問題 | 部署模型 | 測試集 MAE | 測試集 RMSE | 測試集 $R^2$ | 常數基準 Dummy RMSE | 誤差縮減幅度 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Problem 1: Bike Sharing** | M2 (RFE+OLS, $k=30$) | **97.5211 輛** | **133.2088 輛** | **0.6112** | 226.1197 輛 | **-41.09%** |
| **Problem 2: Appliances Energy** | M3 (Ridge, $\alpha=1000$) | **49.2955 Wh** | **83.9152 Wh** | **0.1494** | 91.0182 Wh | **-7.80%** |
| *(Problem 2 備選對照)* | M2 (RFE+OLS, $k=6$) | 50.1420 Wh | 84.1866 Wh | 0.1439 | 91.0182 Wh | -7.51% |

### 5.2 名義 95% 分裂共形預測區間 (Split Conformal Prediction Interval)

為克服計數資料與能耗殘差非高斯分佈之問題，兩大問題均在獨立校準集上計算絕對殘差非整合度分數 $s_i = |y_i - \hat{y}_i|$，依分位數秩次求出共形半徑 $q$：

| 評估指標項目 | Problem 1: Bike Sharing | Problem 2: Appliances Energy |
|:---|:---:|:---:|
| **校準集樣本數 ($n$)** | 2,640 筆 | 3,024 筆 |
| **分位數秩次計算公式** | $\lceil (2640+1) \times 0.95 \rceil = 2509$ | $\lceil (3024+1) \times 0.95 \rceil = 2874$ |
| **校準共形分位數 ($q$)** | **302.6015 輛** | **143.8311 Wh** |
| **名義目標覆蓋率 (Nominal Target)** | 95.00% | 95.00% |
| **測試集實證覆蓋率 (未截斷)** | **94.84%** (2,443 / 2,576) | **95.25%** (2,710 / 2,845) |
| **測試集實證覆蓋率 (非負截斷)** | **94.84%** (2,443 / 2,576) | **95.25%** (2,710 / 2,845) |
| **平均預測區間寬度 (未截斷)** | 605.20 輛 | 287.66 Wh |
| **平均預測區間寬度 (非負截斷 $\max(0, \hat{y}-q)$)** | **513.25 輛** (縮減 15.2%) | **245.98 Wh** (縮減 14.5%) |

> [!WARNING]
> **時間序列可交換性前提宣告 (Exchangeability Caveat):**  
> 標準 Split Conformal 預測之理論無分佈覆蓋保證建立於樣本具備「可交換性 (Exchangeability)」或獨立同分佈 (i.i.d.) 之假設。在時間序列環境中，感測數據與生活用電存在時序自相關性與季節性概念漂移，可交換性無法在數學上嚴格成立。因此，本研究中的 95% 嚴謹定義為**名義設定目標 (Nominal Target)**，模型的真實可靠度必須以獨立測試集上的實證經驗覆蓋率（Bike: 94.84%, Appliances: 95.25%）為最終科學驗收標準。

---

## 6. 網站導覽、下拉選單與互動推論指引

網站現已正式部署於 Vercel，提供直覺且學術專業的交互式視覺化體驗：

### 1. 全域導航與 Problem Selector 下拉選單
- 位於網站右上角導覽列 (`Navbar.tsx`)，實作為原生 HTML `<select>` 互動下拉選單：
  1. `Problem 1 — Bike Sharing Demand`: 進入自行車租借需求預測完整研究頁面（涵蓋 `/`, `/dataset`, `/models`, `/evaluation`, `/predict`, `/methodology`）。
  2. `Problem 2 — Appliances Energy Prediction`: 切換至家電能耗預測儀表板 (`/problem2`)。
  3. `Custom Dataset — Universal MLR`: 切換至加分功能通用資料集多元線性迴歸 Studio (`/custom-dataset`)。

### 2. Problem 2 專屬儀表板特色 (`/problem2`)
- **CRISP-DM 6 階段流程導覽:** 清楚陳列商業目標、資料審計、時序日曆切分、模型比較、盲測成果與部署架構。
- **高解析度學術圖表展示:** 內嵌標的偏度分布圖、138 天時間序列切分邊界圖、Pearson 相關係數熱力圖、RFE 淘汰曲線、四模型 CV 對比圖、殘差診斷圖、時序區間追蹤圖以及 24 小時覆蓋率分析圖。
- **26 特徵分組互動預測器 (Interactive Predictor):**
  - **Lighting (燈光):** `lights` (室內燈具耗電量)。
  - **Indoor Temperature (9 區域溫度):** 廚房 (T1)、客廳 (T2)、洗衣間 (T3)、辦公室 (T4)、浴室 (T5)、外牆 (T6)、燙衣間 (T7)、次臥 (T8)、主臥 (T9)。
  - **Indoor Humidity (9 區域濕度):** RH_1 ~ RH_9 相對濕度 (%)。
  - **Outdoor Weather (6 大氣象因子):** 戶外氣溫、氣壓、戶外濕度、風速、能見度、露點溫度。
  - **Time Schedule (時序狀態):** 時段選擇 (0~23 小時) 與平日/週末開關。
- **雙模型切換:** 支援於部署首選 M3 (Ridge 29 維) 與最優稀疏 M2 (RFE+OLS 6 維) 之間切換即時推論。
- **即時預測輸出:** 顯示原生估計值、非負截斷估算值、95% 名義共形區間 $[\text{Lower}, \text{Upper}]$，以及直觀的**特徵貢獻度瀑布圖 (Waterfall Chart)**。

---

## 7. 加分功能：通用資料集多元線性迴歸 Studio (Bonus)

獨立建置於 `/custom-dataset` 頁面，提供端到端通用多元線性迴歸與特徵工程計算能力：

### 1. 核心功能與工作流程
1. **資料輸入途徑:**
   - **公開 CSV URL 擷取:** 支援貼上任意公開 Web CSV 連結（預設提供 1-Click 載入 50 Startups 基準資料）。
   - **本地 CSV 檔案上傳:** 透過瀏覽器安全上傳本機資料集。
2. **安全防護 (SSRF Hardened):**
   - 伺服器端路由 (`/api/fetch-csv`) 具備嚴格的安全審計過濾：封鎖本機 Loopback (`127.0.0.1`, `localhost`)、私有子網 (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`)、鏈路本地 (`169.254.0.0/16`) 與雲端 Metadata 服務 (`metadata.google.internal`)；強制設定 8 秒請求逾時與 5MB 檔案大小上限。
3. **資料檢視與動態編碼:**
   - 即時解析 CSV Header 與前 5 筆預覽。
   - 自動辨識數值型與低基數類別型欄位（自動進行 One-Hot Encoding）。
4. **動態矩陣迴歸求解器:**
   - 採用正規方程正則化求解器 $(X^T X + \lambda I)^{-1} X^T y$，即時計算 OLS / Ridge 解，並依據標準化係數絕對值輸出特徵重要度排序。
5. **評估與共形區間:**
   - 輸出 Train/Test 之 MAE、RMSE 與 $R^2$ 指標。
   - 繪製互動式 Actual vs Predicted SVG 散佈圖。
   - 依據殘差分位數計算經驗共形預測區間。

### 2. 50 Startups 官方基準實測成果
- **資料來源:** [Avik-Jain 100-Days-Of-ML-Code 50_Startups.csv](https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv)
- **標的變數:** `Profit` (淨利潤，數值連續)
- **實測指標:**
  - **Test RMSE:** **$10,476.76**
  - **Test $R^2$:** **0.9439** (展現高達 94.4% 之極高線性解釋力)
  - **主要貢獻特徵:** R&D Spend (+$14,746.70), Marketing Spend (+$4,109.24)
- **驗證狀態:** 通過 `tests/test_custom_solver.ts` 自動化回歸測試，確認模型計算嚴格真實，絕無使用 Bike Sharing 或 Appliances 之硬編碼係數偽造。

---

## 8. 可重現性驗證、單元測試與 Parity 測試

本專案所有程式碼、模型訓練與數值推論皆經過嚴格測試驗證：

```bash
# 1. 執行 Python 研究實驗管線 (重新產出資料、圖表與模型規格)
python problem2/run_experiment.py

# 2. 自動構建並執行 Jupyter Notebook (確保全部 Cell 正常執行並輸出成果)
python problem2/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace problem2/appliances_energy_prediction.ipynb

# 3. 執行 Python–TypeScript 雙向推論一致性測試 (Parity Tests)
cd website
npm test

# 4. 執行通用資料集求解器單元測試 (50 Startups Benchmark)
npx tsx tests/test_custom_solver.ts

# 5. 執行 ESLint 程式碼規範審查 (0 errors, 0 warnings)
npm run lint

# 6. 執行 TypeScript 嚴格型別檢查
npx tsc --noEmit

# 7. 執行 Next.js 生產環境完整編譯 (12/12 頁面靜態生成成功)
npm run build
```

### 浮點數精度對齊成果 (Floating-Point Parity Summary)
- **Problem 1 (M2 Model):** 最大絕對誤差 **$5.6843 \times 10^{-14}$** ($< 1.0 \times 10^{-12}$，達 IEEE-754 機器精度極限)。
- **Problem 2 (M3 & M2 Models):** 最大絕對誤差 **$0.0000 \times 10^0$** ($< 1.0 \times 10^{-12}$，完全吻合)。

---

## 9. 學術誠信、NotebookLM 與 AI 輔助開發聲明

### 1. NotebookLM 研讀與知識提煉摘要
在進行本作業之研究設計時，使用 NotebookLM 研讀 Luis M. Candanedo 於 2017 年發表於 *Energy and Buildings* 之原始論文《Data driven prediction models of energy use of appliances in a low-energy house》：
- **核心洞察 1:** 家用電器能耗具備劇烈的非平穩隨機尖峰，主要由家庭成員之居住行為（烹飪、烘衣、洗滌）驅動，單純線性迴歸在此類高雜訊資料中 $R^2$ 通常介於 0.10 ~ 0.20 之間，此為該資料集的物理特性，不應刻意過度擬合。
- **核心洞察 2:** 全戶溫濕度感測器（9 間居室）之間具有顯著的空間相關性，造成嚴重的多重共線性。Ridge 正則化或挑選特定區域（如廚房濕度 RH_1 與客廳溫度 T2）是穩定模型係數的最佳實務。
- **核心洞察 3:** 論文中設計隨機變數 `rv1` 與 `rv2` 用作特徵篩選濾鏡，任何合格的特徵選擇演算法皆應在早期將其剔除。

### 2. 人工智慧 (AI) 工具輔助宣告
本專案在實作過程中遵循學術倫理規範，合規使用 AI 工具（Google Antigravity / Claude / GPT）進行輔助開發：
- **AI 輔助範疇:** 輔助編寫 Next.js 前端 UI 元件排版、Tailwind CSS 樣式微調、TypeScript 型別定義與自動化測試腳本框架。
- **學生獨立職責與把關:** 所有資料審計程序、時間序列防洩漏切分邊界定義、CRISP-DM 研究架構、特徵選取超參數設定、數學公式推導與實測數據分析，均由研究者嚴格主導與獨立審核，絕無任何由 AI 虛構或編造數據之行為。

### 3. 原始研究文件與連結
- [Problem 1 專屬技術文件 (7115093001_hw2/README.md)](7115093001_hw2/README.md)
- [Problem 1 研究 Notebook (7115093001_hw2/7115093001_hw2.ipynb)](7115093001_hw2/7115093001_hw2.ipynb)
- [Problem 2 專屬技術文件 (problem2/README.md)](problem2/README.md)
- [Problem 2 研究 Notebook (problem2/appliances_energy_prediction.ipynb)](problem2/appliances_energy_prediction.ipynb)
- [正式可重現性報告 (FINAL_REPRODUCIBILITY_REPORT.md)](FINAL_REPRODUCIBILITY_REPORT.md)

### 4. 課程繳交與 Kaggle 名次宣告
- **課程正式繳交內容:**
  1. 主程式檔案：[`7115093001_hw2/7115093001_hw2.ipynb`](7115093001_hw2/7115093001_hw2.ipynb) 及 [`problem2/appliances_energy_prediction.ipynb`](problem2/appliances_energy_prediction.ipynb)
  2. 整合型 PDF 報告：`7115093001_hw2.pdf`
  3. GitHub 儲存庫與 Vercel 線上展示網站
- **Kaggle 排行榜名次：`N/A`**  
  *說明：本研究採用 Kaggle/UCI 公開收錄之學術資料集，重點在於特徵選擇、正則化迴歸與共形預測區間之嚴謹統計推論與系統化整合，並未報名參加 Kaggle 即時排行榜競賽。*

---

## 10. 研究限制與未來展望

1. **歷史資料時空範圍局限：**
   - Bike Sharing 觀測期間為 2011–2012 年華盛頓特區，Appliances Energy 觀測期間為 2016 年比利時 Stembert 單一家庭低耗能住宅，皆具備特定地域與建築結構屬性，外推至不同氣候區或建築形式時需重新校準。
2. **非線性模型本地實驗比較未納入：**
   - 本研究聚焦於線性迴歸家族之統計可解釋性與特徵選取機制。雖然於書面報告中針對隨機森林、梯度提升樹及神經網路進行了學術文獻綜述比較，但在本地專案代碼中尚未實作上述非線性模型的實證比對。
3. **時間相依性對共形預測理論假設之影響：**
   - 標準 Split Conformal Prediction 之有限樣本無分佈覆蓋保證，嚴格建立於樣本具備「可交換性」（Exchangeability / i.i.d.）之數學假設。
   - 在時序相依資料下，可交換性無法在數學上嚴格保證，因此 95% 為名義設定水準，真實效能應以測試集實測之經驗覆蓋率（Bike: 94.84%, Appliances: 95.25%）為科學依據。
4. **極端峰值之條件預測不確定性：**
   - 在極端尖峰用車與特定高耗能電器啟動情境下，固定寬度共形區間覆蓋率可能偏緊，未來研究可朝向共形化分位數迴歸（CQR）或時序自適應共形區間演進。
5. **通用資料集 Studio 之理論邊界：**
   - `/custom-dataset` 模組專注於數值型標的之多元線性迴歸，對於高度非線性關聯、高維稀疏文字或巨量影像數據並不適用。

---

## 11. 驗收記錄表 (PASS / FAIL / NOT VERIFIED Checklist)

| 階段別 | 驗收核心規範與目標 | 驗收狀態 | 驗收證據與成果對應 |
|:---|:---|:---:|:---|
| **Stage 1: Dataset Acquisition** | 下載並驗證 Appliances 資料集；確認 19,735 筆、29 欄位、0 缺失；SHA-256 比對與來源釐清。 | **PASS** | `problem2/data/energydata_complete.csv` SHA-256 位元完全校驗；`problem2/results/data_audit.json` 產出。 |
| **Stage 2: CRISP-DM Pipeline** | 建立 `problem2/` 獨立工作區；實作 6 大階段；70/15/15 日曆時序切分；Pipeline 內標準化無洩漏；Expanding CV。 | **PASS** | `run_experiment.py` 與 `appliances_energy_prediction.ipynb` 完整落地，零資料洩漏。 |
| **Stage 3: Feature Selection** | RFE 特徵數搜尋；Ridge/Lasso 對比；rv1/rv2 隨機對照欄位健全性檢驗；產出特徵清單與圖表。 | **PASS** | RFE 篩選出 $k=6$ 最優稀疏子集；Ridge 達成最優 CV RMSE (95.85 Wh)；rv1/rv2 成功於末位淘汰；產出 8 張學術圖表。 |
| **Stage 4: Evaluation & Conformal** | 獨立測試集盲測 (MAE/RMSE/R²)；獨立校準集名義 95% Split Conformal 區間；非負截斷；時序可交換性討論。 | **PASS** | Test MAE 49.30 Wh, RMSE 83.92 Wh, $R^2$ 0.1494；Conformal $q=143.83$ Wh；實證覆蓋率達 **95.25%**。 |
| **Stage 5: Website Integration** | 既有 Bike 頁面完整保留；新增 Problem Selector 下拉選單；Problem 2 完整儀表板；26 特徵分組預測；Parity 測試。 | **PASS** | `Navbar.tsx` 原生 `<select>` 實作；`/problem2` 上線；Python–TS 數值推論 Parity 誤差 $< 10^{-12}$。 |
| **Stage 6: Custom Dataset Bonus** | `/custom-dataset` 頁面；支援公開 URL 與本地 CSV；動態 MLR 與 Conformal；SSRF 安全防護；50 Startups 實測。 | **PASS** | SSRF 防護 API 通過；動態求解器通過單元測試；50 Startups 實測 Test $R^2 = 0.9439$。 |
| **Stage 7: README & Delivery** | 全域 README 整合；列出特徵數與數據；操作指引；保留 NotebookLM 與聲明；CI 測試通過；Git Commit & Push。 | **PASS** | 完整編寫根目錄說明；Lint 0 警告；Build 12/12 成功；測試全數 PASS。 |

---
*本專案由學生 7115093001 嚴謹構建，完整落實 GitHub 與 Vercel 雙交付標準。*
