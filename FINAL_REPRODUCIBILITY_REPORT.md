# AIoT HW2 — 最終高分驗收與可重現性報告
# Final Academic Reproducibility & Verification Report

**學生學號 (Student ID):** 7115093001  
**研究題目 (Research Title):** Predicting Hourly Bike Sharing Demand Using Multiple Linear Regression and Feature Selection  
**儲存庫 (Repository):** [https://github.com/Augustine-723/AIoT_HW2.git](https://github.com/Augustine-723/AIoT_HW2.git)  
**實驗識別碼 (Experiment ID):** `AIoT_HW2_PHASE3_EXP01`  
**驗收日期 (Verification Date):** 2026-10-16  
**驗收結論 (Overall Verdict):** **PASS (全數通過)**

---

## 1. 執行總結與恪守原則 (Executive Summary)

本報告針對 GitHub 儲存庫 `Augustine-723/AIoT_HW2` 進行端到端之學術重現性與作業最終高分驗收。驗收全程嚴格恪守以下準則：
1. **不重新選模:** 維持程式自動評選之最佳模型 **M2 (RFE + OLS, k=30)**。
2. **不使用 Test Set 調參:** 測試集 (2,576 筆) 嚴格於 Phase 3 一次性評估，無任何反向參數微調。
3. **不強制改寫統計結果:** 所有指標皆源自 Python 3.13 虛擬環境中真實管線重新計算，數值與報告完全相符。
4. **完全隔離 50 Startups 專案:** 僅於 `AIoT_HW2` 獨立工作區中進行驗證，不更動先前作業之任何檔案。

---

## 2. 執行環境與軟體規格 (Execution Environment)

| 項目 (Item) | 規格與版本 (Specification & Version) | 備註 (Notes) |
| :--- | :--- | :--- |
| **作業系統 (OS)** | Windows (x86_64) | PowerShell 7 / Windows Terminal |
| **Python 版本** | Python 3.13.15 (64-bit) | 乾淨虛擬環境 (`7115093001_hw2/.venv`) |
| **虛擬環境路徑** | `7115093001_hw2/.venv` | 透過 `python -m venv` 建立 |
| **Jupyter 核心** | `aiot_hw2_venv` (`ipykernel 7.4.0`) | 綁定本專案 Python 3.13 虛擬環境 |

---

## 3. Python 依賴套件與版本清單 (Package Dependencies & Versions)

經靜態語法樹 (AST) 掃描與執行階段審計，`7115093001_hw2/requirements.txt` 現已完整涵蓋所有實際引入之相依套件：

| 套件名稱 (Package) | 實際安裝版本 (Installed) | `requirements.txt` 宣告 | 狀態 (Status) | 於專案中之角色 (Role) |
| :--- | :---: | :---: | :---: | :--- |
| **numpy** | 2.5.4 | `>=1.26.0` | **PASS** | 數值矩陣運算與殘差分位數計算 |
| **pandas** | 3.0.6 | `>=2.0.0` | **PASS** | 資料集載入、特徵工程與時序切分 |
| **scikit-learn** | 1.9.1 | `>=1.3.0` | **PASS** | OLS, RFE, Ridge, Lasso, Pipeline, CV |
| **scipy** | 1.18.1 | `>=1.11.0` | **PASS** | 統計分佈檢定與偏度/峰度運算 |
| **matplotlib** | 3.11.2 | `>=3.8.0` | **PASS** | 300 DPI 學術圖表渲染 (`Agg` 後端) |
| **seaborn** | 0.13.2 | `>=0.13.0` | **PASS** | 熱力圖與分佈視覺化 |
| **statsmodels** | 0.15.0 | `>=0.14.0` | **PASS** | 統計模型相容套件庫 |
| **joblib** | 1.6.0 | `>=1.3.0` | **PASS** *(新增宣告)* | 模型權重序列化與反序列化載入 |
| **nbformat** | 5.11.1 | `>=5.9.0` | **PASS** | Jupyter Notebook JSON 結構生成與維護 |
| **nbconvert** | 7.17.2 | `>=7.0.0` | **PASS** *(新增宣告)* | 命令列 headless Notebook 自動化執行 |
| **ipykernel** | 7.4.0 | `>=6.29.0` | **PASS** | Jupyter 核心執行驅動 |
| **jupyter** | 1.1.1 | `>=1.0.0` | **PASS** | Jupyter 元套件管理 |

---

## 4. 依序執行指令與測試結果 (Sequential Execution & Verification Log)

在全新虛擬環境中依序執行以下 7 項自動化腳本，全數順利完成，無任何異常退出：

```bash
# 1. 執行 Phase 1：資料審計、EDA 與 OLS 基準模型
.\7115093001_hw2\.venv\Scripts\python.exe 7115093001_hw2/run_phase1.py
# 結果：[PASS] (費時 4.13s，成功產生 4 張圖表與 6 份 JSON/CSV 結果)

# 2. 執行 Phase 2：RFE 特徵選取、Ridge/Lasso 調優與 12 原生群組消融
.\7115093001_hw2\.venv\Scripts\python.exe 7115093001_hw2/run_phase2.py
# 結果：[PASS] (費時 32.57s，自動選定 M2 k=30，導出 best_model_pipeline.joblib)

# 3. 執行 Phase 3：最佳模型驗證、Conformal 區間校準與最終測試集評估
.\7115093001_hw2\.venv\Scripts\python.exe 7115093001_hw2/run_phase3.py
# 結果：[PASS] (費時 3.91s，重現差距 0.00000000e+00，產生 4 張 300 DPI 圖表)

# 4. 重新構建 Jupyter Notebook（單一事實來源）
.\7115093001_hw2\.venv\Scripts\python.exe 7115093001_hw2/build_notebook.py
# 結果：[PASS] (費時 1.02s，成功生成 7115093001_hw2.ipynb)

# 5. 端到端執行 Jupyter Notebook 全部儲存格
.\7115093001_hw2\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=aiot_hw2_venv 7115093001_hw2/7115093001_hw2.ipynb
# 結果：[PASS] (費時 41.53s，共 17 個程式儲存格全部執行完畢，0 錯誤輸出)

# 6. 同步實驗凍結數據至前端儀表板
.\7115093001_hw2\.venv\Scripts\python.exe 7115093001_hw2/export_website_data.py
# 結果：[PASS] (費時 0.73s，成功同步 8 份 JSON、3 份 CSV 轉換與 12 張圖檔至 website/)

# 7. 執行推論引擎對齊與邊界測試
.\7115093001_hw2\.venv\Scripts\python.exe 7115093001_hw2/test_parity_reproducible.py
# 結果：[PASS] (費時 0.55s，Python 公式與 M2 規格驗證 100% 通過)
```

---

## 5. 核心指標核對表 (Benchmark Metrics Comparison)

重新執行後所得數值與老師評分標準及專案凍結報告進行逐項逐位數嚴格比對：

| 評估維度 (Metric Dimension) | 官方評分要求 (Expected Target) | 重新執行實測值 (Actual Re-run) | 絕對差值 (Delta) | 驗收結果 (Status) |
| :--- | :---: | :---: | :---: | :---: |
| **Training 樣本數** | **12,163 筆** (512 天) | **12,163 筆** | 0 | **PASS** |
| **Calibration 樣本數** | **2,640 筆** (110 天) | **2,640 筆** | 0 | **PASS** |
| **Test 樣本數** | **2,576 筆** (109 天) | **2,576 筆** | 0 | **PASS** |
| **總樣本筆數** | **17,379 筆** (731 天) | **17,379 筆** | 0 | **PASS** |
| **選定最佳模型** | **M2 (RFE + OLS)** | **M2 (RFE + OLS)** | 完美吻合 | **PASS** |
| **M2 選取特徵數 $k$** | **$k = 30$** (共 53 維) | **$k = 30$** | 0 | **PASS** |
| **5-Fold CV Mean RMSE** | **107.4394 輛** | **107.4394 輛** | $< 10^{-6}$ | **PASS** |
| **Test MAE** | **97.5211 輛** | **97.5211 輛** | $< 10^{-6}$ | **PASS** |
| **Test RMSE** | **133.2088 輛** | **133.2088 輛** | $< 10^{-6}$ | **PASS** |
| **Test $R^2$ 判定係數** | **0.6112** | **0.6112** | $< 10^{-6}$ | **PASS** |
| **Test 共形區間覆蓋率** | **94.84%** (2443/2576) | **94.84%** | $< 10^{-4}$ | **PASS** |
| **共形校準分位數 $q$** | **302.6015 輛** | **302.6015 輛** | $< 10^{-6}$ | **PASS** |
| **截斷區間平均寬度** | **513.1870 輛** | **513.1870 輛** | $< 10^{-6}$ | **PASS** |

---

## 6. 資料集來源與 Kaggle 雜湊值查驗 (Dataset Provenance & Kaggle Verification)

依驗收指示對資料集來源進行嚴格審核：

1. **本專案使用之權威資料 (UCI Primary Source):**
   - 檔案路徑: `7115093001_hw2/data/hour.csv`
   - 維度與欄位: **17,379 列，17 欄** (`['instant', 'dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']`)
   - SHA-256 雜湊碼: `e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f` (**Verified**)
2. **Kaggle 原始 CSV 比對查驗狀態 (Kaggle Status):**
   - **查驗結論:** **`NOT VERIFIED`**
   - **原因說明:**
     - 官方 Kaggle 競賽版本 (`c/bike-sharing-demand`) 係將資料切分為每日前 19 天為 `train.csv` (10,886 列，12 欄)、20 日後為 `test.csv` (6,493 列，9 欄，無 casual/registered 且欄位命名不同)，與 UCI 單一完整每小時 17 欄不同。
     - Kaggle 公開轉載資料集 (`marklvl/bike-sharing-dataset`) 雖宣稱為 UCI 鏡像，但因直接透過程式自動下載需個人 Kaggle API 金鑰認證 (`~/.kaggle/kaggle.json`)，本機環境未設定該認證。
     - 依評分誠信準則，**未取得 Kaggle 原始檔案即明確標記為 `NOT VERIFIED`，絕不虛假宣稱已通過雜湊驗證**。

---

## 7. 最小必要修訂審查 (Minimal Necessary Fixes Audit)

在本次驗收過程中，對程式庫進行了最小必要之相容性與嚴謹度修訂，未變更任何實驗邏輯與模型權重：

1. **`7115093001_hw2/requirements.txt`:**
   - 補全直接引入之 `joblib>=1.3.0` 與命令列執行 Notebook 所需之 `nbconvert>=7.0.0`。
2. **`7115093001_hw2/test_parity_reproducible.py`:**
   - 移除原先寫死之他人主機本機路徑 (`C:\Users\9336d\...`)，改採跨平台 `shutil.which` 自動動態探測 `npx`/`node`。若環境無 Node.js 則優雅跳過 TypeScript 引擎測試，完整保留 Python 浮點與測試集全量驗證。
3. **`7115093001_hw2/run_phase1.py`:**
   - 將 `kaggle_verification_status` 之文字依規定明確格式化為 `NOT VERIFIED`，杜絕模糊空間。
4. **`README.md` & `7115093001_hw2/README.md`:**
   - 調整過往「100% 確定性可重現性」、「保證完全數值收斂」等絕對化措辭，修改為基於經驗證據之客觀表述（如「在固定亂數種子與指定套件環境下經自動化腳本驗證」、「在預設數值容許度下收斂」）。

---

## 8. 統計學嚴謹詮釋與研究局限性宣告 (Limitations & Statistical Caveats)

1. **可交換性假設違背 (Exchangeability Violation):**
   - 分拆共形預測之理論保證依賴於樣本具備可交換性（i.i.d.）。真實世界之每小時自行車租借資料具備自相關性與跨季概念漂移，因此 95% 為名義目標覆蓋率（Nominal Coverage），實證 94.84% 乃基於單一測試期之經驗覆蓋，不可宣稱在所有非平穩情境下皆具備無條件數學保證。
2. **尖峰時段條件覆蓋率弱化 (Marginal vs. Conditional Coverage):**
   - 全域固定臨界值 $q=302.60$ 輛無法局部自適應尖峰時段之劇烈方差膨脹。診斷顯示 08:00 (晨峰) 涵蓋率降至 64.49%、17:00 (晚峰) 降至 75.00%，而離峰夜間則達 99%–100%。整體 94.84% 僅具邊際保證（Marginal Guarantee），不代表各條件子集均滿足 95%。
3. **相關性非因果性 (Correlation $\ne$ Causation):**
   - 所有迴歸係數與群組消融 ΔRMSE 衝擊僅代表在特定特徵集與多元線性模型下的敏感度與統計關聯，嚴禁解讀為物理世界的直接因果效應。
