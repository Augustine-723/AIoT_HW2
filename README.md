# AIoT HW2 — Predicting Hourly Bike Sharing Demand Using Multiple Linear Regression and Feature Selection

**學生學號 (Student ID):** 7115093001  
**研究題目 (Research Title):** Predicting Hourly Bike Sharing Demand Using Multiple Linear Regression and Feature Selection  
**GitHub Repository:** [https://github.com/Augustine-723/AIoT_HW2.git](https://github.com/Augustine-723/AIoT_HW2.git)  
**實驗識別碼 (Experiment ID):** `AIoT_HW2_PHASE3_EXP01`  

---

## 專案工作區結構 (Workspace Structure)

本儲存庫為 **AIoT HW2** 之專屬獨立開發工作區，與 50 Startups 作業完全隔離：

- **研究核心工作區:** [`7115093001_hw2/`](7115093001_hw2/)
  - [`7115093001_hw2/7115093001_hw2.ipynb`](7115093001_hw2/7115093001_hw2.ipynb): Phase 1, 2, 3 核心研究 Notebook（單一事實來源，包含完整資料驗證、EDA 圖表、時序切分、預處理管線、RFE 特徵選取、Ridge/Lasso 正則化、群組消融、Split Conformal 預測區間與嚴格測試集評估輸出）
  - [`7115093001_hw2/data/hour.csv`](7115093001_hw2/data/hour.csv): 官方 Bike Sharing 每小時資料集 (17,379 筆)
  - [`7115093001_hw2/figures/`](7115093001_hw2/figures/): 12 張 300 DPI 學術級視覺化圖表 (涵蓋 EDA、特徵選取、正則化、消融與 Phase 3 測試集及預測區間診斷圖)
  - [`7115093001_hw2/results/`](7115093001_hw2/results/): 機器可讀之實驗結果資料庫 (包含模型評比、超參數搜尋紀錄、特徵消融、校準分數、測試集預測、共形預測區間指標與 M2 模型規格 `m2_model_spec.json`)
  - [`7115093001_hw2/models/`](7115093001_hw2/models/): 最佳模型 M2 (RFE + OLS, k=30) 序列化 Pipeline 權重檔
  - [`7115093001_hw2/export_website_data.py`](7115093001_hw2/export_website_data.py): 網站資料同步管線（將 Notebook 凍結成果安全發佈至前端）
  - [`7115093001_hw2/requirements.txt`](7115093001_hw2/requirements.txt): 獨立 Python 環境套件依賴設定
  - [`7115093001_hw2/README.md`](7115093001_hw2/README.md): Phase 1, 2, 3 完整技術報告與獨立執行指南

- **Phase 4 學術研究儀表板與互動預測系統:** [`website/`](website/)
  - **技術堆疊:** Next.js 16 (App Router), TypeScript, Tailwind CSS, Turbopack, 響應式 SVG 圖表系統
  - **推論核心:** 採行 **Option B（確定性 TypeScript 引擎）**，直接執行已擬合之 M2 線性方程（截距 `-76.1581`，30 個選定特徵係數，校準共形分位數 `q = 302.6015`），浮點精度經 17,379 筆樣本檢驗達 `< 2.27 × 10⁻¹³` 機器極限全等。
  - **六大研究頁面:**
    1. `/` (研究總覽 Overview): 展現核心指標（Test MAE 97.5211, RMSE 133.2088, R² 0.6112, 實證覆蓋率 94.84%）、學號、實驗識別與研究架構。
    2. `/dataset` (資料集與 EDA): 呈現時段雙峰需求、目標右偏、氣象相關矩陣、氣溫影響，並標示嚴格排除之洩漏變數（`casual`, `registered`）與 1,135 筆缺漏小時審計。
    3. `/models` (模型比較與消融): 完整展現 M1 ~ M4 在 5-Fold 擴展視窗 CV 下的無偏樣本標準差 (ddof=1) 表現、RFE 淘汰歷程、超參數調校與群組消融重要度。
    4. `/evaluation` (最終測試集評估): 提供互動式實際值 vs 預測值散佈圖、7 天時序共形區間軸（支援原始與非負截斷切換）、殘差診斷與 0–100% 全幅時段覆蓋率分析（標記 08:00、17:00、18:00 尖峰條件覆蓋差異）。
    5. `/predict` (互動預測推論): 支援 12 項特徵輸入調節與 5 大典型情境快速切換，即時計算原生連續預測、非負呈現推估、95% 名義共形預測區間與各特徵貢獻分項。
    6. `/methodology` (研究方法論): CRISP-DM 六大階段、端到端研究管線、分裂共形預測數學公式、可交換性前提與正式學術文獻引用。

---

## 本地開發與建置指南 (Local Development)

### 1. 執行研究實驗與匯出前端資料
```bash
# 在 Python 虛擬環境下執行 Phase 3 實驗並生成成果
python 7115093001_hw2/build_notebook.py

# 同步凍結實驗資料與圖表至網站目錄
python 7115093001_hw2/export_website_data.py
```

### 2. 本地啟動與建置研究網站
```bash
cd website

# 安裝相依套件
npm install

# 執行開發伺服器
npm run dev

# 執行生產建置與 TypeScript 型別驗證
npm run build

# 啟動生產伺服器
npm run start
```

---

## Vercel 獨立部署指南 (Independent Vercel Deployment)

本專案配置為完全獨立之 Vercel 部署，**絕不可與先前的 50 Startups 網站共用專案**：

1. **建立獨立 Vercel 專案:**
   - 登入 [Vercel Dashboard](https://vercel.com)。
   - 點擊 **Add New...** -> **Project**。
   - 選擇匯入 GitHub 儲存庫 `Augustine-723/AIoT_HW2`。

2. **配置專案根目錄 (Root Directory):**
   - 在專案設定中，將 **Root Directory** 設定為：`website`。
   - **Framework Preset** 選擇：`Next.js`。

3. **建置設定 (Build & Development Settings):**
   - Build Command: `npm run build` (或保留預設 `next build`)
   - Output Directory: `.next` (保留預設)
   - Install Command: `npm install` (保留預設)

4. **環境變數:**
   - 本系統前端採用純客戶端確定性推論（Option B）與靜態預先渲染，無須設定外部資料庫金鑰或伺服器密鑰，無機密洩漏風險。

5. **點擊 Deploy:**
   - 專案將在約 1 分鐘內完成建置與邊緣部署，產生獨立之 `aiot-hw2-*.vercel.app` 網址。

---

## 嚴格隔離宣告 (Strict Project Isolation)

- 本儲存庫為 `Augustine-723/AIoT_HW2`，所有成果僅在此專案中開發與提交。
- 未改動、亦未依賴先前的 `Augustine-723/50-Starup` 儲存庫或其既有 Vercel 部署。
