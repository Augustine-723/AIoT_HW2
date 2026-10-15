# AIoT HW2 — Predicting Hourly Bike Sharing Demand Using Multiple Linear Regression and Feature Selection

**學生學號 (Student ID):** 7115093001  
**研究題目 (Research Title):** Predicting Hourly Bike Sharing Demand Using Multiple Linear Regression and Feature Selection  
**GitHub Repository:** [https://github.com/Augustine-723/AIoT_HW2.git](https://github.com/Augustine-723/AIoT_HW2.git)  

---

## 專案工作區結構 (Workspace Structure)

本儲存庫為 **AIoT HW2** 之專屬獨立開發工作區，與 50 Startups 作業完全分離。

- **研究核心工作區:** [`7115093001_hw2/`](7115093001_hw2/)
  - [`7115093001_hw2/7115093001_hw2.ipynb`](7115093001_hw2/7115093001_hw2.ipynb): Phase 1, 2, 3 核心研究 Notebook（單一事實來源，包含完整資料驗證、EDA 圖表、時序切分、預處理管線、RFE 特徵選取、Ridge/Lasso 正則化、群組消融、Split Conformal 預測區間與嚴格測試集評估輸出）
  - [`7115093001_hw2/data/hour.csv`](7115093001_hw2/data/hour.csv): 官方 Bike Sharing 每小時資料集 (17,379 筆)
  - [`7115093001_hw2/figures/`](7115093001_hw2/figures/): 12 張 300 DPI 學術級視覺化圖表 (涵蓋 EDA、特徵選取、正則化、消融與 Phase 3 測試集及預測區間診斷圖)
  - [`7115093001_hw2/results/`](7115093001_hw2/results/): 機器可讀之實驗結果資料庫 (20 個 JSON 與 CSV 檔案，包含模型評比、超參數搜尋紀錄、特徵消融、校準分數、測試集預測與共形預測區間指標)
  - [`7115093001_hw2/models/`](7115093001_hw2/models/): 最佳模型 M2 (RFE + OLS, k=30) 序列化 Pipeline 權重檔
  - [`7115093001_hw2/requirements.txt`](7115093001_hw2/requirements.txt): 獨立環境套件依賴設定
  - [`7115093001_hw2/README.md`](7115093001_hw2/README.md): Phase 1, 2, 3 完整技術報告與獨立執行指南
