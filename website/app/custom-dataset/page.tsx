"use client";

import { useState } from "react";
import { 
  Database, 
  Upload, 
  Globe, 
  Sliders, 
  LineChart, 
  CheckCircle2, 
  AlertTriangle, 
  ShieldCheck, 
  Zap, 
  Sparkles,
  FileSpreadsheet
} from "lucide-react";

import { 
  parseCSV, 
  inspectDataset, 
  trainCustomLinearModel, 
  CustomDatasetSummary, 
  CustomModelEvaluation 
} from "@/lib/custom-dataset-solver";

const DEFAULT_BENCHMARK_URL = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv";

export default function CustomDatasetPage() {
  const [csvUrl, setCsvUrl] = useState(DEFAULT_BENCHMARK_URL);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  // Dataset State
  const [dataset, setDataset] = useState<CustomDatasetSummary | null>(null);
  const [selectedTarget, setSelectedTarget] = useState<string>("");
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [regularization, setRegularization] = useState<number>(1e-4);

  // Model Evaluation State
  const [evaluation, setEvaluation] = useState<CustomModelEvaluation | null>(null);
  const [training, setTraining] = useState(false);

  // Ingest CSV text
  const processCsvText = (text: string, sourceName: string) => {
    try {
      setErrorMsg(null);
      const rows = parseCSV(text);
      const summary = inspectDataset(rows);
      setDataset(summary);

      // Default target selection: look for Profit or last numeric column
      const numCols = summary.columns.filter((c) => c.type === "numeric");
      if (numCols.length === 0) {
        throw new Error("此 CSV 檔案未包含任何連續數值欄位，無法進行線性迴歸。");
      }

      let defaultTarget = numCols.find((c) => c.name.toLowerCase() === "profit")?.name;
      if (!defaultTarget) {
        defaultTarget = numCols[numCols.length - 1].name;
      }

      setSelectedTarget(defaultTarget);
      // Select all other columns as features
      setSelectedFeatures(summary.columns.map((c) => c.name).filter((c) => c !== defaultTarget));
      setEvaluation(null);
      setSuccessMsg(`成功載入「${sourceName}」：共 ${summary.rowCount} 筆觀測、${summary.columnCount} 個欄位！`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "解析失敗";
      setErrorMsg(`CSV 解析錯誤: ${msg}`);
      setDataset(null);
    }
  };

  // Fetch via SSRF-Protected API Route
  const handleFetchUrl = async () => {
    if (!csvUrl.trim()) {
      setErrorMsg("請輸入有效的公開 CSV 檔案網址（URL 不得為空白）。");
      return;
    }
    setLoading(true);
    setErrorMsg(null);
    setSuccessMsg(null);

    try {
      const resp = await fetch("/api/fetch-csv", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: csvUrl.trim() })
      });

      const data = await resp.json();
      if (!resp.ok || !data.success) {
        throw new Error(data.error || "外部 URL 讀取失敗");
      }

      processCsvText(data.csvText, `URL: ${csvUrl}`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "連線逾時或遭防火牆攔截";
      setErrorMsg(`擷取失敗: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  // Handle Local File Upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith(".csv") && file.type !== "text/csv") {
      setErrorMsg("請上傳標準 .csv 格式之文字檔案。");
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      setErrorMsg("檔案容量超過 5MB 上限。");
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      processCsvText(text, file.name);
    };
    reader.onerror = () => {
      setErrorMsg("讀取本機檔案時發生未知錯誤。");
    };
    reader.readAsText(file);
  };

  // Run Real Linear Regression & Feature Selection
  const handleTrainModel = () => {
    if (!dataset || !selectedTarget || selectedFeatures.length === 0) return;
    setTraining(true);
    setErrorMsg(null);

    try {
      // Simulate micro-tick for UI smoothness
      setTimeout(() => {
        try {
          const evalRes = trainCustomLinearModel(
            dataset.rawRows,
            selectedTarget,
            selectedFeatures,
            regularization
          );
          setEvaluation(evalRes);
        } catch (trainErr: unknown) {
          const msg = trainErr instanceof Error ? trainErr.message : "訓練失敗";
          setErrorMsg(`模型擬合異常: ${msg}`);
        } finally {
          setTraining(false);
        }
      }, 50);
    } catch {
      setTraining(false);
    }
  };

  const toggleFeature = (col: string) => {
    if (selectedFeatures.includes(col)) {
      setSelectedFeatures(selectedFeatures.filter((c) => c !== col));
    } else {
      setSelectedFeatures([...selectedFeatures, col]);
    }
  };

  const selectAllFeatures = () => {
    if (!dataset) return;
    setSelectedFeatures(dataset.columns.map((c) => c.name).filter((c) => c !== selectedTarget));
  };

  const deselectAllFeatures = () => {
    setSelectedFeatures([]);
  };

  return (
    <main className="min-h-screen bg-[#F8FAFC] pb-24">
      {/* Header Banner */}
      <section className="bg-gradient-to-b from-slate-900 via-emerald-950 to-[#17243A] text-white pt-12 pb-14 px-4 sm:px-6 lg:px-8 border-b border-emerald-900">
        <div className="max-w-7xl mx-auto space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <span className="px-3 py-1 rounded-full text-xs font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-400/30 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5" />
              <span>AIoT HW2 · 自訂資料分析</span>
            </span>
            <span className="px-3 py-1 rounded-full text-xs font-mono text-slate-300 bg-white/10">
              獨立於 Problem 1 & Problem 2
            </span>
          </div>
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight">
            自訂資料集分析 · Multiple Linear Regression
          </h1>
          <p className="text-slate-300 text-sm sm:text-base max-w-3xl leading-relaxed">
            上傳資料、建立迴歸模型並探索預測結果：支援公開 CSV 網址（含 SSRF 安全防護）與本機 CSV 上傳，動態矩陣求逆計算，即時展示 MAE、RMSE、R²、殘差診斷與 95% 共形預測區間。
          </p>
        </div>
      </section>

      {/* Main Container */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-10 space-y-12">
        
        {/* 1. Data Ingestion Card */}
        <section id="upload" className="bg-white rounded-2xl border border-slate-200/80 p-6 sm:p-8 shadow-xs space-y-6">
          <div className="flex items-center gap-2 border-b border-slate-100 pb-4">
            <span className="p-1.5 rounded-lg bg-emerald-100 text-emerald-700">
              <Database className="w-5 h-5" />
            </span>
            <div>
              <h2 className="text-xl font-extrabold text-slate-900">資料集匯入 (Data Ingestion)</h2>
              <p className="text-xs text-slate-500">可貼上公開 CSV 網址、載入官方 50 Startups 驗收範例，或上傳本機檔案。</p>
            </div>
          </div>

          {/* Input Method 1: Benchmark Preload & URL Input */}
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-3">
              <button
                onClick={() => { setCsvUrl(DEFAULT_BENCHMARK_URL); }}
                className="px-3.5 py-2 rounded-xl text-xs font-bold bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border border-emerald-300 transition-all flex items-center gap-1.5"
              >
                <FileSpreadsheet className="w-4 h-4 text-emerald-600" />
                <span>載入 50 Startups 官方驗收範例</span>
              </button>
              <span className="text-xs text-slate-400">Target: Profit · 50 筆觀測</span>
            </div>

            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2">
              <div className="relative flex-1">
                <Globe className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
                <input
                  type="url"
                  value={csvUrl}
                  onChange={(e) => setCsvUrl(e.target.value)}
                  placeholder="https://example.com/dataset.csv"
                  className="w-full pl-10 pr-4 py-2.5 text-xs rounded-xl border border-slate-300 focus:border-emerald-600 focus:ring-2 focus:ring-emerald-500/20 focus:outline-none font-mono"
                />
              </div>
              <button
                onClick={handleFetchUrl}
                disabled={loading}
                className="px-5 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-300 text-white font-bold text-xs shadow-xs transition-all flex items-center justify-center gap-2 cursor-pointer"
              >
                {loading ? (
                  <>
                    <span className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    <span>安全擷取中...</span>
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4" />
                    <span>擷取並解析 CSV (SSRF 防護)</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Divider */}
          <div className="relative flex items-center justify-center">
            <div className="border-t border-slate-200 w-full" />
            <span className="bg-white px-3 text-xs text-slate-400 font-semibold uppercase tracking-wider absolute">
              或
            </span>
          </div>

          {/* Input Method 2: Local File Upload */}
          <div className="flex items-center justify-center">
            <label className="w-full flex flex-col items-center justify-center px-6 py-6 rounded-2xl border-2 border-dashed border-slate-300 hover:border-emerald-500 bg-slate-50/50 hover:bg-emerald-50/20 transition-all cursor-pointer">
              <Upload className="w-8 h-8 text-slate-400 mb-2" />
              <span className="text-xs font-bold text-slate-700">點擊上傳或拖曳本機 CSV 檔案</span>
              <span className="text-[11px] text-slate-400 mt-0.5">支援最大 5MB 之 UTF-8 逗號分隔格式檔案</span>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="hidden"
              />
            </label>
          </div>

          {/* Messages */}
          {errorMsg && (
            <div className="p-3.5 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-800 flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
              <span>{errorMsg}</span>
            </div>
          )}

          {successMsg && (
            <div className="p-3.5 rounded-xl bg-emerald-50 border border-emerald-200 text-xs text-emerald-800 flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>{successMsg}</span>
            </div>
          )}
        </section>

        {/* 2. Dataset Preview & Configuration */}
        {dataset && (
          <section id="model" className="bg-white rounded-2xl border border-slate-200/80 p-6 sm:p-8 shadow-xs space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-100 pb-4">
              <div className="flex items-center gap-2">
                <span className="p-1.5 rounded-lg bg-blue-100 text-blue-700">
                  <Sliders className="w-5 h-5" />
                </span>
                <div>
                  <h2 className="text-xl font-extrabold text-slate-900">欄位檢視與模型設定 (Configuration)</h2>
                  <p className="text-xs text-slate-500">指定目標變數 (Target)、勾選特徵變數，啟動真實線性模型訓練。</p>
                </div>
              </div>

              <div className="flex items-center gap-2 text-xs">
                <span className="text-slate-500">觀測數: <strong className="text-slate-800 font-mono">{dataset.rowCount}</strong></span>
                <span className="text-slate-300">|</span>
                <span className="text-slate-500">欄位數: <strong className="text-slate-800 font-mono">{dataset.columnCount}</strong></span>
              </div>
            </div>

            {/* Target & Regularization Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-slate-50 p-4 rounded-xl border border-slate-200">
              <div className="space-y-1.5">
                <label htmlFor="target-select" className="text-xs font-bold text-slate-800 block">
                  1. 選擇連續目標變數 (Continuous Target y):
                </label>
                <select
                  id="target-select"
                  value={selectedTarget}
                  onChange={(e) => {
                    const newT = e.target.value;
                    setSelectedTarget(newT);
                    setSelectedFeatures(selectedFeatures.filter((f) => f !== newT));
                  }}
                  className="w-full bg-white border border-slate-300 rounded-lg px-3 py-2 text-xs font-bold text-slate-800 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                >
                  {dataset.columns.filter((c) => c.type === "numeric").map((col) => (
                    <option key={col.name} value={col.name}>
                      {col.name} (數值型 · 均值: {col.mean?.toFixed(2)})
                    </option>
                  ))}
                </select>
                <span className="text-[11px] text-slate-500 block">僅支援連續數值型欄位作為迴歸預測標的。</span>
              </div>

              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-xs font-bold text-slate-800">
                  <label htmlFor="l2-ridge-reg">2. L2 Ridge 正規化係數 (λ):</label>
                  <span className="font-mono text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                    {regularization.toExponential(1)}
                  </span>
                </div>
                <input
                  id="l2-ridge-reg"
                  type="range"
                  min="-6"
                  max="1"
                  step="0.5"
                  value={Math.log10(regularization)}
                  onChange={(e) => setRegularization(Math.pow(10, parseFloat(e.target.value)))}
                  className="w-full accent-emerald-600 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>1e-6 (接近純 OLS)</span>
                  <span>1e-3 (標準穩定)</span>
                  <span>10.0 (強正則化)</span>
                </div>
              </div>
            </div>

            {/* Feature Selection Checkboxes */}
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-xs font-bold text-slate-800 block">
                  3. 指定預測特徵欄位 (Selected Features X, 已選 {selectedFeatures.length} / {dataset.columns.length - 1}):
                </span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={selectAllFeatures}
                    className="text-[11px] font-semibold text-emerald-700 hover:text-emerald-900 px-2 py-1 bg-emerald-50 rounded"
                  >
                    全選
                  </button>
                  <button
                    onClick={deselectAllFeatures}
                    className="text-[11px] font-semibold text-slate-600 hover:text-slate-800 px-2 py-1 bg-slate-100 rounded"
                  >
                    清除
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2.5">
                {dataset.columns.map((col) => {
                  const isTarget = col.name === selectedTarget;
                  const isChecked = selectedFeatures.includes(col.name);

                  return (
                    <label
                      key={col.name}
                      className={`p-2.5 rounded-xl border text-xs flex items-center justify-between cursor-pointer transition-all ${
                        isTarget
                          ? "bg-blue-50/50 border-blue-200 text-blue-800 opacity-60 cursor-not-allowed"
                          : isChecked
                          ? "bg-emerald-50/60 border-emerald-300 text-emerald-950 font-semibold"
                          : "bg-white border-slate-200 text-slate-700 hover:bg-slate-50"
                      }`}
                    >
                      <div className="flex items-center gap-2 truncate">
                        <input
                          type="checkbox"
                          disabled={isTarget}
                          checked={isChecked}
                          onChange={() => toggleFeature(col.name)}
                          className="accent-emerald-600 rounded"
                        />
                        <span className="truncate">{col.name}</span>
                      </div>
                      <span className={`text-[10px] font-mono px-1.5 py-0.2 rounded ${
                        col.type === "numeric" ? "bg-slate-100 text-slate-600" : "bg-purple-100 text-purple-700"
                      }`}>
                        {col.type === "numeric" ? "數值" : "類別"}
                      </span>
                    </label>
                  );
                })}
              </div>
            </div>

            {/* Execute Button */}
            <div className="pt-2">
              <button
                onClick={handleTrainModel}
                disabled={training || selectedFeatures.length === 0}
                className="w-full sm:w-auto px-6 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-300 text-white font-extrabold text-sm shadow-md transition-all flex items-center justify-center gap-2 cursor-pointer"
              >
                {training ? (
                  <>
                    <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    <span>動態矩陣求逆求解中...</span>
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4" />
                    <span>執行多元線性迴歸訓練 (Train Model)</span>
                  </>
                )}
              </button>
            </div>
          </section>
        )}

        {/* 3. Real Training Results & Evaluation */}
        {evaluation && (
          <section id="eval" className="bg-white rounded-2xl border border-slate-200/80 p-6 sm:p-8 shadow-xs space-y-8">
            <div className="flex items-center gap-2 border-b border-slate-100 pb-4">
              <span className="p-1.5 rounded-lg bg-emerald-100 text-emerald-700">
                <LineChart className="w-5 h-5" />
              </span>
              <div>
                <h2 className="text-xl font-extrabold text-slate-900">
                  模型訓練評估與共形區間結果 (Evaluation & Conformal PI)
                </h2>
                <p className="text-xs text-slate-500">
                  嚴格基於所提供之 CSV 進行動態線性迴歸擬合，絕非凍結參數。
                </p>
              </div>
            </div>

            {/* Metric Summary Cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 space-y-1">
                <span className="text-[11px] font-bold text-slate-500 uppercase">Test MAE</span>
                <div className="text-2xl font-black text-slate-900 font-mono">
                  {evaluation.testMetrics.mae.toFixed(2)}
                </div>
                <span className="text-[10px] text-slate-400 block">Dummy: {evaluation.testMetrics.dummyMae.toFixed(2)}</span>
              </div>

              <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 space-y-1">
                <span className="text-[11px] font-bold text-slate-500 uppercase">Test RMSE</span>
                <div className="text-2xl font-black text-slate-900 font-mono">
                  {evaluation.testMetrics.rmse.toFixed(2)}
                </div>
                <span className="text-[10px] text-slate-400 block">Dummy: {evaluation.testMetrics.dummyRmse.toFixed(2)}</span>
              </div>

              <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 space-y-1">
                <span className="text-[11px] font-bold text-slate-500 uppercase">Test R² 決定係數</span>
                <div className="text-2xl font-black text-emerald-600 font-mono">
                  {evaluation.testMetrics.r2.toFixed(4)}
                </div>
                <span className="text-[10px] text-slate-400 block">Train R²: {evaluation.trainMetrics.r2.toFixed(4)}</span>
              </div>

              <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 space-y-1">
                <span className="text-[11px] font-bold text-slate-500 uppercase">95% 共形實證覆蓋率</span>
                <div className="text-2xl font-black text-blue-600 font-mono">
                  {evaluation.conformalInterval ? `${evaluation.conformalInterval.empiricalCoverage.toFixed(1)}%` : "N/A"}
                </div>
                <span className="text-[10px] text-slate-400 block">
                  共形半徑 q: {evaluation.conformalInterval?.conformalQ.toFixed(2)}
                </span>
              </div>
            </div>

            {/* Interactive SVG Scatter Plot: Actual vs Predicted */}
            <div className="space-y-3">
              <span className="text-xs font-bold text-slate-800 uppercase tracking-wider block">
                Actual vs. Predicted 散佈圖 (動態向量繪圖):
              </span>
              <div className="bg-slate-900 p-6 rounded-2xl text-white">
                <div className="flex items-center justify-between text-xs text-slate-400 mb-4 border-b border-slate-800 pb-2">
                  <div className="flex items-center gap-3">
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-blue-400" />
                      <span>訓練集觀測 (n={evaluation.trainCount})</span>
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
                      <span>測試集觀測 (n={evaluation.testCount})</span>
                    </span>
                  </div>
                  <span className="text-[11px] text-slate-500">紅色虛線: 完美預測線 (y = x)</span>
                </div>

                {/* SVG Canvas */}
                <div className="relative w-full h-[320px]">
                  {(() => {
                    const pts = evaluation.scatterPoints;
                    const allVals = pts.flatMap((p) => [p.actual, p.predicted]);
                    const minVal = Math.min(...allVals);
                    const maxVal = Math.max(...allVals);
                    const range = maxVal - minVal || 1;

                    const toX = (val: number) => 40 + ((val - minVal) / range) * 520;
                    const toY = (val: number) => 280 - ((val - minVal) / range) * 240;

                    return (
                      <svg viewBox="0 0 600 320" className="w-full h-full">
                        {/* Axes */}
                        <line x1="40" y1="280" x2="580" y2="280" stroke="#475569" strokeWidth="1" />
                        <line x1="40" y1="40" x2="40" y2="280" stroke="#475569" strokeWidth="1" />
                        
                        {/* Perfect Diagonal Line */}
                        <line
                          x1={toX(minVal)}
                          y1={toY(minVal)}
                          x2={toX(maxVal)}
                          y2={toY(maxVal)}
                          stroke="#ef4444"
                          strokeWidth="1.5"
                          strokeDasharray="4 4"
                        />

                        {/* Data Points */}
                        {pts.map((pt, i) => (
                          <circle
                            key={i}
                            cx={toX(pt.actual)}
                            cy={toY(pt.predicted)}
                            r={pt.isTest ? "4" : "3"}
                            fill={pt.isTest ? "#34d399" : "#60a5fa"}
                            opacity={pt.isTest ? 0.9 : 0.6}
                          >
                            <title>
                              {`Actual: ${pt.actual.toFixed(2)}, Predicted: ${pt.predicted.toFixed(2)}, Residual: ${pt.residual.toFixed(2)} (${pt.isTest ? 'Test' : 'Train'})`}
                            </title>
                          </circle>
                        ))}

                        {/* Labels */}
                        <text x="310" y="310" textAnchor="middle" fill="#94a3b8" fontSize="11">
                          實際值 (Actual {evaluation.targetColumn})
                        </text>
                        <text x="15" y="160" textAnchor="middle" fill="#94a3b8" fontSize="11" transform="rotate(-90 15 160)">
                          預測值 (Predicted)
                        </text>
                      </svg>
                    );
                  })()}
                </div>
              </div>
            </div>

            {/* Feature Weights & Importance Table */}
            <div className="space-y-3">
              <span className="text-xs font-bold text-slate-800 uppercase tracking-wider block">
                特徵重要性與迴歸係數表 (Standardized Feature Weights):
              </span>
              <div className="overflow-x-auto rounded-xl border border-slate-200">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-50 text-slate-600 font-bold border-b border-slate-200">
                    <tr>
                      <th className="py-2.5 px-3">重要性排名</th>
                      <th className="py-2.5 px-3">特徵名稱 (Feature)</th>
                      <th className="py-2.5 px-3">原始欄位</th>
                      <th className="py-2.5 px-3">標準化權重 (Std Beta)</th>
                      <th className="py-2.5 px-3">未標準化斜率 (Slope)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {evaluation.weights.map((w) => (
                      <tr key={w.featureName}>
                        <td className="py-2.5 px-3 font-mono font-bold text-emerald-700">#{w.importanceRank}</td>
                        <td className="py-2.5 px-3 font-bold text-slate-900">{w.featureName}</td>
                        <td className="py-2.5 px-3 text-slate-500">{w.originalColumn}</td>
                        <td className="py-2.5 px-3 font-mono font-bold text-slate-800">
                          {w.stdCoefficient.toFixed(4)}
                        </td>
                        <td className="py-2.5 px-3 font-mono text-slate-500">
                          {w.coefficient.toExponential(4)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        )}

        {/* 4. Technical Safeguards & Limitations Documentation */}
        <section className="bg-white rounded-2xl border border-slate-200/80 p-6 sm:p-8 space-y-4">
          <div className="flex items-center gap-2 text-slate-900 font-extrabold text-base">
            <ShieldCheck className="w-5 h-5 text-emerald-600" />
            <span>安全防護架構與資料支援邊界說明 (Technical Safeguards & Constraints)</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5 text-xs text-slate-600 leading-relaxed">
            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
              <span className="font-bold text-slate-900 block">SSRF (Server-Side Request Forgery) 嚴格防護:</span>
              <ul className="list-disc pl-4 space-y-1 text-slate-600">
                <li>限制僅允許 HTTP 與 HTTPS 協定，阻斷 file:// 與 gopher:// 等協議。</li>
                <li>主動解析 IP 網段，嚴格攔截 127.0.0.0/8 (Loopback)、10.0.0.0/8、172.16.0.0/12、192.168.0.0/16 (RFC-1918 私有網段) 與 169.254.0.0/16 雲端 Metadata 節點。</li>
                <li>強制設定 8 秒請求逾時與 5MB 檔案容量上限，防止 DoS 與記憶體耗盡。</li>
              </ul>
            </div>

            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
              <span className="font-bold text-slate-900 block">通用線性模型支援邊界與誠信宣告:</span>
              <ul className="list-disc pl-4 space-y-1 text-slate-600">
                <li>本工具採用真實正規方程 $(X^T X + \lambda I)^{-1} X^T y$ 求解，並非調用凍結權重。</li>
                <li>支援數值型特徵與低基數類別變數 (自動 One-Hot 編碼，如 50 Startups 之 State)。</li>
                <li>若資料集樣本數少於特徵數 ($N \le P$) 或存在高度奇異矩陣，將啟動 L2 正則化擾動或警示。</li>
                <li>不承諾任何未知結構、超高維稀疏文字或圖像 CSV 均可無痛求解。</li>
              </ul>
            </div>
          </div>
        </section>

      </div>
    </main>
  );
}
