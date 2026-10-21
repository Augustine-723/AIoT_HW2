"use client";

import { useState, useMemo } from "react";
import Image from "next/image";
import { 
  BarChart3, 
  Database, 
  Sliders, 
  LineChart, 
  CheckCircle2, 
  AlertTriangle, 
  ShieldCheck, 
  Layers, 
  ExternalLink,
  RotateCcw,
  Zap,
  HelpCircle,
  Gauge
} from "lucide-react";

import { 
  predictAppliancesEnergy, 
  validateAppliancesInputs, 
  DEFAULT_APPLIANCES_INPUTS, 
  APPLIANCES_PRESETS,
  FEATURE_GROUPS,
  FEATURE_METADATA,
  AppliancesInputFeatures
} from "@/lib/problem2-model";

import modelComparisonRaw from "@/data/problem2/model_comparison.json";

export default function Problem2Page() {
  // Model selection state: M3 Ridge (Primary) or M2 RFE+OLS (Sparse)
  const [selectedModel, setSelectedModel] = useState<"M3_RIDGE" | "M2_RFE">("M3_RIDGE");
  
  // Interactive Prediction Inputs State
  const [inputs, setInputs] = useState<AppliancesInputFeatures>(DEFAULT_APPLIANCES_INPUTS);
  const [activeGroup, setActiveGroup] = useState<string>("lighting");

  // Run Real-time Deterministic Inference
  const prediction = useMemo(() => {
    return predictAppliancesEnergy(inputs, selectedModel);
  }, [inputs, selectedModel]);

  const validation = useMemo(() => {
    return validateAppliancesInputs(inputs);
  }, [inputs]);

  const handleInputChange = (field: keyof AppliancesInputFeatures, val: number) => {
    setInputs((prev) => ({
      ...prev,
      [field]: val
    }));
  };

  const applyPreset = (presetValues: Partial<AppliancesInputFeatures>) => {
    setInputs((prev) => ({
      ...prev,
      ...presetValues
    }));
  };

  const resetDefaults = () => {
    setInputs(DEFAULT_APPLIANCES_INPUTS);
  };

  return (
    <main className="min-h-screen bg-[#F8FAFC] pb-24">
      {/* 1. Header Banner */}
      <section className="bg-gradient-to-b from-blue-900 via-indigo-900 to-[#17243A] text-white pt-12 pb-16 px-4 sm:px-6 lg:px-8 border-b border-blue-950">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="px-3 py-1 rounded-full text-xs font-mono font-bold bg-blue-500/20 text-blue-300 border border-blue-400/30">
                  AIoT HW2 · Problem 2
                </span>
                <span className="px-3 py-1 rounded-full text-xs font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-400/30">
                  CRISP-DM End-to-End
                </span>
                <span className="px-2.5 py-1 rounded-full text-xs font-mono text-slate-300 bg-white/10">
                  ID: 7115093001
                </span>
              </div>
              <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight">
                Appliances Energy Prediction
              </h1>
              <p className="text-slate-300 text-sm sm:text-base max-w-3xl leading-relaxed">
                比利時低耗能住宅 10 分鐘家用電器耗能預測：多元線性迴歸 (MLR)、遞迴特徵淘汰 (RFE)、Ridge/Lasso 正則化比較與 95% 分裂共形預測區間。
              </p>
            </div>

            {/* Quick Actions */}
            <div className="flex flex-wrap items-center gap-3">
              <a
                href="#predict"
                className="px-4 py-2.5 rounded-xl bg-blue-500 hover:bg-blue-400 text-white font-bold text-xs shadow-lg shadow-blue-500/25 transition-all flex items-center gap-2"
              >
                <Zap className="w-4 h-4" />
                <span>立即互動預測 (26特徵)</span>
              </a>
              <a
                href="https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction"
                target="_blank"
                rel="noreferrer"
                className="px-4 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 text-white font-medium text-xs border border-white/20 transition-all flex items-center gap-1.5"
              >
                <span>UCI Repository</span>
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </div>
          </div>

          {/* Key Metrics Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 pt-4">
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-3.5 border border-white/15">
              <span className="text-[11px] font-medium text-blue-200 block">Test MAE (Ridge)</span>
              <span className="text-xl sm:text-2xl font-black tracking-tight text-white">49.30</span>
              <span className="text-[10px] text-slate-300 block">Wh (Dummy: 53.25)</span>
            </div>
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-3.5 border border-white/15">
              <span className="text-[11px] font-medium text-blue-200 block">Test RMSE</span>
              <span className="text-xl sm:text-2xl font-black tracking-tight text-white">83.92</span>
              <span className="text-[10px] text-slate-300 block">Wh (Dummy: 91.02)</span>
            </div>
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-3.5 border border-white/15">
              <span className="text-[11px] font-medium text-blue-200 block">Test R² (Ridge)</span>
              <span className="text-xl sm:text-2xl font-black tracking-tight text-emerald-400">0.1494</span>
              <span className="text-[10px] text-slate-300 block">Dummy: -0.0008</span>
            </div>
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-3.5 border border-white/15">
              <span className="text-[11px] font-medium text-blue-200 block">95% Conformal q</span>
              <span className="text-xl sm:text-2xl font-black tracking-tight text-white">143.83</span>
              <span className="text-[10px] text-slate-300 block">Wh (n=3,024)</span>
            </div>
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-3.5 border border-white/15">
              <span className="text-[11px] font-medium text-blue-200 block">實證覆蓋率 (Test)</span>
              <span className="text-xl sm:text-2xl font-black tracking-tight text-emerald-400">95.25%</span>
              <span className="text-[10px] text-slate-300 block">名義目標: 95.0%</span>
            </div>
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-3.5 border border-white/15">
              <span className="text-[11px] font-medium text-blue-200 block">RFE 精簡特徵</span>
              <span className="text-xl sm:text-2xl font-black tracking-tight text-cyan-300">k = 6</span>
              <span className="text-[10px] text-slate-300 block">CV: 96.20 Wh</span>
            </div>
          </div>
        </div>
      </section>

      {/* Main Container */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-10 space-y-16">

        {/* 2. Interactive Prediction Section (Prominent Anchor) */}
        <section id="predict" className="scroll-mt-20">
          <div className="bg-white rounded-2xl border border-slate-200/80 shadow-md p-6 sm:p-8 space-y-8">
            <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-100 pb-5">
              <div>
                <div className="flex items-center gap-2">
                  <span className="p-1.5 rounded-lg bg-blue-100 text-blue-700">
                    <Zap className="w-5 h-5" />
                  </span>
                  <h2 className="text-xl sm:text-2xl font-extrabold text-slate-900">
                    26 輸入特徵互動預測推論系統
                  </h2>
                </div>
                <p className="text-xs sm:text-sm text-slate-500 mt-1">
                  採 Option B 純客戶端確定性推論。輸入 26 個實體與環境特徵，即時計算瓦時預測值與名義 95% 共形預測區間。
                </p>
              </div>

              {/* Model Selector Toggle */}
              <div className="flex items-center gap-2 bg-slate-100 p-1.5 rounded-xl border border-slate-200">
                <button
                  onClick={() => setSelectedModel("M3_RIDGE")}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                    selectedModel === "M3_RIDGE"
                      ? "bg-blue-600 text-white shadow-xs"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  M3 Ridge (最優 CV, 29特徵)
                </button>
                <button
                  onClick={() => setSelectedModel("M2_RFE")}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                    selectedModel === "M2_RFE"
                      ? "bg-blue-600 text-white shadow-xs"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  M2 RFE+OLS (精簡, 6特徵)
                </button>
              </div>
            </div>

            {/* Prediction Output Display Card */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 bg-gradient-to-br from-slate-900 via-indigo-950 to-blue-950 text-white p-6 rounded-2xl shadow-xl">
              <div className="space-y-2">
                <span className="text-xs font-bold text-blue-300 uppercase tracking-wider flex items-center gap-1.5">
                  <Gauge className="w-4 h-4" />
                  <span>電器耗能點預測 (Point Prediction)</span>
                </span>
                <div className="flex items-baseline gap-2">
                  <span className="text-4xl sm:text-5xl font-black text-white tracking-tight">
                    {prediction.clippedPointPrediction.toFixed(1)}
                  </span>
                  <span className="text-lg font-bold text-blue-300">Wh</span>
                </div>
                <div className="text-[11px] text-slate-400">
                  原生線性方程計算值: <span className="font-mono text-slate-300">{prediction.rawPointPrediction.toFixed(2)} Wh</span>
                  {prediction.rawPointPrediction < 0 && (
                    <span className="text-amber-400 ml-1">(依非負物理邊界鉗制為 0)</span>
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <span className="text-xs font-bold text-emerald-300 uppercase tracking-wider flex items-center gap-1.5">
                  <ShieldCheck className="w-4 h-4" />
                  <span>95% 名義共形預測區間</span>
                </span>
                <div className="flex items-baseline gap-2">
                  <span className="text-2xl sm:text-3xl font-black text-emerald-300 font-mono">
                    [{prediction.piLowerClipped.toFixed(1)}, {prediction.piUpperClipped.toFixed(1)}]
                  </span>
                  <span className="text-sm font-bold text-slate-300">Wh</span>
                </div>
                <div className="text-[11px] text-slate-400">
                  共形半徑 q = <span className="font-mono text-slate-300">{prediction.conformalQuantile.toFixed(2)} Wh</span> · 測試集實證覆蓋率: <span className="text-emerald-400 font-semibold">{prediction.empiricalTestCoveragePct}%</span>
                </div>
              </div>

              <div className="space-y-2 flex flex-col justify-between">
                <div>
                  <span className="text-xs font-bold text-amber-300 uppercase tracking-wider block">當前使用模型</span>
                  <span className="text-base font-extrabold text-white block mt-0.5">{prediction.modelName}</span>
                  <span className="text-[11px] text-slate-300 block">
                    {selectedModel === "M3_RIDGE" ? "全 29 項候選特徵正規化加權" : "RFE 最佳 6 大核心特徵組合"}
                  </span>
                </div>
                <button
                  onClick={resetDefaults}
                  className="self-start text-xs font-semibold text-slate-300 hover:text-white flex items-center gap-1.5 bg-white/10 hover:bg-white/15 px-3 py-1.5 rounded-lg border border-white/10 transition-colors"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>重設為歷史中位數</span>
                </button>
              </div>
            </div>

            {/* Preset Scenarios Buttons */}
            <div className="space-y-2">
              <span className="text-xs font-extrabold text-slate-700 uppercase tracking-wider block">
                快速套用典型作息情境 (Presets):
              </span>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2.5">
                {APPLIANCES_PRESETS.map((preset) => (
                  <button
                    key={preset.name}
                    onClick={() => applyPreset(preset.values)}
                    className="p-2.5 rounded-xl border border-slate-200 hover:border-blue-500 hover:bg-blue-50/50 text-left transition-all group"
                  >
                    <span className="text-xs font-bold text-slate-800 group-hover:text-blue-700 block truncate">
                      {preset.name}
                    </span>
                    <span className="text-[10px] text-slate-500 line-clamp-2 mt-0.5">
                      {preset.desc}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Validation Warnings */}
            {validation.warnings.length > 0 && (
              <div className="bg-amber-50 border border-amber-200 rounded-xl p-3.5 space-y-1.5">
                <div className="flex items-center gap-2 text-amber-800 font-bold text-xs">
                  <AlertTriangle className="w-4 h-4 text-amber-600" />
                  <span>輸入常識審計提醒 ({validation.warnings.length} 項)</span>
                </div>
                <ul className="text-xs text-amber-900 space-y-1 pl-6 list-disc">
                  {validation.warnings.map((w, i) => (
                    <li key={i}>{w.message}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* 26 Features Group Tabs & Sliders */}
            <div className="space-y-4">
              <div className="flex flex-wrap items-center gap-2 border-b border-slate-200 pb-2">
                {Object.values(FEATURE_GROUPS).map((group) => {
                  const isActive = activeGroup === group.id;
                  return (
                    <button
                      key={group.id}
                      onClick={() => setActiveGroup(group.id)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
                        isActive
                          ? "bg-blue-600 text-white shadow-xs"
                          : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"
                      }`}
                    >
                      <span>{group.title}</span>
                      <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-white/20">
                        {group.features.length}
                      </span>
                    </button>
                  );
                })}
              </div>

              {/* Sliders Grid for Active Group */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pt-2">
                {FEATURE_GROUPS[activeGroup as keyof typeof FEATURE_GROUPS].features.map((featureKey) => {
                  const meta = FEATURE_METADATA[featureKey];
                  const currentVal = inputs[featureKey] ?? meta.default;

                  return (
                    <div
                      key={featureKey}
                      className="bg-slate-50/80 p-3.5 rounded-xl border border-slate-200/80 space-y-2 hover:bg-white hover:shadow-xs transition-all"
                    >
                      <div className="flex items-center justify-between">
                        <label htmlFor={`slider-${featureKey}`} className="text-xs font-extrabold text-slate-800">
                          {featureKey} · {meta.label}
                        </label>
                        <span className="text-xs font-mono font-bold text-blue-700 bg-blue-50 px-2 py-0.5 rounded border border-blue-200">
                          {currentVal} {meta.unit}
                        </span>
                      </div>
                      <input
                        id={`slider-${featureKey}`}
                        type="range"
                        min={meta.min}
                        max={meta.max}
                        step={meta.unit === "Wh" || meta.unit === "時" ? 1 : 0.1}
                        value={currentVal}
                        onChange={(e) => handleInputChange(featureKey, parseFloat(e.target.value))}
                        className="w-full accent-blue-600 cursor-pointer"
                      />
                      <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono">
                        <span>{meta.min} {meta.unit}</span>
                        <span className="text-slate-500 truncate max-w-[150px]">{meta.desc}</span>
                        <span>{meta.max} {meta.unit}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Feature Contribution Waterfall / Top Drivers */}
            <div className="space-y-3 pt-4 border-t border-slate-100">
              <span className="text-xs font-extrabold text-slate-700 uppercase tracking-wider block">
                Top 特徵貢獻分解 (Contribution Breakdown, 單位 Wh):
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                {prediction.contributions.slice(0, 8).map((c) => {
                  const isPositive = c.contribution >= 0;
                  return (
                    <div
                      key={c.feature_name}
                      className="p-3 rounded-xl border border-slate-200 bg-slate-50/60 space-y-1"
                    >
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-bold text-slate-800">{c.feature_name}</span>
                        <span className={`font-mono font-bold ${isPositive ? "text-rose-600" : "text-blue-600"}`}>
                          {isPositive ? "+" : ""}{c.contribution.toFixed(2)} Wh
                        </span>
                      </div>
                      <div className="text-[10px] text-slate-500 truncate">
                        {c.description}
                      </div>
                      <div className="text-[9px] font-mono text-slate-400">
                        標準化 z: {c.standardized_value.toFixed(2)} · 係數: {c.coefficient.toFixed(2)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </section>

        {/* 3. Section: Energy Dataset Overview & Audit */}
        <section id="overview" className="scroll-mt-20 space-y-6">
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-indigo-100 text-indigo-700">
              <Database className="w-5 h-5" />
            </span>
            <h2 className="text-2xl font-extrabold text-slate-900">
              Energy Dataset Overview & Audit (資料總覽與審計)
            </h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="bg-white p-6 rounded-2xl border border-slate-200 space-y-4">
              <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">資料品質與真實性校驗</h3>
              <div className="space-y-2.5 text-xs text-slate-600">
                <div className="flex justify-between py-1 border-b border-slate-100">
                  <span className="text-slate-500">觀測筆數 (Observations)</span>
                  <span className="font-mono font-bold text-slate-800">19,735 筆 (100% 吻合)</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-100">
                  <span className="text-slate-500">總欄位數 (Columns)</span>
                  <span className="font-mono font-bold text-slate-800">29 個原始欄位</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-100">
                  <span className="text-slate-500">缺失值 (Null Count)</span>
                  <span className="font-mono font-bold text-emerald-600">0 筆 (完全無缺失)</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-100">
                  <span className="text-slate-500">重複觀測 (Duplicates)</span>
                  <span className="font-mono font-bold text-emerald-600">0 筆</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-100">
                  <span className="text-slate-500">時間戳單調遞增</span>
                  <span className="font-mono font-bold text-emerald-600">嚴格 10 分鐘等距</span>
                </div>
                <div className="flex flex-col py-1 border-b border-slate-100">
                  <span className="text-slate-500">SHA-256 完整性校驗碼</span>
                  <span className="font-mono text-[10px] text-blue-700 truncate mt-0.5">
                    2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-white p-6 rounded-2xl border border-slate-200 space-y-4">
              <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">時序切分 (Full-Date Aligned)</h3>
              <div className="space-y-3 text-xs">
                <div className="p-3 rounded-xl bg-blue-50/70 border border-blue-200">
                  <div className="flex justify-between font-bold text-blue-900">
                    <span>Training Set (70.26%)</span>
                    <span className="font-mono">13,866 筆</span>
                  </div>
                  <span className="text-[11px] text-blue-700 block mt-0.5">
                    2016-01-11 ~ 2016-04-16 (97 天) · Expanding CV 唯一訓練區
                  </span>
                </div>
                <div className="p-3 rounded-xl bg-amber-50/70 border border-amber-200">
                  <div className="flex justify-between font-bold text-amber-900">
                    <span>Calibration Set (15.32%)</span>
                    <span className="font-mono">3,024 筆</span>
                  </div>
                  <span className="text-[11px] text-amber-700 block mt-0.5">
                    2016-04-17 ~ 2016-05-07 (21 天) · 專供 95% 共形分位數校準
                  </span>
                </div>
                <div className="p-3 rounded-xl bg-rose-50/70 border border-rose-200">
                  <div className="flex justify-between font-bold text-rose-900">
                    <span>Test Set (14.42%)</span>
                    <span className="font-mono">2,845 筆</span>
                  </div>
                  <span className="text-[11px] text-rose-700 block mt-0.5">
                    2016-05-08 ~ 2016-05-27 (20 天) · 凍結盲測泛化評估
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-white p-6 rounded-2xl border border-slate-200 space-y-4">
              <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">特徵欄位結構明確區分</h3>
              <div className="space-y-2 text-xs text-slate-600">
                <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                  <span className="font-bold text-slate-800 block">29 個原始欄位:</span>
                  <span className="text-[11px] text-slate-500">date + Appliances + lights + 9室溫 + 9室濕 + 6氣象 + 2隨機</span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                  <span className="font-bold text-slate-800 block">26 個正式原始預測變數:</span>
                  <span className="text-[11px] text-slate-500">25 項實體感測變數 + 1 項時間主特徵 (hour)。排除 rv1/rv2</span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                  <span className="font-bold text-slate-800 block">候選擴充特徵 (29項):</span>
                  <span className="text-[11px] text-slate-500">加入週期編碼 sin_hour, cos_hour 與 is_weekend</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 4. Section: CRISP-DM Six Stages */}
        <section id="crisp-dm" className="scroll-mt-20 space-y-6">
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-blue-100 text-blue-700">
              <Layers className="w-5 h-5" />
            </span>
            <h2 className="text-2xl font-extrabold text-slate-900">
              CRISP-DM 六大研究階段完整落實
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            <div className="bg-white p-5 rounded-2xl border border-slate-200 space-y-2.5">
              <span className="px-2.5 py-1 rounded-md text-[10px] font-extrabold bg-blue-50 text-blue-700 font-mono">
                STAGE 1 · BUSINESS UNDERSTANDING
              </span>
              <h3 className="font-bold text-slate-900 text-base">業務與物理邊界理解</h3>
              <p className="text-xs text-slate-600 leading-relaxed">
                定義「同步遙測能耗估計 (Synchronous Telemetry Estimation)」，釐清室內感測器與氣象觀測皆為同步可用資訊；恪守家用電器能耗非負物理條件 (Wh &ge; 0)。
              </p>
            </div>

            <div className="bg-white p-5 rounded-2xl border border-slate-200 space-y-2.5">
              <span className="px-2.5 py-1 rounded-md text-[10px] font-extrabold bg-indigo-50 text-indigo-700 font-mono">
                STAGE 2 · DATA UNDERSTANDING
              </span>
              <h3 className="font-bold text-slate-900 text-base">資料品質審計與 EDA</h3>
              <p className="text-xs text-slate-600 leading-relaxed">
                嚴格審計 19,735 筆資料、驗證 0 缺失與 SHA-256 位元全等；探索目標變數偏度 (3.39) 與時段相關矩陣；驗證隨機變數 rv1, rv2 無預測力。
              </p>
            </div>

            <div className="bg-white p-5 rounded-2xl border border-slate-200 space-y-2.5">
              <span className="px-2.5 py-1 rounded-md text-[10px] font-extrabold bg-teal-50 text-teal-700 font-mono">
                STAGE 3 · DATA PREPARATION
              </span>
              <h3 className="font-bold text-slate-900 text-base">時序防洩漏切分與管線化</h3>
              <p className="text-xs text-slate-600 leading-relaxed">
                70% / 15% / 15% 全日邊界對齊切分，排除時間穿越洩漏；StandardScaler 標準化皆嚴格封裝於 sklearn Pipeline 內部執行。
              </p>
            </div>

            <div className="bg-white p-5 rounded-2xl border border-slate-200 space-y-2.5">
              <span className="px-2.5 py-1 rounded-md text-[10px] font-extrabold bg-purple-50 text-purple-700 font-mono">
                STAGE 4 · MODELING
              </span>
              <h3 className="font-bold text-slate-900 text-base">擴展視窗 CV 評比與淘汰</h3>
              <p className="text-xs text-slate-600 leading-relaxed">
                僅在 Training 執行 5-Fold Expanding-Window CV；比較 Dummy、Full OLS、RFE+OLS ($k=6$)、Ridge 與 Lasso，依經驗 CV 結果以 Ridge (alpha=1000) 為最佳。
              </p>
            </div>

            <div className="bg-white p-5 rounded-2xl border border-slate-200 space-y-2.5">
              <span className="px-2.5 py-1 rounded-md text-[10px] font-extrabold bg-emerald-50 text-emerald-700 font-mono">
                STAGE 5 · EVALUATION
              </span>
              <h3 className="font-bold text-slate-900 text-base">獨立測試與分裂共形預測</h3>
              <p className="text-xs text-slate-600 leading-relaxed">
                測試集盲測 MAE 49.30 Wh, RMSE 83.92 Wh；獨立校準集建構 95% 共形區間 ($q=143.83$ Wh)，測試集經驗覆蓋率達 95.25%，完整討論時序可交換性。
              </p>
            </div>

            <div className="bg-white p-5 rounded-2xl border border-slate-200 space-y-2.5">
              <span className="px-2.5 py-1 rounded-md text-[10px] font-extrabold bg-amber-50 text-amber-700 font-mono">
                STAGE 6 · DEPLOYMENT
              </span>
              <h3 className="font-bold text-slate-900 text-base">推論規格匯出與雙向對齊</h3>
              <p className="text-xs text-slate-600 leading-relaxed">
                匯出 model_spec.json 與 m2_model_spec.json；Next.js 純客戶端確定性推論；Python 與 TypeScript 雙向 Parity 測試誤差達 0.0000e+0 極限全等。
              </p>
            </div>
          </div>
        </section>

        {/* 5. Section: EDA Charts Gallery */}
        <section className="space-y-6">
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-teal-100 text-teal-700">
              <LineChart className="w-5 h-5" />
            </span>
            <h2 className="text-2xl font-extrabold text-slate-900">
              Exploratory Data Analysis (探索性資料分析圖表)
            </h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white p-5 rounded-2xl border border-slate-200 space-y-3">
              <h3 className="text-sm font-bold text-slate-800">標的變數 Appliances (Wh) 分布右偏特性</h3>
              <div className="relative aspect-video rounded-xl overflow-hidden bg-slate-50 border border-slate-100">
                <Image
                  src="/figures/problem2/eda_appliances_distribution.png"
                  alt="Appliances Distribution"
                  fill
                  className="object-contain"
                />
              </div>
              <p className="text-xs text-slate-500">
                中位數 60.0 Wh，平均值 97.69 Wh，偏度 3.39。呈現高度右偏與長尾特性，高能耗尖峰多集中於傍晚烹飪與洗衣時段。
              </p>
            </div>

            <div className="bg-white p-5 rounded-2xl border border-slate-200 space-y-3">
              <h3 className="text-sm font-bold text-slate-800">138 天時間序列歷程與時序切分界線</h3>
              <div className="relative aspect-video rounded-xl overflow-hidden bg-slate-50 border border-slate-100">
                <Image
                  src="/figures/problem2/eda_time_series_profile.png"
                  alt="Time Series Trajectory"
                  fill
                  className="object-contain"
                />
              </div>
              <p className="text-xs text-slate-500">
                嚴格依完整日曆日邊界對齊：Train (70.3%, 藍色) / Calibration (15.3%, 黃色) / Test (14.4%, 紅色)，杜絕時間穿越洩漏。
              </p>
            </div>

            <div className="bg-white p-5 rounded-2xl border border-slate-200 space-y-3 lg:col-span-2">
              <h3 className="text-sm font-bold text-slate-800">各區溫濕度感測器與氣象觀測之 Pearson 相關係數矩陣</h3>
              <div className="relative h-[380px] rounded-xl overflow-hidden bg-slate-50 border border-slate-100">
                <Image
                  src="/figures/problem2/eda_correlation_matrix.png"
                  alt="Correlation Matrix"
                  fill
                  className="object-contain"
                />
              </div>
              <p className="text-xs text-slate-500">
                室內各區溫度高度正相關 (T1 ~ T9 相關達 0.8+)，濕度亦呈現強共線性。lights 與 hour 對標的有明顯正相關，凸顯正則化或特徵淘汰之必要。
              </p>
            </div>
          </div>
        </section>

        {/* 6. Section: Feature Selection & RFE */}
        <section id="features" className="scroll-mt-20 space-y-6">
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-purple-100 text-purple-700">
              <Sliders className="w-5 h-5" />
            </span>
            <h2 className="text-2xl font-extrabold text-slate-900">
              Feature Selection & RFE Analysis (特徵淘汰與選擇歷程)
            </h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
            <div className="bg-white p-5 rounded-2xl border border-slate-200 space-y-3">
              <h3 className="text-sm font-bold text-slate-800">RFE 特徵數搜尋與 Expanding-Window CV RMSE</h3>
              <div className="relative aspect-video rounded-xl overflow-hidden bg-slate-50 border border-slate-100">
                <Image
                  src="/figures/problem2/feature_selection_rmse.png"
                  alt="Feature Selection RMSE"
                  fill
                  className="object-contain"
                />
              </div>
              <p className="text-xs text-slate-500">
                RFE 依 Expanding CV 搜尋特徵數。當 k=6 時，CV RMSE 達局部最佳值 96.20 Wh (± 5.89 Wh)，相較全特徵 OLS (98.07 Wh) 顯著改善。
              </p>
            </div>

            <div className="bg-white p-6 rounded-2xl border border-slate-200 space-y-4">
              <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">RFE 最佳精簡子集 (k=6) 與健全性查驗</h3>
              <div className="space-y-3">
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-slate-50 text-slate-600 font-bold border-b border-slate-200">
                      <tr>
                        <th className="py-2 px-3">順位</th>
                        <th className="py-2 px-3">特徵名稱</th>
                        <th className="py-2 px-3">標準化係數</th>
                        <th className="py-2 px-3">物理意義與領域詮釋</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      <tr>
                        <td className="py-2 px-3 font-mono font-bold text-blue-600">1</td>
                        <td className="py-2 px-3 font-bold">RH_1</td>
                        <td className="py-2 px-3 font-mono font-bold text-rose-600">+46.60</td>
                        <td className="py-2 px-3 text-slate-500">廚房濕度：烹飪與洗滌為最顯著高耗能行為</td>
                      </tr>
                      <tr>
                        <td className="py-2 px-3 font-mono font-bold text-blue-600">2</td>
                        <td className="py-2 px-3 font-bold">RH_2</td>
                        <td className="py-2 px-3 font-mono font-bold text-blue-600">-43.95</td>
                        <td className="py-2 px-3 text-slate-500">客廳濕度：與廚房濕度形成區域通風梯度</td>
                      </tr>
                      <tr>
                        <td className="py-2 px-3 font-mono font-bold text-blue-600">3</td>
                        <td className="py-2 px-3 font-bold">T3</td>
                        <td className="py-2 px-3 font-mono font-bold text-rose-600">+32.25</td>
                        <td className="py-2 px-3 text-slate-500">洗衣間溫度：烘衣機與洗衣機運轉釋熱</td>
                      </tr>
                      <tr>
                        <td className="py-2 px-3 font-mono font-bold text-blue-600">4</td>
                        <td className="py-2 px-3 font-bold">T2</td>
                        <td className="py-2 px-3 font-mono font-bold text-blue-600">-29.00</td>
                        <td className="py-2 px-3 text-slate-500">客廳溫度：居住活動基準溫度對照</td>
                      </tr>
                      <tr>
                        <td className="py-2 px-3 font-mono font-bold text-blue-600">5</td>
                        <td className="py-2 px-3 font-bold">sin_hour</td>
                        <td className="py-2 px-3 font-mono font-bold text-blue-600">-24.51</td>
                        <td className="py-2 px-3 text-slate-500">晝夜 24 小時正弦週期特徵</td>
                      </tr>
                      <tr>
                        <td className="py-2 px-3 font-mono font-bold text-blue-600">6</td>
                        <td className="py-2 px-3 font-bold">cos_hour</td>
                        <td className="py-2 px-3 font-mono font-bold text-blue-600">-19.25</td>
                        <td className="py-2 px-3 text-slate-500">晝夜 24 小時餘弦週期特徵</td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 space-y-1">
                  <div className="flex items-center gap-1.5 text-xs font-bold text-slate-800">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                    <span>隨機對照變數 rv1, rv2 淘汰健全性檢查 (Sanity Check)</span>
                  </div>
                  <p className="text-[11px] text-slate-600 leading-relaxed">
                    在包含 rv1, rv2 的 RFE 測試中，rv1 排名第 30 / 31，rv2 排名第 31 / 31。兩者皆在最早期階段被優先剔除，證實 RFE 具備強大雜訊辨識能力，未將無關隨機擾動誤認為有效特徵。
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 7. Section: Model Comparison (CV) */}
        <section id="models" className="scroll-mt-20 space-y-6">
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-emerald-100 text-emerald-700">
              <BarChart3 className="w-5 h-5" />
            </span>
            <h2 className="text-2xl font-extrabold text-slate-900">
              Model Comparison (候選模型 5-Fold 擴展視窗 CV 評比)
            </h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
            <div className="bg-white p-5 rounded-2xl border border-slate-200 space-y-3">
              <h3 className="text-sm font-bold text-slate-800">各模型 5-Fold Expanding CV RMSE (Wh) 比較</h3>
              <div className="relative aspect-video rounded-xl overflow-hidden bg-slate-50 border border-slate-100">
                <Image
                  src="/figures/problem2/model_comparison.png"
                  alt="Model Comparison Bar Chart"
                  fill
                  className="object-contain"
                />
              </div>
              <p className="text-xs text-slate-500">
                長條圖標示樣本標準差 ($ddof=1$)。Ridge (alpha=1000) 獲最低 CV 誤差 95.85 Wh，展現 L2 收縮對抑制高維共線性之顯著優勢。
              </p>
            </div>

            <div className="bg-white p-6 rounded-2xl border border-slate-200 space-y-4">
              <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">模型交叉驗證評比總表</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-50 text-slate-600 font-bold border-b border-slate-200">
                    <tr>
                      <th className="py-2.5 px-3">代號</th>
                      <th className="py-2.5 px-3">模型名稱</th>
                      <th className="py-2.5 px-3">特徵數</th>
                      <th className="py-2.5 px-3">CV RMSE (Wh)</th>
                      <th className="py-2.5 px-3">CV 標準差</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {modelComparisonRaw.map((m) => {
                      const isBest = m.model_id === "M3";
                      return (
                        <tr key={m.model_id} className={isBest ? "bg-blue-50/60 font-semibold" : ""}>
                          <td className="py-2.5 px-3 font-mono font-bold text-slate-700">{m.model_id}</td>
                          <td className="py-2.5 px-3">
                            <span className="text-slate-900 block">{m.name}</span>
                            <span className="text-[10px] text-slate-400 font-mono">{m.hyperparameters}</span>
                          </td>
                          <td className="py-2.5 px-3 font-mono">{m.features_count}</td>
                          <td className={`py-2.5 px-3 font-mono font-bold ${isBest ? "text-blue-700" : "text-slate-700"}`}>
                            {m.cv_rmse_mean.toFixed(2)} Wh
                          </td>
                          <td className="py-2.5 px-3 font-mono text-slate-500">
                            ±{m.cv_rmse_std.toFixed(2)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              <div className="p-3.5 rounded-xl bg-blue-50 border border-blue-200 text-xs text-blue-900 space-y-1">
                <span className="font-bold block">模型挑選決策說明 (No Pre-assumed Best Model):</span>
                <p className="leading-relaxed text-[11px] text-blue-800">
                  本研究嚴格恪守規範，不預設 RFE+OLS 一定為最佳模型。由上表可見，面對高相關性 IoT 溫濕度特徵群，Ridge Regression (M3) 透過對所有特徵進行 L2 連續收縮，達成了最佳 CV RMSE (95.85 Wh)，優於單純捨棄變數的 RFE+OLS (96.20 Wh)，故系統將 M3 作為正式生產部署預設模型。
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* 8. Section: Final Test Evaluation & Conformal Prediction Interval */}
        <section id="evaluation" className="scroll-mt-20 space-y-6">
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-rose-100 text-rose-700">
              <LineChart className="w-5 h-5" />
            </span>
            <h2 className="text-2xl font-extrabold text-slate-900">
              Final Test Evaluation & Split Conformal Interval (獨立測試評估與共形區間)
            </h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white p-5 rounded-2xl border border-slate-200 space-y-3">
              <h3 className="text-sm font-bold text-slate-800">實際值 vs. 預測值散佈圖 (Actual vs. Predicted)</h3>
              <div className="relative aspect-video rounded-xl overflow-hidden bg-slate-50 border border-slate-100">
                <Image
                  src="/figures/problem2/actual_vs_predicted.png"
                  alt="Actual vs Predicted Scatter"
                  fill
                  className="object-contain"
                />
              </div>
              <p className="text-xs text-slate-500">
                紅色虛線為完美恆等線 ($y = x$)。測試集 2,845 筆評估呈現點預測均勻環繞主軸，高能耗尖峰處受常態線性假設限制略有收縮。
              </p>
            </div>

            <div className="bg-white p-5 rounded-2xl border border-slate-200 space-y-3">
              <h3 className="text-sm font-bold text-slate-800">殘差診斷 (Residuals vs. Fitted & Distribution)</h3>
              <div className="relative aspect-video rounded-xl overflow-hidden bg-slate-50 border border-slate-100">
                <Image
                  src="/figures/problem2/residual_diagnostics.png"
                  alt="Residual Diagnostics"
                  fill
                  className="object-contain"
                />
              </div>
              <p className="text-xs text-slate-500">
                殘差平均值接近 0 (-11.23 Wh)，分佈呈現典型近常態微右偏，殘差標準差 83.16 Wh。
              </p>
            </div>

            <div className="bg-white p-5 rounded-2xl border border-slate-200 space-y-3">
              <h3 className="text-sm font-bold text-slate-800">測試集首週時序與 95% 共形預測區間 (Prediction Interval)</h3>
              <div className="relative aspect-video rounded-xl overflow-hidden bg-slate-50 border border-slate-100">
                <Image
                  src="/figures/problem2/prediction_interval.png"
                  alt="Prediction Interval Timeline"
                  fill
                  className="object-contain"
                />
              </div>
              <p className="text-xs text-slate-500">
                淺藍色區域為名義 95% 共形預測區間 ($q = 143.83$ Wh)。下界依物理非負原則截斷至 0 Wh，黑線實際觀測值有 95.25% 穩健落在區間內。
              </p>
            </div>

            <div className="bg-white p-5 rounded-2xl border border-slate-200 space-y-3">
              <h3 className="text-sm font-bold text-slate-800">各時段 (0~23時) 實證覆蓋率診斷 (Hourly Coverage)</h3>
              <div className="relative aspect-video rounded-xl overflow-hidden bg-slate-50 border border-slate-100">
                <Image
                  src="/figures/problem2/coverage_by_hour.png"
                  alt="Hourly Coverage Diagnostic"
                  fill
                  className="object-contain"
                />
              </div>
              <p className="text-xs text-slate-500">
                全天 24 小時時段覆蓋率均維持在 91% ~ 98% 之間，紅色虛線為名義 95% 目標，全日實證平均覆蓋率達 95.25%。
              </p>
            </div>
          </div>

          {/* Mathematical & Conformal Formal Explanation Card */}
          <div className="bg-white p-6 sm:p-8 rounded-2xl border border-slate-200 space-y-5">
            <h3 className="text-base font-extrabold text-slate-900 flex items-center gap-2">
              <HelpCircle className="w-5 h-5 text-blue-600" />
              <span>Split Conformal Prediction 數學架構與時序可交換性討論</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs text-slate-600 leading-relaxed">
              <div className="space-y-3 bg-slate-50 p-4 rounded-xl border border-slate-200">
                <span className="font-bold text-slate-900 block">名義 95% 預測區間建構公式:</span>
                <p>
                  給定獨立校準集 {"D_cal = {(x_1, y_1), ..., (x_n, y_n)} (n = 3024)"}，計算絕對殘差非負適應度分數：
                  <span className="font-mono block my-1 font-bold text-slate-800">R_i = |y_i - ŷ_i|</span>
                  設定誤失率 &alpha; = 0.05 (名義 95% 覆蓋)，共形分位數秩次為：
                  <span className="font-mono block my-1 text-slate-700">p = ceil((n+1)(1-&alpha;)) / n = 2874 / 3024 &asymp; 0.9504</span>
                  校準共形量化值計算得 q = 143.8311 Wh。
                </p>
                <p>
                  對任何新觀測 x_new，原生預測區間為 [ŷ - q, ŷ + q]；經非負物理截斷後為：
                  <span className="font-mono block my-1 font-bold text-slate-800">C(x_new) = [max(0, ŷ - q), ŷ + q]</span>
                </p>
              </div>

              <div className="space-y-3 bg-slate-50 p-4 rounded-xl border border-slate-200">
                <span className="font-bold text-slate-900 block">時間序列下之可交換性 (Exchangeability) 嚴謹宣告:</span>
                <p>
                  標準 Split Conformal 理論保證 {"P(y_{n+1} ∈ C(x_{new})) ≥ 1 - α"} 之前提為樣本具備「可交換性 (Exchangeability)」。
                </p>
                <p className="text-amber-900 bg-amber-50/80 p-2.5 rounded-lg border border-amber-200">
                  <strong>學術聲明:</strong> 家用電器能耗與大氣溫濕度時間序列具備顯著的自相關性 (Autocorrelation) 與晝夜作息慣性，嚴格理論上的可交換性並不成立。因此，95% 僅作為名義目標 (Nominal Target)，實際可靠度必須以獨立測試集之經驗覆蓋率 (95.25%) 與每小時覆蓋率診斷進行實證檢驗。
                </p>
              </div>
            </div>
          </div>
        </section>

      </div>
    </main>
  );
}
