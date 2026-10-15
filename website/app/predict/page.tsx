"use client";

import { useState } from "react";
import { 
  predictBikeDemand, 
  BikeInputFeatures, 
  PRESET_SCENARIOS, 
  M2_SPEC 
} from "@/lib/m2-model";
import { 
  Sparkles, 
  Sliders, 
  CheckCircle2, 
  AlertTriangle, 
  Calendar, 
  Clock, 
  Thermometer, 
  Wind, 
  Droplets, 
  SunMedium, 
  Layers
} from "lucide-react";

export default function InteractivePredictionPage() {
  // Default to Morning Commute Peak Scenario
  const [inputs, setInputs] = useState<BikeInputFeatures>(PRESET_SCENARIOS[0].features);
  const [selectedPreset, setSelectedPreset] = useState<string>(PRESET_SCENARIOS[0].name);

  // Execute pure M2 model inference
  const prediction = predictBikeDemand(inputs);

  const handlePresetSelect = (presetName: string) => {
    const target = PRESET_SCENARIOS.find((p) => p.name === presetName);
    if (target) {
      setInputs({ ...target.features });
      setSelectedPreset(target.name);
    }
  };

  const handleInputChange = <K extends keyof BikeInputFeatures>(
    key: K,
    val: BikeInputFeatures[K]
  ) => {
    setInputs((prev) => ({ ...prev, [key]: val }));
    setSelectedPreset(""); // Clear preset selection when manual adjustment occurs
  };

  return (
    <div className="space-y-12">
      {/* Header */}
      <section className="space-y-3">
        <div className="flex items-center gap-2 text-xs font-mono text-cyan-400">
          <Sparkles className="w-4 h-4" />
          <span>REAL-TIME INFERENCE ENGINE · OPTION B VERIFIED</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          M2 模型互動式預測引擎與共形不確定性推論
        </h1>
        <p className="text-sm text-slate-300 max-w-4xl leading-relaxed">
          直接在前端執行已訓練完成之 M2 (RFE + OLS) 權重矩陣，絕非數值模擬或隨機查找。
          具備與 Python Scikit-learn 嚴格驗證之 <strong className="text-emerald-400 font-mono">&lt; 2.27 × 10⁻¹³</strong> 浮點全等精度。
          支援 12 項原始預測變數調節，並輸出 95% 名義共形預測區間（q = 302.60）。
        </p>
      </section>

      {/* Numerical Parity & Architectural Decision Alert */}
      <section className="p-5 rounded-2xl border border-emerald-500/30 bg-emerald-950/20 text-slate-200 space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-emerald-400 font-semibold text-sm">
            <CheckCircle2 className="w-4 h-4" />
            <span>架構可行性決策評估 (Option B: Verified TypeScript Pipeline)</span>
          </div>
          <span className="text-xs font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300">
            Parity Error: 2.27 × 10⁻¹³
          </span>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed">
          經實地測試，本專案採用 <strong>Option B（純 TypeScript 確定性推論引擎）</strong>，
          將凍結訓練之 30 個選定係數與截距以 IEEE-754 雙精度浮點數重構。在歷史全部 17,379 筆樣本上進行數值對齊測試，
          推論結果與原生 Python Scikit-learn Pipeline 之絕對誤差最大值僅為 <strong>2.27 × 10⁻¹³</strong>（機器浮點精度極限），
          免除伺服器 Cold Start、龐大 Python Wheel 依賴與 API 逾時風險，提供 &lt; 1ms 的極速無伺服器推論。
        </p>
      </section>

      {/* Preset Scenarios Selector */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <Sliders className="w-4 h-4 text-indigo-400" />
            <span>典型情境快速載入 (Preset Archetype Scenarios)</span>
          </h2>
          <span className="text-xs text-slate-400">點擊切換標準特徵組合</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {PRESET_SCENARIOS.map((preset) => {
            const isSelected = selectedPreset === preset.name;
            return (
              <button
                key={preset.name}
                onClick={() => handlePresetSelect(preset.name)}
                className={`p-3.5 rounded-xl border text-left transition-all flex flex-col justify-between ${
                  isSelected
                    ? "bg-indigo-600/20 border-indigo-500 text-white shadow-md shadow-indigo-950"
                    : "bg-slate-900/40 border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700"
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
                      {preset.tag}
                    </span>
                    {isSelected && <CheckCircle2 className="w-3.5 h-3.5 text-indigo-400" />}
                  </div>
                  <div className="text-xs font-bold text-white mb-1">
                    {preset.name}
                  </div>
                </div>
                <p className="text-[10px] text-slate-400 leading-tight">
                  {preset.description}
                </p>
              </button>
            );
          })}
        </div>
      </section>

      {/* Main Prediction & Controls Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: 12 Predictor Controls (7 Cols) */}
        <div className="lg:col-span-7 space-y-6">
          <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-6">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                <Calendar className="w-4 h-4 text-cyan-400" />
                <span>12 項原始輸入特徵調節盤 (Feature Controls)</span>
              </h2>
              <span className="text-xs font-mono text-slate-400">12 Predictors</span>
            </div>

            {/* Centralized Calendar Consistency & Validation Warnings */}
            {prediction.validation.warnings.length > 0 && (
              <div className="p-4 rounded-xl border border-amber-500/30 bg-amber-950/20 space-y-2 text-xs">
                <div className="flex items-center gap-2 text-amber-400 font-semibold">
                  <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                  <span>輸入驗證與曆法組合提示 ({prediction.validation.warnings.length} 項警示)</span>
                </div>
                <ul className="space-y-1.5 pl-5 list-disc text-slate-300">
                  {prediction.validation.warnings.map((w, idx) => (
                    <li key={idx}>
                      <strong className="text-amber-300">{w.title}：</strong>
                      <span className="text-slate-400">{w.message}</span>
                    </li>
                  ))}
                </ul>
                <div className="text-[10px] text-slate-400 pt-1 border-t border-amber-500/10">
                  * 說明：上述提示為 UCI 領域語意一致性輔助警示，已確立之 M2 迴歸模型仍忠實依據您的特徵數值進行推論，未改動既定模型結構。
                </div>
              </div>
            )}

            {/* 1. Calendar & Time Features */}
            <div className="space-y-4">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400 block">
                時段與曆法維度 (Calendar & Temporal)
              </span>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {/* Year Indicator */}
                <div className="space-y-1.5">
                  <label className="text-xs text-slate-300 flex justify-between">
                    <span>年份趨勢 (yr)</span>
                    <span className="font-mono text-cyan-400">{inputs.yr === 1 ? "2012 年" : "2011 年"}</span>
                  </label>
                  <select
                    value={inputs.yr}
                    onChange={(e) => handleInputChange("yr", parseInt(e.target.value))}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                  >
                    <option value={0}>2011 年 (基準年度, yr=0)</option>
                    <option value={1}>2012 年 (成長年度, yr=1)</option>
                  </select>
                  <span className="text-[10px] text-amber-400/80 block">歷史編碼僅支援 2011-2012，不可外推</span>
                </div>

                {/* Season */}
                <div className="space-y-1.5">
                  <label className="text-xs text-slate-300 flex justify-between">
                    <span>季節 (season)</span>
                    <span className="font-mono text-cyan-400">
                      {["春季", "夏季", "秋季", "冬季"][inputs.season - 1]}
                    </span>
                  </label>
                  <select
                    value={inputs.season}
                    onChange={(e) => handleInputChange("season", parseInt(e.target.value))}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                  >
                    <option value={1}>1: 春季 (Spring, 基準)</option>
                    <option value={2}>2: 夏季 (Summer)</option>
                    <option value={3}>3: 秋季 (Fall)</option>
                    <option value={4}>4: 冬季 (Winter)</option>
                  </select>
                </div>

                {/* Month */}
                <div className="space-y-1.5">
                  <label className="text-xs text-slate-300 flex justify-between">
                    <span>月份 (mnth)</span>
                    <span className="font-mono text-cyan-400">{inputs.mnth} 月</span>
                  </label>
                  <select
                    value={inputs.mnth}
                    onChange={(e) => handleInputChange("mnth", parseInt(e.target.value))}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                  >
                    {Array.from({ length: 12 }, (_, i) => i + 1).map((m) => (
                      <option key={m} value={m}>
                        {m} 月 {m === 7 || m === 8 ? "★ RFE 選定" : ""}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Hour Slider */}
              <div className="space-y-2 pt-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-300 flex items-center gap-1.5">
                    <Clock className="w-3.5 h-3.5 text-indigo-400" />
                    <span>時段 (hr, 0 到 23 時)</span>
                  </span>
                  <div className="flex items-center gap-2">
                    {inputs.hr === 8 || inputs.hr === 17 || inputs.hr === 18 ? (
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-amber-500/20 text-amber-300 border border-amber-500/30">
                        ⚡ 尖峰通勤時段
                      </span>
                    ) : null}
                    <span className="font-mono text-sm font-bold text-white px-2 py-0.5 rounded bg-slate-800">
                      {inputs.hr < 10 ? `0${inputs.hr}` : inputs.hr}:00
                    </span>
                  </div>
                </div>
                <input
                  type="range"
                  min="0"
                  max="23"
                  step="1"
                  value={inputs.hr}
                  onChange={(e) => handleInputChange("hr", parseInt(e.target.value))}
                  className="w-full accent-indigo-500 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] font-mono text-slate-400">
                  <span>00:00 (基準)</span>
                  <span>08:00 (晨峰)</span>
                  <span>12:00</span>
                  <span>17:00 (傍晚峰)</span>
                  <span>23:00</span>
                </div>
              </div>

              {/* Weekday, Workingday, Holiday */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
                <div className="space-y-1.5">
                  <label className="text-xs text-slate-300">星期 (weekday)</label>
                  <select
                    value={inputs.weekday}
                    onChange={(e) => handleInputChange("weekday", parseInt(e.target.value))}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                  >
                    <option value={0}>週日 (Sunday, 0)</option>
                    <option value={1}>週一 (Monday, 1)</option>
                    <option value={2}>週二 (Tuesday, 2)</option>
                    <option value={3}>週三 (Wednesday, 3)</option>
                    <option value={4}>週四 (Thursday, 4)</option>
                    <option value={5}>週五 (Friday, 5)</option>
                    <option value={6}>週六 (Saturday, 6)</option>
                  </select>
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs text-slate-300">工作日 (workingday)</label>
                  <select
                    value={inputs.workingday}
                    onChange={(e) => handleInputChange("workingday", parseInt(e.target.value))}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                  >
                    <option value={1}>工作日 (1: 既非假日亦非週末)</option>
                    <option value={0}>非工作日 (0: 假日或週末)</option>
                  </select>
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs text-slate-300">國定假日 (holiday)</label>
                  <select
                    value={inputs.holiday}
                    onChange={(e) => handleInputChange("holiday", parseInt(e.target.value))}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                  >
                    <option value={0}>平常日 (0: 否)</option>
                    <option value={1}>法定假日 (1: 是)</option>
                  </select>
                </div>
              </div>
            </div>

            {/* 2. Weather & Climate Features */}
            <div className="space-y-4 pt-4 border-t border-slate-800">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400 block">
                天候與微氣候連續因子 (Weather & Micro-climate)
              </span>

              {/* Weather Situation */}
              <div className="space-y-1.5">
                <label className="text-xs text-slate-300 flex justify-between">
                  <span>天候狀況代碼 (weathersit)</span>
                  <span className="font-mono text-cyan-400">
                    {["晴朗/少雲", "薄霧/多雲", "小雨/小雪", "大豪雨/暴風雪"][inputs.weathersit - 1]}
                  </span>
                </label>
                <select
                  value={inputs.weathersit}
                  onChange={(e) => handleInputChange("weathersit", parseInt(e.target.value))}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                >
                  <option value={1}>1: 晴朗 / 少雲 / 多雲 (Clear/Few clouds, 基準)</option>
                  <option value={2}>2: 薄霧 / 陰天 (Mist + Cloudy)</option>
                  <option value={3}>3: 小雨 / 小雪 / 雷雨 (Light Snow/Rain, ★ RFE -56.41)</option>
                  <option value={4}>4: 暴雨 / 暴風雪 / 冰雹 (Heavy Rain/Snow, ★ RFE -52.82)</option>
                </select>
              </div>

              {/* Temperature Slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-300 flex items-center gap-1.5">
                    <Thermometer className="w-3.5 h-3.5 text-cyan-400" />
                    <span>正規化氣溫 (temp)</span>
                  </span>
                  <div className="flex items-center gap-2 font-mono">
                    <span className="text-slate-400 text-[11px]">
                      實際換算: 約 <strong>{Math.round(inputs.temp * 41)}°C</strong>
                    </span>
                    <span className="text-white font-bold bg-slate-800 px-2 py-0.5 rounded text-xs">
                      {inputs.temp.toFixed(2)}
                    </span>
                  </div>
                </div>
                <input
                  type="range"
                  min="0.02"
                  max="1.00"
                  step="0.01"
                  value={inputs.temp}
                  onChange={(e) => handleInputChange("temp", parseFloat(e.target.value))}
                  className="w-full accent-cyan-400 cursor-pointer"
                />
              </div>

              {/* Apparent Temperature Slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-300 flex items-center gap-1.5">
                    <SunMedium className="w-3.5 h-3.5 text-amber-400" />
                    <span>正規化體感溫度 (atemp)</span>
                  </span>
                  <div className="flex items-center gap-2 font-mono">
                    <span className="text-slate-400 text-[11px]">
                      實際換算: 約 <strong>{Math.round(inputs.atemp * 50)}°C</strong>
                    </span>
                    <span className="text-white font-bold bg-slate-800 px-2 py-0.5 rounded text-xs">
                      {inputs.atemp.toFixed(2)}
                    </span>
                  </div>
                </div>
                <input
                  type="range"
                  min="0.00"
                  max="1.00"
                  step="0.01"
                  value={inputs.atemp}
                  onChange={(e) => handleInputChange("atemp", parseFloat(e.target.value))}
                  className="w-full accent-amber-400 cursor-pointer"
                />
              </div>

              {/* Humidity Slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-300 flex items-center gap-1.5">
                    <Droplets className="w-3.5 h-3.5 text-indigo-400" />
                    <span>正規化相對濕度 (hum)</span>
                  </span>
                  <div className="flex items-center gap-2 font-mono">
                    <span className="text-slate-400 text-[11px]">
                      實際濕度: 約 <strong>{Math.round(inputs.hum * 100)}%</strong>
                    </span>
                    <span className="text-white font-bold bg-slate-800 px-2 py-0.5 rounded text-xs">
                      {inputs.hum.toFixed(2)}
                    </span>
                  </div>
                </div>
                <input
                  type="range"
                  min="0.00"
                  max="1.00"
                  step="0.01"
                  value={inputs.hum}
                  onChange={(e) => handleInputChange("hum", parseFloat(e.target.value))}
                  className="w-full accent-indigo-400 cursor-pointer"
                />
              </div>

              {/* Windspeed Slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-300 flex items-center gap-1.5">
                    <Wind className="w-3.5 h-3.5 text-slate-400" />
                    <span>正規化風速 (windspeed)</span>
                  </span>
                  <div className="flex items-center gap-2 font-mono">
                    <span className="text-slate-400 text-[11px]">
                      實際風速: 約 <strong>{Math.round(inputs.windspeed * 67)} km/h</strong>
                    </span>
                    <span className="text-white font-bold bg-slate-800 px-2 py-0.5 rounded text-xs">
                      {inputs.windspeed.toFixed(2)}
                    </span>
                  </div>
                </div>
                <input
                  type="range"
                  min="0.00"
                  max="1.00"
                  step="0.01"
                  value={inputs.windspeed}
                  onChange={(e) => handleInputChange("windspeed", parseFloat(e.target.value))}
                  className="w-full accent-slate-400 cursor-pointer"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Prediction Output Card (5 Cols) */}
        <div className="lg:col-span-5 space-y-6">
          <div className="p-6 rounded-2xl border border-indigo-500/40 bg-gradient-to-b from-indigo-950/40 via-slate-900/60 to-slate-950 shadow-2xl space-y-6">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <span className="text-xs font-mono text-cyan-400 uppercase tracking-wider font-semibold">
                推論輸出結果 (Prediction Output)
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                {M2_SPEC.modelId}
              </span>
            </div>

            {/* Headline Display Point Prediction */}
            <div className="space-y-1">
              <span className="text-xs text-slate-400 block">
                非負呈現推估值 (Nonnegative Display Count)
              </span>
              <div className="text-4xl sm:text-5xl font-black font-mono text-white tracking-tight flex items-baseline gap-2">
                <span>{Math.round(prediction.clippedPointPrediction)}</span>
                <span className="text-sm font-sans font-medium text-slate-400">輛 / 小時</span>
              </div>
              <div className="text-[11px] font-mono text-slate-400 pt-1">
                模型原生連續點預測：
                <strong className={prediction.rawPointPrediction < 0 ? "text-amber-400" : "text-cyan-400"}>
                  {prediction.rawPointPrediction.toFixed(4)}
                </strong>
                {prediction.rawPointPrediction < 0 && (
                  <span className="ml-1 text-amber-400 text-[10px]">
                    (截距小於零導致原生未截斷為負，已進行非負呈現轉換)
                  </span>
                )}
              </div>
            </div>

            {/* 95% Conformal Prediction Interval */}
            <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/80 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-white flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  <span>95% 名義共形預測區間 (95% Nominal PI)</span>
                </span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  q = 302.60
                </span>
              </div>

              {/* Clipped Interval Display */}
              <div className="text-xl font-bold font-mono text-emerald-300">
                [ {prediction.piLowerClipped.toFixed(1)} , {prediction.piUpperClipped.toFixed(1)} ]
              </div>
              <span className="text-[10px] text-slate-400 block">
                非負截斷區間：[max(0, ŷ - 302.60), ŷ + 302.60]
              </span>

              {/* Raw Interval */}
              <div className="pt-2 border-t border-slate-800/60 text-[11px] font-mono text-slate-400 flex justify-between">
                <span>原始對稱區間 [ŷ - q, ŷ + q]</span>
                <span className="text-slate-300">
                  [{prediction.piLowerRaw.toFixed(1)}, {prediction.piUpperRaw.toFixed(1)}]
                </span>
              </div>
            </div>

            {/* Feature Contributions Breakdown */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-indigo-400" />
                  <span>當前特徵貢獻度分項 (Active Contributions)</span>
                </h3>
                <span className="text-[10px] font-mono text-slate-400">
                  {prediction.featureContributions.length} 啟用項目
                </span>
              </div>

              <div className="overflow-y-auto max-h-56 rounded-xl border border-slate-800/80 bg-slate-950/60 divide-y divide-slate-800/40 text-xs font-mono">
                {prediction.featureContributions.map((fc, idx) => (
                  <div key={idx} className="p-2.5 flex items-center justify-between hover:bg-slate-900/40">
                    <div className="flex flex-col">
                      <span className="text-white font-medium">{fc.feature_name}</span>
                      <span className="text-[10px] text-slate-400 font-sans">{fc.description}</span>
                    </div>
                    <div className="text-right">
                      <span className={`font-bold ${
                        fc.contribution >= 0 ? "text-cyan-400" : "text-amber-400"
                      }`}>
                        {fc.contribution > 0 ? "+" : ""}
                        {fc.contribution.toFixed(2)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Experiment Metadata */}
            <div className="pt-2 border-t border-slate-800 text-[11px] font-mono text-slate-400 space-y-1">
              <div className="flex justify-between">
                <span>實驗識別 ID</span>
                <span className="text-cyan-400">{prediction.experimentId}</span>
              </div>
              <div className="flex justify-between">
                <span>推論引擎架構</span>
                <span className="text-slate-300">Option B (TypeScript Deterministic)</span>
              </div>
              <div className="flex justify-between">
                <span>共形校準資料集</span>
                <span className="text-slate-300">Calibration Set (n = 2,640)</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
