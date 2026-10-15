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
        <div className="flex items-center gap-2 text-xs font-mono font-semibold text-teal-800 bg-teal-50 border border-teal-200 px-3 py-1 rounded-full w-fit">
          <Sparkles className="w-4 h-4 text-teal-700" />
          <span>REAL-TIME INFERENCE ENGINE · OPTION B VERIFIED</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-[#17243A] tracking-tight">
          M2 模型互動式預測引擎與共形不確定性推論
        </h1>
        <p className="text-sm sm:text-base text-[#334155] max-w-4xl leading-relaxed">
          直接在前端執行已訓練完成之 M2 (RFE + OLS) 權重矩陣，絕非數值模擬或隨機查找。
          具備與 Python Scikit-learn 嚴格驗證之 <strong className="text-emerald-700 font-mono font-bold">&lt; 2.27 × 10⁻¹³</strong> 浮點全等精度。
          支援 12 項原始預測變數調節，並輸出 95% 名義共形預測區間（q = 302.60）。
        </p>
      </section>

      {/* Numerical Parity & Architectural Decision Alert */}
      <section className="p-5 rounded-2xl border border-emerald-200 bg-emerald-50 text-emerald-950 shadow-xs space-y-2">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-emerald-900 font-bold text-sm sm:text-base">
            <CheckCircle2 className="w-4 h-4 text-emerald-700 flex-shrink-0" />
            <span>架構可行性決策評估 (Option B: Verified TypeScript Pipeline)</span>
          </div>
          <span className="text-xs font-mono font-semibold px-2.5 py-0.5 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-300 w-fit">
            Parity Error: 2.27 × 10⁻¹³
          </span>
        </div>
        <p className="text-xs sm:text-sm text-emerald-900/90 leading-relaxed">
          經實地測試，本專案採用 <strong>Option B（純 TypeScript 確定性推論引擎）</strong>，
          將凍結訓練之 30 個選定係數與截距以 IEEE-754 雙精度浮點數重構。在歷史全部 17,379 筆樣本上進行數值對齊測試，
          推論結果與原生 Python Scikit-learn Pipeline 之絕對誤差最大值僅為 <strong>2.27 × 10⁻¹³</strong>（機器浮點精度極限），
          免除伺服器 Cold Start、龐大 Python Wheel 依賴與 API 逾時風險，提供 &lt; 1ms 的極速無伺服器推論。
        </p>
      </section>

      {/* Preset Scenarios Selector */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-base sm:text-lg font-bold text-[#17243A] flex items-center gap-2">
            <Sliders className="w-4 h-4 text-blue-600" />
            <span>典型情境快速載入 (Preset Archetype Scenarios)</span>
          </h2>
          <span className="text-xs text-[#475569]">點擊切換標準特徵組合</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {PRESET_SCENARIOS.map((preset) => {
            const isSelected = selectedPreset === preset.name;
            return (
              <button
                key={preset.name}
                onClick={() => handlePresetSelect(preset.name)}
                className={`p-3.5 rounded-xl border text-left transition-all flex flex-col justify-between cursor-pointer ${
                  isSelected
                    ? "bg-blue-50 border-2 border-blue-600 text-[#17243A] shadow-xs"
                    : "bg-white border-[#DFE5ED] text-[#475569] hover:text-[#17243A] hover:border-slate-300 hover:bg-slate-50/60"
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-100 text-[#334155] border border-[#CBD5E1] font-semibold">
                      {preset.tag}
                    </span>
                    {isSelected && <CheckCircle2 className="w-4 h-4 text-blue-600" />}
                  </div>
                  <div className="text-xs sm:text-sm font-bold text-[#17243A] mb-1">
                    {preset.name}
                  </div>
                </div>
                <p className="text-[11px] text-[#475569] leading-tight mt-1">
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
          <div className="p-6 rounded-2xl border border-[#DFE5ED] bg-white shadow-xs space-y-6">
            <div className="flex items-center justify-between border-b border-[#DFE5ED] pb-3">
              <h2 className="text-base sm:text-lg font-bold text-[#17243A] flex items-center gap-2">
                <Calendar className="w-4 h-4 text-blue-600" />
                <span>12 項原始輸入特徵調節盤 (Feature Controls)</span>
              </h2>
              <span className="text-xs font-mono font-semibold text-[#475569] bg-slate-100 px-2.5 py-0.5 rounded border border-[#DFE5ED]">12 Predictors</span>
            </div>

            {/* Centralized Calendar Consistency & Validation Warnings */}
            {prediction.validation.warnings.length > 0 && (
              <div className="p-4 rounded-xl border border-amber-200 bg-amber-50 space-y-2 text-xs sm:text-sm text-amber-950">
                <div className="flex items-center gap-2 text-amber-900 font-bold">
                  <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                  <span>輸入驗證與曆法組合提示 ({prediction.validation.warnings.length} 項警示)</span>
                </div>
                <ul className="space-y-1.5 pl-5 list-disc text-amber-900/90">
                  {prediction.validation.warnings.map((w, idx) => (
                    <li key={idx}>
                      <strong className="text-amber-950">{w.title}：</strong>
                      <span className="text-amber-900/80">{w.message}</span>
                    </li>
                  ))}
                </ul>
                <div className="text-[11px] text-amber-900/70 pt-1 border-t border-amber-200/60">
                  * 說明：上述提示為 UCI 領域語意一致性輔助警示，已確立之 M2 迴歸模型仍忠實依據您的特徵數值進行推論，未改動既定模型結構。
                </div>
              </div>
            )}

            {/* 1. Calendar & Time Features */}
            <div className="space-y-4">
              <span className="text-xs font-bold uppercase tracking-wider text-[#475569] block">
                時段與曆法維度 (Calendar &amp; Temporal)
              </span>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {/* Year Indicator */}
                <div className="space-y-1.5">
                  <label className="text-xs sm:text-sm font-semibold text-[#17243A] flex justify-between">
                    <span>年份趨勢 (yr)</span>
                    <span className="font-mono text-blue-700">{inputs.yr === 1 ? "2012 年" : "2011 年"}</span>
                  </label>
                  <select
                    value={inputs.yr}
                    onChange={(e) => handleInputChange("yr", parseInt(e.target.value))}
                    className="w-full bg-white border border-[#CBD5E1] rounded-lg px-3 py-2 text-xs sm:text-sm text-[#17243A] font-medium focus:outline-none focus:border-blue-600 focus:ring-1 focus:ring-blue-600"
                  >
                    <option value={0}>2011 年 (基準年度, yr=0)</option>
                    <option value={1}>2012 年 (成長年度, yr=1)</option>
                  </select>
                  <span className="text-[10px] text-amber-800 block">歷史編碼僅支援 2011-2012，不可外推</span>
                </div>

                {/* Season */}
                <div className="space-y-1.5">
                  <label className="text-xs sm:text-sm font-semibold text-[#17243A] flex justify-between">
                    <span>季節 (season)</span>
                    <span className="font-mono text-blue-700">
                      {["春季", "夏季", "秋季", "冬季"][inputs.season - 1]}
                    </span>
                  </label>
                  <select
                    value={inputs.season}
                    onChange={(e) => handleInputChange("season", parseInt(e.target.value))}
                    className="w-full bg-white border border-[#CBD5E1] rounded-lg px-3 py-2 text-xs sm:text-sm text-[#17243A] font-medium focus:outline-none focus:border-blue-600 focus:ring-1 focus:ring-blue-600"
                  >
                    <option value={1}>1: 春季 (Spring, 基準)</option>
                    <option value={2}>2: 夏季 (Summer)</option>
                    <option value={3}>3: 秋季 (Fall)</option>
                    <option value={4}>4: 冬季 (Winter)</option>
                  </select>
                </div>

                {/* Month */}
                <div className="space-y-1.5">
                  <label className="text-xs sm:text-sm font-semibold text-[#17243A] flex justify-between">
                    <span>月份 (mnth)</span>
                    <span className="font-mono text-blue-700">{inputs.mnth} 月</span>
                  </label>
                  <select
                    value={inputs.mnth}
                    onChange={(e) => handleInputChange("mnth", parseInt(e.target.value))}
                    className="w-full bg-white border border-[#CBD5E1] rounded-lg px-3 py-2 text-xs sm:text-sm text-[#17243A] font-medium focus:outline-none focus:border-blue-600 focus:ring-1 focus:ring-blue-600"
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
                <div className="flex justify-between items-center text-xs sm:text-sm">
                  <span className="text-[#17243A] font-semibold flex items-center gap-1.5">
                    <Clock className="w-4 h-4 text-blue-600" />
                    <span>時段 (hr, 0 到 23 時)</span>
                  </span>
                  <div className="flex items-center gap-2">
                    {inputs.hr === 8 || inputs.hr === 17 || inputs.hr === 18 ? (
                      <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-amber-100 text-amber-900 border border-amber-300 font-bold">
                        ⚡ 尖峰通勤時段
                      </span>
                    ) : null}
                    <span className="font-mono text-sm font-bold text-blue-800 px-2.5 py-0.5 rounded bg-blue-50 border border-blue-200">
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
                  className="w-full accent-blue-600 cursor-pointer"
                />
                <div className="flex justify-between text-[11px] font-mono text-[#64748B]">
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
                  <label className="text-xs sm:text-sm font-semibold text-[#17243A]">星期 (weekday)</label>
                  <select
                    value={inputs.weekday}
                    onChange={(e) => handleInputChange("weekday", parseInt(e.target.value))}
                    className="w-full bg-white border border-[#CBD5E1] rounded-lg px-3 py-2 text-xs sm:text-sm text-[#17243A] font-medium focus:outline-none focus:border-blue-600 focus:ring-1 focus:ring-blue-600"
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
                  <label className="text-xs sm:text-sm font-semibold text-[#17243A]">工作日 (workingday)</label>
                  <select
                    value={inputs.workingday}
                    onChange={(e) => handleInputChange("workingday", parseInt(e.target.value))}
                    className="w-full bg-white border border-[#CBD5E1] rounded-lg px-3 py-2 text-xs sm:text-sm text-[#17243A] font-medium focus:outline-none focus:border-blue-600 focus:ring-1 focus:ring-blue-600"
                  >
                    <option value={1}>工作日 (1: 既非假日亦非週末)</option>
                    <option value={0}>非工作日 (0: 假日或週末)</option>
                  </select>
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs sm:text-sm font-semibold text-[#17243A]">國定假日 (holiday)</label>
                  <select
                    value={inputs.holiday}
                    onChange={(e) => handleInputChange("holiday", parseInt(e.target.value))}
                    className="w-full bg-white border border-[#CBD5E1] rounded-lg px-3 py-2 text-xs sm:text-sm text-[#17243A] font-medium focus:outline-none focus:border-blue-600 focus:ring-1 focus:ring-blue-600"
                  >
                    <option value={0}>平常日 (0: 否)</option>
                    <option value={1}>法定假日 (1: 是)</option>
                  </select>
                </div>
              </div>
            </div>

            {/* 2. Weather & Climate Features */}
            <div className="space-y-4 pt-4 border-t border-[#DFE5ED]">
              <span className="text-xs font-bold uppercase tracking-wider text-[#475569] block">
                天候與微氣候連續因子 (Weather &amp; Micro-climate)
              </span>

              {/* Weather Situation */}
              <div className="space-y-1.5">
                <label className="text-xs sm:text-sm font-semibold text-[#17243A] flex justify-between">
                  <span>天候狀況代碼 (weathersit)</span>
                  <span className="font-mono text-blue-700">
                    {["晴朗/少雲", "薄霧/多雲", "小雨/小雪", "大豪雨/暴風雪"][inputs.weathersit - 1]}
                  </span>
                </label>
                <select
                  value={inputs.weathersit}
                  onChange={(e) => handleInputChange("weathersit", parseInt(e.target.value))}
                  className="w-full bg-white border border-[#CBD5E1] rounded-lg px-3 py-2 text-xs sm:text-sm text-[#17243A] font-medium focus:outline-none focus:border-blue-600 focus:ring-1 focus:ring-blue-600"
                >
                  <option value={1}>1: 晴朗 / 少雲 / 多雲 (Clear/Few clouds, 基準)</option>
                  <option value={2}>2: 薄霧 / 陰天 (Mist + Cloudy)</option>
                  <option value={3}>3: 小雨 / 小雪 / 雷雨 (Light Snow/Rain, ★ RFE -56.41)</option>
                  <option value={4}>4: 暴雨 / 暴風雪 / 冰雹 (Heavy Rain/Snow, ★ RFE -52.82)</option>
                </select>
              </div>

              {/* Temperature Slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-xs sm:text-sm">
                  <span className="text-[#17243A] font-semibold flex items-center gap-1.5">
                    <Thermometer className="w-4 h-4 text-blue-600" />
                    <span>正規化氣溫 (temp = t / 41)</span>
                  </span>
                  <div className="flex items-center gap-2 font-mono">
                    <span className="text-[#475569] text-xs">
                      實際換算: 約 <strong>{Math.round(inputs.temp * 41)}°C</strong>
                    </span>
                    <span className="text-blue-800 font-bold bg-blue-50 border border-blue-200 px-2 py-0.5 rounded text-xs">
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
                  className="w-full accent-blue-600 cursor-pointer"
                />
              </div>

              {/* Apparent Temperature Slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-xs sm:text-sm">
                  <span className="text-[#17243A] font-semibold flex items-center gap-1.5">
                    <SunMedium className="w-4 h-4 text-amber-600" />
                    <span>正規化體感溫度 (atemp = at / 50)</span>
                  </span>
                  <div className="flex items-center gap-2 font-mono">
                    <span className="text-[#475569] text-xs">
                      實際換算: 約 <strong>{Math.round(inputs.atemp * 50)}°C</strong>
                    </span>
                    <span className="text-amber-900 font-bold bg-amber-50 border border-amber-200 px-2 py-0.5 rounded text-xs">
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
                  className="w-full accent-amber-600 cursor-pointer"
                />
              </div>

              {/* Humidity Slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-xs sm:text-sm">
                  <span className="text-[#17243A] font-semibold flex items-center gap-1.5">
                    <Droplets className="w-4 h-4 text-teal-600" />
                    <span>正規化相對濕度 (hum / 100)</span>
                  </span>
                  <div className="flex items-center gap-2 font-mono">
                    <span className="text-[#475569] text-xs">
                      實際濕度: 約 <strong>{Math.round(inputs.hum * 100)}%</strong>
                    </span>
                    <span className="text-teal-900 font-bold bg-teal-50 border border-teal-200 px-2 py-0.5 rounded text-xs">
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
                  className="w-full accent-teal-600 cursor-pointer"
                />
              </div>

              {/* Windspeed Slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-xs sm:text-sm">
                  <span className="text-[#17243A] font-semibold flex items-center gap-1.5">
                    <Wind className="w-4 h-4 text-[#475569]" />
                    <span>正規化風速 (windspeed / 67)</span>
                  </span>
                  <div className="flex items-center gap-2 font-mono">
                    <span className="text-[#475569] text-xs">
                      實際風速: 約 <strong>{Math.round(inputs.windspeed * 67)} km/h</strong>
                    </span>
                    <span className="text-[#17243A] font-bold bg-slate-100 border border-[#CBD5E1] px-2 py-0.5 rounded text-xs">
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
                  className="w-full accent-slate-500 cursor-pointer"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Prediction Output Card (5 Cols) */}
        <div className="lg:col-span-5 space-y-6">
          <div className="p-6 rounded-2xl border-2 border-blue-500 bg-white shadow-md space-y-6 sticky top-20">
            <div className="flex items-center justify-between border-b border-[#DFE5ED] pb-3">
              <span className="text-xs font-mono text-blue-700 uppercase tracking-wider font-bold">
                推論輸出結果 (Prediction Output)
              </span>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-blue-50 text-blue-800 border border-blue-200 font-bold">
                {M2_SPEC.modelId}
              </span>
            </div>

            {/* Headline Display Point Prediction */}
            <div className="space-y-1">
              <span className="text-xs font-bold uppercase tracking-wider text-[#475569] block">
                非負呈現推估值 (Nonnegative Display Count)
              </span>
              <div className="text-4xl sm:text-5xl font-black font-mono text-[#17243A] tracking-tight flex items-baseline gap-2">
                <span>{Math.round(prediction.clippedPointPrediction)}</span>
                <span className="text-sm font-sans font-medium text-[#475569]">輛 / 小時</span>
              </div>
              <div className="text-xs font-mono text-[#475569] pt-1">
                模型原生連續點預測：
                <strong className={prediction.rawPointPrediction < 0 ? "text-amber-800" : "text-blue-700"}>
                  {prediction.rawPointPrediction.toFixed(4)}
                </strong>
                {prediction.rawPointPrediction < 0 && (
                  <span className="ml-1 text-amber-800 text-[11px] block sm:inline">
                    (截距小於零導致原生未截斷為負，已進行非負呈現轉換)
                  </span>
                )}
              </div>
            </div>

            {/* 95% Conformal Prediction Interval */}
            <div className="p-4 rounded-xl border border-blue-200 bg-blue-50/70 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-blue-900 flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4 text-emerald-700" />
                  <span>95% 名義共形預測區間 (95% Nominal PI)</span>
                </span>
                <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-blue-100 text-blue-800 border border-blue-200 font-bold">
                  q = 302.60
                </span>
              </div>

              {/* Clipped Interval Display */}
              <div className="text-2xl font-black font-mono text-blue-950">
                [ {prediction.piLowerClipped.toFixed(1)} , {prediction.piUpperClipped.toFixed(1)} ]
              </div>
              <span className="text-xs text-blue-900/80 block">
                非負截斷區間：[max(0, ŷ - 302.60), ŷ + 302.60]
              </span>

              {/* Raw Interval */}
              <div className="pt-2 border-t border-blue-200 text-xs font-mono text-[#475569] flex justify-between">
                <span>原始對稱區間 [ŷ - q, ŷ + q]</span>
                <span className="text-[#17243A] font-semibold">
                  [{prediction.piLowerRaw.toFixed(1)}, {prediction.piUpperRaw.toFixed(1)}]
                </span>
              </div>
            </div>

            {/* Feature Contributions Breakdown */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold uppercase tracking-wider text-[#17243A] flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-blue-600" />
                  <span>當前特徵貢獻度分項 (Active Contributions)</span>
                </h3>
                <span className="text-xs font-mono text-[#475569]">
                  {prediction.featureContributions.length} 啟用項目
                </span>
              </div>

              <div className="overflow-y-auto max-h-56 rounded-xl border border-[#DFE5ED] bg-[#F8FAFC] divide-y divide-[#DFE5ED] text-xs font-mono">
                {prediction.featureContributions.map((fc, idx) => (
                  <div key={idx} className="p-2.5 flex items-center justify-between hover:bg-slate-100/60">
                    <div className="flex flex-col">
                      <span className="text-[#17243A] font-semibold">{fc.feature_name}</span>
                      <span className="text-[11px] text-[#475569] font-sans">{fc.description}</span>
                    </div>
                    <div className="text-right">
                      <span className={`font-bold ${
                        fc.contribution >= 0 ? "text-blue-700" : "text-amber-800"
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
            <div className="pt-2 border-t border-[#DFE5ED] text-xs font-mono text-[#64748B] space-y-1">
              <div className="flex justify-between">
                <span>實驗識別 ID</span>
                <span className="text-blue-700 font-semibold">{prediction.experimentId}</span>
              </div>
              <div className="flex justify-between">
                <span>推論引擎架構</span>
                <span className="text-[#334155]">Option B (TypeScript Deterministic)</span>
              </div>
              <div className="flex justify-between">
                <span>共形校準資料集</span>
                <span className="text-[#334155]">Calibration Set (n = 2,640)</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
