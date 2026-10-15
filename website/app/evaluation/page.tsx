"use client";

import Image from "next/image";
import { useState, useMemo } from "react";
import { 
  finalTestMetrics, 
  predictionIntervalMetrics, 
  hourlyCoverageDiagnostic,
  testTimelineAll,
  testScatterSamples,
  residualDistributionBins,
  HourlyCoverageRow,
  TimelinePoint,
  ScatterSample,
  ResidualBin
} from "@/lib/data";
import { 
  LineChart, 
  Target, 
  ShieldAlert, 
  Activity, 
  Clock, 
  Calendar
} from "lucide-react";

export default function FinalEvaluationPage() {
  const [hoveredScatter, setHoveredScatter] = useState<ScatterSample | null>(null);
  const [hoveredTimeline, setHoveredTimeline] = useState<TimelinePoint | null>(null);
  const [useClippedInterval, setUseClippedInterval] = useState(true);
  const [activeTab, setActiveTab] = useState<"INTERACTIVE" | "STATIC">("INTERACTIVE");

  const allTimeline = testTimelineAll as TimelinePoint[];
  const scatterData = testScatterSamples as ScatterSample[];
  const hourlyData = hourlyCoverageDiagnostic as HourlyCoverageRow[];
  const residualBins = residualDistributionBins as ResidualBin[];

  // Presets and date range states for timeline exploration
  const [selectedPreset, setSelectedPreset] = useState<string>("W1");
  const [startDate, setStartDate] = useState<string>("2012-09-14");
  const [endDate, setEndDate] = useState<string>("2012-09-20");

  const timelinePresets = [
    { id: "W1", label: "首週 7 天 (Sep 14–20)", start: "2012-09-14", end: "2012-09-20", desc: "測試期初秋常態通勤" },
    { id: "W2", label: "次週 7 天 (Sep 21–27)", start: "2012-09-21", end: "2012-09-27", desc: "初秋連續晴朗週" },
    { id: "OCT1", label: "10 月首週 (Oct 01–07)", start: "2012-10-01", end: "2012-10-07", desc: "秋季高運量時段" },
    { id: "SANDY", label: "桑迪颶風期 (Oct 28–Nov 01)", start: "2012-10-28", end: "2012-11-01", desc: "強烈颶風極端天氣衝擊" },
    { id: "THANKSGIVING", label: "感恩節週 (Nov 19–25)", start: "2012-11-19", end: "2012-11-25", desc: "節慶休假模式與通勤落差" },
    { id: "CHRISTMAS", label: "聖誕冬期 (Dec 22–28)", start: "2012-12-22", end: "2012-12-28", desc: "冬季嚴寒氣溫驟降" },
    { id: "ALL", label: "完整測試期 (Sep 14–Dec 31)", start: "2012-09-14", end: "2012-12-31", desc: "全部 2,576 筆觀測時序全貌" },
  ];

  const handleSelectPreset = (presetId: string) => {
    setSelectedPreset(presetId);
    const p = timelinePresets.find((item) => item.id === presetId);
    if (p) {
      setStartDate(p.start);
      setEndDate(p.end);
    }
  };

  const handleDateChange = (type: "start" | "end", value: string) => {
    setSelectedPreset("CUSTOM");
    if (type === "start") setStartDate(value);
    else setEndDate(value);
  };

  const timelineData = useMemo(() => {
    const filtered = allTimeline.filter((pt) => {
      const d = pt.dteday || pt.timestamp.slice(0, 10);
      return d >= startDate && d <= endDate;
    });
    return filtered.length > 0 ? filtered : allTimeline.slice(0, 168);
  }, [allTimeline, startDate, endDate]);

  const sliceStats = useMemo(() => {
    const count = timelineData.length;
    if (count === 0) return { count: 0, mae: 0, rmse: 0, coveragePct: 0, maxY: 800 };
    let sumAbsErr = 0;
    let sumSqErr = 0;
    let coveredCount = 0;
    let maxVal = 0;

    timelineData.forEach((pt) => {
      const err = pt.actual_cnt - pt.predicted_cnt;
      sumAbsErr += Math.abs(err);
      sumSqErr += err * err;
      if (useClippedInterval) {
        if (pt.actual_cnt >= pt.pi_lower_clipped && pt.actual_cnt <= pt.pi_upper_clipped) {
          coveredCount += 1;
        }
      } else {
        if (pt.actual_cnt >= pt.pi_lower_raw && pt.actual_cnt <= pt.pi_upper_raw) {
          coveredCount += 1;
        }
      }
      const upper = useClippedInterval ? pt.pi_upper_clipped : pt.pi_upper_raw;
      maxVal = Math.max(maxVal, pt.actual_cnt, upper);
    });

    return {
      count,
      mae: sumAbsErr / count,
      rmse: Math.sqrt(sumSqErr / count),
      coveragePct: (coveredCount / count) * 100,
      maxY: Math.max(600, Math.ceil((maxVal + 60) / 100) * 100),
    };
  }, [timelineData, useClippedInterval]);

  // Max values for scatter SVG scaling
  const maxActual = 980;
  const scatterSvgSize = 500;

  // Max values for timeline SVG scaling
  const maxTimelineY = sliceStats.maxY;
  const timelineSvgW = Math.max(800, timelineData.length * (timelineData.length > 300 ? 3 : 5));
  const timelineSvgH = 320;

  // Hourly coverage chart dimensions
  const hourlySvgW = 800;
  const hourlySvgH = 280;

  return (
    <div className="space-y-12">
      {/* Header */}
      <section className="space-y-3">
        <div className="flex items-center gap-2 text-xs font-mono font-semibold text-emerald-800 bg-emerald-50 border border-emerald-200 px-3 py-1 rounded-full w-fit">
          <LineChart className="w-4 h-4 text-emerald-700" />
          <span>FINAL TEST EVALUATION &amp; CONFORMAL UNCERTAINTY</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-[#17243A] tracking-tight">
          最終測試集評估、分裂共形預測區間與時段覆蓋率分析
        </h1>
        <p className="text-sm sm:text-base text-[#334155] max-w-4xl leading-relaxed">
          以鎖定之 Phase 3 測試集（2012-09-14 至 2012-12-31，共 2,576 筆真實觀測）驗證 M2 模型泛化能力。
          本頁面全面整合實際值 vs 預測值散佈圖、歷史測試期預測區間時序歷程探索器、殘差分佈診斷與 0–100% 全幅時段實證覆蓋率。
        </p>
      </section>

      {/* Top Test Metric Cards */}
      <section className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="p-4 rounded-xl border border-[#DFE5ED] bg-white shadow-xs">
          <span className="text-xs sm:text-sm font-bold uppercase tracking-wider text-[#334155]">Test MAE</span>
          <div className="text-xl sm:text-2xl font-extrabold font-mono text-[#17243A] mt-1">
            {finalTestMetrics.test_mae.toFixed(4)}
          </div>
          <span className="text-xs text-[#475569] font-medium">輛 / 小時</span>
        </div>

        <div className="p-4 rounded-xl border border-[#DFE5ED] bg-white shadow-xs">
          <span className="text-xs sm:text-sm font-bold uppercase tracking-wider text-[#334155]">Test RMSE</span>
          <div className="text-xl sm:text-2xl font-extrabold font-mono text-[#17243A] mt-1">
            {finalTestMetrics.test_rmse.toFixed(4)}
          </div>
          <span className="text-xs text-[#475569] font-medium">輛 / 小時</span>
        </div>

        <div className="p-4 rounded-xl border border-[#DFE5ED] bg-white shadow-xs">
          <span className="text-xs sm:text-sm font-bold uppercase tracking-wider text-[#334155]">Test R²</span>
          <div className="text-xl sm:text-2xl font-extrabold font-mono text-blue-700 mt-1">
            {finalTestMetrics.test_r2.toFixed(4)}
          </div>
          <span className="text-xs text-[#475569] font-medium">解釋變異量</span>
        </div>

        <div className="p-4 rounded-xl border border-[#DFE5ED] bg-white shadow-xs">
          <span className="text-xs sm:text-sm font-bold uppercase tracking-wider text-[#334155]">Nominal Coverage</span>
          <div className="text-xl sm:text-2xl font-extrabold font-mono text-[#17243A] mt-1">
            {(predictionIntervalMetrics.nominal_coverage * 100).toFixed(0)}%
          </div>
          <span className="text-xs text-[#475569] font-medium">1 - α 目標</span>
        </div>

        <div className="p-4 rounded-xl border border-[#DFE5ED] bg-white shadow-xs">
          <span className="text-xs sm:text-sm font-bold uppercase tracking-wider text-[#334155]">Empirical Coverage</span>
          <div className="text-xl sm:text-2xl font-extrabold font-mono text-emerald-800 mt-1">
            {(predictionIntervalMetrics.empirical_coverage_clipped * 100).toFixed(2)}%
          </div>
          <span className="text-xs text-[#475569] font-medium">2,576 筆觀測</span>
        </div>

        <div className="p-4 rounded-xl border border-[#DFE5ED] bg-white shadow-xs">
          <span className="text-xs sm:text-sm font-bold uppercase tracking-wider text-[#334155]">Conformal Quantile (q)</span>
          <div className="text-xl sm:text-2xl font-extrabold font-mono text-teal-800 mt-1">
            {predictionIntervalMetrics.conformal_quantile.toFixed(2)}
          </div>
          <span className="text-xs text-[#475569] font-medium">半區間長度</span>
        </div>
      </section>

      {/* Interactive / Static Tab Switcher */}
      <div className="flex items-center justify-between border-b border-[#DFE5ED] pb-3 print-hide">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setActiveTab("INTERACTIVE")}
            className={`px-4 py-2 rounded-lg text-xs sm:text-sm font-semibold transition-all cursor-pointer ${
              activeTab === "INTERACTIVE"
                ? "bg-blue-600 text-white shadow-xs"
                : "bg-slate-100 text-[#475569] hover:text-[#17243A]"
            }`}
          >
            互動式圖表儀表板 (Interactive Exploration)
          </button>
          <button
            onClick={() => setActiveTab("STATIC")}
            className={`px-4 py-2 rounded-lg text-xs sm:text-sm font-semibold transition-all cursor-pointer ${
              activeTab === "STATIC"
                ? "bg-blue-600 text-white shadow-xs"
                : "bg-slate-100 text-[#475569] hover:text-[#17243A]"
            }`}
          >
            靜態論文級圖表 (Phase 3 High-Res Figures)
          </button>
        </div>
        <span className="text-xs text-[#475569] hidden sm:block font-mono">
          所有數據皆直接源自 test_predictions.csv
        </span>
      </div>

      {activeTab === "INTERACTIVE" ? (
        <div className="space-y-12">
          {/* 1. Actual vs Predicted Interactive Scatter Plot */}
          <section className="p-6 rounded-2xl border border-[#DFE5ED] bg-white shadow-xs space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
              <div>
                <h3 className="text-base font-bold text-[#17243A] flex items-center gap-2">
                  <Target className="w-5 h-5 text-blue-600" />
                  <span>實際租借量 vs 預測租借量散佈圖 (Actual vs Predicted)</span>
                </h3>
                <p className="text-xs sm:text-sm text-[#475569]">
                  X 軸：實際租借量 (Actual Rental Count) · Y 軸：M2 預測租借量 (Predicted Count) · 綠線為 y = x 完美預測基準線
                </p>
              </div>
              {hoveredScatter && (
                <div className="px-3 py-1.5 rounded-lg bg-slate-100 text-xs font-mono text-[#17243A] border border-[#CBD5E1] shadow-xs flex items-center gap-3">
                  <span>時間: {hoveredScatter.timestamp}</span>
                  <span>實際: {hoveredScatter.actual}</span>
                  <span>預測: {hoveredScatter.predicted.toFixed(1)}</span>
                  <span>殘差: {hoveredScatter.residual.toFixed(1)}</span>
                </div>
              )}
            </div>

            <div className="relative w-full aspect-square max-w-xl mx-auto border border-[#DFE5ED] rounded-xl bg-[#F8FAFC] p-4">
              <svg
                viewBox={`0 0 ${scatterSvgSize} ${scatterSvgSize}`}
                className="w-full h-full overflow-visible"
              >
                {/* Grid Lines */}
                {[0, 200, 400, 600, 800].map((val) => {
                  const pos = (val / maxActual) * (scatterSvgSize - 50);
                  const yPos = scatterSvgSize - 30 - pos;
                  const xPos = 40 + pos;
                  return (
                    <g key={val}>
                      {/* Horizontal Grid */}
                      <line
                        x1={40}
                        y1={yPos}
                        x2={scatterSvgSize}
                        y2={yPos}
                        stroke="#E2E8F0"
                        strokeDasharray="3 3"
                        strokeWidth="1"
                      />
                      <text
                        x={35}
                        y={yPos + 4}
                        fill="#475569"
                        fontSize="10"
                        textAnchor="end"
                        fontFamily="monospace"
                      >
                        {val}
                      </text>

                      {/* Vertical Grid */}
                      <line
                        x1={xPos}
                        y1={0}
                        x2={xPos}
                        y2={scatterSvgSize - 30}
                        stroke="#E2E8F0"
                        strokeDasharray="3 3"
                        strokeWidth="1"
                      />
                      <text
                        x={xPos}
                        y={scatterSvgSize - 15}
                        fill="#475569"
                        fontSize="10"
                        textAnchor="middle"
                        fontFamily="monospace"
                      >
                        {val}
                      </text>
                    </g>
                  );
                })}

                {/* y = x Ideal Diagonal Reference Line */}
                <line
                  x1={40}
                  y1={scatterSvgSize - 30}
                  x2={40 + ((scatterSvgSize - 50) * 800) / maxActual}
                  y2={scatterSvgSize - 30 - ((scatterSvgSize - 50) * 800) / maxActual}
                  stroke="#059669"
                  strokeWidth="2"
                  strokeDasharray="5 5"
                />

                {/* y = 0 Horizontal baseline */}
                <line
                  x1={40}
                  y1={scatterSvgSize - 30}
                  x2={scatterSvgSize}
                  y2={scatterSvgSize - 30}
                  stroke="#94A3B8"
                  strokeWidth="1.5"
                />

                {/* Scatter Points */}
                {scatterData.map((pt, idx) => {
                  const cx = 40 + (pt.actual / maxActual) * (scatterSvgSize - 50);
                  const cy = scatterSvgSize - 30 - (pt.predicted / maxActual) * (scatterSvgSize - 50);
                  const isHovered = hoveredScatter?.timestamp === pt.timestamp;

                  return (
                    <circle
                      key={idx}
                      cx={cx}
                      cy={cy}
                      r={isHovered ? 5 : 2.5}
                      fill={pt.predicted < 0 ? "#D97706" : pt.covered ? "#0284C7" : "#DC2626"}
                      opacity={isHovered ? 1 : 0.75}
                      className="cursor-pointer transition-all hover:scale-150"
                      onMouseEnter={() => setHoveredScatter(pt)}
                    />
                  );
                })}
              </svg>
            </div>

            <div className="flex flex-wrap items-center justify-between text-xs text-[#475569] pt-2 border-t border-[#DFE5ED]">
              <div className="flex items-center gap-4">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-[#0284C7]" />
                  <span>區間內點位 (Covered)</span>
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-[#DC2626]" />
                  <span>區間外逸脫點 (Violated, 5.16%)</span>
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-[#D97706]" />
                  <span>未截斷負值預測 (Negative Predictions)</span>
                </span>
              </div>
              <span className="text-[11px] font-mono text-[#64748B]">
                點位呈現 815 筆跨時段代表性測試集抽樣
              </span>
            </div>
          </section>

          {/* 2. Prediction Interval Timeline with Selectable Date Ranges */}
          <section className="p-6 rounded-2xl border border-[#DFE5ED] bg-white shadow-xs space-y-5">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
              <div>
                <h3 className="text-base font-bold text-[#17243A] flex items-center gap-2">
                  <Clock className="w-5 h-5 text-blue-600" />
                  <span>測試集時序預測區間帶與歷程探索 (Test Prediction Interval Explorer)</span>
                </h3>
                <p className="text-xs sm:text-sm text-[#475569] mt-0.5">
                  支援自由切換 2012-09-14 至 2012-12-31 測試集全貌（共 2,576 筆觀測），或選取歷史典型情境探索 M2 模型與 95% 共形區間之動態表現。
                </p>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => setUseClippedInterval(!useClippedInterval)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold border transition-colors cursor-pointer ${
                    useClippedInterval
                      ? "bg-blue-50 text-blue-800 border-blue-300 shadow-2xs"
                      : "bg-slate-100 text-[#475569] border-[#DFE5ED]"
                  }`}
                >
                  {useClippedInterval ? "顯示：非負截斷區間 [max(0, y-q), y+q]" : "顯示：原始對稱區間 [y-q, y+q]"}
                </button>
              </div>
            </div>

            {/* Range Presets and Date Controls */}
            <div className="p-4 rounded-xl border border-[#DFE5ED] bg-[#F8FAFC] space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <span className="text-xs font-bold text-[#17243A] flex items-center gap-1.5">
                  <Calendar className="w-4 h-4 text-blue-600" />
                  <span>典型情境預設區間 (Scenario Presets)：</span>
                </span>
                <span className="text-xs font-mono text-[#64748B]">
                  測試集完整範圍：2012-09-14 至 2012-12-31
                </span>
              </div>

              {/* Preset buttons */}
              <div className="flex flex-wrap gap-2">
                {timelinePresets.map((preset) => (
                  <button
                    key={preset.id}
                    onClick={() => handleSelectPreset(preset.id)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                      selectedPreset === preset.id
                        ? "bg-blue-600 text-white shadow-xs"
                        : "bg-white text-[#334155] hover:bg-slate-50 border border-[#DFE5ED]"
                    }`}
                    title={preset.desc}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>

              {/* Custom Date Pickers */}
              <div className="flex flex-wrap items-center gap-3 pt-2 border-t border-[#DFE5ED] text-xs text-[#334155]">
                <span className="font-semibold text-[#475569]">自訂起訖日期：</span>
                <div className="flex items-center gap-2">
                  <label className="text-[#475569]">開始：</label>
                  <input
                    type="date"
                    min="2012-09-14"
                    max="2012-12-31"
                    value={startDate}
                    onChange={(e) => handleDateChange("start", e.target.value)}
                    className="px-2.5 py-1 rounded-md bg-white border border-[#CBD5E1] text-[#17243A] font-mono text-xs focus:outline-none focus:border-blue-600"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-[#475569]">結束：</label>
                  <input
                    type="date"
                    min="2012-09-14"
                    max="2012-12-31"
                    value={endDate}
                    onChange={(e) => handleDateChange("end", e.target.value)}
                    className="px-2.5 py-1 rounded-md bg-white border border-[#CBD5E1] text-[#17243A] font-mono text-xs focus:outline-none focus:border-blue-600"
                  />
                </div>
                {startDate > endDate && (
                  <span className="text-red-700 text-xs font-semibold">開始日期不能晚於結束日期</span>
                )}
              </div>
            </div>

            {/* Slice Diagnostics Metric Summary */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="p-3 rounded-lg border border-[#DFE5ED] bg-white shadow-2xs">
                <span className="text-[11px] text-[#475569] font-medium block">選取範圍樣本數</span>
                <span className="text-xl font-bold font-mono text-[#17243A]">
                  {sliceStats.count.toLocaleString()}
                </span>
                <span className="text-[11px] text-[#64748B] ml-1">小時觀測</span>
              </div>
              <div className="p-3 rounded-lg border border-[#DFE5ED] bg-white shadow-2xs">
                <span className="text-[11px] text-[#475569] font-medium block">範圍切片實證覆蓋率</span>
                <span className={`text-xl font-bold font-mono ${
                  sliceStats.coveragePct >= 94.0 ? "text-emerald-700" : "text-amber-800"
                }`}>
                  {sliceStats.coveragePct.toFixed(2)}%
                </span>
                <span className="text-[11px] text-[#64748B] ml-1">名義 95%</span>
              </div>
              <div className="p-3 rounded-lg border border-[#DFE5ED] bg-white shadow-2xs">
                <span className="text-[11px] text-[#475569] font-medium block">範圍切片 MAE</span>
                <span className="text-xl font-bold font-mono text-blue-700">
                  {sliceStats.mae.toFixed(2)}
                </span>
                <span className="text-[11px] text-[#64748B] ml-1">輛 / 小時</span>
              </div>
              <div className="p-3 rounded-lg border border-[#DFE5ED] bg-white shadow-2xs">
                <span className="text-[11px] text-[#475569] font-medium block">範圍切片 RMSE</span>
                <span className="text-xl font-bold font-mono text-teal-800">
                  {sliceStats.rmse.toFixed(2)}
                </span>
                <span className="text-[11px] text-[#64748B] ml-1">輛 / 小時</span>
              </div>
            </div>

            {/* Hover tooltip readout */}
            {hoveredTimeline && (
              <div className="p-3 rounded-xl bg-white border border-[#DFE5ED] shadow-md text-xs font-mono grid grid-cols-2 sm:grid-cols-5 gap-2 text-[#334155]">
                <div>
                  <span className="text-[#64748B] block text-[10px]">時戳 Timestamp</span>
                  <span className="text-[#17243A] font-bold">{hoveredTimeline.timestamp}</span>
                </div>
                <div>
                  <span className="text-[#64748B] block text-[10px]">實際值 Actual</span>
                  <span className="text-[#17243A] font-bold">{hoveredTimeline.actual_cnt}</span>
                </div>
                <div>
                  <span className="text-[#64748B] block text-[10px]">預測值 Predicted</span>
                  <span className="text-blue-700 font-bold">{hoveredTimeline.predicted_cnt.toFixed(1)}</span>
                </div>
                <div>
                  <span className="text-[#64748B] block text-[10px]">95% 預測區間</span>
                  <span className="text-[#17243A] font-semibold">
                    [
                    {useClippedInterval
                      ? hoveredTimeline.pi_lower_clipped.toFixed(0)
                      : hoveredTimeline.pi_lower_raw.toFixed(0)}
                    , {hoveredTimeline.pi_upper_clipped.toFixed(0)}]
                  </span>
                </div>
                <div>
                  <span className="text-[#64748B] block text-[10px]">覆蓋狀態 Covered</span>
                  <span className={hoveredTimeline.covered ? "text-emerald-700 font-bold" : "text-red-700 font-bold"}>
                    {hoveredTimeline.covered ? "✓ 區間內" : "✕ 超出區間"}
                  </span>
                </div>
              </div>
            )}

            {/* Responsive Timeline SVG with horizontal scroll for wide ranges */}
            <div className="w-full overflow-x-auto rounded-xl border border-[#DFE5ED] bg-[#F8FAFC] p-2">
              <svg
                viewBox={`0 0 ${timelineSvgW} ${timelineSvgH}`}
                style={{ minWidth: timelineData.length > 168 ? `${Math.min(2400, timelineSvgW)}px` : "700px" }}
                className="w-full h-auto"
              >
                {/* Horizontal Grid */}
                {[0, 200, 400, 600, 800].map((val) => {
                  if (val > maxTimelineY) return null;
                  const y = timelineSvgH - 30 - (val / maxTimelineY) * (timelineSvgH - 50);
                  return (
                    <g key={val}>
                      <line
                        x1={45}
                        y1={y}
                        x2={timelineSvgW - 10}
                        y2={y}
                        stroke="#E2E8F0"
                        strokeDasharray="3 3"
                      />
                      <text
                        x={40}
                        y={y + 4}
                        fill="#475569"
                        fontSize="10"
                        textAnchor="end"
                        fontFamily="monospace"
                      >
                        {val}
                      </text>
                    </g>
                  );
                })}

                {/* Day Boundary Separators with Date Markers */}
                {timelineData.map((pt, hrIndex) => {
                  if (hrIndex > 0 && pt.timestamp.endsWith("00:00:00")) {
                    const x = 45 + (hrIndex / (timelineData.length - 1 || 1)) * (timelineSvgW - 55);
                    return (
                      <g key={`day-${hrIndex}`}>
                        <line
                          x1={x}
                          y1={10}
                          x2={x}
                          y2={timelineSvgH - 30}
                          stroke="#CBD5E1"
                          strokeDasharray="3 3"
                        />
                        <text
                          x={x + 3}
                          y={20}
                          fill="#475569"
                          fontSize="9"
                          fontFamily="monospace"
                          fontWeight="bold"
                        >
                          {pt.timestamp.slice(5, 10)}
                        </text>
                      </g>
                    );
                  }
                  return null;
                })}

                {/* Shaded Prediction Interval Band Polygon */}
                {(() => {
                  const pointsUpper: string[] = [];
                  const pointsLower: string[] = [];

                  timelineData.forEach((pt, i) => {
                    const x = 45 + (i / (timelineData.length - 1 || 1)) * (timelineSvgW - 55);
                    const upper = useClippedInterval ? pt.pi_upper_clipped : pt.pi_upper_raw;
                    const lower = useClippedInterval ? pt.pi_lower_clipped : pt.pi_lower_raw;
                    const yUpper = timelineSvgH - 30 - (Math.max(0, upper) / maxTimelineY) * (timelineSvgH - 50);
                    const yLower = timelineSvgH - 30 - (Math.max(-100, lower) / maxTimelineY) * (timelineSvgH - 50);

                    pointsUpper.push(`${x},${yUpper}`);
                    pointsLower.unshift(`${x},${yLower}`);
                  });

                  return (
                    <polygon
                      points={`${pointsUpper.join(" ")} ${pointsLower.join(" ")}`}
                      fill="rgba(37, 99, 235, 0.12)"
                      stroke="rgba(37, 99, 235, 0.45)"
                      strokeWidth="1.5"
                    />
                  );
                })()}

                {/* Predicted Polyline */}
                {(() => {
                  const pts = timelineData.map((pt, i) => {
                    const x = 45 + (i / (timelineData.length - 1 || 1)) * (timelineSvgW - 55);
                    const y = timelineSvgH - 30 - (pt.predicted_cnt / maxTimelineY) * (timelineSvgH - 50);
                    return `${x},${y}`;
                  }).join(" ");
                  return (
                    <polyline
                      points={pts}
                      fill="none"
                      stroke="#2563EB"
                      strokeWidth="1.8"
                      strokeDasharray="4 2"
                    />
                  );
                })()}

                {/* Actual Polyline & Data Points */}
                {(() => {
                  const pts = timelineData.map((pt, i) => {
                    const x = 45 + (i / (timelineData.length - 1 || 1)) * (timelineSvgW - 55);
                    const y = timelineSvgH - 30 - (pt.actual_cnt / maxTimelineY) * (timelineSvgH - 50);
                    return `${x},${y}`;
                  }).join(" ");
                  return (
                    <polyline
                      points={pts}
                      fill="none"
                      stroke="#0F172A"
                      strokeWidth="2.5"
                    />
                  );
                })()}

                {/* Hover trigger columns */}
                {timelineData.map((pt, i) => {
                  const x = 45 + (i / (timelineData.length - 1 || 1)) * (timelineSvgW - 55);
                  const isHovered = hoveredTimeline?.timestamp === pt.timestamp;
                  return (
                    <rect
                      key={i}
                      x={x - Math.max(2, (timelineSvgW - 55) / (timelineData.length * 2))}
                      y={10}
                      width={Math.max(4, (timelineSvgW - 55) / timelineData.length)}
                      height={timelineSvgH - 40}
                      fill={isHovered ? "rgba(37, 99, 235, 0.15)" : "transparent"}
                      className="cursor-crosshair"
                      onMouseEnter={() => setHoveredTimeline(pt)}
                    />
                  );
                })}
              </svg>
            </div>

            <div className="flex flex-wrap items-center justify-between text-xs text-[#475569] pt-2 border-t border-[#DFE5ED]">
              <div className="flex items-center gap-4">
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-1 bg-[#0F172A] rounded-full" />
                  <span className="text-[#17243A] font-semibold">實際借車量 (Actual Count)</span>
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-1 bg-blue-600 rounded-full border-t border-dashed border-white" />
                  <span className="text-blue-800 font-semibold">M2 點預測 (Predicted Count)</span>
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-3 bg-blue-50 border border-blue-400 rounded" />
                  <span className="text-[#334155] font-semibold">95% 名義共形預測區間 (Conformal PI Band)</span>
                </span>
              </div>
              <span className="text-[11px] font-mono text-[#64748B]">
                可水平滾動查看更多時序，滑鼠懸停於圖表上可即時檢視該小時數值
              </span>
            </div>
          </section>

          {/* 3. Hourly Empirical Coverage (0–100% Full Scale) */}
          <section className="p-6 rounded-2xl border border-[#DFE5ED] bg-white shadow-xs space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
              <div>
                <h3 className="text-base font-bold text-[#17243A] flex items-center gap-2">
                  <Activity className="w-5 h-5 text-amber-600" />
                  <span>24 小時時段實證覆蓋率全幅診斷 (Hourly Empirical Coverage, 0–100%)</span>
                </h3>
                <p className="text-xs sm:text-sm text-[#475569]">
                  全幅 Y 軸（0% 至 100%）清楚展示各小時之條件覆蓋率分佈，突顯早晚通勤尖峰時段覆蓋率下滑現象。
                </p>
              </div>
              <div className="px-3 py-1 rounded-md bg-amber-50 text-amber-900 border border-amber-200 text-xs font-mono font-semibold">
                95% 為整體邊際覆蓋率目標 (Marginal Coverage)
              </div>
            </div>

            {/* Hourly Coverage SVG Bar Chart */}
            <div className="w-full overflow-x-auto rounded-xl border border-[#DFE5ED] bg-[#F8FAFC] p-2">
              <svg
                viewBox={`0 0 ${hourlySvgW} ${hourlySvgH}`}
                className="w-full h-auto min-w-[700px]"
              >
                {/* Horizontal Grid (0%, 25%, 50%, 75%, 100%) */}
                {[0, 25, 50, 75, 100].map((pct) => {
                  const y = hourlySvgH - 30 - (pct / 100) * (hourlySvgH - 50);
                  return (
                    <g key={pct}>
                      <line
                        x1={45}
                        y1={y}
                        x2={hourlySvgW - 10}
                        y2={y}
                        stroke="#E2E8F0"
                        strokeDasharray="2 2"
                      />
                      <text
                        x={40}
                        y={y + 4}
                        fill="#475569"
                        fontSize="10"
                        textAnchor="end"
                        fontFamily="monospace"
                      >
                        {pct}%
                      </text>
                    </g>
                  );
                })}

                {/* 95% Nominal Reference Line */}
                {(() => {
                  const y95 = hourlySvgH - 30 - (95 / 100) * (hourlySvgH - 50);
                  return (
                    <g>
                      <line
                        x1={45}
                        y1={y95}
                        x2={hourlySvgW - 10}
                        y2={y95}
                        stroke="#DC2626"
                        strokeWidth="1.8"
                        strokeDasharray="6 3"
                      />
                      <text
                        x={hourlySvgW - 15}
                        y={y95 - 6}
                        fill="#DC2626"
                        fontSize="10"
                        fontFamily="monospace"
                        textAnchor="end"
                        fontWeight="bold"
                      >
                        95% Nominal Target
                      </text>
                    </g>
                  );
                })()}

                {/* Bars for 0 to 23 hours */}
                {hourlyData.map((row) => {
                  const colW = (hourlySvgW - 65) / 24;
                  const x = 50 + row.hr * colW;
                  const barH = (row.coverage_pct / 100) * (hourlySvgH - 50);
                  const y = hourlySvgH - 30 - barH;

                  // Highlight weaker coverage at 8, 17, 18
                  const isWeakPeak = row.hr === 8 || row.hr === 17 || row.hr === 18;

                  return (
                    <g key={row.hr} className="group">
                      <rect
                        x={x + 3}
                        y={y}
                        width={colW - 6}
                        height={barH}
                        rx="3"
                        fill={isWeakPeak ? "#D97706" : row.coverage_pct >= 95 ? "#0F766E" : "#2563EB"}
                        className="transition-all hover:opacity-85"
                      />

                      {/* Percentage label above weak peak bars */}
                      {isWeakPeak && (
                        <text
                          x={x + colW / 2}
                          y={y - 6}
                          fill="#B45309"
                          fontSize="9"
                          fontFamily="monospace"
                          fontWeight="bold"
                          textAnchor="middle"
                        >
                          {row.coverage_pct}%
                        </text>
                      )}

                      {/* Hour label below */}
                      <text
                        x={x + colW / 2}
                        y={hourlySvgH - 15}
                        fill={isWeakPeak ? "#B45309" : "#475569"}
                        fontSize="9"
                        fontFamily="monospace"
                        fontWeight={isWeakPeak ? "bold" : "normal"}
                        textAnchor="middle"
                      >
                        {row.hr < 10 ? `0${row.hr}` : row.hr}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>

            {/* In-depth Peak Hours Coverage Commentary */}
            <div className="p-5 rounded-xl border border-amber-200 bg-amber-50 space-y-2 text-sm text-amber-950">
              <div className="flex items-center gap-2 text-amber-900 font-bold text-sm sm:text-base">
                <ShieldAlert className="w-4 h-4 flex-shrink-0" />
                <span>尖峰時段條件覆蓋率下滑現象深入解析 (Conditional Coverage Disparity)</span>
              </div>
              <p className="leading-relaxed text-amber-950">
                雖然 M2 模型在整個測試集 2,576 筆樣本上達成了 <strong>94.84%</strong> 的實證覆蓋率，近乎完美貼合 95% 的名義目標；
                但在逐時段（Conditional on Hour）診斷中，可觀察到嚴重的時段條件覆蓋差異：
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1 font-mono text-xs">
                <div className="p-3 rounded-lg bg-white border border-amber-200 shadow-2xs">
                  <span className="text-amber-800 block font-bold text-xs sm:text-sm">08:00 (早通勤峰)</span>
                  <span className="text-[#17243A] text-sm font-extrabold">覆蓋率 64.49%</span>
                  <span className="text-[#475569] block text-xs mt-0.5 font-sans">實際均值 366.5 vs 預測 324.9</span>
                </div>
                <div className="p-3 rounded-lg bg-white border border-amber-200 shadow-2xs">
                  <span className="text-amber-800 block font-bold text-xs sm:text-sm">17:00 (晚通勤峰)</span>
                  <span className="text-[#17243A] text-sm font-extrabold">覆蓋率 75.00%</span>
                  <span className="text-[#475569] block text-xs mt-0.5 font-sans">實際均值 479.2 vs 預測 432.4</span>
                </div>
                <div className="p-3 rounded-lg bg-white border border-amber-200 shadow-2xs">
                  <span className="text-amber-800 block font-bold text-xs sm:text-sm">18:00 (晚通勤峰)</span>
                  <span className="text-[#17243A] text-sm font-extrabold">覆蓋率 80.56%</span>
                  <span className="text-[#475569] block text-xs mt-0.5 font-sans">實際均值 432.8 vs 預測 398.2</span>
                </div>
              </div>
              <p className="text-sm text-amber-950 pt-1 leading-relaxed">
                <strong>統計學核心意涵：</strong>共形預測的邊際保證（Marginal Guarantee）意味著在全樣本總體空間期望覆蓋率大於等於 95%，但<strong>不保證</strong>在任何特定子群體或時段內具備均等的條件覆蓋率（Equal Conditional Coverage）。由於通勤尖峰需求受天氣良好度與平日突發活動之交互放大效應影響，線性模型的加性常數無法完全吸納高階非線性波動，導致殘差擴大並突破固定半徑 q = 302.60 的區間邊界。
              </p>
            </div>
          </section>

          {/* 4. Residual Diagnostics */}
          <section className="p-6 rounded-2xl border border-[#DFE5ED] bg-white shadow-xs space-y-4">
            <h3 className="text-base font-bold text-[#17243A] flex items-center gap-2">
              <Activity className="w-5 h-5 text-blue-600" />
              <span>殘差分佈柱狀圖診斷 (Residual Distribution Analysis)</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Residual Histogram */}
              <div className="space-y-2">
                <span className="text-xs text-[#475569] font-mono font-semibold">殘差直方圖 (Residual Bins, n=35)</span>
                <div className="h-48 rounded-xl bg-[#F8FAFC] border border-[#DFE5ED] p-2 flex items-end gap-1">
                  {residualBins.map((bin, i) => {
                    const maxCount = 280;
                    const hPct = (bin.count / maxCount) * 100;
                    return (
                      <div
                        key={i}
                        className="flex-1 bg-blue-600 hover:bg-blue-700 transition-colors rounded-t group relative cursor-pointer"
                        style={{ height: `${Math.max(4, hPct)}%` }}
                        title={`殘差區間: ${bin.bin_start} ~ ${bin.bin_end}, 頻次: ${bin.count}`}
                      />
                    );
                  })}
                </div>
                <div className="flex justify-between text-[11px] font-mono text-[#64748B]">
                  <span>-400 (過度高估)</span>
                  <span>0 (無偏差核心)</span>
                  <span>+600 (極端低估)</span>
                </div>
              </div>

              {/* Residual Interpretation */}
              <div className="space-y-3 text-xs sm:text-sm text-[#334155]">
                <h4 className="font-bold text-[#17243A]">殘差診斷統計結論</h4>
                <ul className="space-y-2 text-[#475569]">
                  <li className="flex items-start gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-blue-600 mt-2 flex-shrink-0" />
                    <span>
                      <strong className="text-[#17243A]">近對稱鐘形分佈：</strong>殘差在 0 附近高度聚集，平均殘差接近 0，證實線性回歸具備良好無偏性 (Unbiasedness)。
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-600 mt-2 flex-shrink-0" />
                    <span>
                      <strong className="text-[#17243A]">厚尾現象 (Heavy Right Tail)：</strong>右側出現長拖尾（正殘差高達 +500 以上），這對應到晴朗溫和的好天氣上下班尖峰，實際租借人數暴衝遠超線性預測上限。
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-teal-600 mt-2 flex-shrink-0" />
                    <span>
                      <strong className="text-[#17243A]">異質變異 (Heteroscedasticity)：</strong>高預測量時殘差離散程度隨之擴大，此為計數資料（Count Data）典型的波松/負二項過度離散特徵。
                    </span>
                  </li>
                </ul>
              </div>
            </div>
          </section>
        </div>
      ) : (
        /* Static High-Res Figures */
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="p-6 rounded-2xl border border-[#DFE5ED] bg-white shadow-xs space-y-3">
            <h3 className="text-base font-bold text-[#17243A]">實際值 vs 預測值靜態圖 (actual_vs_predicted.png)</h3>
            <div className="relative aspect-[16/10] w-full rounded-xl overflow-hidden border border-[#DFE5ED] bg-[#F8FAFC]">
              <Image
                src="/figures/actual_vs_predicted.png"
                alt="Actual vs Predicted"
                fill
                className="object-contain"
                sizes="(max-width: 1024px) 100vw, 50vw"
              />
            </div>
          </div>

          <div className="p-6 rounded-2xl border border-[#DFE5ED] bg-white shadow-xs space-y-3">
            <h3 className="text-base font-bold text-[#17243A]">共形預測區間時序軸 (prediction_interval.png)</h3>
            <div className="relative aspect-[16/10] w-full rounded-xl overflow-hidden border border-[#DFE5ED] bg-[#F8FAFC]">
              <Image
                src="/figures/prediction_interval.png"
                alt="Prediction Interval Timeline"
                fill
                className="object-contain"
                sizes="(max-width: 1024px) 100vw, 50vw"
              />
            </div>
          </div>

          <div className="p-6 rounded-2xl border border-[#DFE5ED] bg-white shadow-xs space-y-3">
            <h3 className="text-base font-bold text-[#17243A]">時段覆蓋率 0–100% 修正圖 (interval_coverage_by_hour.png)</h3>
            <div className="relative aspect-[16/10] w-full rounded-xl overflow-hidden border border-[#DFE5ED] bg-[#F8FAFC]">
              <Image
                src="/figures/interval_coverage_by_hour.png"
                alt="Hourly Interval Coverage 0-100%"
                fill
                className="object-contain"
                sizes="(max-width: 1024px) 100vw, 50vw"
              />
            </div>
          </div>

          <div className="p-6 rounded-2xl border border-[#DFE5ED] bg-white shadow-xs space-y-3">
            <h3 className="text-base font-bold text-[#17243A]">殘差診斷分析圖 (residual_analysis.png)</h3>
            <div className="relative aspect-[16/10] w-full rounded-xl overflow-hidden border border-[#DFE5ED] bg-[#F8FAFC]">
              <Image
                src="/figures/residual_analysis.png"
                alt="Residual Analysis"
                fill
                className="object-contain"
                sizes="(max-width: 1024px) 100vw, 50vw"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
