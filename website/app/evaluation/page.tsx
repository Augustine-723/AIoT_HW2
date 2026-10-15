"use client";

import Image from "next/image";
import { useState } from "react";
import { 
  finalTestMetrics, 
  predictionIntervalMetrics, 
  hourlyCoverageDiagnostic,
  testTimeline7d,
  testScatterSamples,
  residualDistributionBins,
  HourlyCoverageRow,
  TimelinePoint,
  ScatterSample,
  ResidualBin
} from "@/lib/data";
import { 
  LineChart, 
  BarChart3, 
  Target, 
  ShieldAlert, 
  CheckCircle2, 
  Activity, 
  Clock, 
  Info,
  Calendar,
  Layers
} from "lucide-react";

export default function FinalEvaluationPage() {
  const [hoveredScatter, setHoveredScatter] = useState<ScatterSample | null>(null);
  const [hoveredTimeline, setHoveredTimeline] = useState<TimelinePoint | null>(null);
  const [useClippedInterval, setUseClippedInterval] = useState(true);
  const [activeTab, setActiveTab] = useState<"INTERACTIVE" | "STATIC">("INTERACTIVE");

  const timelineData = testTimeline7d as TimelinePoint[];
  const scatterData = testScatterSamples as ScatterSample[];
  const hourlyData = hourlyCoverageDiagnostic as HourlyCoverageRow[];
  const residualBins = residualDistributionBins as ResidualBin[];

  // Max values for scatter SVG scaling
  const maxActual = 980;
  const maxPred = 750;
  const scatterSvgSize = 500;

  // Max values for timeline SVG scaling
  const maxTimelineY = 900;
  const timelineSvgW = 800;
  const timelineSvgH = 320;

  // Hourly coverage chart dimensions
  const hourlySvgW = 800;
  const hourlySvgH = 280;

  return (
    <div className="space-y-12">
      {/* Header */}
      <section className="space-y-3">
        <div className="flex items-center gap-2 text-xs font-mono text-emerald-400">
          <LineChart className="w-4 h-4" />
          <span>FINAL TEST EVALUATION & CONFORMAL UNCERTAINTY</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          最終測試集評估、分裂共形預測區間與時段覆蓋率分析
        </h1>
        <p className="text-sm text-slate-300 max-w-4xl leading-relaxed">
          以鎖定之 Phase 3 測試集（2012-09-14 至 2012-12-31，共 2,576 筆真實觀測）驗證 M2 模型泛化能力。
          本頁面全面整合實際值 vs 預測值散佈圖、7 天時序共形區間、殘差分佈診斷與 0–100% 全幅時段實證覆蓋率。
        </p>
      </section>

      {/* Top Test Metric Cards */}
      <section className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40">
          <span className="text-[11px] text-slate-400">Test MAE</span>
          <div className="text-xl font-bold font-mono text-white mt-1">
            {finalTestMetrics.test_mae.toFixed(4)}
          </div>
          <span className="text-[10px] text-slate-400">輛 / 小時</span>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40">
          <span className="text-[11px] text-slate-400">Test RMSE</span>
          <div className="text-xl font-bold font-mono text-white mt-1">
            {finalTestMetrics.test_rmse.toFixed(4)}
          </div>
          <span className="text-[10px] text-slate-400">輛 / 小時</span>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40">
          <span className="text-[11px] text-slate-400">Test R²</span>
          <div className="text-xl font-bold font-mono text-cyan-400 mt-1">
            {finalTestMetrics.test_r2.toFixed(4)}
          </div>
          <span className="text-[10px] text-slate-400">解釋變異量 61.12%</span>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40">
          <span className="text-[11px] text-slate-400">校準共形分位數 q</span>
          <div className="text-xl font-bold font-mono text-indigo-300 mt-1">
            {predictionIntervalMetrics.conformal_quantile.toFixed(2)}
          </div>
          <span className="text-[10px] text-slate-400">Cal 樣本 n=2,640</span>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40">
          <span className="text-[11px] text-slate-400">名義覆蓋率目標</span>
          <div className="text-xl font-bold font-mono text-slate-300 mt-1">
            {(predictionIntervalMetrics.nominal_coverage * 100).toFixed(0)}%
          </div>
          <span className="text-[10px] text-slate-400">1 - α (α = 0.05)</span>
        </div>

        <div className="p-4 rounded-xl border border-emerald-500/30 bg-emerald-950/20">
          <span className="text-[11px] text-emerald-400 font-semibold">實證測試覆蓋率</span>
          <div className="text-xl font-bold font-mono text-emerald-300 mt-1">
            {(predictionIntervalMetrics.empirical_coverage_clipped * 100).toFixed(2)}%
          </div>
          <span className="text-[10px] text-emerald-400/80">94.84% 邊際達成</span>
        </div>
      </section>

      {/* Switch Interactive / Static Visualizations */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setActiveTab("INTERACTIVE")}
            className={`px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
              activeTab === "INTERACTIVE"
                ? "bg-indigo-600 text-white shadow-md shadow-indigo-900"
                : "bg-slate-900 text-slate-400 hover:text-white"
            }`}
          >
            互動式圖表儀表板 (Interactive Exploration)
          </button>
          <button
            onClick={() => setActiveTab("STATIC")}
            className={`px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
              activeTab === "STATIC"
                ? "bg-indigo-600 text-white shadow-md shadow-indigo-900"
                : "bg-slate-900 text-slate-400 hover:text-white"
            }`}
          >
            靜態論文級圖表 (Phase 3 High-Res Figures)
          </button>
        </div>
        <span className="text-xs text-slate-400 hidden sm:block">
          所有數據皆直接源自 test_predictions.csv
        </span>
      </div>

      {activeTab === "INTERACTIVE" ? (
        <div className="space-y-12">
          {/* 1. Actual vs Predicted Interactive Scatter Plot */}
          <section className="p-6 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <Target className="w-5 h-5 text-cyan-400" />
                  <span>實際租借量 vs 預測租借量散佈圖 (Actual vs Predicted)</span>
                </h3>
                <p className="text-xs text-slate-400">
                  X 軸：實際租借量 (Actual Rental Count) · Y 軸：M2 預測租借量 (Predicted Count) · 綠線為 y = x 完美預測基準線
                </p>
              </div>
              {hoveredScatter && (
                <div className="px-3 py-1.5 rounded-lg bg-slate-800 text-xs font-mono text-cyan-300 border border-slate-700 flex items-center gap-3">
                  <span>時間: {hoveredScatter.timestamp}</span>
                  <span>實際: {hoveredScatter.actual}</span>
                  <span>預測: {hoveredScatter.predicted.toFixed(1)}</span>
                  <span>殘差: {hoveredScatter.residual.toFixed(1)}</span>
                </div>
              )}
            </div>

            <div className="relative w-full aspect-square max-w-xl mx-auto border border-slate-800 rounded-xl bg-slate-950 p-4">
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
                    <g key={val} className="text-slate-800">
                      {/* Horizontal Grid */}
                      <line
                        x1={40}
                        y1={yPos}
                        x2={scatterSvgSize}
                        y2={yPos}
                        stroke="currentColor"
                        strokeDasharray="3 3"
                        strokeWidth="1"
                      />
                      <text
                        x={35}
                        y={yPos + 4}
                        fill="#64748B"
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
                        stroke="currentColor"
                        strokeDasharray="3 3"
                        strokeWidth="1"
                      />
                      <text
                        x={xPos}
                        y={scatterSvgSize - 15}
                        fill="#64748B"
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
                  stroke="#10B981"
                  strokeWidth="2"
                  strokeDasharray="5 5"
                />

                {/* y = 0 Horizontal baseline (negative predicted zone) */}
                <line
                  x1={40}
                  y1={scatterSvgSize - 30}
                  x2={scatterSvgSize}
                  y2={scatterSvgSize - 30}
                  stroke="#475569"
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
                      fill={pt.predicted < 0 ? "#F59E0B" : pt.covered ? "#06B6D4" : "#EF4444"}
                      opacity={isHovered ? 1 : 0.65}
                      className="cursor-pointer transition-all hover:scale-150"
                      onMouseEnter={() => setHoveredScatter(pt)}
                    />
                  );
                })}
              </svg>
            </div>

            <div className="flex flex-wrap items-center justify-between text-xs text-slate-400 pt-2 border-t border-slate-800">
              <div className="flex items-center gap-4">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-cyan-400" />
                  <span>區間內點位 (Covered)</span>
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-red-500" />
                  <span>區間外逸脫點 (Violated, 5.16%)</span>
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-amber-400" />
                  <span>未截斷負值預測 (Negative Point Predictions)</span>
                </span>
              </div>
              <span className="text-[11px] font-mono text-slate-400">
                點位呈現 815 筆跨時段代表性測試集抽樣
              </span>
            </div>
          </section>

          {/* 2. Prediction Interval Timeline (7-Day Slice) */}
          <section className="p-6 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <Clock className="w-5 h-5 text-indigo-400" />
                  <span>測試集前 7 天時序預測區間帶 (7-Day Prediction Interval Timeline)</span>
                </h3>
                <p className="text-xs text-slate-400">
                  2012-09-14 至 2012-09-20 連續 168 小時之實際值、M2 預測值與 95% 名義預測區間陰影帶
                </p>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => setUseClippedInterval(!useClippedInterval)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium border transition-colors ${
                    useClippedInterval
                      ? "bg-indigo-600/20 text-indigo-300 border-indigo-500/40"
                      : "bg-slate-800 text-slate-400 border-slate-700"
                  }`}
                >
                  {useClippedInterval ? "顯示：非負截斷區間 [max(0, y-q), y+q]" : "顯示：原始對稱區間 [y-q, y+q]"}
                </button>
              </div>
            </div>

            {/* Hover tooltip readout */}
            {hoveredTimeline && (
              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-xs font-mono grid grid-cols-2 sm:grid-cols-5 gap-2 text-slate-300">
                <div>
                  <span className="text-slate-400 block text-[10px]">時戳 Timestamp</span>
                  <span className="text-white font-bold">{hoveredTimeline.timestamp}</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px]">實際值 Actual</span>
                  <span className="text-cyan-400 font-bold">{hoveredTimeline.actual_cnt}</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px]">預測值 Predicted</span>
                  <span className="text-indigo-300 font-bold">{hoveredTimeline.predicted_cnt.toFixed(1)}</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px]">95% 預測區間</span>
                  <span className="text-white">
                    [
                    {useClippedInterval
                      ? hoveredTimeline.pi_lower_clipped.toFixed(0)
                      : hoveredTimeline.pi_lower_raw.toFixed(0)}
                    , {hoveredTimeline.pi_upper_clipped.toFixed(0)}]
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px]">覆蓋狀態 Covered</span>
                  <span className={hoveredTimeline.covered ? "text-emerald-400 font-bold" : "text-red-400 font-bold"}>
                    {hoveredTimeline.covered ? "✓ 區間內" : "✕ 超出區間"}
                  </span>
                </div>
              </div>
            )}

            {/* Responsive Timeline SVG */}
            <div className="w-full overflow-x-auto rounded-xl border border-slate-800 bg-slate-950 p-2">
              <svg
                viewBox={`0 0 ${timelineSvgW} ${timelineSvgH}`}
                className="w-full h-auto min-w-[700px]"
              >
                {/* Horizontal Grid */}
                {[0, 200, 400, 600, 800].map((val) => {
                  const y = timelineSvgH - 30 - (val / maxTimelineY) * (timelineSvgH - 50);
                  return (
                    <g key={val}>
                      <line
                        x1={45}
                        y1={y}
                        x2={timelineSvgW - 10}
                        y2={y}
                        stroke="#1E293B"
                        strokeDasharray="3 3"
                      />
                      <text
                        x={40}
                        y={y + 4}
                        fill="#64748B"
                        fontSize="10"
                        textAnchor="end"
                        fontFamily="monospace"
                      >
                        {val}
                      </text>
                    </g>
                  );
                })}

                {/* Day Boundary Separators */}
                {[24, 48, 72, 96, 120, 144].map((hrIndex) => {
                  const x = 45 + (hrIndex / timelineData.length) * (timelineSvgW - 55);
                  return (
                    <line
                      key={hrIndex}
                      x1={x}
                      y1={10}
                      x2={x}
                      y2={timelineSvgH - 30}
                      stroke="#334155"
                      strokeDasharray="4 4"
                    />
                  );
                })}

                {/* Shaded Prediction Interval Band Polygon */}
                {(() => {
                  const pointsUpper: string[] = [];
                  const pointsLower: string[] = [];

                  timelineData.forEach((pt, i) => {
                    const x = 45 + (i / (timelineData.length - 1)) * (timelineSvgW - 55);
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
                      fill="rgba(99, 102, 241, 0.15)"
                      stroke="rgba(99, 102, 241, 0.35)"
                      strokeWidth="1"
                    />
                  );
                })()}

                {/* Predicted Polyline */}
                {(() => {
                  const pts = timelineData.map((pt, i) => {
                    const x = 45 + (i / (timelineData.length - 1)) * (timelineSvgW - 55);
                    const y = timelineSvgH - 30 - (pt.predicted_cnt / maxTimelineY) * (timelineSvgH - 50);
                    return `${x},${y}`;
                  }).join(" ");
                  return (
                    <polyline
                      points={pts}
                      fill="none"
                      stroke="#818CF8"
                      strokeWidth="1.8"
                      strokeDasharray="4 2"
                    />
                  );
                })()}

                {/* Actual Polyline & Data Points */}
                {(() => {
                  const pts = timelineData.map((pt, i) => {
                    const x = 45 + (i / (timelineData.length - 1)) * (timelineSvgW - 55);
                    const y = timelineSvgH - 30 - (pt.actual_cnt / maxTimelineY) * (timelineSvgH - 50);
                    return `${x},${y}`;
                  }).join(" ");
                  return (
                    <polyline
                      points={pts}
                      fill="none"
                      stroke="#06B6D4"
                      strokeWidth="2"
                    />
                  );
                })()}

                {/* Hover trigger columns */}
                {timelineData.map((pt, i) => {
                  const x = 45 + (i / (timelineData.length - 1)) * (timelineSvgW - 55);
                  const isHovered = hoveredTimeline?.timestamp === pt.timestamp;
                  return (
                    <rect
                      key={i}
                      x={x - 2.5}
                      y={10}
                      width={5}
                      height={timelineSvgH - 40}
                      fill={isHovered ? "rgba(255, 255, 255, 0.15)" : "transparent"}
                      className="cursor-crosshair"
                      onMouseEnter={() => setHoveredTimeline(pt)}
                    />
                  );
                })}
              </svg>
            </div>

            <div className="flex flex-wrap items-center justify-between text-xs text-slate-400 pt-2 border-t border-slate-800">
              <div className="flex items-center gap-4">
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-1 bg-cyan-400 rounded-full" />
                  <span className="text-slate-300">實際每小時借車量 (Actual Count)</span>
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-1 bg-indigo-400 rounded-full border-t border-dashed border-white" />
                  <span className="text-slate-300">M2 模型點預測 (Predicted Count)</span>
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-3 bg-indigo-500/20 border border-indigo-500/40 rounded" />
                  <span className="text-slate-300">95% 名義共形預測區間 (Conformal PI Band)</span>
                </span>
              </div>
              <span className="text-[11px] font-mono text-slate-400">
                滑鼠懸停於圖表上可即時檢視該小時數值
              </span>
            </div>
          </section>

          {/* 3. Hourly Empirical Coverage (0–100% Full Scale) */}
          <section className="p-6 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <Activity className="w-5 h-5 text-amber-400" />
                  <span>24 小時時段實證覆蓋率全幅診斷 (Hourly Empirical Coverage, 0–100%)</span>
                </h3>
                <p className="text-xs text-slate-400">
                  全幅 Y 軸（0% 至 100%）清楚展示各小時之條件覆蓋率分佈，突顯早晚通勤尖峰時段覆蓋率下滑現象。
                </p>
              </div>
              <div className="px-3 py-1 rounded-md bg-amber-500/10 text-amber-300 border border-amber-500/20 text-xs font-mono">
                95% 為整體邊際覆蓋率目標 (Marginal Coverage)
              </div>
            </div>

            {/* Hourly Coverage SVG Bar Chart */}
            <div className="w-full overflow-x-auto rounded-xl border border-slate-800 bg-slate-950 p-2">
              <svg
                viewBox={`0 0 ${hourlySvgW} ${hourlySvgH}`}
                className="w-full h-auto min-w-[700px]"
              >
                {/* Horizontal Grid (0%, 25%, 50%, 75%, 95%, 100%) */}
                {[0, 25, 50, 75, 100].map((pct) => {
                  const y = hourlySvgH - 30 - (pct / 100) * (hourlySvgH - 50);
                  return (
                    <g key={pct}>
                      <line
                        x1={45}
                        y1={y}
                        x2={hourlySvgW - 10}
                        y2={y}
                        stroke="#1E293B"
                        strokeDasharray="2 2"
                      />
                      <text
                        x={40}
                        y={y + 4}
                        fill="#64748B"
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
                        stroke="#10B981"
                        strokeWidth="1.5"
                        strokeDasharray="6 3"
                      />
                      <text
                        x={hourlySvgW - 15}
                        y={y95 - 6}
                        fill="#10B981"
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
                        fill={isWeakPeak ? "#F59E0B" : row.coverage_pct >= 95 ? "#06B6D4" : "#6366F1"}
                        className="transition-all hover:opacity-80"
                      />

                      {/* Percentage label above weak peak bars */}
                      {isWeakPeak && (
                        <text
                          x={x + colW / 2}
                          y={y - 6}
                          fill="#F59E0B"
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
                        fill={isWeakPeak ? "#F59E0B" : "#94A3B8"}
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
            <div className="p-4 rounded-xl border border-amber-500/20 bg-amber-950/10 space-y-2 text-xs text-slate-300">
              <div className="flex items-center gap-2 text-amber-400 font-semibold">
                <ShieldAlert className="w-4 h-4" />
                <span>尖峰時段條件覆蓋率下滑現象深入解析 (Conditional Coverage Disparity)</span>
              </div>
              <p className="leading-relaxed text-slate-400">
                雖然 M2 模型在整個測試集 2,576 筆樣本上達成了 <strong>94.84%</strong> 的實證覆蓋率，近乎完美貼合 95% 的名義目標；
                但在逐時段（Conditional on Hour）診斷中，可觀察到嚴重的時段條件覆蓋差異：
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1 font-mono text-[11px]">
                <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                  <span className="text-amber-400 block font-bold">08:00 (早通勤峰)</span>
                  <span className="text-white text-sm">覆蓋率 64.49%</span>
                  <span className="text-slate-400 block text-[10px] mt-0.5">實際均值 366.5 vs 預測 324.9</span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                  <span className="text-amber-400 block font-bold">17:00 (晚通勤峰)</span>
                  <span className="text-white text-sm">覆蓋率 75.00%</span>
                  <span className="text-slate-400 block text-[10px] mt-0.5">實際均值 479.2 vs 預測 432.4</span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                  <span className="text-amber-400 block font-bold">18:00 (晚通勤峰)</span>
                  <span className="text-white text-sm">覆蓋率 80.56%</span>
                  <span className="text-slate-400 block text-[10px] mt-0.5">實際均值 432.8 vs 預測 398.2</span>
                </div>
              </div>
              <p className="text-[11px] text-slate-400 pt-1">
                <strong>統計學核心意涵：</strong>共形預測的邊際保證（Marginal Guarantee）意味著在全樣本總體空間期望覆蓋率大於等於 95%，但<strong>不保證</strong>在任何特定子群體或時段內具備均等的條件覆蓋率（Equal Conditional Coverage）。由於通勤尖峰需求受天氣良好度與平日突發活動之交互放大效應影響，線性模型的加性常數無法完全吸納高階非線性波動，導致殘差擴大並突破固定半徑 q = 302.60 的區間邊界。
              </p>
            </div>
          </section>

          {/* 4. Residual Diagnostics */}
          <section className="p-6 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-4">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Activity className="w-5 h-5 text-cyan-400" />
              <span>殘差分佈柱狀圖診斷 (Residual Distribution Analysis)</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Residual Histogram */}
              <div className="space-y-2">
                <span className="text-xs text-slate-400 font-mono">殘差直方圖 (Residual Bins, n=35)</span>
                <div className="h-48 rounded-xl bg-slate-950 border border-slate-800 p-2 flex items-end gap-1">
                  {residualBins.map((bin, i) => {
                    const maxCount = 280;
                    const hPct = (bin.count / maxCount) * 100;
                    return (
                      <div
                        key={i}
                        className="flex-1 bg-indigo-500/70 hover:bg-cyan-400 transition-colors rounded-t group relative cursor-pointer"
                        style={{ height: `${Math.max(4, hPct)}%` }}
                        title={`殘差區間: ${bin.bin_start} ~ ${bin.bin_end}, 頻次: ${bin.count}`}
                      />
                    );
                  })}
                </div>
                <div className="flex justify-between text-[10px] font-mono text-slate-400">
                  <span>-400 (過度高估)</span>
                  <span>0 (無偏差核心)</span>
                  <span>+600 (極端低估)</span>
                </div>
              </div>

              {/* Residual Interpretation */}
              <div className="space-y-3 text-xs text-slate-300">
                <h4 className="font-bold text-white">殘差診斷統計結論</h4>
                <ul className="space-y-2 text-slate-400">
                  <li className="flex items-start gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5 flex-shrink-0" />
                    <span>
                      <strong>近對稱鐘形分佈：</strong>殘差在 0 附近高度聚集，平均殘差接近 0，證實線性回歸具備良好無偏性 (Unbiasedness)。
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-1.5 flex-shrink-0" />
                    <span>
                      <strong>厚尾現象 (Heavy Right Tail)：</strong>右側出現長拖尾（正殘差高達 +500 以上），這對應到晴朗溫和的好天氣上下班尖峰，實際租借人數暴衝遠超線性預測上限。
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 mt-1.5 flex-shrink-0" />
                    <span>
                      <strong>異質變異 (Heteroscedasticity)：</strong>高預測量時殘差離散程度隨之擴大，此為計數資料（Count Data）典型的波松/負二項過度離散特徵。
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
          <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-3">
            <h3 className="text-sm font-bold text-white">實際值 vs 預測值靜態圖 (actual_vs_predicted.png)</h3>
            <div className="relative aspect-[16/10] w-full rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
              <Image
                src="/figures/actual_vs_predicted.png"
                alt="Actual vs Predicted"
                fill
                className="object-contain"
                sizes="(max-width: 1024px) 100vw, 50vw"
              />
            </div>
          </div>

          <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-3">
            <h3 className="text-sm font-bold text-white">共形預測區間時序軸 (prediction_interval.png)</h3>
            <div className="relative aspect-[16/10] w-full rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
              <Image
                src="/figures/prediction_interval.png"
                alt="Prediction Interval Timeline"
                fill
                className="object-contain"
                sizes="(max-width: 1024px) 100vw, 50vw"
              />
            </div>
          </div>

          <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-3">
            <h3 className="text-sm font-bold text-white">時段覆蓋率 0–100% 修正圖 (interval_coverage_by_hour.png)</h3>
            <div className="relative aspect-[16/10] w-full rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
              <Image
                src="/figures/interval_coverage_by_hour.png"
                alt="Hourly Interval Coverage 0-100%"
                fill
                className="object-contain"
                sizes="(max-width: 1024px) 100vw, 50vw"
              />
            </div>
          </div>

          <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-3">
            <h3 className="text-sm font-bold text-white">殘差診斷分析圖 (residual_analysis.png)</h3>
            <div className="relative aspect-[16/10] w-full rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
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
