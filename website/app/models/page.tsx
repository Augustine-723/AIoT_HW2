"use client";

import Image from "next/image";
import { useState } from "react";
import { 
  modelComparison, 
  groupedFeatureAblation, 
  selectedFeatures,
  ModelComparisonRow,
  GroupedAblationRow,
  SelectedFeatureRow
} from "@/lib/data";
import { 
  Sliders, 
  Trophy, 
  CheckCircle2, 
  Filter, 
  Layers, 
  BarChart2, 
  Search,
  Sparkles,
  Info,
  ChevronDown
} from "lucide-react";

export default function ModelComparisonPage() {
  const [filterType, setFilterType] = useState<"ALL" | "SELECTED" | "ELIMINATED">("ALL");
  const [searchTerm, setSearchTerm] = useState("");

  const m2Features = (selectedFeatures as unknown as SelectedFeatureRow[]).filter(
    (feat) => feat.model_id === "M2"
  );

  const filteredFeatures = m2Features.filter((feat) => {
    const isSelected = feat.selection_status === "selected";
    if (filterType === "SELECTED" && !isSelected) return false;
    if (filterType === "ELIMINATED" && isSelected) return false;
    if (searchTerm) {
      const q = searchTerm.toLowerCase();
      return (
        feat.encoded_feature_name.toLowerCase().includes(q) ||
        feat.original_feature_name.toLowerCase().includes(q)
      );
    }
    return true;
  });

  return (
    <div className="space-y-12">
      {/* Header */}
      <section className="space-y-3">
        <div className="flex items-center gap-2 text-xs font-mono text-indigo-400">
          <Sliders className="w-4 h-4" />
          <span>MODEL BENCHMARKING & FEATURE SELECTION</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          模型交叉驗證評比、特徵選擇 (RFE) 與消融實驗
        </h1>
        <p className="text-sm text-slate-300 max-w-4xl leading-relaxed">
          以嚴格的 5-Fold 擴展視窗交叉驗證（Expanding Window CV, ddof=1 樣本標準差）評估 4 大候選模型。
          M2 藉由遞歸特徵消除（RFE）在剔除 18 個編碼維度下達成最低 CV RMSE，獲得最高泛化效能。
        </p>
      </section>

      {/* Model Comparison Table */}
      <section className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Trophy className="w-5 h-5 text-amber-400" />
              <span>4 大候選模型綜合評比 (Candidate Model Leaderboard)</span>
            </h2>
            <p className="text-xs text-slate-400">
              資料來源權威依據：results/model_comparison.csv（採用無偏樣本標準差 ddof=1）
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs px-2.5 py-1 rounded-md bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 font-mono">
              ★ M2 為官方選定模型
            </span>
          </div>
        </div>

        <div className="overflow-x-auto rounded-2xl border border-slate-800 bg-slate-900/40">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/90 text-slate-400 font-mono">
                <th className="py-3 px-4 font-semibold">排名</th>
                <th className="py-3 px-4 font-semibold">模型代號與名稱</th>
                <th className="py-3 px-4 font-semibold text-right">編碼特徵數</th>
                <th className="py-3 px-4 font-semibold text-right">Mean CV RMSE</th>
                <th className="py-3 px-4 font-semibold text-right">CV RMSE Std (ddof=1)</th>
                <th className="py-3 px-4 font-semibold text-right">Mean CV MAE</th>
                <th className="py-3 px-4 font-semibold text-right">Mean CV R²</th>
                <th className="py-3 px-4 font-semibold text-right">Train RMSE</th>
                <th className="py-3 px-4 font-semibold text-right">最佳超參數 (Hyperparameters)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-sans">
              {(modelComparison as unknown as ModelComparisonRow[]).map((m, idx) => {
                const isSelected = m.model_id === "M2";
                return (
                  <tr
                    key={m.model_id}
                    className={`transition-colors ${
                      isSelected
                        ? "bg-indigo-950/30 hover:bg-indigo-950/50 font-medium"
                        : "hover:bg-slate-800/30"
                    }`}
                  >
                    <td className="py-3.5 px-4 font-mono">
                      {isSelected ? (
                        <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-amber-400 text-slate-950 font-bold text-[10px]">
                          1
                        </span>
                      ) : (
                        <span className="text-slate-400 font-bold">{idx + 1}</span>
                      )}
                    </td>
                    <td className="py-3.5 px-4">
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-bold text-white text-sm">
                          {m.model_id}
                        </span>
                        <span className="text-slate-300">{m.model_name}</span>
                        {isSelected && (
                          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                            Selected
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="py-3.5 px-4 text-right font-mono text-slate-300">
                      {m.selected_features_count}
                    </td>
                    <td className={`py-3.5 px-4 text-right font-mono font-bold ${
                      isSelected ? "text-cyan-400 text-sm" : "text-white"
                    }`}>
                      {m.cv_rmse_mean.toFixed(4)}
                    </td>
                    <td className="py-3.5 px-4 text-right font-mono text-slate-400">
                      ±{m.cv_rmse_std.toFixed(4)}
                    </td>
                    <td className="py-3.5 px-4 text-right font-mono text-slate-300">
                      {m.cv_mae_mean.toFixed(4)}
                    </td>
                    <td className="py-3.5 px-4 text-right font-mono text-slate-300">
                      {m.cv_r2_mean.toFixed(4)}
                    </td>
                    <td className="py-3.5 px-4 text-right font-mono text-slate-400">
                      {m.training_rmse.toFixed(4)}
                    </td>
                    <td className="py-3.5 px-4 text-right font-mono text-indigo-300">
                      {m.best_hyperparameters}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* Model Selection Discussion */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-2">
          <div className="text-xs font-semibold uppercase tracking-wider text-cyan-400">
            最優 RMSE 與精簡特徵
          </div>
          <div className="text-base font-bold text-white">M2 (RFE + OLS)</div>
          <p className="text-xs text-slate-400 leading-relaxed">
            M2 在 5-Fold 擴展視窗 CV 評比中取得最低之平均 CV RMSE (107.44)，
            成功剔除 18 個貢獻極低之冗餘特徵，僅保留 30 個特徵，兼具最低均方根誤差與最優模型精簡性 (Parsimony)。
          </p>
        </div>

        <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-2">
          <div className="text-xs font-semibold uppercase tracking-wider text-indigo-400">
            最優 MAE 絕對誤差
          </div>
          <div className="text-base font-bold text-white">M4 (Lasso Regression)</div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Lasso 在最佳懲罰係數 α = 0.1 下取得最低之平均 CV MAE。
            L1 正則化能將部分不重要係數自動歸零，對常態分佈誤差中位數展現絕佳適應度。
          </p>
        </div>

        <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-2">
          <div className="text-xs font-semibold uppercase tracking-wider text-emerald-400">
            最低摺間變異度
          </div>
          <div className="text-base font-bold text-white">M1 (Standard OLS)</div>
          <p className="text-xs text-slate-400 leading-relaxed">
            未篩選特徵之 OLS 在 5-Fold CV 展現最低之 RMSE 標準差 (15.36)。
            各模型間 CV RMSE 差異幅度相對較小，顯示線性結構在該問題具備極佳的基礎適應力。
          </p>
        </div>
      </section>

      {/* Phase 2 Figures Grid */}
      <section className="space-y-6">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <BarChart2 className="w-5 h-5 text-indigo-400" />
          <span>特徵淘汰歷程與正規化路徑視覺化 (Phase 2 Visualizations)</span>
        </h2>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Feature Selection Curve */}
          <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-white">RFE 特徵數量 vs CV RMSE 變化趨勢</h3>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">feature_selection_rmse.png</span>
            </div>
            <div className="relative aspect-[16/10] w-full rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
              <Image
                src="/figures/feature_selection_rmse.png"
                alt="RFE Feature Count vs CV RMSE"
                fill
                className="object-contain"
                sizes="(max-width: 1024px) 100vw, 50vw"
              />
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              當編碼特徵由完整維度逐步剔除至 30 時，CV RMSE 呈現平穩最佳化；若特徵數進一步縮減至 20 以下，CV 誤差顯著攀升，證實 30 個特徵為兼顧精度與簡潔之最佳切點。
            </p>
          </div>

          {/* Model Comparison Bar */}
          <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-white">4 大候選模型 CV RMSE 柱狀比較</h3>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">model_comparison.png</span>
            </div>
            <div className="relative aspect-[16/10] w-full rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
              <Image
                src="/figures/model_comparison.png"
                alt="Model Comparison Bar Chart"
                fill
                className="object-contain"
                sizes="(max-width: 1024px) 100vw, 50vw"
              />
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              呈現各模型在 Expanding Window 5 個摺上的跨時段表現。M2 在保持最精簡架構下達到整體均值最優，展現穩固泛化能力。
            </p>
          </div>

          {/* Regularization Tuning */}
          <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-white">正則化超參數 α 調校軌跡 (Ridge & Lasso Tuning)</h3>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">regularization_alpha_tuning.png</span>
            </div>
            <div className="relative aspect-[16/10] w-full rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
              <Image
                src="/figures/regularization_alpha_tuning.png"
                alt="Regularization Alpha Tuning"
                fill
                className="object-contain"
                sizes="(max-width: 1024px) 100vw, 50vw"
              />
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              對 Ridge 與 Lasso 在跨數個數量級之超參數空間搜尋。適度的懲罰強度能有效平抑共線性帶來的方差放大，過大的正則化強度則會過度壓抑真實信號。
            </p>
          </div>

          {/* Grouped Feature Ablation */}
          <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-white">群組特徵消融重要度 (Grouped Feature Ablation)</h3>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">feature_group_ablation.png</span>
            </div>
            <div className="relative aspect-[16/10] w-full rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
              <Image
                src="/figures/feature_group_ablation.png"
                alt="Grouped Feature Ablation"
                fill
                className="object-contain"
                sizes="(max-width: 1024px) 100vw, 50vw"
              />
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              透過分組剔除評估各特徵組的 Δ RMSE。時段特徵（Hour）移除後 RMSE 激增 +34.19，為最關鍵之決定性維度；年度與季節特徵亦展現顯著衝擊。
            </p>
          </div>
        </div>
      </section>

      {/* Grouped Feature Ablation Table */}
      <section className="space-y-4">
        <h2 className="text-lg font-bold text-white">群組消融實驗詳細數據 (Grouped Feature Ablation Results)</h2>
        <div className="overflow-x-auto rounded-2xl border border-slate-800 bg-slate-900/40">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/80 text-slate-400 font-mono">
                <th className="py-3 px-4 font-semibold">排名</th>
                <th className="py-3 px-4 font-semibold">剔除特徵群組 (Dropped Group)</th>
                <th className="py-3 px-4 font-semibold text-right">CV RMSE (剔除後)</th>
                <th className="py-3 px-4 font-semibold text-right text-amber-400">Δ RMSE (性能懲罰增量)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-sans">
              {(groupedFeatureAblation as unknown as GroupedAblationRow[]).map((ab, idx) => (
                <tr key={ab.dropped_group} className="hover:bg-slate-800/30 transition-colors">
                  <td className="py-3 px-4 font-mono font-bold text-white">
                    #{idx + 1}
                  </td>
                  <td className="py-3 px-4 font-mono font-semibold text-indigo-300">
                    {ab.dropped_group}
                  </td>
                  <td className="py-3 px-4 text-right font-mono font-bold text-white">
                    {ab.cv_rmse.toFixed(4)}
                  </td>
                  <td className="py-3 px-4 text-right font-mono font-bold text-amber-400">
                    +{ab.delta_rmse.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Selected Features Explorer */}
      <section className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Layers className="w-5 h-5 text-cyan-400" />
              <span>M2 模型之特徵係數清單 (M2 Selected Features Explorer)</span>
            </h2>
            <p className="text-xs text-slate-400">
              呈現 M2 模型中選定之特徵與遭 RFE 淘汰特徵之完整擬合係數
            </p>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="搜尋特徵代碼..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-8 pr-3 py-1.5 rounded-lg text-xs bg-slate-900 border border-slate-700 text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 w-44"
              />
            </div>

            <div className="flex rounded-lg border border-slate-800 p-0.5 bg-slate-900">
              <button
                onClick={() => setFilterType("ALL")}
                className={`px-2.5 py-1 text-xs rounded-md font-medium transition-colors ${
                  filterType === "ALL" ? "bg-indigo-600 text-white" : "text-slate-400 hover:text-white"
                }`}
              >
                全部 ({m2Features.length})
              </button>
              <button
                onClick={() => setFilterType("SELECTED")}
                className={`px-2.5 py-1 text-xs rounded-md font-medium transition-colors ${
                  filterType === "SELECTED" ? "bg-indigo-600 text-white" : "text-slate-400 hover:text-white"
                }`}
              >
                選定 ({m2Features.filter((f) => f.selection_status === "selected").length})
              </button>
              <button
                onClick={() => setFilterType("ELIMINATED")}
                className={`px-2.5 py-1 text-xs rounded-md font-medium transition-colors ${
                  filterType === "ELIMINATED" ? "bg-indigo-600 text-white" : "text-slate-400 hover:text-white"
                }`}
              >
                淘汰 ({m2Features.filter((f) => f.selection_status === "eliminated").length})
              </button>
            </div>
          </div>
        </div>

        <div className="overflow-x-auto rounded-2xl border border-slate-800 bg-slate-900/40 max-h-96">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="sticky top-0 bg-slate-900 border-b border-slate-800 text-slate-400 font-mono z-10">
              <tr>
                <th className="py-2.5 px-4 font-semibold">編碼特徵識別代號</th>
                <th className="py-2.5 px-4 font-semibold">原始特徵</th>
                <th className="py-2.5 px-4 font-semibold text-center">狀態</th>
                <th className="py-2.5 px-4 font-semibold text-right">M2 擬合迴歸係數 (β)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-sans">
              {filteredFeatures.map((f, idx) => {
                const isSelected = f.selection_status === "selected";
                return (
                  <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                    <td className="py-2 px-4 font-mono font-bold text-white">
                      {f.encoded_feature_name}
                    </td>
                    <td className="py-2 px-4 font-mono text-slate-400">
                      {f.original_feature_name}
                    </td>
                    <td className="py-2 px-4 text-center">
                      {isSelected ? (
                        <span className="inline-block px-2 py-0.5 rounded-full text-[10px] font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                          保留 (Selected)
                        </span>
                      ) : (
                        <span className="inline-block px-2 py-0.5 rounded-full text-[10px] font-mono bg-slate-800 text-slate-400 border border-slate-700">
                          RFE 淘汰
                        </span>
                      )}
                    </td>
                    <td className="py-2 px-4 text-right font-mono font-semibold">
                      {f.coefficient !== null && f.coefficient !== undefined ? (
                        <span className={f.coefficient >= 0 ? "text-cyan-400" : "text-amber-400"}>
                          {f.coefficient > 0 ? "+" : ""}
                          {f.coefficient.toFixed(4)}
                        </span>
                      ) : (
                        <span className="text-slate-600 font-normal">N/A (淘汰無係數)</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
