import Link from "next/link";
import { 
  finalTestMetrics, 
  predictionIntervalMetrics, 
  bestModelMetadata, 
  datasetMetadata,
  phase3ExperimentMetadata 
} from "@/lib/data";
import { 
  ArrowRight, 
  Calendar, 
  Activity, 
  Sliders, 
  Sparkles, 
  ShieldAlert,
  Database,
  LineChart
} from "lucide-react";

export default function OverviewPage() {
  const splitBoundaries = phase3ExperimentMetadata.exact_date_boundaries;
  const trainBound = splitBoundaries.training;
  const calBound = splitBoundaries.calibration;
  const testBound = splitBoundaries.test;

  const headlineCards = [
    {
      label: "Test MAE",
      sublabel: "平均絕對誤差",
      value: finalTestMetrics.test_mae.toFixed(4),
      unit: "輛 / 小時",
      desc: "反映平均預測偏離實際租借車輛數之絕對量級",
      highlight: false,
    },
    {
      label: "Test RMSE",
      sublabel: "均方根誤差",
      value: finalTestMetrics.test_rmse.toFixed(4),
      unit: "輛 / 小時",
      desc: "對尖峰通勤時段的大幅波動誤差賦予更高懲罰",
      highlight: false,
    },
    {
      label: "Test R²",
      sublabel: "判定係數",
      value: finalTestMetrics.test_r2.toFixed(4),
      unit: "解釋變異量",
      desc: "解釋了未見測試集逾 61.12% 的需求變異量",
      highlight: true,
    },
    {
      label: "Nominal Coverage",
      sublabel: "名義共形覆蓋率目標",
      value: `${(predictionIntervalMetrics.nominal_coverage * 100).toFixed(0)}%`,
      unit: "1 - α (α = 0.05)",
      desc: "基於 Calibration 集非符合性分數之分位數校準目標",
      highlight: false,
    },
    {
      label: "Empirical Coverage",
      sublabel: "實證測試集覆蓋率",
      value: `${(predictionIntervalMetrics.empirical_coverage_clipped * 100).toFixed(2)}%`,
      unit: "Test Set (2,576 樣本)",
      desc: "名義 95% 目標下，實證覆蓋率達 94.84%，展現高邊際有效性",
      highlight: true,
    },
  ];

  return (
    <div className="space-y-12">
      {/* Hero Header */}
      <section className="relative overflow-hidden rounded-3xl border border-[#DFE5ED] bg-[#F7F9FC] p-6 sm:p-10 shadow-xs">
        <div className="relative space-y-6 max-w-4xl">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="px-3 py-1 rounded-full font-mono font-semibold bg-teal-50 text-teal-800 border border-teal-200">
              NCU AIoT 課程研究
            </span>
            <span className="px-3 py-1 rounded-full font-mono font-semibold bg-blue-50 text-blue-800 border border-blue-200">
              學號: 7115093001
            </span>
            <span className="px-3 py-1 rounded-full font-mono font-semibold bg-slate-100 text-[#334155] border border-[#CBD5E1]">
              EXP-ID: {phase3ExperimentMetadata.experiment_id}
            </span>
          </div>

          <div className="space-y-2">
            <h1 className="text-2xl sm:text-4xl font-extrabold tracking-tight text-[#17243A] leading-tight">
              以多元線性迴歸與特徵選擇預測共享單車每小時租借需求
            </h1>
            <p className="text-base sm:text-lg text-[#475569] font-medium">
              Predicting Hourly Bike Sharing Demand Using Multiple Linear Regression and Feature Selection
            </p>
          </div>

          <p className="text-sm sm:text-base text-[#334155] leading-relaxed">
            本學術研究專案針對都市共享單車系統的資源動態調度與調度風險控制，
            採用 UCI Capital Bikeshare 歷史資料集（2011–2012 年，共 17,379 筆觀測值），
            以嚴格防止目標洩漏（Target Leakage）之特徵工程為基礎，透過時間序列擴展視窗交叉驗證
            （Expanding Window CV）評估多元線性模型，並結合遞歸特徵消除（RFE）選出最佳 30 個編碼特徵。
            最終引入傳統分裂共形預測（Split Conformal Prediction），在未見測試集上提供具嚴謹統計意義的 95% 名義區間不確定性量化。
          </p>

          <div className="pt-2 flex flex-wrap items-center gap-4">
            <Link
              href="/predict"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold text-white bg-blue-600 hover:bg-blue-700 shadow-sm transition-all"
            >
              <Sparkles className="w-4 h-4" />
              <span>啟動互動預測推論</span>
              <ArrowRight className="w-4 h-4" />
            </Link>

            <Link
              href="/evaluation"
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold text-[#17243A] bg-white hover:bg-slate-50 border border-[#DFE5ED] shadow-xs transition-all"
            >
              <LineChart className="w-4 h-4 text-teal-700" />
              <span>檢視最終測試集評估</span>
            </Link>
          </div>
        </div>
      </section>

      {/* Authoritative Headline Metrics Grid */}
      <section className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-xl sm:text-2xl font-bold text-[#17243A] flex items-center gap-2">
              <Activity className="w-5 h-5 text-blue-600" />
              <span>權威凍結實驗指標 (Authoritative Headline Metrics)</span>
            </h2>
            <p className="text-xs sm:text-sm text-[#475569]">
              直接讀取自 Phase 3 凍結之成果檔案 (final_test_metrics.json &amp; prediction_interval_metrics.json)
            </p>
          </div>
          <span className="text-xs font-mono font-semibold px-2.5 py-1 rounded bg-[#F7F9FC] border border-[#DFE5ED] text-[#475569] self-start sm:self-auto">
            {phase3ExperimentMetadata.experiment_id}
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {headlineCards.map((card, idx) => (
            <div
              key={idx}
              className={`p-5 rounded-2xl border transition-all ${
                card.highlight
                  ? "bg-blue-50/50 border-blue-200 shadow-xs"
                  : "bg-white border-[#DFE5ED] shadow-xs hover:border-slate-300"
              }`}
            >
              <div className="flex items-baseline justify-between mb-1">
                <span className="text-xs sm:text-sm font-bold uppercase tracking-wider text-[#334155]">
                  {card.label}
                </span>
                <span className="text-xs text-[#475569] font-mono">
                  {card.unit}
                </span>
              </div>
              <div className="text-2xl sm:text-3xl font-extrabold tracking-tight text-[#17243A] my-1 font-mono">
                {card.value}
              </div>
              <div className="text-sm font-bold text-blue-800 mb-1.5">
                {card.sublabel}
              </div>
              <p className="text-sm text-[#334155] leading-relaxed">
                {card.desc}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Dataset & Model Specification Summary Cards */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Research Dataset Spec */}
        <div className="p-6 rounded-2xl border border-[#DFE5ED] bg-white shadow-xs space-y-4">
          <div className="w-10 h-10 rounded-xl bg-teal-50 border border-teal-200 flex items-center justify-center text-teal-700">
            <Database className="w-5 h-5" />
          </div>
          <div className="space-y-1">
            <h3 className="text-lg font-bold text-[#17243A]">研究資料集架構</h3>
            <p className="text-sm text-[#475569]">UCI Bike Sharing Dataset (Hourly)</p>
          </div>
          <ul className="space-y-2.5 text-sm text-[#334155]">
            <li className="flex justify-between py-1.5 border-b border-[#DFE5ED]">
              <span className="text-[#334155]">總歷史觀測筆數</span>
              <strong className="font-mono text-[#17243A]">{datasetMetadata.number_of_rows.toLocaleString()} 筆</strong>
            </li>
            <li className="flex justify-between py-1.5 border-b border-[#DFE5ED]">
              <span className="text-[#334155]">原始輸入預測變數</span>
              <strong className="font-mono text-[#17243A]">12 個 (12 Predictors)</strong>
            </li>
            <li className="flex justify-between py-1.5 border-b border-[#DFE5ED]">
              <span className="text-[#334155]">預測目標變數</span>
              <strong className="font-mono text-teal-800">cnt (每小時總租借量)</strong>
            </li>
            <li className="flex justify-between py-1.5 border-b border-[#DFE5ED]">
              <span className="text-[#334155]">防範目標洩漏排除變數</span>
              <span className="font-mono text-amber-800 font-semibold">casual, registered</span>
            </li>
            <li className="flex justify-between py-1.5">
              <span className="text-[#334155]">跨年度時間跨度</span>
              <span className="font-mono text-[#334155]">2011-01-01 至 2012-12-31</span>
            </li>
          </ul>
          <Link
            href="/dataset"
            className="inline-flex items-center gap-1.5 text-sm font-semibold text-blue-700 hover:text-blue-800 transition-colors pt-2"
          >
            <span>檢視資料集特徵與 EDA 分析</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        {/* Selected Model Spec */}
        <div className="p-6 rounded-2xl border border-[#DFE5ED] bg-white shadow-xs space-y-4">
          <div className="w-10 h-10 rounded-xl bg-blue-50 border border-blue-200 flex items-center justify-center text-blue-600">
            <Sliders className="w-5 h-5" />
          </div>
          <div className="space-y-1">
            <h3 className="text-lg font-bold text-[#17243A]">選定最佳模型架構</h3>
            <p className="text-sm text-[#475569]">M2 — RFE + OLS (最優泛化模型)</p>
          </div>
          <ul className="space-y-2.5 text-sm text-[#334155]">
            <li className="flex justify-between py-1.5 border-b border-[#DFE5ED]">
              <span className="text-[#334155]">模型識別名稱</span>
              <strong className="font-mono text-blue-700">{bestModelMetadata.chosen_model.model_name}</strong>
            </li>
            <li className="flex justify-between py-1.5 border-b border-[#DFE5ED]">
              <span className="text-[#334155]">RFE 篩選編碼特徵數</span>
              <strong className="font-mono text-[#17243A]">{bestModelMetadata.chosen_model.selected_encoded_feature_count} / {bestModelMetadata.chosen_model.candidate_encoded_feature_count} ({bestModelMetadata.chosen_model.eliminated_encoded_feature_count} eliminated)</strong>
            </li>
            <li className="flex justify-between py-1.5 border-b border-[#DFE5ED]">
              <span className="text-[#334155]">5-Fold CV 平均 RMSE</span>
              <strong className="font-mono text-[#17243A]">{bestModelMetadata.chosen_model.cv_performance.cv_rmse_mean.toFixed(4)}</strong>
            </li>
            <li className="flex justify-between py-1.5 border-b border-[#DFE5ED]">
              <span className="text-[#334155]">CV RMSE 樣本標準差</span>
              <span className="font-mono text-[#334155]">{bestModelMetadata.chosen_model.cv_performance.cv_rmse_std.toFixed(4)} (ddof=1)</span>
            </li>
            <li className="flex justify-between py-1.5">
              <span className="text-[#334155]">數值推論精確度</span>
              <span className="font-mono text-emerald-700 font-semibold">&lt; 2.27 × 10⁻¹³ (全等)</span>
            </li>
          </ul>
          <Link
            href="/models"
            className="inline-flex items-center gap-1.5 text-sm font-semibold text-blue-700 hover:text-blue-800 transition-colors pt-2"
          >
            <span>檢視 4 大候選模型比較與消融實驗</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        {/* Chronological Partitioning */}
        <div className="p-6 rounded-2xl border border-[#DFE5ED] bg-white shadow-xs space-y-4">
          <div className="w-10 h-10 rounded-xl bg-emerald-50 border border-emerald-200 flex items-center justify-center text-emerald-700">
            <Calendar className="w-5 h-5" />
          </div>
          <div className="space-y-1">
            <h3 className="text-lg font-bold text-[#17243A]">嚴謹時序三階段切分</h3>
            <p className="text-sm text-[#475569]">無未來資訊洩漏之標準劃分</p>
          </div>
          <ul className="space-y-2.5 text-sm text-[#334155]">
            <li className="flex justify-between py-1.5 border-b border-[#DFE5ED]">
              <span className="text-[#334155]">訓練集 (Training Set)</span>
              <div className="text-right">
                <span className="font-mono font-bold text-[#17243A]">{trainBound.observations.toLocaleString()} 筆</span>
                <div className="text-xs text-[#475569]">{trainBound.date_range[0]} ~ {trainBound.date_range[1]}</div>
              </div>
            </li>
            <li className="flex justify-between py-1.5 border-b border-[#DFE5ED]">
              <span className="text-[#334155]">校準集 (Calibration Set)</span>
              <div className="text-right">
                <span className="font-mono font-bold text-[#17243A]">{calBound.observations.toLocaleString()} 筆</span>
                <div className="text-xs text-[#475569]">{calBound.date_range[0]} ~ {calBound.date_range[1]}</div>
              </div>
            </li>
            <li className="flex justify-between py-1.5">
              <span className="text-[#334155]">最終測試集 (Test Set)</span>
              <div className="text-right">
                <span className="font-mono text-emerald-800 font-bold">{testBound.observations.toLocaleString()} 筆</span>
                <div className="text-xs text-[#475569]">{testBound.date_range[0]} ~ {testBound.date_range[1]}</div>
              </div>
            </li>
          </ul>
          <Link
            href="/evaluation"
            className="inline-flex items-center gap-1.5 text-sm font-semibold text-emerald-700 hover:text-emerald-800 transition-colors pt-2"
          >
            <span>檢視共形預測區間與時序覆蓋分析</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      </section>

      {/* Critical Scientific Disclaimer */}
      <section className="p-5 rounded-2xl border border-amber-200 bg-amber-50 text-amber-950 shadow-xs space-y-2">
        <div className="flex items-center gap-2 text-amber-900 text-sm font-bold">
          <ShieldAlert className="w-4 h-4 flex-shrink-0" />
          <span>研究方法論重要限制與統計假設說明</span>
        </div>
        <p className="text-sm sm:text-base text-amber-950 leading-relaxed">
          1. <strong>歷史回歸本質：</strong>本系統基於 2011–2012 年之歷史資料進行條件需求推估，非即時感測資料流，且氣象輸入為正規化後之氣溫、體感溫度、濕度與風速，不可視為即時氣象預報。<br />
          2. <strong>共形保證前提：</strong>傳統分裂共形預測 (Split Conformal Prediction) 所提供之有限樣本邊際覆蓋率保證嚴格仰賴校準集與測試集之可交換性 (Exchangeability) 假設；本每小時資料集具備顯著之時間相依性與自相關性，因此 95% 為名義覆蓋目標 (Nominal Target)，實證測試集觀察值約為 94.84%，且條件覆蓋率因時段而異（如 08:00、17:00 等上下班通勤尖峰時段覆蓋率下滑），歷史觀察結果不構成對未來時間區段之無條件保證。
        </p>
      </section>

      {/* Quick Navigation Cards */}
      <section className="space-y-4">
        <h2 className="text-xl sm:text-2xl font-bold text-[#17243A]">研究導覽模組 (Research Navigation Modules)</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <Link
            href="/dataset"
            className="p-5 rounded-xl border border-[#DFE5ED] bg-white hover:bg-slate-50/80 hover:border-blue-300 transition-all group shadow-xs hover:shadow-sm"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-base sm:text-lg font-bold text-[#17243A] group-hover:text-blue-700 transition-colors">
                1. 資料集與 EDA
              </span>
              <ArrowRight className="w-4 h-4 text-[#64748B] group-hover:text-blue-700 group-hover:translate-x-1 transition-all" />
            </div>
            <p className="text-sm text-[#334155] leading-relaxed">
              探索 17,379 筆觀測值特徵分佈、時段雙峰現象、相關性熱圖、氣溫影響及 165 筆缺漏小時審計。
            </p>
          </Link>

          <Link
            href="/models"
            className="p-5 rounded-xl border border-[#DFE5ED] bg-white hover:bg-slate-50/80 hover:border-blue-300 transition-all group shadow-xs hover:shadow-sm"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-base sm:text-lg font-bold text-[#17243A] group-hover:text-blue-700 transition-colors">
                2. 模型比較與特徵消融
              </span>
              <ArrowRight className="w-4 h-4 text-[#64748B] group-hover:text-blue-700 group-hover:translate-x-1 transition-all" />
            </div>
            <p className="text-sm text-[#334155] leading-relaxed">
              比較 M1 (OLS)、M2 (RFE+OLS)、M3 (Ridge)、M4 (Lasso) 在擴展視窗 CV 下的穩定度與群組消融分析。
            </p>
          </Link>

          <Link
            href="/evaluation"
            className="p-5 rounded-xl border border-[#DFE5ED] bg-white hover:bg-slate-50/80 hover:border-blue-300 transition-all group shadow-xs hover:shadow-sm"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-base sm:text-lg font-bold text-[#17243A] group-hover:text-blue-700 transition-colors">
                3. 最終測試集評估
              </span>
              <ArrowRight className="w-4 h-4 text-[#64748B] group-hover:text-blue-700 group-hover:translate-x-1 transition-all" />
            </div>
            <p className="text-sm text-[#334155] leading-relaxed">
              互動式實際值 vs 預測值散佈圖、7 天共形區間時序軸、殘差分佈與 24 小時覆蓋率全幅診斷。
            </p>
          </Link>

          <Link
            href="/predict"
            className="p-5 rounded-xl border border-[#DFE5ED] bg-white hover:bg-slate-50/80 hover:border-blue-300 transition-all group shadow-xs hover:shadow-sm"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-base sm:text-lg font-bold text-[#17243A] group-hover:text-blue-700 transition-colors">
                4. 互動式模型預測引擎
              </span>
              <ArrowRight className="w-4 h-4 text-[#64748B] group-hover:text-blue-700 group-hover:translate-x-1 transition-all" />
            </div>
            <p className="text-sm text-[#334155] leading-relaxed">
              以真實 M2 權重與校準分位數進行 12 項特徵即時推論，提供連續值、非負截斷與 95% 預測區間。
            </p>
          </Link>

          <Link
            href="/methodology"
            className="p-5 rounded-xl border border-[#DFE5ED] bg-white hover:bg-slate-50/80 hover:border-blue-300 transition-all group sm:col-span-2 lg:col-span-2 shadow-xs hover:shadow-sm"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-base sm:text-lg font-bold text-[#17243A] group-hover:text-blue-700 transition-colors">
                5. 完整 CRISP-DM 研究方法論
              </span>
              <ArrowRight className="w-4 h-4 text-[#64748B] group-hover:text-blue-700 group-hover:translate-x-1 transition-all" />
            </div>
            <p className="text-sm text-[#334155] leading-relaxed">
              詳解業務理解、資料準備（防洩漏與單熱編碼）、時序擴展視窗 CV、RFE 特徵淘汰機制、分裂共形數學校準、交換性前提與論文參考文獻。
            </p>
          </Link>
        </div>
      </section>
    </div>
  );
}
