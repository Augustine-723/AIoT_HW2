import Image from "next/image";
import { 
  Database, 
  Calendar, 
  AlertTriangle, 
  Clock, 
  Info
} from "lucide-react";
import { phase3ExperimentMetadata } from "@/lib/data";

export default function DatasetPage() {
  const splitBoundaries = phase3ExperimentMetadata.exact_date_boundaries;
  const trainBound = splitBoundaries.training;
  const calBound = splitBoundaries.calibration;
  const testBound = splitBoundaries.test;
  const totalObs = trainBound.observations + calBound.observations + testBound.observations;

  const featureDefinitions = [
    { name: "instant", type: "Index", role: "排除", desc: "資料流水號，無預測價值" },
    { name: "dteday", type: "Date", role: "切分邊界", desc: "日期標記 (YYYY-MM-DD)，用於嚴格按日期切分 Expanding Window CV" },
    { name: "season", type: "Categorical", role: "預測特徵 (4類)", desc: "四季：1:春季 (Spring), 2:夏季 (Summer), 3:秋季 (Fall), 4:冬季 (Winter)" },
    { name: "yr", type: "Binary", role: "預測特徵 (2類)", desc: "年份指標：0: 2011 年, 1: 2012 年 (反映系統擴張趨勢，不可外推)" },
    { name: "mnth", type: "Categorical", role: "預測特徵 (12類)", desc: "月份 (1 到 12 月)，捕捉季節交替之次要氣候模式" },
    { name: "hr", type: "Categorical", role: "預測特徵 (24類)", desc: "小時時段 (0 到 23 時)，主導通勤雙峰與夜間平谷核心模式" },
    { name: "holiday", type: "Binary", role: "預測特徵 (2類)", desc: "國定假日 (0: 否, 1: 是)，依據華盛頓特區法定公眾假期標註" },
    { name: "weekday", type: "Categorical", role: "預測特徵 (7類)", desc: "星期 (0: 週日, 1: 週一, ..., 6: 週六)" },
    { name: "workingday", type: "Binary", role: "預測特徵 (2類)", desc: "工作日標記 (1: 既非週末亦非假日, 0: 其餘非工作日)" },
    { name: "weathersit", type: "Categorical", role: "預測特徵 (4類)", desc: "天候狀況代碼：1:晴朗/少雲, 2:薄霧/多雲, 3:小雨/小雪, 4:大豪雨/暴風雪" },
    { name: "temp", type: "Numeric", role: "預測特徵 (連續)", desc: "正規化攝氏氣溫：temp = t / 41，其中 t 為攝氏溫度 (°C，有效範圍 0.00 ~ 1.00，例如 0.50 對應約 20.5°C)" },
    { name: "atemp", type: "Numeric", role: "預測特徵 (連續)", desc: "正規化攝氏體感溫度：atemp = at / 50，其中 at 為攝氏體感溫度 (°C，有效範圍 0.00 ~ 1.00，例如 0.50 對應約 25.0°C)" },
    { name: "hum", type: "Numeric", role: "預測特徵 (連續)", desc: "相對濕度經正規化：hum / 100，有效範圍介於 0.00 ~ 1.00" },
    { name: "windspeed", type: "Numeric", role: "預測特徵 (連續)", desc: "風速經正規化：windspeed / 67 (km/h)，有效範圍介於 0.00 ~ 1.00" },
    { name: "casual", type: "Target Component", role: "嚴格排除 (目標洩漏)", desc: "臨時租借會員使用量。因 casual + registered = cnt，測試時不可得，納入將導致嚴重 Data Leakage" },
    { name: "registered", type: "Target Component", role: "嚴格排除 (目標洩漏)", desc: "註冊註記會員使用量。同上，與 casual 加總即為真實目標值，必須在建模前完全剔除" },
    { name: "cnt", type: "Numeric", role: "預測目標 (Target)", desc: "每小時總租借量 (casual + registered)，為本研究預測之連續目標變數" },
  ];

  return (
    <div className="space-y-10">
      {/* Page Header */}
      <section className="space-y-3">
        <div className="flex items-center gap-2 text-xs font-mono text-cyan-400">
          <Database className="w-4 h-4" />
          <span>DATASET & EXPLORATORY DATA ANALYSIS</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          研究資料集結構、探索性分析 (EDA) 與時序劃分
        </h1>
        <p className="text-sm text-slate-300 max-w-4xl leading-relaxed">
          深入探討 Capital Bikeshare 每小時資料集之統計特徵、雙峰分佈樣態、氣候變數相關性，
          並落實嚴謹防止資料洩漏之特徵剔除機制與時序擴展劃分策略。
        </p>
      </section>

      {/* Dataset Overview Cards */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-1">
          <span className="text-xs text-slate-400">資料集來源</span>
          <div className="text-lg font-bold text-white">UCI ML Repository</div>
          <p className="text-[11px] text-slate-400">Capital Bikeshare, Washington D.C.</p>
        </div>

        <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-1">
          <span className="text-xs text-slate-400">有效觀測筆數</span>
          <div className="text-lg font-bold text-white font-mono">17,379 筆</div>
          <p className="text-[11px] text-slate-400">完整跨越 2011/01/01 ~ 2012/12/31</p>
        </div>

        <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-1">
          <span className="text-xs text-slate-400">原始輸入預測因子</span>
          <div className="text-lg font-bold text-cyan-400 font-mono">12 項可用特徵</div>
          <p className="text-[11px] text-slate-400">涵蓋曆法、時段與正規化氣象變數</p>
        </div>

        <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-1">
          <span className="text-xs text-slate-400">缺漏小時審計</span>
          <div className="text-lg font-bold text-amber-400 font-mono">165 小時</div>
          <p className="text-[11px] text-slate-400">為系統連線/維護中斷，非需求為零</p>
        </div>
      </section>

      {/* Strict Leakage Prevention Alert */}
      <section className="p-5 rounded-2xl border border-amber-500/30 bg-amber-950/20 text-slate-200 space-y-3">
        <div className="flex items-center gap-2 text-amber-400 font-semibold text-sm">
          <AlertTriangle className="w-5 h-5 flex-shrink-0" />
          <span>嚴格防範目標洩漏 (Target Leakage Prevention)</span>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed">
          原始資料集中包含 <code className="px-1.5 py-0.5 rounded bg-slate-900 font-mono text-amber-300">casual</code> (臨時散客借車數) 與 <code className="px-1.5 py-0.5 rounded bg-slate-900 font-mono text-amber-300">registered</code> (註冊會員借車數)。
          由於數學恆等式：
          <span className="block my-2 text-center font-mono text-sm text-cyan-300 bg-slate-900/80 py-1.5 rounded-lg border border-slate-800">
            cnt = casual + registered
          </span>
          在實際營運與未來的任何時點，營運商在尚未結算之前<strong>絕對不可能預先得知</strong>散客與會員的分項數值。若將這兩個欄位納入模型訓練，迴歸模型將學會簡單將兩者相加，造成 R² = 1.0 的<strong>偽完美預測（Data Leakage）</strong>。本研究在進入任何特徵工程與交叉驗證前，已徹底自特徵矩陣中剔除該二變數。
        </p>
      </section>

      {/* Chronological Three-Split Architecture */}
      <section className="space-y-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Calendar className="w-5 h-5 text-indigo-400" />
            <span>三階段時序切分架構 (Chronological Split Boundaries)</span>
          </h2>
          <p className="text-xs text-slate-400">
            為了杜絕未來資訊洩漏 (Data Snooping)，所有切分均以自然日期邊界為依據，嚴禁進行時序洗牌 (No Shuffling)。
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-5 rounded-2xl border border-indigo-500/30 bg-indigo-950/20 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-mono">
                Training Set (訓練集)
              </span>
              <span className="text-xs font-mono text-slate-400">
                {((trainBound.observations / totalObs) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="text-2xl font-bold font-mono text-white">
              {trainBound.observations.toLocaleString()} 筆
            </div>
            <div className="text-xs space-y-1 text-slate-300">
              <div className="flex justify-between">
                <span className="text-slate-400">起始日期</span>
                <span className="font-mono">{trainBound.date_range[0]} 00:00</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">截止日期</span>
                <span className="font-mono">{trainBound.date_range[1]} 23:00</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">自然天數</span>
                <span className="font-mono">{trainBound.calendar_days} 天</span>
              </div>
              <div className="flex justify-between pt-1 border-t border-slate-800">
                <span className="text-slate-400">主要職責</span>
                <span className="text-indigo-300">5-Fold 擴展視窗 CV、RFE 特徵淘汰與模型擬合</span>
              </div>
            </div>
          </div>

          <div className="p-5 rounded-2xl border border-cyan-500/30 bg-cyan-950/20 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 font-mono">
                Calibration Set (校準集)
              </span>
              <span className="text-xs font-mono text-slate-400">
                {((calBound.observations / totalObs) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="text-2xl font-bold font-mono text-white">
              {calBound.observations.toLocaleString()} 筆
            </div>
            <div className="text-xs space-y-1 text-slate-300">
              <div className="flex justify-between">
                <span className="text-slate-400">起始日期</span>
                <span className="font-mono">{calBound.date_range[0]} 00:00</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">截止日期</span>
                <span className="font-mono">{calBound.date_range[1]} 23:00</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">自然天數</span>
                <span className="font-mono">{calBound.calendar_days} 天</span>
              </div>
              <div className="flex justify-between pt-1 border-t border-slate-800">
                <span className="text-slate-400">主要職責</span>
                <span className="text-cyan-300">共形預測非符合性分數校準 (Conformal q = 302.60)</span>
              </div>
            </div>
          </div>

          <div className="p-5 rounded-2xl border border-emerald-500/30 bg-emerald-950/20 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-mono">
                Test Set (最終測試集)
              </span>
              <span className="text-xs font-mono text-slate-400">
                {((testBound.observations / totalObs) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="text-2xl font-bold font-mono text-emerald-400">
              {testBound.observations.toLocaleString()} 筆
            </div>
            <div className="text-xs space-y-1 text-slate-300">
              <div className="flex justify-between">
                <span className="text-slate-400">起始日期</span>
                <span className="font-mono">{testBound.date_range[0]} 00:00</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">截止日期</span>
                <span className="font-mono">{testBound.date_range[1]} 23:00</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">自然天數</span>
                <span className="font-mono">{testBound.calendar_days} 天</span>
              </div>
              <div className="flex justify-between pt-1 border-t border-slate-800">
                <span className="text-slate-400">主要職責</span>
                <span className="text-emerald-300">最終鎖定之未見測試集獨立泛化評估</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* EDA Visualizations Grid */}
      <section className="space-y-6">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Clock className="w-5 h-5 text-indigo-400" />
            <span>探索性資料分析核心圖表 (Phase 1 EDA Visualizations)</span>
          </h2>
          <p className="text-xs text-slate-400">
            呈現資料集內在時段規律、右偏目標分佈、氣候相關矩陣以及氣溫與需求的關聯性。
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* 1. Hourly Demand */}
          <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-white">1. 時段平均需求分佈 (Hourly Demand Distribution)</h3>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">Figure: hourly_demand.png</span>
            </div>
            <div className="relative aspect-[16/10] w-full rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
              <Image
                src="/figures/hourly_demand.png"
                alt="Hourly Bike Sharing Demand"
                fill
                className="object-contain"
                sizes="(max-width: 1024px) 100vw, 50vw"
              />
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              <strong>觀察：</strong>時段呈现極為鮮明的<strong>通勤雙峰（Bimodal Pattern）</strong>：早尖峰落在 08:00（平均需求約 360 輛），晚尖峰落在 17:00–18:00（平均需求突破 460 輛）。深夜 01:00–05:00 需求降至 10~30 輛的極低基準。此現象證明時段（hr）為整個預測模型中最重要的主導因子。
            </p>
          </div>

          {/* 2. Target Distribution */}
          <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-white">2. 目標變數分佈 (Target Count Distribution)</h3>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">Figure: target_distribution.png</span>
            </div>
            <div className="relative aspect-[16/10] w-full rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
              <Image
                src="/figures/target_distribution.png"
                alt="Target Distribution"
                fill
                className="object-contain"
                sizes="(max-width: 1024px) 100vw, 50vw"
              />
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              <strong>觀察：</strong>每小時總租借量 <code className="text-cyan-400 font-mono">cnt</code> 呈現強烈<strong>右偏態分佈（Right-skewed, positive skewness）</strong>，中位數約 142 輛，且有大量低需求夜間與惡劣天候樣本，極端高值可達 977 輛。由於線性迴歸假設加性誤差，此右偏結構在極端尖峰時段容易出現預測下凹或夜間預測值偶爾小於零的現象。
            </p>
          </div>

          {/* 3. Correlation Heatmap */}
          <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-white">3. 特徵相關性熱圖 (Correlation Heatmap)</h3>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">Figure: correlation_heatmap.png</span>
            </div>
            <div className="relative aspect-[16/10] w-full rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
              <Image
                src="/figures/correlation_heatmap.png"
                alt="Correlation Heatmap"
                fill
                className="object-contain"
                sizes="(max-width: 1024px) 100vw, 50vw"
              />
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              <strong>觀察：</strong>氣溫 <code className="text-cyan-400 font-mono">temp</code> 與體感溫度 <code className="text-cyan-400 font-mono">atemp</code> 呈現極高的共線性（相關係數高達 <strong>0.99</strong>），在傳統未正則化的 OLS 模型中容易造成係數方差膨脹。此外，濕度（hum）與需求呈負相關 (-0.32)，風速與天候代碼亦呈現統計抑制效果。
            </p>
          </div>

          {/* 4. Temperature vs Demand */}
          <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-white">4. 氣溫與租借需求關係 (Temperature vs Demand)</h3>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">Figure: temperature_vs_demand.png</span>
            </div>
            <div className="relative aspect-[16/10] w-full rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
              <Image
                src="/figures/temperature_vs_demand.png"
                alt="Temperature vs Demand"
                fill
                className="object-contain"
                sizes="(max-width: 1024px) 100vw, 50vw"
              />
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              <strong>觀察：</strong>散佈圖與擬合趨勢顯示在低溫至適溫區間（0°C ~ 25°C），需求隨氣溫升高呈現顯著正向成長；然而當氣溫過高（&gt; 35°C）時，高溫曝曬反而使部分騎乘意願趨緩，顯示氣溫影響具備輕微非線性特徵。
            </p>
          </div>
        </div>
      </section>

      {/* Feature Definitions Table */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Info className="w-5 h-5 text-cyan-400" />
            <span>特徵欄位定義清單 (Feature Dictionary & Transformation Reference)</span>
          </h2>
          <span className="text-xs text-slate-400 font-mono">17 Total Fields</span>
        </div>

        <div className="overflow-x-auto rounded-2xl border border-slate-800 bg-slate-900/40">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/80 text-slate-400 font-mono">
                <th className="py-3 px-4 font-semibold">特徵名稱</th>
                <th className="py-3 px-4 font-semibold">型態</th>
                <th className="py-3 px-4 font-semibold">角色/用途</th>
                <th className="py-3 px-4 font-semibold">數值範圍與詳細規格說明</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-sans">
              {featureDefinitions.map((feat, idx) => (
                <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                  <td className="py-3 px-4 font-mono font-bold text-white">
                    {feat.name}
                  </td>
                  <td className="py-3 px-4 font-mono text-slate-400">
                    {feat.type}
                  </td>
                  <td className="py-3 px-4">
                    <span className={`px-2 py-0.5 rounded text-[11px] font-mono ${
                      feat.role.includes("排除")
                        ? "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                        : feat.role.includes("目標")
                        ? "bg-cyan-500/10 text-cyan-400 border border-cyan-500/20"
                        : "bg-indigo-500/10 text-indigo-300 border border-indigo-500/20"
                    }`}>
                      {feat.role}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-slate-300 leading-relaxed">
                    {feat.desc}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
