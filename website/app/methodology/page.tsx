import { 
  BookOpen, 
  GitBranch, 
  ShieldCheck, 
  CheckCircle2, 
  AlertTriangle, 
  Layers, 
  ExternalLink,
  ArrowRight
} from "lucide-react";

export default function MethodologyPage() {
  const crispStages = [
    {
      step: "01",
      name: "Business Understanding (業務理解)",
      desc: "界定都市共享單車動態調度問題。車輛供需失衡會導致無車可借或無位可還，亟需在保護歷史邊界下精準推估未來每小時租借量，並量化其預測不確定性以進行調度風險管理。",
    },
    {
      step: "02",
      name: "Data Understanding (資料理解)",
      desc: "審查 UCI Capital Bikeshare 17,379 筆逐時紀錄。探討時段雙峰效應、目標右偏態分佈、氣溫體感共線性，並完成 165 筆缺漏小時系統審計，證實缺漏源於站點通訊維護而非需求歸零。",
    },
    {
      step: "03",
      name: "Data Preparation (資料準備)",
      desc: "徹底剔除 casual 與 registered 防止致命目標洩漏。落實無未來資訊洩漏之時序三階段切分（訓練 12,163 筆、校準 2,640 筆、測試 2,576 筆），並採用 drop='first' 之一熱編碼消除虛擬變數陷阱。",
    },
    {
      step: "04",
      name: "Modeling (模型建構)",
      desc: "以 Expanding Window Cross-Validation (5 摺) 評估候選模型。運用 RFE 自 53 個候選編碼特徵中淘汰 23 個低貢獻特徵收斂至 M2 (30 特徵)，並與 L2 正則化 Ridge (α=1.0) 及 L1 稀疏化 Lasso (α=0.1) 進行交叉評比。",
    },
    {
      step: "05",
      name: "Evaluation (模型評估)",
      desc: "在鎖定之未見測試集評估泛化性能 (Test MAE 97.52, RMSE 133.21, R² 0.6112)。導入分裂共形預測 (Split Conformal) 計算有限樣本分位數 q=302.60，達成 94.84% 實證覆蓋率並診斷尖峰時段條件覆蓋弱點。",
    },
    {
      step: "06",
      name: "Deployment (系統部署)",
      desc: "經數值對齊驗證確立 Option B（純 TypeScript 確定性推論架構），達成 2.27 × 10⁻¹³ 浮點全等精度，打造無伺服器 Cold Start、具備互動式情境模擬與研究視覺化之獨立學術研究儀表板。",
    },
  ];

  return (
    <div className="space-y-12">
      {/* Header */}
      <section className="space-y-3">
        <div className="flex items-center gap-2 text-xs font-mono font-semibold text-blue-800 bg-blue-50 border border-blue-200 px-3 py-1 rounded-full w-fit">
          <BookOpen className="w-4 h-4 text-blue-600" />
          <span>RESEARCH METHODOLOGY &amp; STATISTICAL FRAMEWORK</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-[#17243A] tracking-tight">
          研究方法論、CRISP-DM 流程與共形統計理論
        </h1>
        <p className="text-sm sm:text-base text-[#334155] max-w-4xl leading-relaxed">
          全面解析本研究自問題定義、防洩漏資料工程、擴展視窗交叉驗證、遞歸特徵消除、
          正則化超參數調校至傳統分裂共形預測（Split Conformal Prediction）之嚴謹學術流程與數學基礎。
        </p>
      </section>

      {/* Research Pipeline Diagram */}
      <section className="p-6 rounded-2xl border border-[#DFE5ED] bg-white shadow-xs space-y-4">
        <h2 className="text-base sm:text-lg font-bold text-[#17243A] flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-blue-600" />
          <span>端到端研究執行流程管線 (End-to-End Research Pipeline)</span>
        </h2>
        <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-[#334155]">
          <span className="px-3 py-1.5 rounded-lg bg-slate-100 text-[#17243A] border border-[#CBD5E1] font-semibold">
            UCI Dataset (17,379)
          </span>
          <ArrowRight className="w-3.5 h-3.5 text-[#64748B]" />
          <span className="px-3 py-1.5 rounded-lg bg-slate-100 text-[#17243A] border border-[#CBD5E1] font-semibold">
            EDA &amp; Missing Audit
          </span>
          <ArrowRight className="w-3.5 h-3.5 text-[#64748B]" />
          <span className="px-3 py-1.5 rounded-lg bg-amber-50 text-amber-900 border border-amber-200 font-semibold">
            Leakage Prevention
          </span>
          <ArrowRight className="w-3.5 h-3.5 text-[#64748B]" />
          <span className="px-3 py-1.5 rounded-lg bg-blue-50 text-blue-800 border border-blue-200 font-semibold">
            Chronological Split
          </span>
          <ArrowRight className="w-3.5 h-3.5 text-[#64748B]" />
          <span className="px-3 py-1.5 rounded-lg bg-slate-100 text-[#17243A] border border-[#CBD5E1] font-semibold">
            Expanding Window CV
          </span>
          <ArrowRight className="w-3.5 h-3.5 text-[#64748B]" />
          <span className="px-3 py-1.5 rounded-lg bg-blue-100 text-blue-900 border border-blue-300 font-bold">
            M2 (RFE 30 Features)
          </span>
          <ArrowRight className="w-3.5 h-3.5 text-[#64748B]" />
          <span className="px-3 py-1.5 rounded-lg bg-teal-50 text-teal-800 border border-teal-200 font-semibold">
            Conformal Calibration
          </span>
          <ArrowRight className="w-3.5 h-3.5 text-[#64748B]" />
          <span className="px-3 py-1.5 rounded-lg bg-emerald-50 text-emerald-800 border border-emerald-200 font-bold">
            Option B Deployed
          </span>
        </div>
      </section>

      {/* CRISP-DM 6 Stages Grid */}
      <section className="space-y-4">
        <h2 className="text-xl sm:text-2xl font-bold text-[#17243A] flex items-center gap-2">
          <Layers className="w-5 h-5 text-blue-600" />
          <span>CRISP-DM 六大階段完整落實架構</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {crispStages.map((stage) => (
            <div
              key={stage.step}
              className="p-6 rounded-2xl border border-[#DFE5ED] bg-white hover:border-blue-300 shadow-xs hover:shadow-sm transition-all space-y-2.5"
            >
              <div className="flex items-center justify-between">
                <span className="text-2xl font-black font-mono text-blue-600">
                  {stage.step}
                </span>
                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
              </div>
              <h3 className="text-base font-bold text-[#17243A]">
                {stage.name}
              </h3>
              <p className="text-sm text-[#334155] leading-relaxed">
                {stage.desc}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Split Conformal Prediction Theory Card */}
      <section className="p-6 rounded-2xl border border-blue-200 bg-blue-50/40 shadow-xs space-y-4">
        <div className="flex items-center gap-2 text-blue-900 font-bold text-base sm:text-lg">
          <ShieldCheck className="w-5 h-5 text-blue-700" />
          <span>分裂共形預測數學理論 (Split Conformal Prediction Framework)</span>
        </div>

        <p className="text-xs sm:text-sm text-[#334155] leading-relaxed">
          為克服傳統線性迴歸常態誤差假設在右偏計數資料中的失真，本研究導入歸納式分裂共形預測（Inductive / Split Conformal Prediction）。
          該方法在非參數條件下提供具備<strong>有限樣本邊際有效性（Finite-sample Marginal Validity）</strong>之預測區間：
        </p>

        {/* Mathematical Formulas */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono text-xs">
          <div className="p-4 rounded-xl bg-white border border-blue-200 shadow-2xs space-y-1">
            <span className="text-xs text-[#475569] block font-sans font-semibold">1. 非符合性分數 (Non-conformity Score)</span>
            <div className="text-base text-blue-800 py-1 font-bold">
              s_i = | y_i - ŷ_i |
            </div>
            <p className="text-xs text-[#475569] font-sans leading-normal">
              以校準集（Calibration Set, n=2,640）上擬合模型之絕對殘差衡量預測難度。
            </p>
          </div>

          <div className="p-4 rounded-xl bg-white border border-blue-200 shadow-2xs space-y-1">
            <span className="text-xs text-[#475569] block font-sans font-semibold">2. 有限樣本經驗分位數 (Finite-sample Quantile)</span>
            <div className="text-base text-blue-800 py-1 font-bold">
              p = ⌈ (n + 1)(1 - α) ⌉ / n
            </div>
            <p className="text-xs text-[#475569] font-sans leading-normal">
              在 α = 0.05 下，取第 95.04% 分位數以校正有限樣本邊界，得 q = 302.6015。
            </p>
          </div>

          <div className="p-4 rounded-xl bg-white border border-blue-200 shadow-2xs space-y-1">
            <span className="text-xs text-[#475569] block font-sans font-semibold">3. 預測區間與非負截斷 (Prediction Interval)</span>
            <div className="text-base text-emerald-800 py-1 font-bold">
              C(X) = [ max(0, ŷ - q), ŷ + q ]
            </div>
            <p className="text-xs text-[#475569] font-sans leading-normal">
              原始區間為對稱加減 q，為符合物理租借計數非負約束，下界截斷至 0。
            </p>
          </div>
        </div>
      </section>

      {/* Critical Statistical Limitations & Exchangeability */}
      <section className="p-6 rounded-2xl border border-amber-200 bg-amber-50 shadow-xs space-y-3">
        <div className="flex items-center gap-2 text-amber-900 font-bold text-sm sm:text-base">
          <AlertTriangle className="w-5 h-5 flex-shrink-0" />
          <span>統計限制：可交換性前提、時間序列依賴與條件覆蓋率差異</span>
        </div>

        <div className="space-y-3 text-xs sm:text-sm text-amber-900/90 leading-relaxed">
          <p>
            1. <strong>可交換性前提 (Exchangeability Assumption)：</strong>
            傳統分裂共形預測 (Split Conformal Prediction) 之有限樣本邊際覆蓋率保證嚴格仰賴校準集與測試集之觀測值具備<strong>可交換性（Exchangeability）</strong>假設。
            本歷史每小時時間序列資料具備顯著的時間相依性（Temporal Dependence）與自相關結構。
            因此，<strong>95% 為名義目標覆蓋率 (Nominal Target Coverage)</strong>，實證觀察之測試集覆蓋率約為 <strong>94.84%</strong>。
            各時段之條件覆蓋率存在顯著差異，歷史實證觀察成果不構成對未來時間區段之無條件保證（Unconditional Guarantee）。
          </p>

          <p>
            2. <strong>邊際覆蓋 vs 條件覆蓋 (Marginal vs. Conditional Coverage)：</strong>
            邊際覆蓋保證的是全母體期望值，但無法約束局部子空間的條件覆蓋率。
            實證顯示：在 <strong>08:00（早通勤峰，覆蓋率 64.49%）</strong>與 <strong>17:00–18:00（傍晚通勤峰，覆蓋率 75.00%~80.56%）</strong>，
            因人潮集中與突發氣候交互影響，單車需求變異急劇放大，線性模型產生較大之殘差，造成該特定時段之預測區間逸脫率顯著高於平均。
          </p>
        </div>
      </section>

      {/* Academic Citations & Bibliography */}
      <section className="p-6 rounded-2xl border border-[#DFE5ED] bg-white shadow-xs space-y-4">
        <h2 className="text-base sm:text-lg font-bold text-[#17243A] flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-blue-600" />
          <span>正式學術文獻引用清單 (Academic References &amp; Attribution)</span>
        </h2>

        <div className="space-y-4 text-sm text-[#334155] leading-relaxed divide-y divide-[#DFE5ED]">
          <div className="pt-2">
            <p className="text-[#17243A] font-bold text-sm sm:text-base">
              [1] UCI Machine Learning Repository: Bike Sharing Dataset
            </p>
            <p className="text-sm text-[#334155] mt-1">
              Fanaee-T, H., &amp; Gama, J. (2014). Event labeling combining ensemble detectors and background knowledge. <em>Progress in Artificial Intelligence</em>, 2(2-4), 113-127.
              <a
                href="https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
                target="_blank"
                rel="noopener noreferrer"
                className="bib-link text-blue-700 hover:underline inline-flex items-center gap-1 ml-2 font-medium"
              >
                <span>[Link]</span>
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </p>
          </div>

          <div className="pt-3">
            <p className="text-[#17243A] font-bold text-sm sm:text-base">
              [2] Algorithmic Learning in a Random World (Conformal Prediction Foundations)
            </p>
            <p className="text-sm text-[#334155] mt-1">
              Vovk, V., Gammerman, A., &amp; Shafer, G. (2005). <em>Algorithmic Learning in a Random World</em>. Springer Science &amp; Business Media.
            </p>
          </div>

          <div className="pt-3">
            <p className="text-[#17243A] font-bold text-sm sm:text-base">
              [3] Distribution-Free Uncertainty Quantification &amp; Split Conformal Prediction
            </p>
            <p className="text-sm text-[#334155] mt-1">
              Angelopoulos, A. N., &amp; Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. <em>arXiv preprint arXiv:2107.07511</em>.
            </p>
          </div>

          <div className="pt-3">
            <p className="text-[#17243A] font-bold text-sm sm:text-base">
              [4] Scikit-Learn: Machine Learning in Python
            </p>
            <p className="text-sm text-[#334155] mt-1">
              Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... &amp; Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. <em>Journal of Machine Learning Research</em>, 12, 2825-2830.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
