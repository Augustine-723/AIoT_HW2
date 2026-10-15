import { Code2, ExternalLink, ShieldCheck } from "lucide-react";

export default function Footer() {
  return (
    <footer className="w-full border-t border-slate-800/80 bg-slate-950 py-10 mt-auto text-slate-400">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Research Summary */}
          <div className="md:col-span-2 space-y-3">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-slate-100 text-sm">
                AIoT HW2 研究儀表板與預測系統
              </span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                Phase 4 Deployed
              </span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed max-w-lg">
              本研究基於 UCI Machine Learning Repository 之 Capital Bikeshare 每小時資料集 (17,379 筆)，
              採用時間序列擴展視窗交叉驗證 (Expanding Window CV)、遞歸特徵消除 (RFE)、
              以及分裂共形預測 (Split Conformal Prediction) 建構具備分佈自由 (Distribution-Free) 邊際覆蓋保證之每小時單車需求推論模型。
            </p>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2 pt-1 text-[11px] font-mono text-slate-400">
              <span>學號: <strong className="text-slate-300">7115093001</strong></span>
              <span>·</span>
              <span>實驗識別: <strong className="text-cyan-400">AIoT_HW2_PHASE3_EXP01</strong></span>
              <span>·</span>
              <span>選定模型: <strong className="text-indigo-300">M2 (RFE + OLS, 30 Features)</strong></span>
            </div>
          </div>

          {/* Academic Attributions & References */}
          <div className="space-y-2">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
              學術引用與來源
            </h4>
            <ul className="space-y-1.5 text-xs">
              <li>
                <a
                  href="https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-slate-200 flex items-center gap-1 transition-colors"
                >
                  <span>UCI Bike Sharing Dataset</span>
                  <ExternalLink className="w-3 h-3 text-slate-400" />
                </a>
              </li>
              <li>
                <span className="text-slate-400">Fanaee-T & Gama (2014)</span>
              </li>
              <li>
                <span className="text-slate-400">Vovk et al. (2005) Conformal</span>
              </li>
              <li>
                <span className="text-slate-400">Angelopoulos & Bates (2021)</span>
              </li>
            </ul>
          </div>

          {/* Links & Repository Isolation Notice */}
          <div className="space-y-2">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
              專案資源與隔離
            </h4>
            <ul className="space-y-1.5 text-xs">
              <li>
                <a
                  href="https://github.com/Augustine-723/AIoT_HW2.git"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-slate-200 flex items-center gap-1.5 transition-colors"
                >
                  <Code2 className="w-3.5 h-3.5" />
                  <span>GitHub Repository</span>
                </a>
              </li>
              <li className="pt-2 text-[11px] text-slate-400 leading-tight">
                <span className="inline-block px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700 mr-1">
                  嚴格隔離
                </span>
                本專案為獨立 AIoT HW2 提交，與先前的 50 Startups 作業完全隔離，未改動任何既有專案與部署。
              </li>
            </ul>
          </div>
        </div>

        <div className="pt-6 border-t border-slate-900 flex flex-col sm:flex-row items-center justify-between gap-4 text-[11px] text-slate-400">
          <div>
            © 2026 AIoT Course Research Dashboard. Student ID: 7115093001. All rights reserved.
          </div>
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-500/80" />
            <span>所有呈現數值皆來自凍結實驗結果 (Frozen Artifacts)，無隨機模擬假造數據。</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
