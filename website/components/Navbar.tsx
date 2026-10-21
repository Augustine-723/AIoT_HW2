"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";
import { 
  BarChart3, 
  Database, 
  Sliders, 
  LineChart, 
  Sparkles, 
  BookOpen, 
  Menu, 
  X, 
  CheckCircle2, 
  Cpu,
  Layers,
  Zap,
  FolderPlus
} from "lucide-react";

const P1_NAV_LINKS = [
  { href: "/", label: "研究總覽", enLabel: "Overview", icon: BarChart3 },
  { href: "/dataset", label: "資料與 EDA", enLabel: "Dataset & EDA", icon: Database },
  { href: "/models", label: "模型比較", enLabel: "Model Comparison", icon: Sliders },
  { href: "/evaluation", label: "最終評估", enLabel: "Final Evaluation", icon: LineChart },
  { href: "/predict", label: "互動預測", enLabel: "Interactive Prediction", icon: Sparkles },
  { href: "/methodology", label: "研究方法論", enLabel: "Methodology", icon: BookOpen },
];

const P2_NAV_LINKS = [
  { href: "/problem2#overview", label: "P2 資料與審計", enLabel: "Dataset & Audit", icon: Database },
  { href: "/problem2#crisp-dm", label: "CRISP-DM 歷程", enLabel: "CRISP-DM", icon: Layers },
  { href: "/problem2#features", label: "RFE 特徵淘汰", enLabel: "Feature Selection", icon: Sliders },
  { href: "/problem2#models", label: "模型比較 (CV)", enLabel: "Model Comparison", icon: BarChart3 },
  { href: "/problem2#evaluation", label: "測試集與共形區間", enLabel: "Conformal PI", icon: LineChart },
  { href: "/problem2#predict", label: "26特徵互動預測", enLabel: "Interactive Predict", icon: Zap },
];

const CUSTOM_NAV_LINKS = [
  { href: "/custom-dataset#upload", label: "資料匯入 (URL/本機)", enLabel: "Data Ingestion", icon: Database },
  { href: "/custom-dataset#model", label: "動態線性建模", enLabel: "Universal MLR", icon: Sliders },
  { href: "/custom-dataset#eval", label: "測試評估與共形區間", enLabel: "Evaluation & PI", icon: LineChart },
];

export default function Navbar() {
  const pathname = usePathname();
  const router = useRouter();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  // Determine active problem from pathname
  const isProblem2 = pathname.startsWith("/problem2");
  const isCustom = pathname.startsWith("/custom-dataset");
  const currentProblem = isProblem2 ? "p2" : isCustom ? "custom" : "p1";

  const expId = isProblem2 ? "AIoT_HW2_P2_EXP01" : isCustom ? "AIoT_HW2_BONUS_EXP" : "AIoT_HW2_PHASE3_EXP01";
  const modelTag = isProblem2 ? "M3 · Ridge (p=29)" : isCustom ? "Universal · MLR" : "M2 · RFE (k=30)";
  const problemSubtitle = isProblem2 
    ? "Appliances Energy Prediction (10-min Wh)" 
    : isCustom 
    ? "Universal MLR Playground (+2 Bonus)" 
    : "Bike Sharing Demand Prediction (Hourly)";

  const copyExpId = () => {
    navigator.clipboard.writeText(expId);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleProblemChange = (val: string) => {
    if (val === "p1") router.push("/");
    else if (val === "p2") router.push("/problem2");
    else if (val === "custom") router.push("/custom-dataset");
  };

  const activeNavLinks = isProblem2 ? P2_NAV_LINKS : isCustom ? CUSTOM_NAV_LINKS : P1_NAV_LINKS;

  return (
    <header className="sticky top-0 z-50 w-full border-b border-[#DFE5ED] bg-white/95 backdrop-blur-md shadow-xs print-hide">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Logo & Title */}
          <div className="flex items-center gap-3">
            <Link href={isProblem2 ? "/problem2" : isCustom ? "/custom-dataset" : "/"} className="flex items-center gap-2.5 group">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-teal-500 p-[1px] shadow-sm">
                <div className="w-full h-full bg-white rounded-[11px] flex items-center justify-center group-hover:bg-blue-50/50 transition-colors">
                  <Cpu className="w-5 h-5 text-blue-600 group-hover:rotate-12 transition-transform duration-300" />
                </div>
              </div>
              <div className="flex flex-col">
                <div className="flex items-center gap-2">
                  <span className="font-extrabold text-[#17243A] tracking-tight text-base sm:text-lg">
                    AIoT HW2
                  </span>
                  <span className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200">
                    {modelTag}
                  </span>
                </div>
                <span className="text-xs text-[#475569] hidden sm:block truncate max-w-xs font-medium">
                  {problemSubtitle}
                </span>
              </div>
            </Link>
          </div>

          {/* Genuine Problem Selector Dropdown */}
          <div className="flex items-center gap-2">
            <label htmlFor="problem-selector" className="text-xs font-bold text-slate-700 hidden md:flex items-center gap-1">
              <Layers className="w-3.5 h-3.5 text-blue-600" />
              <span>切換題目:</span>
            </label>
            <select
              id="problem-selector"
              value={currentProblem}
              onChange={(e) => handleProblemChange(e.target.value)}
              aria-label="選擇研究問題"
              className="bg-blue-50/80 hover:bg-blue-100/80 border-2 border-blue-600 text-blue-950 text-xs font-bold rounded-lg px-2.5 py-1.5 shadow-xs focus:ring-2 focus:ring-blue-500 focus:outline-none cursor-pointer transition-all"
            >
              <option value="p1">Problem 1 — Bike Sharing Demand</option>
              <option value="p2">Problem 2 — Appliances Energy Prediction</option>
              <option value="custom">Bonus — Custom Dataset MLR (+2分)</option>
            </select>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden xl:flex items-center gap-1">
            {activeNavLinks.map((link) => {
              const Icon = link.icon;
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                    isActive
                      ? "bg-blue-50 text-blue-700 font-semibold border border-blue-200 shadow-xs"
                      : "text-[#475569] hover:text-[#17243A] hover:bg-slate-100/70"
                  }`}
                >
                  <Icon className={`w-3.5 h-3.5 ${isActive ? "text-blue-600" : "text-[#64748B]"}`} />
                  <span>{link.label}</span>
                </Link>
              );
            })}
          </nav>

          {/* Right Badges & Controls */}
          <div className="hidden sm:flex items-center gap-2">
            {/* Quick Bonus Nav Button */}
            {!isCustom && (
              <Link
                href="/custom-dataset"
                className="hidden lg:flex items-center gap-1 px-2 py-1 rounded-md text-xs font-semibold bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border border-emerald-200 transition-colors"
                title="前往自訂資料集通用迴歸加分功能"
              >
                <FolderPlus className="w-3.5 h-3.5 text-emerald-600" />
                <span>+2分 加分題</span>
              </Link>
            )}

            {/* Experiment ID Tag with Copy */}
            <button
              onClick={copyExpId}
              title="點擊複製實驗識別碼"
              className="flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-mono bg-[#F7F9FC] hover:bg-slate-200/70 text-[#334155] border border-[#DFE5ED] transition-colors cursor-pointer"
            >
              <span className="text-[#64748B] text-[10px]">EXP:</span>
              <span className="text-blue-700 font-semibold text-[11px] truncate max-w-[110px]">
                {isProblem2 ? "P2_EXP01" : isCustom ? "BONUS_EXP" : "PHASE3_EXP01"}
              </span>
              {copied ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              ) : (
                <span className="text-[10px] text-[#64748B]">📋</span>
              )}
            </button>

            {/* Student ID */}
            <div className="px-2 py-1 rounded-md text-xs font-mono font-semibold bg-blue-50 text-blue-800 border border-blue-200">
              7115093001
            </div>
          </div>

          {/* Mobile Menu Button */}
          <div className="flex xl:hidden items-center gap-2">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-lg bg-slate-100 text-[#334155] hover:text-[#17243A] hover:bg-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-600"
              aria-label="Toggle Menu"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Navigation Dropdown */}
      {mobileMenuOpen && (
        <div className="xl:hidden border-b border-[#DFE5ED] bg-white px-4 pt-2 pb-4 space-y-2 shadow-lg">
          <div className="flex items-center justify-between py-2 border-b border-[#DFE5ED]">
            <span className="text-xs font-mono font-semibold text-blue-800">學號: 7115093001</span>
            <span className="text-xs font-mono font-semibold text-[#475569]">EXP: {expId}</span>
          </div>

          {/* Mobile Problem Selector */}
          <div className="py-2">
            <label className="text-xs font-bold text-slate-700 block mb-1">當前題目切換:</label>
            <div className="grid grid-cols-1 gap-1.5">
              <button
                onClick={() => { handleProblemChange("p1"); setMobileMenuOpen(false); }}
                className={`text-left px-3 py-2 rounded-lg text-xs font-semibold flex items-center justify-between ${
                  currentProblem === "p1" ? "bg-blue-600 text-white" : "bg-slate-100 text-slate-700"
                }`}
              >
                <span>Problem 1 — Bike Sharing</span>
                {currentProblem === "p1" && <CheckCircle2 className="w-4 h-4 text-white" />}
              </button>
              <button
                onClick={() => { handleProblemChange("p2"); setMobileMenuOpen(false); }}
                className={`text-left px-3 py-2 rounded-lg text-xs font-semibold flex items-center justify-between ${
                  currentProblem === "p2" ? "bg-blue-600 text-white" : "bg-slate-100 text-slate-700"
                }`}
              >
                <span>Problem 2 — Appliances Energy</span>
                {currentProblem === "p2" && <CheckCircle2 className="w-4 h-4 text-white" />}
              </button>
              <button
                onClick={() => { handleProblemChange("custom"); setMobileMenuOpen(false); }}
                className={`text-left px-3 py-2 rounded-lg text-xs font-semibold flex items-center justify-between ${
                  currentProblem === "custom" ? "bg-emerald-600 text-white" : "bg-emerald-50 text-emerald-800 border border-emerald-200"
                }`}
              >
                <span>Bonus — Custom Dataset (+2分)</span>
                {currentProblem === "custom" && <CheckCircle2 className="w-4 h-4 text-white" />}
              </button>
            </div>
          </div>

          <div className="border-t border-slate-200 pt-2 space-y-1">
            <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block px-1">本題分項導覽</span>
            {activeNavLinks.map((link) => {
              const Icon = link.icon;
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={() => setMobileMenuOpen(false)}
                  className={`flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium ${
                    isActive
                      ? "bg-blue-50 text-blue-700 font-semibold border border-blue-200"
                      : "text-[#334155] hover:bg-slate-50"
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <Icon className="w-4 h-4 text-blue-600" />
                    <span>{link.label}</span>
                  </div>
                  <span className="text-[11px] text-[#64748B]">{link.enLabel}</span>
                </Link>
              );
            })}
          </div>
        </div>
      )}
    </header>
  );
}
