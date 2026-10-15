"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
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
  ExternalLink,
  Cpu
} from "lucide-react";

const NAV_LINKS = [
  { href: "/", label: "研究總覽", enLabel: "Overview", icon: BarChart3 },
  { href: "/dataset", label: "資料與 EDA", enLabel: "Dataset & EDA", icon: Database },
  { href: "/models", label: "模型比較", enLabel: "Model Comparison", icon: Sliders },
  { href: "/evaluation", label: "最終評估", enLabel: "Final Evaluation", icon: LineChart },
  { href: "/predict", label: "互動預測", enLabel: "Interactive Prediction", icon: Sparkles },
  { href: "/methodology", label: "研究方法論", enLabel: "Methodology", icon: BookOpen },
];

export default function Navbar() {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const copyExpId = () => {
    navigator.clipboard.writeText("AIoT_HW2_PHASE3_EXP01");
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-800/80 bg-slate-950/85 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Logo & Title */}
          <div className="flex items-center gap-3">
            <Link href="/" className="flex items-center gap-2.5 group">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 p-[1px] shadow-lg shadow-indigo-500/20">
                <div className="w-full h-full bg-slate-950 rounded-[11px] flex items-center justify-center group-hover:bg-slate-900 transition-colors">
                  <Cpu className="w-5 h-5 text-cyan-400 group-hover:rotate-12 transition-transform duration-300" />
                </div>
              </div>
              <div className="flex flex-col">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-slate-100 tracking-tight text-sm sm:text-base">
                    AIoT HW2
                  </span>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                    M2 · RFE
                  </span>
                </div>
                <span className="text-[11px] text-slate-400 hidden sm:block truncate max-w-xs">
                  Bike Sharing Demand Prediction
                </span>
              </div>
            </Link>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden lg:flex items-center gap-1">
            {NAV_LINKS.map((link) => {
              const Icon = link.icon;
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                    isActive
                      ? "bg-indigo-600/15 text-indigo-300 border border-indigo-500/30 shadow-sm shadow-indigo-950"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/60"
                  }`}
                >
                  <Icon className={`w-3.5 h-3.5 ${isActive ? "text-indigo-400" : "text-slate-400"}`} />
                  <span>{link.label}</span>
                </Link>
              );
            })}
          </nav>

          {/* Right Badges & Controls */}
          <div className="hidden sm:flex items-center gap-2.5">
            {/* Experiment ID Tag with Copy */}
            <button
              onClick={copyExpId}
              title="點擊複製實驗識別碼"
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono bg-slate-900/90 hover:bg-slate-800 text-slate-300 border border-slate-700/60 transition-colors"
            >
              <span className="text-slate-400 text-[10px]">EXP:</span>
              <span className="text-cyan-400 font-semibold">PHASE3_EXP01</span>
              {copied ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              ) : (
                <span className="text-[10px] text-slate-400 hover:text-slate-300">📋</span>
              )}
            </button>

            {/* Student ID */}
            <div className="px-2.5 py-1 rounded-md text-xs font-mono bg-indigo-950/40 text-indigo-300 border border-indigo-800/40">
              ID: 7115093001
            </div>
          </div>

          {/* Mobile Menu Button */}
          <div className="flex lg:hidden items-center gap-2">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-lg bg-slate-900 text-slate-300 hover:text-white hover:bg-slate-800 focus:outline-none"
              aria-label="Toggle Menu"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Navigation Dropdown */}
      {mobileMenuOpen && (
        <div className="lg:hidden border-b border-slate-800 bg-slate-950/95 px-4 pt-2 pb-4 space-y-1">
          <div className="flex items-center justify-between py-2 border-b border-slate-800/60 mb-2">
            <span className="text-xs font-mono text-slate-400">ID: 7115093001</span>
            <span className="text-xs font-mono text-cyan-400">EXP: PHASE3_EXP01</span>
          </div>
          {NAV_LINKS.map((link) => {
            const Icon = link.icon;
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                onClick={() => setMobileMenuOpen(false)}
                className={`flex items-center justify-between px-3 py-2 rounded-lg text-sm font-medium ${
                  isActive
                    ? "bg-indigo-600/20 text-indigo-300 border border-indigo-500/30"
                    : "text-slate-300 hover:bg-slate-900"
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <Icon className="w-4 h-4 text-indigo-400" />
                  <span>{link.label}</span>
                </div>
                <span className="text-xs text-slate-400">{link.enLabel}</span>
              </Link>
            );
          })}
        </div>
      )}
    </header>
  );
}
