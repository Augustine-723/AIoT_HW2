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
    <header className="sticky top-0 z-50 w-full border-b border-[#DFE5ED] bg-white/95 backdrop-blur-md shadow-xs print-hide">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Logo & Title */}
          <div className="flex items-center gap-3">
            <Link href="/" className="flex items-center gap-2.5 group">
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
                    M2 · RFE
                  </span>
                </div>
                <span className="text-xs text-[#475569] hidden sm:block truncate max-w-xs font-medium">
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
          <div className="hidden sm:flex items-center gap-2.5">
            {/* Experiment ID Tag with Copy */}
            <button
              onClick={copyExpId}
              title="點擊複製實驗識別碼"
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono bg-[#F7F9FC] hover:bg-slate-200/70 text-[#334155] border border-[#DFE5ED] transition-colors cursor-pointer"
            >
              <span className="text-[#64748B] text-[10px]">EXP:</span>
              <span className="text-blue-700 font-semibold">PHASE3_EXP01</span>
              {copied ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              ) : (
                <span className="text-[10px] text-[#64748B] hover:text-[#17243A]">📋</span>
              )}
            </button>

            {/* Student ID */}
            <div className="px-2.5 py-1 rounded-md text-xs font-mono font-semibold bg-blue-50 text-blue-800 border border-blue-200">
              ID: 7115093001
            </div>
          </div>

          {/* Mobile Menu Button */}
          <div className="flex lg:hidden items-center gap-2">
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
        <div className="lg:hidden border-b border-[#DFE5ED] bg-white px-4 pt-2 pb-4 space-y-1 shadow-lg">
          <div className="flex items-center justify-between py-2 border-b border-[#DFE5ED] mb-2">
            <span className="text-xs font-mono font-semibold text-blue-800">ID: 7115093001</span>
            <span className="text-xs font-mono font-semibold text-[#475569]">EXP: PHASE3_EXP01</span>
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
                    ? "bg-blue-50 text-blue-700 font-semibold border border-blue-200"
                    : "text-[#334155] hover:bg-slate-50"
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <Icon className="w-4 h-4 text-blue-600" />
                  <span>{link.label}</span>
                </div>
                <span className="text-xs text-[#64748B]">{link.enLabel}</span>
              </Link>
            );
          })}
        </div>
      )}
    </header>
  );
}
