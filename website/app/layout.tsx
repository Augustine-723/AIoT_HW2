import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "以多元線性迴歸與特徵選擇預測共享單車每小時租借需求 | AIoT HW2 Dashboard",
  description:
    "Predicting Hourly Bike Sharing Demand Using Multiple Linear Regression and Feature Selection. AIoT HW2 Academic Research Dashboard (Student ID: 7115093001, Experiment: AIoT_HW2_PHASE3_EXP01).",
  keywords: [
    "Bike Sharing Demand",
    "Multiple Linear Regression",
    "Feature Selection",
    "RFE",
    "Split Conformal Prediction",
    "AIoT",
    "7115093001",
  ],
  authors: [{ name: "7115093001" }],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="zh-Hant"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased dark`}
    >
      <body className="min-h-full flex flex-col bg-slate-950 text-slate-100 font-sans selection:bg-indigo-500/30 selection:text-indigo-200">
        <Navbar />
        <main className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  );
}
