import os
import json
import shutil
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
WEBSITE_DIR = os.path.join(os.path.dirname(BASE_DIR), "website")
DATA_OUT_DIR = os.path.join(WEBSITE_DIR, "data")
PUBLIC_FIGS_DIR = os.path.join(WEBSITE_DIR, "public", "figures")

os.makedirs(DATA_OUT_DIR, exist_ok=True)
os.makedirs(PUBLIC_FIGS_DIR, exist_ok=True)

print("--- [Website Data Exporter] Synchronizing Phase 3 Artifacts ---")

# 1. Copy JSON result files
json_files = [
    "final_test_metrics.json",
    "prediction_interval_metrics.json",
    "best_model_metadata.json",
    "phase3_experiment_metadata.json",
    "dataset_metadata.json",
    "m2_model_spec.json",
    "missing_timestamps.json",
    "data_split_summary.json"
]

for jf in json_files:
    src = os.path.join(RESULTS_DIR, jf)
    if os.path.exists(src):
        dst = os.path.join(DATA_OUT_DIR, jf)
        shutil.copy2(src, dst)
        print(f"[OK] Copied {jf}")

# 2. Parse model_comparison.csv to model_comparison.json
mc_df = pd.read_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))
mc_json_path = os.path.join(DATA_OUT_DIR, "model_comparison.json")
with open(mc_json_path, "w", encoding="utf-8") as f:
    text = json.dumps(mc_df.to_dict(orient="records"), indent=2, ensure_ascii=False).replace(": NaN", ": null")
    f.write(text)
print("[OK] Converted model_comparison.csv -> model_comparison.json")

# 3. Parse selected_features.csv to selected_features.json
sf_df = pd.read_csv(os.path.join(RESULTS_DIR, "selected_features.csv"))
sf_json_path = os.path.join(DATA_OUT_DIR, "selected_features.json")
with open(sf_json_path, "w", encoding="utf-8") as f:
    text = json.dumps(sf_df.to_dict(orient="records"), indent=2, ensure_ascii=False).replace(": NaN", ": null")
    f.write(text)
print("[OK] Converted selected_features.csv -> selected_features.json")

# 4. Parse grouped_feature_ablation.csv to grouped_feature_ablation.json
ab_df = pd.read_csv(os.path.join(RESULTS_DIR, "grouped_feature_ablation.csv"))
ab_json_path = os.path.join(DATA_OUT_DIR, "grouped_feature_ablation.json")
with open(ab_json_path, "w", encoding="utf-8") as f:
    text = json.dumps(ab_df.to_dict(orient="records"), indent=2, ensure_ascii=False).replace(": NaN", ": null")
    f.write(text)
print("[OK] Converted grouped_feature_ablation.csv -> grouped_feature_ablation.json")

# 5. Process test_predictions.csv for interactive timeline & scatter plots
tp_df = pd.read_csv(os.path.join(RESULTS_DIR, "test_predictions.csv"))
tp_df["datetime"] = pd.to_datetime(tp_df["timestamp"])
tp_df["hr"] = tp_df["datetime"].dt.hour
tp_df["dteday"] = tp_df["datetime"].dt.strftime("%Y-%m-%d")

# 5a. Hourly coverage diagnostic table
hourly_stats = tp_df.groupby("hr").agg(
    empirical_coverage=("covered_clipped", "mean"),
    sample_count=("covered_clipped", "count"),
    actual_mean=("actual_cnt", "mean"),
    pred_mean=("predicted_cnt", "mean")
).reset_index()
hourly_stats["coverage_pct"] = (hourly_stats["empirical_coverage"] * 100.0).round(2)
hourly_stats_path = os.path.join(DATA_OUT_DIR, "hourly_coverage_diagnostic.json")
with open(hourly_stats_path, "w", encoding="utf-8") as f:
    json.dump(hourly_stats.to_dict(orient="records"), f, indent=2)
print("[OK] Generated hourly_coverage_diagnostic.json")

# 5b. First 7 complete calendar days timeline (2012-09-14 to 2012-09-20)
test_dates = sorted(tp_df["dteday"].unique())
first_7_days = set(test_dates[:7])
timeline_7d = tp_df[tp_df["dteday"].isin(first_7_days)].copy()
timeline_records = []
for _, r in timeline_7d.iterrows():
    timeline_records.append({
        "timestamp": r["timestamp"],
        "dteday": r["dteday"],
        "actual_cnt": float(r["actual_cnt"]),
        "predicted_cnt": float(r["predicted_cnt"]),
        "pi_lower_raw": float(r["pi_lower_raw"]),
        "pi_upper_raw": float(r["pi_upper_raw"]),
        "pi_lower_clipped": float(r["pi_lower_clipped"]),
        "pi_upper_clipped": float(r["pi_upper_clipped"]),
        "residual": float(r["residual"]),
        "covered": bool(r["covered_clipped"])
    })
timeline_path = os.path.join(DATA_OUT_DIR, "test_timeline_7d.json")
with open(timeline_path, "w", encoding="utf-8") as f:
    json.dump(timeline_records, f, indent=2)
print(f"[OK] Generated test_timeline_7d.json ({len(timeline_records)} hours)")

# 5b2. Full test timeline records (all 2,576 observations for selectable historical date ranges)
timeline_all = []
for _, r in tp_df.iterrows():
    timeline_all.append({
        "timestamp": r["timestamp"],
        "dteday": r["dteday"],
        "actual_cnt": float(r["actual_cnt"]),
        "predicted_cnt": float(r["predicted_cnt"]),
        "pi_lower_raw": float(r["pi_lower_raw"]),
        "pi_upper_raw": float(r["pi_upper_raw"]),
        "pi_lower_clipped": float(r["pi_lower_clipped"]),
        "pi_upper_clipped": float(r["pi_upper_clipped"]),
        "residual": float(r["residual"]),
        "covered": bool(r["covered_clipped"])
    })
timeline_all_path = os.path.join(DATA_OUT_DIR, "test_timeline_all.json")
with open(timeline_all_path, "w", encoding="utf-8") as f:
    json.dump(timeline_all, f, indent=2)
print(f"[OK] Generated test_timeline_all.json ({len(timeline_all)} total test hours)")

# 5c. All available test dates list
dates_summary = {
    "total_test_days": len(test_dates),
    "start_date": test_dates[0],
    "end_date": test_dates[-1],
    "available_dates": test_dates
}
with open(os.path.join(DATA_OUT_DIR, "test_dates_summary.json"), "w", encoding="utf-8") as f:
    json.dump(dates_summary, f, indent=2)

# 5d. Scatter plot samples (subsample 600 representative points across the range for fast interactive canvas/svg rendering)
# Include all negative predictions and peak extremes
np.random.seed(42)
indices_step = np.linspace(0, len(tp_df) - 1, 650, dtype=int)
neg_indices = tp_df.index[tp_df["predicted_cnt"] < 0].tolist()
extreme_indices = tp_df.index[tp_df["actual_cnt"] > 600].tolist()
selected_indices = sorted(list(set(indices_step.tolist() + neg_indices + extreme_indices)))

scatter_samples = []
for idx in selected_indices:
    r = tp_df.iloc[idx]
    scatter_samples.append({
        "actual": float(r["actual_cnt"]),
        "predicted": float(r["predicted_cnt"]),
        "residual": float(r["residual"]),
        "timestamp": r["timestamp"],
        "covered": bool(r["covered_clipped"])
    })

scatter_path = os.path.join(DATA_OUT_DIR, "test_scatter_samples.json")
with open(scatter_path, "w", encoding="utf-8") as f:
    json.dump(scatter_samples, f, indent=2)
print(f"[OK] Generated test_scatter_samples.json ({len(scatter_samples)} points)")

# 5e. Residual distribution histogram bins
residuals = tp_df["residual"].values
counts, bin_edges = np.histogram(residuals, bins=35)
res_bins = []
for i in range(len(counts)):
    res_bins.append({
        "bin_start": round(float(bin_edges[i]), 1),
        "bin_end": round(float(bin_edges[i+1]), 1),
        "bin_center": round(float((bin_edges[i] + bin_edges[i+1]) / 2.0), 1),
        "count": int(counts[i])
    })
with open(os.path.join(DATA_OUT_DIR, "residual_distribution_bins.json"), "w", encoding="utf-8") as f:
    json.dump(res_bins, f, indent=2)
print("[OK] Generated residual_distribution_bins.json")

# 6. Copy figures
fig_files = [f for f in os.listdir(FIGURES_DIR) if f.endswith(".png")]
for ff in fig_files:
    shutil.copy2(os.path.join(FIGURES_DIR, ff), os.path.join(PUBLIC_FIGS_DIR, ff))
    print(f"[OK] Copied figure {ff} to website/public/figures/")

print("--- [Website Data Exporter] All artifacts synchronized successfully! ---")
