"""
AIoT HW2 Phase 1: Complete Verification and Artifact Generation Script
Author: Student 7115093001
Title: Predicting Hourly Bike Sharing Demand Using Multiple Linear Regression and Feature Selection
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "hour.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"[Phase 1] Execution Directory: {BASE_DIR}")
print(f"[Phase 1] Data Path: {DATA_PATH}")

# 1. Dataset Acquisition & Metadata Recording
with open(DATA_PATH, "rb") as f:
    file_bytes = f.read()
    sha256_hash = hashlib.sha256(file_bytes).hexdigest()

df_raw = pd.read_csv(DATA_PATH)
n_rows, n_cols = df_raw.shape

dataset_metadata = {
    "student_id": "7115093001",
    "research_title": "Predicting Hourly Bike Sharing Demand Using Multiple Linear Regression and Feature Selection",
    "dataset_name": "Bike Sharing Dataset (Hourly Version)",
    "official_kaggle_url": "https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset",
    "original_source_url": "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
    "filename": "hour.csv",
    "sha256_checksum": sha256_hash,
    "number_of_rows": int(n_rows),
    "number_of_columns": int(n_cols),
    "column_names": list(df_raw.columns),
    "data_types": {col: str(dtype) for col, dtype in df_raw.dtypes.items()},
    "environment_versions": {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__
    }
}

with open(os.path.join(RESULTS_DIR, "dataset_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(dataset_metadata, f, indent=2, ensure_ascii=False)

print(f"[Phase 1] Saved dataset_metadata.json (Rows: {n_rows}, Cols: {n_cols})")

# 2. Step 1: Dataset Validation
missing_values = df_raw.isnull().sum().to_dict()
total_missing = int(df_raw.isnull().sum().sum())
duplicate_rows = int(df_raw.duplicated().sum())

# Timestamp construction
df = df_raw.copy()
df['datetime'] = pd.to_datetime(df['dteday'] + ' ' + df['hr'].astype(str).str.zfill(2) + ':00:00')
duplicate_timestamps = int(df['datetime'].duplicated().sum())

# Chronological sorting check
is_initially_sorted = bool(df['datetime'].is_monotonic_increasing)
df = df.sort_values('datetime').reset_index(drop=True)

# Missing hours / Gaps check
full_date_range = pd.date_range(start=df['datetime'].min(), end=df['datetime'].max(), freq='h')
expected_hours_count = len(full_date_range)
missing_hours_count = expected_hours_count - len(df)
missing_timestamps_sample = [str(ts) for ts in full_date_range.difference(df['datetime'])[:10]]

# Relationship verification: casual + registered == cnt
sum_matches = bool((df['casual'] + df['registered'] == df['cnt']).all())
non_negative_target = bool((df['cnt'] >= 0).all())

# Category and range validations
range_checks = {
    "season_valid": bool(df['season'].isin([1, 2, 3, 4]).all()),
    "yr_valid": bool(df['yr'].isin([0, 1]).all()),
    "mnth_valid": bool(df['mnth'].between(1, 12).all()),
    "hr_valid": bool(df['hr'].between(0, 23).all()),
    "holiday_valid": bool(df['holiday'].isin([0, 1]).all()),
    "weekday_valid": bool(df['weekday'].between(0, 6).all()),
    "workingday_valid": bool(df['workingday'].isin([0, 1]).all()),
    "weathersit_valid": bool(df['weathersit'].isin([1, 2, 3, 4]).all()),
    "temp_valid": bool(((df['temp'] >= 0.0) & (df['temp'] <= 1.0)).all()),
    "atemp_valid": bool(((df['atemp'] >= 0.0) & (df['atemp'] <= 1.0)).all()),
    "hum_valid": bool(((df['hum'] >= 0.0) & (df['hum'] <= 1.0)).all()),
    "windspeed_valid": bool(((df['windspeed'] >= 0.0) & (df['windspeed'] <= 1.0)).all())
}

validation_results = {
    "total_rows": n_rows,
    "total_columns": n_cols,
    "total_missing_values": total_missing,
    "missing_values_per_column": missing_values,
    "duplicate_records": duplicate_rows,
    "duplicate_timestamps": duplicate_timestamps,
    "is_monotonic_increasing_chronological": is_initially_sorted,
    "timestamp_min": str(df['datetime'].min()),
    "timestamp_max": str(df['datetime'].max()),
    "expected_hours_in_calendar_span": expected_hours_count,
    "actual_recorded_hours": len(df),
    "gap_hours_count": missing_hours_count,
    "gap_hours_sample": missing_timestamps_sample,
    "casual_plus_registered_equals_cnt": sum_matches,
    "cnt_non_negative": non_negative_target,
    "cnt_min": int(df['cnt'].min()),
    "cnt_max": int(df['cnt'].max()),
    "cnt_mean": float(df['cnt'].mean()),
    "cnt_median": float(df['cnt'].median()),
    "cnt_std": float(df['cnt'].std()),
    "cnt_skewness": float(df['cnt'].skew()),
    "feature_range_checks": range_checks
}

with open(os.path.join(RESULTS_DIR, "data_validation.json"), "w", encoding="utf-8") as f:
    json.dump(validation_results, f, indent=2, ensure_ascii=False)

print("[Phase 1] Saved data_validation.json")

# Descriptive statistics
desc_stats = df.describe()
desc_stats.to_csv(os.path.join(RESULTS_DIR, "descriptive_statistics.csv"))
print("[Phase 1] Saved descriptive_statistics.csv")

# 3. Step 2: Exploratory Data Analysis & Visualizations
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Figure 1: Hourly Rental Demand
fig1, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
hourly_avg = df.groupby('hr')['cnt'].mean()
ax1.plot(hourly_avg.index, hourly_avg.values, marker='o', color='#1f77b4', linewidth=2.5, markersize=6)
ax1.set_title("Average Bike Rental Demand by Hour", fontsize=14, fontweight='bold', pad=12)
ax1.set_xlabel("Hour of Day", fontsize=12, labelpad=8)
ax1.set_ylabel("Average Rental Count", fontsize=12, labelpad=8)
ax1.set_xticks(range(0, 24))
ax1.grid(True, linestyle='--', alpha=0.6)
fig1.tight_layout()
fig1_path = os.path.join(FIGURES_DIR, "hourly_demand.png")
fig1.savefig(fig1_path, dpi=300)
plt.close(fig1)
print(f"[Phase 1] Saved {fig1_path}")

# Figure 2: Correlation Heatmap
# Exclude instant, casual, registered, dteday, datetime
numeric_cols_for_corr = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
                         'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt']
corr_matrix = df[numeric_cols_for_corr].corr()

fig2, ax2 = plt.subplots(figsize=(11, 9), dpi=300)
im = ax2.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax2.set_xticks(range(len(numeric_cols_for_corr)))
ax2.set_yticks(range(len(numeric_cols_for_corr)))
ax2.set_xticklabels(numeric_cols_for_corr, rotation=45, ha='right', fontsize=10)
ax2.set_yticklabels(numeric_cols_for_corr, fontsize=10)
ax2.set_title("Correlation Heatmap of Predictors and Target Variable", fontsize=14, fontweight='bold', pad=12)
fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

for i in range(len(numeric_cols_for_corr)):
    for j in range(len(numeric_cols_for_corr)):
        val = corr_matrix.iloc[i, j]
        text_color = "white" if abs(val) > 0.55 else "black"
        ax2.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8.5)

fig2.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
fig2.savefig(fig2_path, dpi=300)
plt.close(fig2)
print(f"[Phase 1] Saved {fig2_path}")

# Figure 3: Target Distribution Histogram
fig3, ax3 = plt.subplots(figsize=(9, 5), dpi=300)
ax3.hist(df['cnt'], bins=50, color='#3498db', edgecolor='black', alpha=0.75, density=True)
df['cnt'].plot(kind='kde', ax=ax3, color='#e74c3c', linewidth=2)
ax3.set_title("Distribution of Hourly Bike Rental Demand (cnt)", fontsize=14, fontweight='bold', pad=12)
ax3.set_xlabel("Rental Count (cnt)", fontsize=12, labelpad=8)
ax3.set_ylabel("Density", fontsize=12, labelpad=8)
ax3.grid(True, linestyle='--', alpha=0.6)
fig3.tight_layout()
fig3_path = os.path.join(FIGURES_DIR, "target_distribution.png")
fig3.savefig(fig3_path, dpi=300)
plt.close(fig3)
print(f"[Phase 1] Saved {fig3_path}")

# Figure 4: Temperature vs Demand Scatter Plot
fig4, ax4 = plt.subplots(figsize=(9, 5), dpi=300)
ax4.scatter(df['temp'], df['cnt'], alpha=0.25, color='#2ca02c', s=12, edgecolors='none')
m, b = np.polyfit(df['temp'], df['cnt'], 1)
x_vals = np.linspace(df['temp'].min(), df['temp'].max(), 100)
ax4.plot(x_vals, m * x_vals + b, color='#d62728', linewidth=2.5, label=f'OLS Trendline: cnt = {m:.1f}*temp + {b:.1f}')
ax4.set_title("Normalized Temperature vs. Hourly Rental Demand", fontsize=14, fontweight='bold', pad=12)
ax4.set_xlabel("Normalized Temperature (temp)", fontsize=12, labelpad=8)
ax4.set_ylabel("Rental Count (cnt)", fontsize=12, labelpad=8)
ax4.legend(frameon=True, facecolor='white', loc='upper left')
ax4.grid(True, linestyle='--', alpha=0.6)
fig4.tight_layout()
fig4_path = os.path.join(FIGURES_DIR, "temperature_vs_demand.png")
fig4.savefig(fig4_path, dpi=300)
plt.close(fig4)
print(f"[Phase 1] Saved {fig4_path}")

# 4. Step 3: Chronological Data Split
# Splitting on complete calendar day boundaries
unique_dates = df['dteday'].drop_duplicates().sort_values().reset_index(drop=True)
total_days = len(unique_dates)

# Target ~70% Train, ~15% Calibration, ~15% Test
n_train_days = int(round(total_days * 0.70))
n_calib_days = int(round(total_days * 0.15))
n_test_days = total_days - n_train_days - n_calib_days

train_dates = set(unique_dates.iloc[:n_train_days])
calib_dates = set(unique_dates.iloc[n_train_days:n_train_days + n_calib_days])
test_dates = set(unique_dates.iloc[n_train_days + n_calib_days:])

train_df = df[df['dteday'].isin(train_dates)].copy()
calib_df = df[df['dteday'].isin(calib_dates)].copy()
test_df = df[df['dteday'].isin(test_dates)].copy()

# Integrity checks
assert len(train_dates.intersection(calib_dates)) == 0, "Train and Calib dates overlap!"
assert len(train_dates.intersection(test_dates)) == 0, "Train and Test dates overlap!"
assert len(calib_dates.intersection(test_dates)) == 0, "Calib and Test dates overlap!"
assert train_df['datetime'].max() < calib_df['datetime'].min(), "Chronological sequence violated between Train & Calib!"
assert calib_df['datetime'].max() < test_df['datetime'].min(), "Chronological sequence violated between Calib & Test!"

data_split_summary = {
    "total_observations": len(df),
    "total_calendar_days": total_days,
    "target_ratios": {"train": 0.70, "calibration": 0.15, "test": 0.15},
    "subsets": {
        "train": {
            "n_days": len(train_dates),
            "n_observations": len(train_df),
            "percentage_obs": round(len(train_df) / len(df) * 100, 2),
            "start_date": str(train_df['dteday'].min()),
            "end_date": str(train_df['dteday'].max()),
            "start_timestamp": str(train_df['datetime'].min()),
            "end_timestamp": str(train_df['datetime'].max())
        },
        "calibration": {
            "n_days": len(calib_dates),
            "n_observations": len(calib_df),
            "percentage_obs": round(len(calib_df) / len(df) * 100, 2),
            "start_date": str(calib_df['dteday'].min()),
            "end_date": str(calib_df['dteday'].max()),
            "start_timestamp": str(calib_df['datetime'].min()),
            "end_timestamp": str(calib_df['datetime'].max())
        },
        "test": {
            "n_days": len(test_dates),
            "n_observations": len(test_df),
            "percentage_obs": round(len(test_df) / len(df) * 100, 2),
            "start_date": str(test_df['dteday'].min()),
            "end_date": str(test_df['dteday'].max()),
            "start_timestamp": str(test_df['datetime'].min()),
            "end_timestamp": str(test_df['datetime'].max())
        }
    },
    "verification": {
        "disjoint_calendar_days": True,
        "chronological_order_strict": True,
        "train_max_before_calib_min": True,
        "calib_max_before_test_min": True
    }
}

with open(os.path.join(RESULTS_DIR, "data_split_summary.json"), "w", encoding="utf-8") as f:
    json.dump(data_split_summary, f, indent=2, ensure_ascii=False)

print(f"[Phase 1] Saved data_split_summary.json:")
print(f"  Train: {len(train_df)} rows ({data_split_summary['subsets']['train']['percentage_obs']}%), {data_split_summary['subsets']['train']['start_date']} to {data_split_summary['subsets']['train']['end_date']}")
print(f"  Calib: {len(calib_df)} rows ({data_split_summary['subsets']['calibration']['percentage_obs']}%), {data_split_summary['subsets']['calibration']['start_date']} to {data_split_summary['subsets']['calibration']['end_date']}")
print(f"  Test:  {len(test_df)} rows ({data_split_summary['subsets']['test']['percentage_obs']}%), {data_split_summary['subsets']['test']['start_date']} to {data_split_summary['subsets']['test']['end_date']}")

# 5. Step 4: Preprocessing Pipeline & Feature Definitions
categorical_features = ['season', 'mnth', 'hr', 'weekday', 'weathersit']
numeric_binary_features = ['yr', 'holiday', 'workingday', 'temp', 'atemp', 'hum', 'windspeed']
all_12_predictors = categorical_features + numeric_binary_features
target_col = 'cnt'

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features),
        ('num', 'passthrough', numeric_binary_features)
    ],
    remainder='drop'
)

# Fit preprocessor strictly on Training data
X_train = train_df[all_12_predictors]
y_train = train_df[target_col]

preprocessor.fit(X_train)

# Extract encoded feature names
cat_encoder = preprocessor.named_transformers_['cat']
cat_encoded_names = list(cat_encoder.get_feature_names_out(categorical_features))
all_encoded_names = cat_encoded_names + numeric_binary_features

encoded_df_info = pd.DataFrame({
    "encoded_column_index": range(len(all_encoded_names)),
    "encoded_feature_name": all_encoded_names,
    "original_feature": [col.split('_')[0] if col.startswith(tuple(categorical_features)) else col for col in all_encoded_names],
    "type": ["categorical_dummy" if i < len(cat_encoded_names) else "numeric_or_binary" for i in range(len(all_encoded_names))]
})
encoded_df_info.to_csv(os.path.join(RESULTS_DIR, "encoded_feature_names.csv"), index=False)
print(f"[Phase 1] Saved encoded_feature_names.csv: {len(all_encoded_names)} transformed input columns from 12 original predictors.")

# 6. Step 5: Baseline OLS Model & TimeSeriesSplit CV
ols_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train on full Training set
ols_pipeline.fit(X_train, y_train)
y_train_pred = ols_pipeline.predict(X_train)

train_mae = float(mean_absolute_error(y_train, y_train_pred))
train_mse = float(mean_squared_error(y_train, y_train_pred))
train_rmse = float(np.sqrt(train_mse))
train_r2 = float(r2_score(y_train, y_train_pred))

# Reference DummyRegressor (predicts mean)
dummy_reg = DummyRegressor(strategy="mean")
dummy_reg.fit(X_train, y_train)
y_train_dummy_pred = dummy_reg.predict(X_train)
dummy_train_mae = float(mean_absolute_error(y_train, y_train_dummy_pred))
dummy_train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_dummy_pred)))
dummy_train_r2 = float(r2_score(y_train, y_train_dummy_pred))

# TimeSeriesSplit Cross-Validation on Training set only
tscv = TimeSeriesSplit(n_splits=5)
cv_results = []

for fold_idx, (cv_train_idx, cv_val_idx) in enumerate(tscv.split(X_train), 1):
    X_cv_train = X_train.iloc[cv_train_idx]
    y_cv_train = y_train.iloc[cv_train_idx]
    X_cv_val = X_train.iloc[cv_val_idx]
    y_cv_val = y_train.iloc[cv_val_idx]
    
    # New pipeline per fold to ensure ZERO preprocessing leakage
    fold_preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features),
            ('num', 'passthrough', numeric_binary_features)
        ],
        remainder='drop'
    )
    fold_pipeline = Pipeline(steps=[
        ('preprocessor', fold_preprocessor),
        ('regressor', LinearRegression())
    ])
    
    fold_pipeline.fit(X_cv_train, y_cv_train)
    y_cv_val_pred = fold_pipeline.predict(X_cv_val)
    
    fold_mae = float(mean_absolute_error(y_cv_val, y_cv_val_pred))
    fold_mse = float(mean_squared_error(y_cv_val, y_cv_val_pred))
    fold_rmse = float(np.sqrt(fold_mse))
    fold_r2 = float(r2_score(y_cv_val, y_cv_val_pred))
    
    # Dummy on this fold
    fold_dummy = DummyRegressor(strategy="mean")
    fold_dummy.fit(X_cv_train, y_cv_train)
    y_fold_dummy_pred = fold_dummy.predict(X_cv_val)
    fold_dummy_rmse = float(np.sqrt(mean_squared_error(y_cv_val, y_fold_dummy_pred)))
    
    cv_results.append({
        "fold": fold_idx,
        "n_train_samples": len(X_cv_train),
        "n_val_samples": len(X_cv_val),
        "val_mae": fold_mae,
        "val_mse": fold_mse,
        "val_rmse": fold_rmse,
        "val_r2": fold_r2,
        "dummy_val_rmse": fold_dummy_rmse
    })

cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv(os.path.join(RESULTS_DIR, "baseline_cv_results.csv"), index=False)
print("[Phase 1] Saved baseline_cv_results.csv")

cv_rmse_mean = float(cv_results_df['val_rmse'].mean())
cv_rmse_std = float(cv_results_df['val_rmse'].std())
cv_mae_mean = float(cv_results_df['val_mae'].mean())
cv_r2_mean = float(cv_results_df['val_r2'].mean())

baseline_metrics = {
    "model_name": "Multiple Linear Regression (OLS Baseline)",
    "estimator": "sklearn.linear_model.LinearRegression",
    "preprocessing": "ColumnTransformer (OneHotEncoder drop='first' for 5 categoricals + passthrough for 7 numerics/binaries)",
    "training_samples": len(X_train),
    "original_predictors_count": len(all_12_predictors),
    "encoded_input_columns_count": len(all_encoded_names),
    "excluded_columns": ["instant", "dteday", "casual", "registered"],
    "target": "cnt",
    "training_metrics": {
        "MAE": train_mae,
        "MSE": train_mse,
        "RMSE": train_rmse,
        "R2": train_r2
    },
    "dummy_mean_reference_training": {
        "MAE": dummy_train_mae,
        "RMSE": dummy_train_rmse,
        "R2": dummy_train_r2
    },
    "time_series_cv_5fold": {
        "splits": 5,
        "validation_rmse_mean": cv_rmse_mean,
        "validation_rmse_std": cv_rmse_std,
        "validation_mae_mean": cv_mae_mean,
        "validation_r2_mean": cv_r2_mean,
        "dummy_rmse_mean": float(cv_results_df['dummy_val_rmse'].mean())
    },
    "notes": "Baseline model evaluated strictly on Training set and chronological TimeSeriesSplit folds. Calibration and Test sets remain strictly unobserved in Phase 1."
}

with open(os.path.join(RESULTS_DIR, "baseline_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(baseline_metrics, f, indent=2, ensure_ascii=False)

print("[Phase 1] Saved baseline_metrics.json")
print(f"  Training RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}")
print(f"  5-Fold CV RMSE Mean: {cv_rmse_mean:.4f} (std: {cv_rmse_std:.4f})")
print(f"  Dummy Mean CV RMSE: {cv_results_df['dummy_val_rmse'].mean():.4f}")
print("[Phase 1] All computations and artifacts successfully completed!")
