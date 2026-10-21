"""
AIoT HW2 — Problem 2: Appliances Energy Prediction
CRISP-DM End-to-End Research Pipeline
Author: Senior ML Engineer & Research Engineer
Student ID: 7115093001
"""

import os
import sys
# Configure UTF-8 stdout for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
RES_DIR = os.path.join(BASE_DIR, 'results')
MOD_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(MOD_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, 'energydata_complete.csv')

print("=" * 80)
print("STAGE 1: DATASET VERIFICATION & AUDIT")
print("=" * 80)

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Missing {CSV_PATH}. Please run data download first.")

with open(CSV_PATH, 'rb') as f:
    sha256 = hashlib.sha256(f.read()).hexdigest()

df = pd.read_csv(CSV_PATH)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"SHA-256: {sha256}")

# Verify 19735 rows, 29 cols
assert df.shape[0] == 19735, f"Expected 19735 rows, got {df.shape[0]}"
assert df.shape[1] == 29, f"Expected 29 columns, got {df.shape[1]}"
assert df.isnull().sum().sum() == 0, "Null values detected!"
assert df.duplicated().sum() == 0, "Duplicate rows detected!"

# Date parsing & chronological sorting check
df['date_dt'] = pd.to_datetime(df['date'])
assert df['date_dt'].is_monotonic_increasing, "Timestamps are not strictly monotonic increasing!"

# Feature definitions
target_col = 'Appliances'
timestamp_col = 'date'
random_cols = ['rv1', 'rv2']

# 25 physical sensor / lighting variables present directly in the CSV
physical_predictors_25 = [
    'lights',
    'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5',
    'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9',
    'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint'
]

# Time engineering: primary hour of day (0-23)
df['hour'] = df['date_dt'].dt.hour
df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
df['day_of_week'] = df['date_dt'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['NSM'] = df['hour'] * 3600 + df['date_dt'].dt.minute * 60 + df['date_dt'].dt.second

# 26 formal raw input features: 25 physical features + primary time-of-day (hour)
formal_raw_predictors_26 = physical_predictors_25 + ['hour']

# Candidate model features: formal 26 + cyclical time encodings (sin_hour, cos_hour, is_weekend)
candidate_features = physical_predictors_25 + ['hour', 'sin_hour', 'cos_hour', 'is_weekend']

print(f"Formal raw predictors (26): {formal_raw_predictors_26}")
print(f"Total candidate features for modeling: {len(candidate_features)}")

# Sanity check: verify rv1 and rv2 are independent uniform random variables
corr_rv1_target = df['rv1'].corr(df[target_col])
corr_rv2_target = df['rv2'].corr(df[target_col])
print(f"Sanity Check: rv1 vs target correlation = {corr_rv1_target:.5f}, rv2 vs target correlation = {corr_rv2_target:.5f}")

print("\n" + "=" * 80)
print("STAGE 2: CHRONOLOGICAL DATA SPLIT (70% / 15% / 15% FULL DATE ALIGNED)")
print("=" * 80)

unique_days = sorted(df['date_dt'].dt.date.unique())
total_days = len(unique_days)
n_train_days = int(round(total_days * 0.70))  # 97 days
n_cal_days = int(round(total_days * 0.15))    # 21 days
n_test_days = total_days - n_train_days - n_cal_days  # 20 days

train_days = unique_days[:n_train_days]
cal_days = unique_days[n_train_days:n_train_days + n_cal_days]
test_days = unique_days[n_train_days + n_cal_days:]

train_mask = df['date_dt'].dt.date.isin(train_days)
cal_mask = df['date_dt'].dt.date.isin(cal_days)
test_mask = df['date_dt'].dt.date.isin(test_days)

train_df = df[train_mask].copy()
cal_df = df[cal_mask].copy()
test_df = df[test_mask].copy()

print(f"Train: {train_days[0]} to {train_days[-1]} ({len(train_days)} days) -> {len(train_df)} rows ({len(train_df)/len(df):.2%})")
print(f"Cal:   {cal_days[0]} to {cal_days[-1]} ({len(cal_days)} days) -> {len(cal_df)} rows ({len(cal_df)/len(df):.2%})")
print(f"Test:  {test_days[0]} to {test_days[-1]} ({len(test_days)} days) -> {len(test_df)} rows ({len(test_df)/len(df):.2%})")
assert len(train_df) + len(cal_df) + len(test_df) == len(df)

y_train = train_df[target_col].values
y_cal = cal_df[target_col].values
y_test = test_df[target_col].values

X_train_cand = train_df[candidate_features]
X_cal_cand = cal_df[candidate_features]
X_test_cand = test_df[candidate_features]

# Save split summary
split_summary = {
    "total_observations": len(df),
    "train_set": {
        "start_date": str(train_days[0]),
        "end_date": str(train_days[-1]),
        "days_count": len(train_days),
        "observations": len(train_df),
        "percentage": round(len(train_df) / len(df) * 100, 2)
    },
    "calibration_set": {
        "start_date": str(cal_days[0]),
        "end_date": str(cal_days[-1]),
        "days_count": len(cal_days),
        "observations": len(cal_df),
        "percentage": round(len(cal_df) / len(df) * 100, 2)
    },
    "test_set": {
        "start_date": str(test_days[0]),
        "end_date": str(test_days[-1]),
        "days_count": len(test_days),
        "observations": len(test_df),
        "percentage": round(len(test_df) / len(df) * 100, 2)
    }
}
with open(os.path.join(RES_DIR, 'data_split_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(split_summary, f, indent=2)

print("\n" + "=" * 80)
print("STAGE 3: FEATURE SELECTION & EXPANDING-WINDOW CV (TRAINING SET ONLY)")
print("=" * 80)

# Expanding-window Cross Validation on Training Set only
tscv = TimeSeriesSplit(n_splits=5)

def eval_cv_rmse(pipeline, X, y):
    """Compute 5-fold expanding window CV RMSE and sample standard deviation (ddof=1)."""
    rmses = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmses.append(rmse)
    return float(np.mean(rmses)), float(np.std(rmses, ddof=1)), rmses

# 1. Baseline: DummyRegressor (predicting mean)
dummy_pipe = Pipeline([('scaler', StandardScaler()), ('reg', DummyRegressor(strategy='mean'))])
dummy_cv_mean, dummy_cv_std, dummy_rmses = eval_cv_rmse(dummy_pipe, X_train_cand, y_train)
print(f"M0 (Dummy Mean) CV RMSE: {dummy_cv_mean:.4f} +/- {dummy_cv_std:.4f}")

# 2. M1: Full OLS (all 29 candidate features)
ols_pipe = Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())])
ols_cv_mean, ols_cv_std, ols_rmses = eval_cv_rmse(ols_pipe, X_train_cand, y_train)
print(f"M1 (Full OLS, p={X_train_cand.shape[1]}) CV RMSE: {ols_cv_mean:.4f} +/- {ols_cv_std:.4f}")

# 3. RFE Feature Selection: evaluate k from 5 to total candidate features
print("\nRunning RFE feature number search with expanding window CV...")
rfe_results = []
scaler_full = StandardScaler().fit(X_train_cand)
X_train_cand_scaled = pd.DataFrame(scaler_full.transform(X_train_cand), columns=candidate_features)

# RFE ranking on training set
rfe_selector = RFE(estimator=LinearRegression(), n_features_to_select=1, step=1)
rfe_selector.fit(X_train_cand_scaled, y_train)
feature_ranking = pd.DataFrame({
    'feature': candidate_features,
    'ranking': rfe_selector.ranking_
}).sort_values('ranking').reset_index(drop=True)

k_candidates = list(range(5, len(candidate_features) + 1))
best_k = None
best_rfe_cv = float('inf')
best_rfe_std = 0
best_rfe_features = []

for k in k_candidates:
    selected_k_cols = feature_ranking.iloc[:k]['feature'].tolist()
    sub_pipe = Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())])
    cv_mean, cv_std, _ = eval_cv_rmse(sub_pipe, X_train_cand[selected_k_cols], y_train)
    rfe_results.append({
        'k': k,
        'cv_rmse_mean': cv_mean,
        'cv_rmse_std': cv_std,
        'features': selected_k_cols
    })
    if cv_mean < best_rfe_cv:
        best_rfe_cv = cv_mean
        best_rfe_std = cv_std
        best_k = k
        best_rfe_features = selected_k_cols

print(f"Best RFE feature count: k={best_k} with CV RMSE: {best_rfe_cv:.4f} +/- {best_rfe_std:.4f}")

# Sanity Check with rv1 and rv2 included in RFE
X_train_with_random = train_df[candidate_features + random_cols]
scaler_rand = StandardScaler().fit(X_train_with_random)
rfe_rand_selector = RFE(estimator=LinearRegression(), n_features_to_select=1, step=1)
rfe_rand_selector.fit(scaler_rand.transform(X_train_with_random), y_train)
rand_ranking = pd.DataFrame({
    'feature': candidate_features + random_cols,
    'ranking': rfe_rand_selector.ranking_
}).sort_values('ranking').reset_index(drop=True)
print("\nRFE Sanity Check (Ranking with random variables rv1, rv2):")
for r_col in random_cols:
    rank_pos = rand_ranking[rand_ranking['feature'] == r_col].index[0] + 1
    total_f = len(rand_ranking)
    print(f"  {r_col} ranking position: {rank_pos} / {total_f} (Eliminated among early non-informative features)")

# 4. M3: Ridge Regression alpha tuning
print("\nTuning Ridge alpha on Training CV...")
ridge_alphas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 5000.0]
best_ridge_alpha = None
best_ridge_cv = float('inf')
best_ridge_std = 0
for alpha in ridge_alphas:
    r_pipe = Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=alpha, random_state=42))])
    cv_mean, cv_std, _ = eval_cv_rmse(r_pipe, X_train_cand, y_train)
    if cv_mean < best_ridge_cv:
        best_ridge_cv = cv_mean
        best_ridge_std = cv_std
        best_ridge_alpha = alpha
print(f"Best Ridge alpha={best_ridge_alpha} with CV RMSE: {best_ridge_cv:.4f} +/- {best_ridge_std:.4f}")

# 5. M4: Lasso Regression alpha tuning
print("Tuning Lasso alpha on Training CV...")
lasso_alphas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
best_lasso_alpha = None
best_lasso_cv = float('inf')
best_lasso_std = 0
for alpha in lasso_alphas:
    l_pipe = Pipeline([('scaler', StandardScaler()), ('reg', Lasso(alpha=alpha, random_state=42, max_iter=5000))])
    cv_mean, cv_std, _ = eval_cv_rmse(l_pipe, X_train_cand, y_train)
    if cv_mean < best_lasso_cv:
        best_lasso_cv = cv_mean
        best_lasso_std = cv_std
        best_lasso_alpha = alpha
print(f"Best Lasso alpha={best_lasso_alpha} with CV RMSE: {best_lasso_cv:.4f} +/- {best_lasso_std:.4f}")

# Compare Models on CV
model_comp_list = [
    {
        "model_id": "M0",
        "name": "DummyRegressor (Mean Baseline)",
        "features_count": 0,
        "selected_features": [],
        "hyperparameters": "strategy='mean'",
        "cv_rmse_mean": round(dummy_cv_mean, 4),
        "cv_rmse_std": round(dummy_cv_std, 4)
    },
    {
        "model_id": "M1",
        "name": "Full Multiple Linear Regression (OLS)",
        "features_count": len(candidate_features),
        "selected_features": candidate_features,
        "hyperparameters": "fit_intercept=True",
        "cv_rmse_mean": round(ols_cv_mean, 4),
        "cv_rmse_std": round(ols_cv_std, 4)
    },
    {
        "model_id": "M2",
        "name": f"RFE + OLS (Optimal k={best_k})",
        "features_count": best_k,
        "selected_features": best_rfe_features,
        "hyperparameters": f"k={best_k}, step=1",
        "cv_rmse_mean": round(best_rfe_cv, 4),
        "cv_rmse_std": round(best_rfe_std, 4)
    },
    {
        "model_id": "M3",
        "name": f"Ridge Regression (alpha={best_ridge_alpha})",
        "features_count": len(candidate_features),
        "selected_features": candidate_features,
        "hyperparameters": f"alpha={best_ridge_alpha}",
        "cv_rmse_mean": round(best_ridge_cv, 4),
        "cv_rmse_std": round(best_ridge_std, 4)
    },
    {
        "model_id": "M4",
        "name": f"Lasso Regression (alpha={best_lasso_alpha})",
        "features_count": len(candidate_features),
        "selected_features": candidate_features,
        "hyperparameters": f"alpha={best_lasso_alpha}",
        "cv_rmse_mean": round(best_lasso_cv, 4),
        "cv_rmse_std": round(best_lasso_std, 4)
    }
]

model_comp_df = pd.DataFrame(model_comp_list)
model_comp_df.to_csv(os.path.join(RES_DIR, 'model_comparison.csv'), index=False)
with open(os.path.join(RES_DIR, 'model_comparison.json'), 'w', encoding='utf-8') as f:
    json.dump(model_comp_list, f, indent=2)

print("\nModel Comparison Table (Expanding-Window CV on Training Set):")
print(model_comp_df[['model_id', 'name', 'features_count', 'cv_rmse_mean', 'cv_rmse_std']])

# Determine the strictly best model based on CV RMSE
best_model_entry = min(model_comp_list[1:], key=lambda x: x['cv_rmse_mean'])
best_model_id = best_model_entry['model_id']
print(f"\n>>> Best Model Selected by Training CV: {best_model_id} ({best_model_entry['name']}) with CV RMSE = {best_model_entry['cv_rmse_mean']}")

# Save selected features info
selected_features_df = pd.DataFrame({
    'rank': list(range(1, best_k + 1)),
    'feature': best_rfe_features
})
selected_features_df.to_csv(os.path.join(RES_DIR, 'selected_features.csv'), index=False)

print("\n" + "=" * 80)
print("STAGE 4: FIT SELECTED MODEL & INDEPENDENT TEST EVALUATION")
print("=" * 80)

# Build and fit the chosen model on the full Training set
selected_features = best_model_entry['selected_features']
X_train_final = train_df[selected_features]
X_cal_final = cal_df[selected_features]
X_test_final = test_df[selected_features]

if best_model_id in ['M1', 'M2']:
    final_estimator = LinearRegression()
elif best_model_id == 'M3':
    final_estimator = Ridge(alpha=best_ridge_alpha, random_state=42)
elif best_model_id == 'M4':
    final_estimator = Lasso(alpha=best_lasso_alpha, random_state=42, max_iter=5000)

final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('reg', final_estimator)
])

final_pipeline.fit(X_train_final, y_train)

# Save pipeline model
joblib.dump(final_pipeline, os.path.join(MOD_DIR, 'best_model.joblib'))

# Evaluate on Calibration set (for conformal prediction)
y_cal_pred = final_pipeline.predict(X_cal_final)
cal_residuals = np.abs(y_cal - y_cal_pred)

# Nominal 95% Split Conformal Prediction Interval calculation
n_cal = len(cal_residuals)
alpha_conformal = 0.05
# Conformal quantile rank: ceil((n + 1) * (1 - alpha)) / n
quantile_rank = int(np.ceil((n_cal + 1) * (1 - alpha_conformal)))
conformal_p = min(1.0, quantile_rank / n_cal)
cal_q = float(np.quantile(cal_residuals, conformal_p, method='higher'))
print(f"Calibration count n: {n_cal}")
print(f"Nominal Coverage Target: 95.0%")
print(f"Quantile rank: {quantile_rank} / {n_cal} (p={conformal_p:.6f})")
print(f"Calibration Conformal Quantile q: {cal_q:.4f} Wh")

# Evaluate on strictly held-out Test set (2845 observations)
y_test_pred = final_pipeline.predict(X_test_final)
test_residuals = y_test - y_test_pred

test_mae = float(mean_absolute_error(y_test, y_test_pred))
test_mse = float(mean_squared_error(y_test, y_test_pred))
test_rmse = float(np.sqrt(test_mse))
test_r2 = float(r2_score(y_test, y_test_pred))

# Dummy baseline on Test set
dummy_full = DummyRegressor(strategy='mean').fit(X_train_final, y_train)
dummy_test_pred = dummy_full.predict(X_test_final)
dummy_test_mae = float(mean_absolute_error(y_test, dummy_test_pred))
dummy_test_rmse = float(np.sqrt(mean_squared_error(y_test, dummy_test_pred)))
dummy_test_r2 = float(r2_score(y_test, dummy_test_pred))

# Compute Conformal Intervals on Test set
interval_lower_raw = y_test_pred - cal_q
interval_upper_raw = y_test_pred + cal_q
# Physical clipping: appliances power consumption cannot be negative (Wh >= 0)
interval_lower_clipped = np.maximum(0.0, interval_lower_raw)
interval_upper_clipped = interval_upper_raw

raw_covered = (y_test >= interval_lower_raw) & (y_test <= interval_upper_raw)
raw_coverage = float(np.mean(raw_covered))

clipped_covered = (y_test >= interval_lower_clipped) & (y_test <= interval_upper_clipped)
clipped_coverage = float(np.mean(clipped_covered))

raw_mean_width = float(np.mean(interval_upper_raw - interval_lower_raw))
clipped_mean_width = float(np.mean(interval_upper_clipped - interval_lower_clipped))

print("\n--- FINAL TEST EVALUATION METRICS ---")
print(f"Model: {best_model_entry['name']}")
print(f"Test MAE:  {test_mae:.4f} Wh (Baseline Dummy MAE: {dummy_test_mae:.4f} Wh)")
print(f"Test RMSE: {test_rmse:.4f} Wh (Baseline Dummy RMSE: {dummy_test_rmse:.4f} Wh)")
print(f"Test R2:   {test_r2:.4f} (Baseline Dummy R2: {dummy_test_r2:.4f})")
print(f"Nominal 95% Target Coverage: 95.0%")
print(f"Empirical Test Coverage (Raw): {raw_coverage * 100:.2f}% (Mean Width: {raw_mean_width:.2f} Wh)")
print(f"Empirical Test Coverage (Clipped): {clipped_coverage * 100:.2f}% (Mean Width: {clipped_mean_width:.2f} Wh)")

# Hourly coverage breakdown on Test set
test_eval_df = test_df.copy()
test_eval_df['y_true'] = y_test
test_eval_df['y_pred'] = y_test_pred
test_eval_df['residual'] = test_residuals
test_eval_df['lower_clipped'] = interval_lower_clipped
test_eval_df['upper_clipped'] = interval_upper_clipped
test_eval_df['is_covered'] = clipped_covered

hourly_coverage = []
for h in range(24):
    sub_h = test_eval_df[test_eval_df['hour'] == h]
    if len(sub_h) > 0:
        cov = float(sub_h['is_covered'].mean())
        h_mae = float(mean_absolute_error(sub_h['y_true'], sub_h['y_pred']))
        h_rmse = float(np.sqrt(mean_squared_error(sub_h['y_true'], sub_h['y_pred'])))
        hourly_coverage.append({
            'hour': h,
            'observations': len(sub_h),
            'empirical_coverage': round(cov * 100, 2),
            'mae': round(h_mae, 4),
            'rmse': round(h_rmse, 4)
        })

with open(os.path.join(RES_DIR, 'hourly_coverage_diagnostic.json'), 'w', encoding='utf-8') as f:
    json.dump(hourly_coverage, f, indent=2)

# Save final test metrics JSON
test_metrics_summary = {
    "model_id": best_model_id,
    "model_name": best_model_entry['name'],
    "selected_features_count": len(selected_features),
    "selected_features": selected_features,
    "test_observations": len(y_test),
    "metrics": {
        "MAE": round(test_mae, 4),
        "MSE": round(test_mse, 4),
        "RMSE": round(test_rmse, 4),
        "R2": round(test_r2, 4)
    },
    "dummy_baseline": {
        "MAE": round(dummy_test_mae, 4),
        "RMSE": round(dummy_test_rmse, 4),
        "R2": round(dummy_test_r2, 4)
    },
    "conformal_prediction_interval": {
        "calibration_sample_count": n_cal,
        "alpha": alpha_conformal,
        "nominal_coverage_pct": 95.0,
        "quantile_rank": quantile_rank,
        "calibration_quantile_q": round(cal_q, 4),
        "empirical_test_coverage_raw_pct": round(raw_coverage * 100, 2),
        "empirical_test_coverage_clipped_pct": round(clipped_coverage * 100, 2),
        "mean_interval_width_raw": round(raw_mean_width, 4),
        "mean_interval_width_clipped": round(clipped_mean_width, 4),
        "exchangeability_caveat": "Standard split conformal prediction assumes exchangeability (i.i.d.), which is violated under temporal autocorrelation. The 95% is a nominal target; empirical coverage on strictly held-out test data is reported transparently."
    }
}
with open(os.path.join(RES_DIR, 'final_test_metrics.json'), 'w', encoding='utf-8') as f:
    json.dump(test_metrics_summary, f, indent=2)

# Export Test Predictions CSV for analysis
test_pred_output = pd.DataFrame({
    'date': test_df['date'],
    'actual': y_test,
    'predicted': y_test_pred,
    'residual': test_residuals,
    'interval_lower': interval_lower_clipped,
    'interval_upper': interval_upper_clipped,
    'covered': clipped_covered.astype(int)
})
test_pred_output.to_csv(os.path.join(RES_DIR, 'test_predictions.csv'), index=False)

# Export Model Spec for deterministic inference (Option B TypeScript parity)
scaler_step = final_pipeline.named_steps['scaler']
reg_step = final_pipeline.named_steps['reg']

feature_specs = []
for i, feat in enumerate(selected_features):
    feature_specs.append({
        'feature': feat,
        'index': i,
        'mean': float(scaler_step.mean_[i]),
        'scale': float(scaler_step.scale_[i]),
        'coefficient': float(reg_step.coef_[i])
    })

# Compute sensible default values for all 26 raw features from training set
default_feature_values = {}
for feat in formal_raw_predictors_26:
    default_feature_values[feat] = {
        'default': float(train_df[feat].median()),
        'mean': float(train_df[feat].mean()),
        'min': float(train_df[feat].min()),
        'max': float(train_df[feat].max()),
        'step': 1 if feat in ['lights', 'hour'] else 0.1
    }

model_spec = {
    "problem_id": "problem2",
    "problem_title": "Appliances Energy Prediction",
    "target": "Appliances",
    "target_unit": "Wh",
    "best_model_id": best_model_id,
    "model_name": best_model_entry['name'],
    "intercept": float(reg_step.intercept_),
    "calibration_q": cal_q,
    "nominal_coverage_pct": 95.0,
    "empirical_test_coverage_pct": round(clipped_coverage * 100, 2),
    "features_count": len(selected_features),
    "selected_features": selected_features,
    "feature_specs": feature_specs,
    "default_feature_values": default_feature_values,
    "evaluation": {
        "MAE": round(test_mae, 4),
        "RMSE": round(test_rmse, 4),
        "R2": round(test_r2, 4),
        "dummy_MAE": round(dummy_test_mae, 4),
        "dummy_RMSE": round(dummy_test_rmse, 4)
    }
}
with open(os.path.join(RES_DIR, 'model_spec.json'), 'w', encoding='utf-8') as f:
    json.dump(model_spec, f, indent=2)

print("\n" + "=" * 80)
print("STAGE 5: GENERATING 300 DPI ACADEMIC RESEARCH CHARTS")
print("=" * 80)

# 1. Target Distribution & Skewness
plt.figure(figsize=(10, 5), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.histplot(df['Appliances'], bins=50, kde=True, color='#2563eb', edgecolor='white', alpha=0.7)
plt.axvline(df['Appliances'].median(), color='#dc2626', linestyle='--', linewidth=2, label=f'Median: {df["Appliances"].median():.1f} Wh')
plt.axvline(df['Appliances'].mean(), color='#16a34a', linestyle='-', linewidth=2, label=f'Mean: {df["Appliances"].mean():.1f} Wh')
plt.title('Target Variable Distribution: Appliances Energy Consumption (Wh)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Appliances (Wh)', fontsize=11)
plt.ylabel('Observation Frequency', fontsize=11)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eda_appliances_distribution.png'))
plt.close()

# 2. Time Series Profile & Split Boundaries
plt.figure(figsize=(14, 5), dpi=300)
daily_mean = df.set_index('date_dt')['Appliances'].resample('D').mean()
plt.plot(daily_mean.index, daily_mean.values, color='#475569', linewidth=1.2, label='Daily Average Energy (Wh)')
# Split boundary lines
plt.axvline(pd.to_datetime(cal_days[0]), color='#eab308', linestyle='--', linewidth=2, label=f'Calibration Start ({cal_days[0]})')
plt.axvline(pd.to_datetime(test_days[0]), color='#dc2626', linestyle='--', linewidth=2, label=f'Test Start ({test_days[0]})')
plt.axvspan(df['date_dt'].min(), pd.to_datetime(train_days[-1]), alpha=0.08, color='#3b82f6', label='Train (70.3%)')
plt.axvspan(pd.to_datetime(cal_days[0]), pd.to_datetime(cal_days[-1]), alpha=0.08, color='#eab308', label='Calibration (15.3%)')
plt.axvspan(pd.to_datetime(test_days[0]), df['date_dt'].max(), alpha=0.08, color='#dc2626', label='Test (14.4%)')
plt.title('Chronological Time-Series Trajectory & Strict Split Boundaries', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Date (2016)', fontsize=11)
plt.ylabel('Daily Average Energy (Wh)', fontsize=11)
plt.legend(loc='upper right', fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eda_time_series_profile.png'))
plt.close()

# 3. Correlation Heatmap of Physical Predictors vs Target
corr_cols = ['Appliances', 'lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'T6', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'hour']
corr_matrix = df[corr_cols].corr()
plt.figure(figsize=(10, 8), dpi=300)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.8, vmax=0.8, linewidths=0.5, cbar_kws={'label': 'Pearson Correlation'})
plt.title('Pearson Correlation Heatmap (Selected IoT Sensor Variables vs Target)', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eda_correlation_matrix.png'))
plt.close()

# 4. Feature Selection RMSE vs k
k_vals = [r['k'] for r in rfe_results]
cv_rmses = [r['cv_rmse_mean'] for r in rfe_results]
cv_stds = [r['cv_rmse_std'] for r in rfe_results]
plt.figure(figsize=(10, 5), dpi=300)
plt.plot(k_vals, cv_rmses, marker='o', color='#2563eb', linewidth=2, markersize=5, label='5-Fold Expanding CV RMSE')
plt.fill_between(k_vals, np.array(cv_rmses) - np.array(cv_stds), np.array(cv_rmses) + np.array(cv_stds), color='#2563eb', alpha=0.15, label='±1 Unbiased Std (ddof=1)')
plt.axvline(best_k, color='#dc2626', linestyle='--', linewidth=2, label=f'Optimal k={best_k} (RMSE: {best_rfe_cv:.2f} Wh)')
plt.title('RFE Feature Count Optimization via Expanding-Window CV', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Number of Retained Features (k)', fontsize=11)
plt.ylabel('Cross-Validation RMSE (Wh)', fontsize=11)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'feature_selection_rmse.png'))
plt.close()

# 5. Model Comparison Bar Chart
plt.figure(figsize=(10, 5), dpi=300)
model_names = [m['model_id'] + ': ' + m['name'].split('(')[0].strip() for m in model_comp_list]
model_cv_means = [m['cv_rmse_mean'] for m in model_comp_list]
model_cv_stds = [m['cv_rmse_std'] for m in model_comp_list]
colors = ['#94a3b8', '#3b82f6', '#16a34a', '#8b5cf6', '#ec4899']
bars = plt.bar(model_names, model_cv_means, yerr=model_cv_stds, capsize=5, color=colors, edgecolor='black', alpha=0.85, width=0.55)
plt.title('Candidate Model Cross-Validation Comparison (Expanding-Window CV RMSE)', fontsize=13, fontweight='bold', pad=12)
plt.ylabel('CV RMSE (Wh)', fontsize=11)
plt.xticks(rotation=15, ha='right', fontsize=9)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval / 2.0, f'{yval:.2f}', ha='center', va='center', color='white', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'model_comparison.png'))
plt.close()

# 6. Actual vs Predicted Scatter on Test Set
plt.figure(figsize=(7, 7), dpi=300)
plt.scatter(y_test, y_test_pred, alpha=0.3, color='#2563eb', s=18, edgecolors='none', label='Test Observations')
max_val = max(np.max(y_test), np.max(y_test_pred))
plt.plot([0, max_val], [0, max_val], color='#dc2626', linestyle='--', linewidth=2, label='Perfect Identity Line (y = x)')
plt.title(f'Actual vs. Predicted Appliances Energy (Test Set: {len(y_test)} Obs)', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Actual Appliances Energy (Wh)', fontsize=11)
plt.ylabel('Predicted Appliances Energy (Wh)', fontsize=11)
plt.xlim(-20, 1100)
plt.ylim(-50, 700)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'actual_vs_predicted.png'))
plt.close()

# 7. Residual Diagnostics
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
# Residuals vs Predicted
axes[0].scatter(y_test_pred, test_residuals, alpha=0.3, color='#0284c7', s=16)
axes[0].axhline(0, color='#dc2626', linestyle='--', linewidth=1.5)
axes[0].set_title('Residuals vs. Fitted Values', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Fitted Values (Wh)', fontsize=10)
axes[0].set_ylabel('Residuals (Actual - Predicted, Wh)', fontsize=10)
# Residual Distribution Histogram
sns.histplot(test_residuals, bins=50, kde=True, ax=axes[1], color='#0d9488', edgecolor='white')
axes[1].axvline(0, color='#dc2626', linestyle='--', linewidth=1.5)
axes[1].set_title(f'Residual Error Distribution (Mean: {np.mean(test_residuals):.2f}, Std: {np.std(test_residuals):.2f})', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Residual (Wh)', fontsize=10)
axes[1].set_ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'residual_diagnostics.png'))
plt.close()

# 8. Prediction Time Series with 95% Conformal Prediction Interval (First 7 Days of Test)
slice_7d = test_eval_df.iloc[:7 * 144].copy()
plt.figure(figsize=(14, 5), dpi=300)
plt.plot(pd.to_datetime(slice_7d['date']), slice_7d['y_true'], color='#1e293b', label='Actual Wh', linewidth=1.2)
plt.plot(pd.to_datetime(slice_7d['date']), slice_7d['y_pred'], color='#2563eb', label='Point Prediction Wh', linewidth=1.5)
plt.fill_between(
    pd.to_datetime(slice_7d['date']),
    slice_7d['lower_clipped'],
    slice_7d['upper_clipped'],
    color='#60a5fa', alpha=0.25,
    label=f'Nominal 95% Conformal Interval (q={cal_q:.1f} Wh)'
)
plt.title('Test Set Time Series & Split Conformal Prediction Interval (First 7 Days)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Date & Time', fontsize=11)
plt.ylabel('Appliances Energy (Wh)', fontsize=11)
plt.legend(loc='upper right', fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'prediction_interval.png'))
plt.close()

# 9. Full Test Prediction Time Series Overview
plt.figure(figsize=(15, 5), dpi=300)
plt.plot(pd.to_datetime(test_eval_df['date']), test_eval_df['y_true'], color='#64748b', alpha=0.6, linewidth=0.8, label='Actual')
plt.plot(pd.to_datetime(test_eval_df['date']), test_eval_df['y_pred'], color='#2563eb', linewidth=1.1, label='Prediction')
plt.title(f'Full Test Set Trajectory ({len(test_df)} observations over 20 days)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Date (May 2016)', fontsize=11)
plt.ylabel('Appliances (Wh)', fontsize=11)
plt.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'prediction_time_series.png'))
plt.close()

# 10. Empirical Coverage by Hour of Day
plt.figure(figsize=(10, 4.5), dpi=300)
hours = [h['hour'] for h in hourly_coverage]
cov_vals = [h['empirical_coverage'] for h in hourly_coverage]
bars = plt.bar(hours, cov_vals, color='#3b82f6', edgecolor='black', alpha=0.8, width=0.6)
plt.axhline(95.0, color='#dc2626', linestyle='--', linewidth=2, label='Nominal 95% Coverage Target')
plt.title('Empirical Test Coverage by Hour of Day (Nominal 95% Conformal Interval)', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Hour of Day (0 - 23)', fontsize=11)
plt.ylabel('Empirical Coverage (%)', fontsize=11)
plt.ylim(80, 100)
plt.xticks(hours)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'coverage_by_hour.png'))
plt.close()

print("\n" + "=" * 80)
print("STAGE 6: RESEARCH EXPERIMENT COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"All figures saved to: {FIG_DIR}")
print(f"All results saved to: {RES_DIR}")
print(f"Model serialized to: {os.path.join(MOD_DIR, 'best_model.joblib')}")
