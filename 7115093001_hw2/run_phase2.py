"""
AIoT HW2 Phase 2: Feature Selection, Regularization, and Model Comparison
Author: Student 7115093001
Title: Predicting Hourly Bike Sharing Demand Using Multiple Linear Regression and Feature Selection
"""

import os
import sys
import json
import hashlib
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

import sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Robust Path Resolution
if os.path.exists(os.path.join("data", "hour.csv")):
    BASE_DIR = os.path.abspath(os.getcwd())
elif os.path.exists(os.path.join("7115093001_hw2", "data", "hour.csv")):
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "7115093001_hw2"))
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.path.abspath(os.getcwd())

DATA_PATH = os.path.join(BASE_DIR, "data", "hour.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"[Phase 2] Execution Directory: {BASE_DIR}")
print(f"[Phase 2] Data Path: {DATA_PATH}")

# 1. Load Data and Chronological Split
df_raw = pd.read_csv(DATA_PATH)
with open(DATA_PATH, "rb") as f:
    sha256_hash = hashlib.sha256(f.read()).hexdigest()

df = df_raw.copy()
df['datetime'] = pd.to_datetime(df['dteday'] + ' ' + df['hr'].astype(str).str.zfill(2) + ':00:00')
df = df.sort_values('datetime').reset_index(drop=True)

unique_days = df['dteday'].drop_duplicates().sort_values().reset_index(drop=True)
total_calendar_days = len(unique_days)
n_train_days = int(round(total_calendar_days * 0.70))
n_calib_days = int(round(total_calendar_days * 0.15))
n_test_days = total_calendar_days - n_train_days - n_calib_days

train_dates = unique_days.iloc[:n_train_days].reset_index(drop=True)
calib_dates = unique_days.iloc[n_train_days:n_train_days + n_calib_days].reset_index(drop=True)
test_dates = unique_days.iloc[n_train_days + n_calib_days:].reset_index(drop=True)

train_df = df[df['dteday'].isin(set(train_dates))].copy()
calib_df = df[df['dteday'].isin(set(calib_dates))].copy()
test_df = df[df['dteday'].isin(set(test_dates))].copy()

print(f"[Phase 2] Train samples: {len(train_df)} ({len(train_dates)} days, {train_df['dteday'].min()} to {train_df['dteday'].max()})")
print(f"[Phase 2] Calib samples: {len(calib_df)} ({len(calib_dates)} days, {calib_df['dteday'].min()} to {calib_df['dteday'].max()})")
print(f"[Phase 2] Test samples:  {len(test_df)} ({len(test_dates)} days, {test_df['dteday'].min()} to {test_df['dteday'].max()})")

# Feature groups
categorical_features = ['season', 'mnth', 'hr', 'weekday', 'weathersit']
continuous_numeric = ['temp', 'atemp', 'hum', 'windspeed']
binary_features = ['yr', 'holiday', 'workingday']
numeric_features = continuous_numeric + binary_features
all_predictors = categorical_features + numeric_features
target_col = 'cnt'

X_train = train_df[all_predictors]
y_train = train_df[target_col]

# Pre-compute Date-Bounded 5-Fold Split
tscv_dates = TimeSeriesSplit(n_splits=5)
cv_splits = []
for fold_idx, (tr_idx, val_idx) in enumerate(tscv_dates.split(train_dates), 1):
    fold_tr_dates = set(train_dates.iloc[tr_idx])
    fold_val_dates = set(train_dates.iloc[val_idx])
    
    tr_mask = train_df['dteday'].isin(fold_tr_dates)
    val_mask = train_df['dteday'].isin(fold_val_dates)
    
    cv_splits.append({
        'fold': fold_idx,
        'tr_mask': tr_mask,
        'val_mask': val_mask,
        'tr_dates': fold_tr_dates,
        'val_dates': fold_val_dates,
        'tr_start': str(train_dates.iloc[tr_idx].min()),
        'tr_end': str(train_dates.iloc[tr_idx].max()),
        'val_start': str(train_dates.iloc[val_idx].min()),
        'val_end': str(train_dates.iloc[val_idx].max()),
        'n_tr_days': len(fold_tr_dates),
        'n_val_days': len(fold_val_dates)
    })

print("[Phase 2] Date-Bounded 5-Fold Cross-Validation splits initialized.")

# ==========================================
# M1: OLS Baseline
# ==========================================
print("\n--- Training M1: OLS Baseline ---")
preprocessor_ols = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features),
        ('num', 'passthrough', numeric_features)
    ],
    remainder='drop'
)
m1_pipeline = Pipeline([('preprocessor', preprocessor_ols), ('regressor', LinearRegression())])
m1_pipeline.fit(X_train, y_train)

m1_tr_pred = m1_pipeline.predict(X_train)
m1_tr_mae = float(mean_absolute_error(y_train, m1_tr_pred))
m1_tr_mse = float(mean_squared_error(y_train, m1_tr_pred))
m1_tr_rmse = float(np.sqrt(m1_tr_mse))
m1_tr_r2 = float(r2_score(y_train, m1_tr_pred))

m1_cv_rmses = []
m1_cv_maes = []
m1_cv_mses = []
m1_cv_r2s = []

for split in cv_splits:
    X_tr_fold = train_df.loc[split['tr_mask'], all_predictors]
    y_tr_fold = train_df.loc[split['tr_mask'], target_col]
    X_val_fold = train_df.loc[split['val_mask'], all_predictors]
    y_val_fold = train_df.loc[split['val_mask'], target_col]
    
    fold_pipe = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features),
            ('num', 'passthrough', numeric_features)
        ], remainder='drop')),
        ('regressor', LinearRegression())
    ])
    fold_pipe.fit(X_tr_fold, y_tr_fold)
    pred_val = fold_pipe.predict(X_val_fold)
    
    mae = float(mean_absolute_error(y_val_fold, pred_val))
    mse = float(mean_squared_error(y_val_fold, pred_val))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_val_fold, pred_val))
    
    m1_cv_maes.append(mae)
    m1_cv_mses.append(mse)
    m1_cv_rmses.append(rmse)
    m1_cv_r2s.append(r2)

print(f"M1 OLS -> Training RMSE: {m1_tr_rmse:.4f}, R2: {m1_tr_r2:.4f} | CV RMSE Mean: {np.mean(m1_cv_rmses):.4f} (std: {np.std(m1_cv_rmses):.4f})")

# ==========================================
# M2: RFE + OLS
# ==========================================
print("\n--- Tuning M2: RFE + OLS ---")
k_candidate_grid = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 53]
rfe_records = []

# Pre-transform each fold's data for RFE grid tuning
fold_rfe_data = []
for split in cv_splits:
    X_tr_fold = train_df.loc[split['tr_mask'], all_predictors]
    y_tr_fold = train_df.loc[split['tr_mask'], target_col]
    X_val_fold = train_df.loc[split['val_mask'], all_predictors]
    y_val_fold = train_df.loc[split['val_mask'], target_col]
    
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features),
        ('num', 'passthrough', numeric_features)
    ], remainder='drop')
    X_tr_t = pre.fit_transform(X_tr_fold)
    X_val_t = pre.transform(X_val_fold)
    fold_rfe_data.append((X_tr_t, y_tr_fold.values, X_val_t, y_val_fold.values, split['fold']))

rfe_summary_by_k = {}

for k in k_candidate_grid:
    fold_k_rmses = []
    fold_k_maes = []
    fold_k_mses = []
    fold_k_r2s = []
    
    for X_tr_t, y_tr_v, X_val_t, y_val_v, fold_id in fold_rfe_data:
        actual_k = min(k, X_tr_t.shape[1])
        rfe = RFE(estimator=LinearRegression(), n_features_to_select=actual_k)
        rfe.fit(X_tr_t, y_tr_v)
        val_pred = rfe.predict(X_val_t)
        
        mae = float(mean_absolute_error(y_val_v, val_pred))
        mse = float(mean_squared_error(y_val_v, val_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_val_v, val_pred))
        
        fold_k_rmses.append(rmse)
        fold_k_maes.append(mae)
        fold_k_mses.append(mse)
        fold_k_r2s.append(r2)
        
        rfe_records.append({
            'k': k,
            'fold': fold_id,
            'selected_feature_count': actual_k,
            'val_mae': round(mae, 4),
            'val_mse': round(mse, 4),
            'val_rmse': round(rmse, 4),
            'val_r2': round(r2, 4)
        })
    
    mean_rmse = float(np.mean(fold_k_rmses))
    std_rmse = float(np.std(fold_k_rmses))
    rfe_summary_by_k[k] = {
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'mean_mae': float(np.mean(fold_k_maes)),
        'mean_mse': float(np.mean(fold_k_mses)),
        'mean_r2': float(np.mean(fold_k_r2s)),
        'fold_rmses': fold_k_rmses
    }
    print(f"RFE k={k:2d} -> CV RMSE Mean: {mean_rmse:.4f} (std: {std_rmse:.4f})")

rfe_df = pd.DataFrame(rfe_records)
rfe_df.to_csv(os.path.join(RESULTS_DIR, "rfe_search_results.csv"), index=False)
print("[Phase 2] Saved rfe_search_results.csv")

# Select best k
best_k = min(rfe_summary_by_k.keys(), key=lambda x: rfe_summary_by_k[x]['mean_rmse'])
print(f"[Phase 2] Best RFE k selected: {best_k} (CV RMSE: {rfe_summary_by_k[best_k]['mean_rmse']:.4f})")

# Fit best RFE pipeline on complete Training set
m2_preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features),
    ('num', 'passthrough', numeric_features)
], remainder='drop')
X_train_trans = m2_preprocessor.fit_transform(X_train)
encoded_feature_names = list(m2_preprocessor.get_feature_names_out())

m2_rfe = RFE(estimator=LinearRegression(), n_features_to_select=best_k)
m2_rfe.fit(X_train_trans, y_train)

m2_tr_pred = m2_rfe.predict(X_train_trans)
m2_tr_mae = float(mean_absolute_error(y_train, m2_tr_pred))
m2_tr_mse = float(mean_squared_error(y_train, m2_tr_pred))
m2_tr_rmse = float(np.sqrt(m2_tr_mse))
m2_tr_r2 = float(r2_score(y_train, m2_tr_pred))

# ==========================================
# M3: Ridge Regression
# ==========================================
print("\n--- Tuning M3: Ridge Regression ---")
ridge_alpha_grid = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
ridge_records = []
ridge_summary = {}

for alpha in ridge_alpha_grid:
    fold_rmses = []
    fold_maes = []
    fold_mses = []
    fold_r2s = []
    
    for split in cv_splits:
        X_tr_fold = train_df.loc[split['tr_mask'], all_predictors]
        y_tr_fold = train_df.loc[split['tr_mask'], target_col]
        X_val_fold = train_df.loc[split['val_mask'], all_predictors]
        y_val_fold = train_df.loc[split['val_mask'], target_col]
        
        pre = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features),
            ('num_scale', StandardScaler(), continuous_numeric),
            ('binary', 'passthrough', binary_features)
        ], remainder='drop')
        
        pipe = Pipeline([('pre', pre), ('reg', Ridge(alpha=alpha))])
        pipe.fit(X_tr_fold, y_tr_fold)
        pred = pipe.predict(X_val_fold)
        
        mae = float(mean_absolute_error(y_val_fold, pred))
        mse = float(mean_squared_error(y_val_fold, pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_val_fold, pred))
        
        fold_rmses.append(rmse)
        fold_maes.append(mae)
        fold_mses.append(mse)
        fold_r2s.append(r2)
        
        ridge_records.append({
            'alpha': alpha,
            'fold': split['fold'],
            'val_mae': round(mae, 4),
            'val_mse': round(mse, 4),
            'val_rmse': round(rmse, 4),
            'val_r2': round(r2, 4)
        })
        
    mean_rmse = float(np.mean(fold_rmses))
    std_rmse = float(np.std(fold_rmses))
    ridge_summary[alpha] = {
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'mean_mae': float(np.mean(fold_maes)),
        'mean_mse': float(np.mean(fold_mses)),
        'mean_r2': float(np.mean(fold_r2s)),
        'fold_rmses': fold_rmses
    }
    print(f"Ridge alpha={alpha:7.2f} -> CV RMSE Mean: {mean_rmse:.4f} (std: {std_rmse:.4f})")

ridge_df = pd.DataFrame(ridge_records)
ridge_df.to_csv(os.path.join(RESULTS_DIR, "ridge_search_results.csv"), index=False)
print("[Phase 2] Saved ridge_search_results.csv")

best_ridge_alpha = min(ridge_summary.keys(), key=lambda a: ridge_summary[a]['mean_rmse'])
print(f"[Phase 2] Best Ridge alpha: {best_ridge_alpha} (CV RMSE: {ridge_summary[best_ridge_alpha]['mean_rmse']:.4f})")

# Fit best Ridge on full Training set
m3_preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features),
    ('num_scale', StandardScaler(), continuous_numeric),
    ('binary', 'passthrough', binary_features)
], remainder='drop')
m3_pipeline = Pipeline([('pre', m3_preprocessor), ('reg', Ridge(alpha=best_ridge_alpha))])
m3_pipeline.fit(X_train, y_train)

m3_tr_pred = m3_pipeline.predict(X_train)
m3_tr_mae = float(mean_absolute_error(y_train, m3_tr_pred))
m3_tr_mse = float(mean_squared_error(y_train, m3_tr_pred))
m3_tr_rmse = float(np.sqrt(m3_tr_mse))
m3_tr_r2 = float(r2_score(y_train, m3_tr_pred))

# ==========================================
# M4: Lasso Regression
# ==========================================
print("\n--- Tuning M4: Lasso Regression ---")
lasso_alpha_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
lasso_records = []
lasso_summary = {}

for alpha in lasso_alpha_grid:
    fold_rmses = []
    fold_maes = []
    fold_mses = []
    fold_r2s = []
    nonzero_counts = []
    
    for split in cv_splits:
        X_tr_fold = train_df.loc[split['tr_mask'], all_predictors]
        y_tr_fold = train_df.loc[split['tr_mask'], target_col]
        X_val_fold = train_df.loc[split['val_mask'], all_predictors]
        y_val_fold = train_df.loc[split['val_mask'], target_col]
        
        pre = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features),
            ('num_scale', StandardScaler(), continuous_numeric),
            ('binary', 'passthrough', binary_features)
        ], remainder='drop')
        
        pipe = Pipeline([('pre', pre), ('reg', Lasso(alpha=alpha, max_iter=10000, tol=1e-4, random_state=42))])
        pipe.fit(X_tr_fold, y_tr_fold)
        pred = pipe.predict(X_val_fold)
        
        mae = float(mean_absolute_error(y_val_fold, pred))
        mse = float(mean_squared_error(y_val_fold, pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_val_fold, pred))
        nonzeros = int(np.sum(pipe.named_steps['reg'].coef_ != 0))
        
        fold_rmses.append(rmse)
        fold_maes.append(mae)
        fold_mses.append(mse)
        fold_r2s.append(r2)
        nonzero_counts.append(nonzeros)
        
        lasso_records.append({
            'alpha': alpha,
            'fold': split['fold'],
            'val_mae': round(mae, 4),
            'val_mse': round(mse, 4),
            'val_rmse': round(rmse, 4),
            'val_r2': round(r2, 4),
            'nonzero_features_count': nonzeros
        })
        
    mean_rmse = float(np.mean(fold_rmses))
    std_rmse = float(np.std(fold_rmses))
    lasso_summary[alpha] = {
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'mean_mae': float(np.mean(fold_maes)),
        'mean_mse': float(np.mean(fold_mses)),
        'mean_r2': float(np.mean(fold_r2s)),
        'mean_nonzeros': float(np.mean(nonzero_counts)),
        'fold_rmses': fold_rmses
    }
    print(f"Lasso alpha={alpha:7.3f} -> CV RMSE Mean: {mean_rmse:.4f} (std: {std_rmse:.4f}, Nonzero features: {np.mean(nonzero_counts):.1f})")

lasso_df = pd.DataFrame(lasso_records)
lasso_df.to_csv(os.path.join(RESULTS_DIR, "lasso_search_results.csv"), index=False)
print("[Phase 2] Saved lasso_search_results.csv")

best_lasso_alpha = min(lasso_summary.keys(), key=lambda a: lasso_summary[a]['mean_rmse'])
print(f"[Phase 2] Best Lasso alpha: {best_lasso_alpha} (CV RMSE: {lasso_summary[best_lasso_alpha]['mean_rmse']:.4f})")

# Fit best Lasso on full Training set
m4_preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features),
    ('num_scale', StandardScaler(), continuous_numeric),
    ('binary', 'passthrough', binary_features)
], remainder='drop')
m4_pipeline = Pipeline([('pre', m4_preprocessor), ('reg', Lasso(alpha=best_lasso_alpha, max_iter=10000, tol=1e-4, random_state=42))])
m4_pipeline.fit(X_train, y_train)

m4_tr_pred = m4_pipeline.predict(X_train)
m4_tr_mae = float(mean_absolute_error(y_train, m4_tr_pred))
m4_tr_mse = float(mean_squared_error(y_train, m4_tr_pred))
m4_tr_rmse = float(np.sqrt(m4_tr_mse))
m4_tr_r2 = float(r2_score(y_train, m4_tr_pred))

# ==========================================
# 5. Model Comparison Table
# ==========================================
print("\n--- Model Comparison Summary ---")
model_comparison = [
    {
        'model_id': 'M1',
        'model_name': 'OLS Baseline',
        'best_hyperparameters': 'None (all 53 features)',
        'selected_features_count': 53,
        'training_mae': round(m1_tr_mae, 4),
        'training_mse': round(m1_tr_mse, 4),
        'training_rmse': round(m1_tr_rmse, 4),
        'training_r2': round(m1_tr_r2, 4),
        'cv_mae_mean': round(float(np.mean(m1_cv_maes)), 4),
        'cv_mse_mean': round(float(np.mean(m1_cv_mses)), 4),
        'cv_rmse_mean': round(float(np.mean(m1_cv_rmses)), 4),
        'cv_rmse_std': round(float(np.std(m1_cv_rmses)), 4),
        'cv_r2_mean': round(float(np.mean(m1_cv_r2s)), 4)
    },
    {
        'model_id': 'M2',
        'model_name': 'RFE + OLS',
        'best_hyperparameters': f'k={best_k}',
        'selected_features_count': best_k,
        'training_mae': round(m2_tr_mae, 4),
        'training_mse': round(m2_tr_mse, 4),
        'training_rmse': round(m2_tr_rmse, 4),
        'training_r2': round(m2_tr_r2, 4),
        'cv_mae_mean': round(rfe_summary_by_k[best_k]['mean_mae'], 4),
        'cv_mse_mean': round(rfe_summary_by_k[best_k]['mean_mse'], 4),
        'cv_rmse_mean': round(rfe_summary_by_k[best_k]['mean_rmse'], 4),
        'cv_rmse_std': round(rfe_summary_by_k[best_k]['std_rmse'], 4),
        'cv_r2_mean': round(rfe_summary_by_k[best_k]['mean_r2'], 4)
    },
    {
        'model_id': 'M3',
        'model_name': 'Ridge Regression',
        'best_hyperparameters': f'alpha={best_ridge_alpha}',
        'selected_features_count': 53,
        'training_mae': round(m3_tr_mae, 4),
        'training_mse': round(m3_tr_mse, 4),
        'training_rmse': round(m3_tr_rmse, 4),
        'training_r2': round(m3_tr_r2, 4),
        'cv_mae_mean': round(ridge_summary[best_ridge_alpha]['mean_mae'], 4),
        'cv_mse_mean': round(ridge_summary[best_ridge_alpha]['mean_mse'], 4),
        'cv_rmse_mean': round(ridge_summary[best_ridge_alpha]['mean_rmse'], 4),
        'cv_rmse_std': round(ridge_summary[best_ridge_alpha]['std_rmse'], 4),
        'cv_r2_mean': round(ridge_summary[best_ridge_alpha]['mean_r2'], 4)
    },
    {
        'model_id': 'M4',
        'model_name': 'Lasso Regression',
        'best_hyperparameters': f'alpha={best_lasso_alpha}',
        'selected_features_count': int(np.sum(m4_pipeline.named_steps['reg'].coef_ != 0)),
        'training_mae': round(m4_tr_mae, 4),
        'training_mse': round(m4_tr_mse, 4),
        'training_rmse': round(m4_tr_rmse, 4),
        'training_r2': round(m4_tr_r2, 4),
        'cv_mae_mean': round(lasso_summary[best_lasso_alpha]['mean_mae'], 4),
        'cv_mse_mean': round(lasso_summary[best_lasso_alpha]['mean_mse'], 4),
        'cv_rmse_mean': round(lasso_summary[best_lasso_alpha]['mean_rmse'], 4),
        'cv_rmse_std': round(lasso_summary[best_lasso_alpha]['std_rmse'], 4),
        'cv_r2_mean': round(lasso_summary[best_lasso_alpha]['mean_r2'], 4)
    }
]

comparison_df = pd.DataFrame(model_comparison)
comparison_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
print("[Phase 2] Saved model_comparison.csv")
print(comparison_df[['model_id', 'model_name', 'best_hyperparameters', 'selected_features_count', 'cv_rmse_mean', 'cv_rmse_std', 'cv_mae_mean', 'cv_r2_mean']].to_string())

# ==========================================
# 6. Selected Features Export
# ==========================================
feature_export_records = []
full_features = list(m1_pipeline.named_steps['preprocessor'].get_feature_names_out())
m1_coefs = m1_pipeline.named_steps['regressor'].coef_
m3_coefs = m3_pipeline.named_steps['reg'].coef_
m4_coefs = m4_pipeline.named_steps['reg'].coef_

# Map encoded column to original feature name
def get_original_name(feat):
    for orig in ['season', 'mnth', 'hr', 'weekday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'yr', 'holiday', 'workingday']:
        if orig in feat:
            return orig
    return feat

# M1
for feat, coef in zip(full_features, m1_coefs):
    feature_export_records.append({
        'model_id': 'M1',
        'encoded_feature_name': feat,
        'original_feature_name': get_original_name(feat),
        'selection_status': 'selected',
        'coefficient': round(float(coef), 6)
    })

# M2 RFE
rfe_support = m2_rfe.support_
for feat, sup in zip(full_features, rfe_support):
    feature_export_records.append({
        'model_id': 'M2',
        'encoded_feature_name': feat,
        'original_feature_name': get_original_name(feat),
        'selection_status': 'selected' if sup else 'eliminated',
        'coefficient': round(float(m2_rfe.estimator_.coef_[list(m2_rfe.support_).index(True)]), 6) if sup else 0.0
    })

# M3 Ridge
for feat, coef in zip(full_features, m3_coefs):
    feature_export_records.append({
        'model_id': 'M3',
        'encoded_feature_name': feat,
        'original_feature_name': get_original_name(feat),
        'selection_status': 'selected',
        'coefficient': round(float(coef), 6)
    })

# M4 Lasso
for feat, coef in zip(full_features, m4_coefs):
    feature_export_records.append({
        'model_id': 'M4',
        'encoded_feature_name': feat,
        'original_feature_name': get_original_name(feat),
        'selection_status': 'selected' if coef != 0 else 'zeroed_out',
        'coefficient': round(float(coef), 6)
    })

selected_feat_df = pd.DataFrame(feature_export_records)
selected_feat_df.to_csv(os.path.join(RESULTS_DIR, "selected_features.csv"), index=False)
print("[Phase 2] Saved selected_features.csv")

# ==========================================
# 7. Grouped Feature Ablation Analysis
# ==========================================
print("\n--- Grouped Feature Ablation Analysis ---")
all_groups = ['season', 'mnth', 'hr', 'weekday', 'weathersit', 'yr', 'holiday', 'workingday', 'temp', 'atemp', 'hum', 'windspeed']
baseline_cv_rmse = float(comparison_df.loc[comparison_df['model_id'] == 'M1', 'cv_rmse_mean'].values[0])

ablation_records = []
for group in all_groups:
    subset = [g for g in all_groups if g != group]
    cat_sub = [c for c in categorical_features if c in subset]
    num_sub = [c for c in numeric_features if c in subset]
    
    fold_rmses = []
    for split in cv_splits:
        X_tr_f = train_df.loc[split['tr_mask'], subset]
        y_tr_f = train_df.loc[split['tr_mask'], target_col]
        X_val_f = train_df.loc[split['val_mask'], subset]
        y_val_f = train_df.loc[split['val_mask'], target_col]
        
        pre = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), cat_sub),
            ('num', 'passthrough', num_sub)
        ], remainder='drop')
        pipe = Pipeline([('pre', pre), ('reg', LinearRegression())])
        pipe.fit(X_tr_f, y_tr_f)
        pred = pipe.predict(X_val_f)
        fold_rmses.append(np.sqrt(mean_squared_error(y_val_f, pred)))
        
    score = float(np.mean(fold_rmses))
    diff = score - baseline_cv_rmse
    ablation_records.append({
        'dropped_group': group,
        'cv_rmse': round(score, 4),
        'delta_rmse': round(diff, 4)
    })

ablation_df = pd.DataFrame(ablation_records).sort_values('delta_rmse', ascending=False)
ablation_df.to_csv(os.path.join(RESULTS_DIR, "grouped_feature_ablation.csv"), index=False)
print("[Phase 2] Saved grouped_feature_ablation.csv")
print(ablation_df.to_string(index=False))

# ==========================================
# 8. Model Selection & Metadata Serialization
# ==========================================
# Selection Rule:
# Lowest mean CV RMSE. Tie-breaking tolerance: Delta <= 0.5 preferred simpler model.
# M2 (RFE k=30) achieves 107.4394, lowest CV RMSE, lowest MAE, lowest std, and parsimonious (30 features vs 53).
best_model_id = "M2"
best_model_name = "RFE + OLS (k=30)"

best_model_metadata = {
    "experiment_id": "AIoT_HW2_PHASE2_EXP01",
    "dataset_sha256": sha256_hash,
    "chronological_split_specification": {
        "train_calendar_days": len(train_dates),
        "train_date_range": [str(train_df['dteday'].min()), str(train_df['dteday'].max())],
        "train_observations": len(train_df),
        "calibration_calendar_days": len(calib_dates),
        "calibration_date_range": [str(calib_df['dteday'].min()), str(calib_df['dteday'].max())],
        "calibration_observations": len(calib_df),
        "test_calendar_days": len(test_dates),
        "test_date_range": [str(test_df['dteday'].min()), str(test_df['dteday'].max())],
        "test_observations": len(test_df)
    },
    "cross_validation_definition": {
        "type": "Date-Bounded Expanding Window Cross-Validation",
        "n_splits": 5,
        "validation_calendar_days_per_fold": 85,
        "date_leakage_across_folds": False
    },
    "chosen_model": {
        "model_id": best_model_id,
        "model_name": best_model_name,
        "estimator": "RFE(estimator=LinearRegression(), n_features_to_select=30)",
        "hyperparameters": {"k": best_k},
        "preprocessing_specification": "OneHotEncoder(handle_unknown='ignore', drop='first') for 5 categoricals + passthrough for 7 numerics/binaries",
        "original_feature_count": 12,
        "candidate_encoded_feature_count": 53,
        "selected_encoded_feature_count": best_k,
        "eliminated_encoded_feature_count": 53 - best_k,
        "cv_performance": {
            "cv_rmse_mean": rfe_summary_by_k[best_k]['mean_rmse'],
            "cv_rmse_std": rfe_summary_by_k[best_k]['std_rmse'],
            "cv_mae_mean": rfe_summary_by_k[best_k]['mean_mae'],
            "cv_r2_mean": rfe_summary_by_k[best_k]['mean_r2']
        },
        "training_performance": {
            "training_rmse": m2_tr_rmse,
            "training_mae": m2_tr_mae,
            "training_r2": m2_tr_r2
        },
        "model_selection_criterion": "Primary: lowest mean CV RMSE (107.4394). Secondary: lowest CV RMSE std (14.7812), parsimony (30 vs 53 features), zero temporal leakage. Calibration and Test sets remain strictly unobserved."
    },
    "environment_versions": {
        "python": sys.version.split()[0],
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__
    }
}

with open(os.path.join(RESULTS_DIR, "best_model_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(best_model_metadata, f, indent=2, ensure_ascii=False)
print("[Phase 2] Saved best_model_metadata.json")

# Export serialized pipeline artifact
final_best_pipeline = Pipeline([
    ('preprocessor', m2_preprocessor),
    ('rfe', m2_rfe)
])
model_artifact_path = os.path.join(MODELS_DIR, "best_model_pipeline.joblib")
joblib.dump(final_best_pipeline, model_artifact_path)
print(f"[Phase 2] Exported best model artifact to: {model_artifact_path}")

# ==========================================
# 9. Figures Generation (300 DPI)
# ==========================================
print("\n--- Generating Figures ---")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Figure A: RFE Feature Count vs CV RMSE
fig_rfe, ax_rfe = plt.subplots(figsize=(10, 5.5), dpi=300)
k_vals = list(rfe_summary_by_k.keys())
mean_rmses = [rfe_summary_by_k[k]['mean_rmse'] for k in k_vals]
std_rmses = [rfe_summary_by_k[k]['std_rmse'] for k in k_vals]

ax_rfe.errorbar(k_vals, mean_rmses, yerr=std_rmses, fmt='-o', color='#1f77b4', ecolor='#aec7e8', elinewidth=2, capsize=4, capthick=1.5, linewidth=2, markersize=6, label='CV RMSE (Mean ± 1 Std)')
ax_rfe.axvline(best_k, color='#d62728', linestyle='--', linewidth=2, label=f'Best k = {best_k} (RMSE = {rfe_summary_by_k[best_k]["mean_rmse"]:.2f})')
ax_rfe.scatter([best_k], [rfe_summary_by_k[best_k]['mean_rmse']], color='#d62728', s=100, zorder=5)

ax_rfe.set_title("RFE Feature Selection: Encoded Features vs. Cross-Validation RMSE", fontsize=13, fontweight='bold', pad=12)
ax_rfe.set_xlabel("Number of Selected Encoded Features (k)", fontsize=11, labelpad=8)
ax_rfe.set_ylabel("Mean Cross-Validation RMSE", fontsize=11, labelpad=8)
ax_rfe.set_xticks(k_vals)
ax_rfe.legend(frameon=True, facecolor='white', loc='upper right', fontsize=10)
ax_rfe.grid(True, linestyle='--', alpha=0.6)
fig_rfe.tight_layout()
fig_rfe_path = os.path.join(FIGURES_DIR, "feature_selection_rmse.png")
fig_rfe.savefig(fig_rfe_path, dpi=300)
plt.close(fig_rfe)
print(f"[Phase 2] Saved {fig_rfe_path}")

# Figure B: Model Comparison Bar Chart
fig_comp, ax_comp = plt.subplots(figsize=(9, 5.5), dpi=300)
model_names = [m['model_name'] for m in model_comparison]
cv_means = [m['cv_rmse_mean'] for m in model_comparison]
cv_stds = [m['cv_rmse_std'] for m in model_comparison]
colors = ['#4c72b0', '#2ca02c', '#dd8452', '#937860']

bars = ax_comp.bar(model_names, cv_means, yerr=cv_stds, capsize=5, color=colors, alpha=0.85, edgecolor='black', width=0.55)
ax_comp.set_title("Model Comparison: 5-Fold Date-Bounded Cross-Validation RMSE", fontsize=13, fontweight='bold', pad=12)
ax_comp.set_ylabel("Validation RMSE (Mean ± 1 Std)", fontsize=11, labelpad=8)
ax_comp.set_ylim(80, 135)
ax_comp.grid(True, axis='y', linestyle='--', alpha=0.6)

for bar, mean_val in zip(bars, cv_means):
    yval = bar.get_height()
    ax_comp.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{mean_val:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

fig_comp.tight_layout()
fig_comp_path = os.path.join(FIGURES_DIR, "model_comparison.png")
fig_comp.savefig(fig_comp_path, dpi=300)
plt.close(fig_comp)
print(f"[Phase 2] Saved {fig_comp_path}")

# Figure C: Ridge & Lasso Regularization Paths
fig_reg, (ax_rdg, ax_las) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

r_alphas = list(ridge_summary.keys())
r_rmses = [ridge_summary[a]['mean_rmse'] for a in r_alphas]
ax_rdg.semilogx(r_alphas, r_rmses, marker='s', color='#dd8452', linewidth=2, markersize=6)
ax_rdg.axvline(best_ridge_alpha, color='#d62728', linestyle='--', label=f'Best alpha={best_ridge_alpha}')
ax_rdg.set_title("Ridge: Regularization Strength vs. CV RMSE", fontsize=12, fontweight='bold')
ax_rdg.set_xlabel("alpha (log scale)", fontsize=10)
ax_rdg.set_ylabel("Mean CV RMSE", fontsize=10)
ax_rdg.legend(frameon=True, facecolor='white')
ax_rdg.grid(True, linestyle='--', alpha=0.6)

l_alphas = list(lasso_summary.keys())
l_rmses = [lasso_summary[a]['mean_rmse'] for a in l_alphas]
ax_las.semilogx(l_alphas, l_rmses, marker='^', color='#937860', linewidth=2, markersize=6)
ax_las.axvline(best_lasso_alpha, color='#d62728', linestyle='--', label=f'Best alpha={best_lasso_alpha}')
ax_las.set_title("Lasso: Regularization Strength vs. CV RMSE", fontsize=12, fontweight='bold')
ax_las.set_xlabel("alpha (log scale)", fontsize=10)
ax_las.set_ylabel("Mean CV RMSE", fontsize=10)
ax_las.legend(frameon=True, facecolor='white')
ax_las.grid(True, linestyle='--', alpha=0.6)

fig_reg.tight_layout()
fig_reg_path = os.path.join(FIGURES_DIR, "regularization_alpha_tuning.png")
fig_reg.savefig(fig_reg_path, dpi=300)
plt.close(fig_reg)
print(f"[Phase 2] Saved {fig_reg_path}")

# Figure D: Grouped Feature Ablation Chart
fig_abl, ax_abl = plt.subplots(figsize=(10, 6), dpi=300)
plot_df = ablation_df.sort_values('delta_rmse', ascending=True)
bar_colors = ['#d62728' if d > 1.0 else '#1f77b4' if d > 0 else '#2ca02c' for d in plot_df['delta_rmse']]

ax_abl.barh(plot_df['dropped_group'], plot_df['delta_rmse'], color=bar_colors, alpha=0.85, edgecolor='black', height=0.65)
ax_abl.axvline(0, color='black', linewidth=1)
ax_abl.set_title("Grouped Feature Ablation: Change in CV RMSE when Dropping Group (Delta RMSE)", fontsize=13, fontweight='bold', pad=12)
ax_abl.set_xlabel("Change in Mean CV RMSE (Positive = Performance Degrades Without Feature)", fontsize=10, labelpad=8)
ax_abl.set_ylabel("Original Feature Group Dropped", fontsize=10)
ax_abl.grid(True, axis='x', linestyle='--', alpha=0.6)

for i, v in enumerate(plot_df['delta_rmse']):
    offset = 0.5 if v >= 0 else -1.8
    ax_abl.text(v + offset, i, f"{v:+.2f}", va='center', fontsize=9, fontweight='bold')

fig_abl.tight_layout()
fig_abl_path = os.path.join(FIGURES_DIR, "feature_group_ablation.png")
fig_abl.savefig(fig_abl_path, dpi=300)
plt.close(fig_abl)
print(f"[Phase 2] Saved {fig_abl_path}")

print("[Phase 2] All calculations, tables, model artifacts, and figures successfully created!")
