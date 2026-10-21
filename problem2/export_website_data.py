"""
Export Problem 2 research results, figures, and model specs to website/
"""

import os
import shutil
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
WEBSITE_DIR = os.path.join(PROJECT_DIR, 'website')
WEB_DATA_DIR = os.path.join(WEBSITE_DIR, 'data', 'problem2')
WEB_FIG_DIR = os.path.join(WEBSITE_DIR, 'public', 'figures', 'problem2')

os.makedirs(WEB_DATA_DIR, exist_ok=True)
os.makedirs(WEB_FIG_DIR, exist_ok=True)

# 1. Copy JSON result files
src_results = os.path.join(BASE_DIR, 'results')
for fname in os.listdir(src_results):
    src_f = os.path.join(src_results, fname)
    if fname.endswith('.json') or fname.endswith('.csv'):
        shutil.copy2(src_f, os.path.join(WEB_DATA_DIR, fname))
print(f"Copied results from {src_results} to {WEB_DATA_DIR}")

# 2. Copy figures
src_figures = os.path.join(BASE_DIR, 'figures')
for fname in os.listdir(src_figures):
    src_f = os.path.join(src_figures, fname)
    if fname.endswith('.png'):
        shutil.copy2(src_f, os.path.join(WEB_FIG_DIR, fname))
print(f"Copied figures from {src_figures} to {WEB_FIG_DIR}")

# 3. Create sample test points for lightweight interactive SVG chart rendering
preds_path = os.path.join(src_results, 'test_predictions.csv')
if os.path.exists(preds_path):
    df_preds = pd.read_csv(preds_path)
    
    # Timeline samples (First 7 days = 7 * 144 = 1008 points)
    df_7d = df_preds.iloc[:1008].copy()
    timeline_7d = df_7d.to_dict(orient='records')
    with open(os.path.join(WEB_DATA_DIR, 'test_timeline_7d.json'), 'w', encoding='utf-8') as f:
        json.dump(timeline_7d, f, indent=2)
    
    # Scatter samples: sample 500 representative points from test set
    step = max(1, len(df_preds) // 500)
    df_scatter = df_preds.iloc[::step].copy()
    scatter_samples = df_scatter[['date', 'actual', 'predicted', 'residual', 'covered']].to_dict(orient='records')
    with open(os.path.join(WEB_DATA_DIR, 'test_scatter_samples.json'), 'w', encoding='utf-8') as f:
        json.dump(scatter_samples, f, indent=2)
        
    print(f"Generated test_timeline_7d.json ({len(timeline_7d)} pts) and test_scatter_samples.json ({len(scatter_samples)} pts)")

print("Successfully exported all Problem 2 data and assets to website!")
