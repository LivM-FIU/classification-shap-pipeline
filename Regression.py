#!/usr/bin/env python3
# ==================================================
# REGRESSION PIPELINE (HW3 Task 3 & 4)
# GPU Acceleration + SHAP + Fold Metrics + Timing
# ==================================================
import os
import re
import time
import glob
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# -------- CONFIG --------
DATA_PATH = "hw3-drug-screening-data.csv"
USE_GPU = False
GPU_ID = 0
N_FOLDS = 5
RANDOM_STATE = 42

def minutes(sec): return sec / 60.0
def safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(s)).strip('_')

# -------- LOAD DATA --------
start_total = time.perf_counter()
gdsc = pd.read_csv(DATA_PATH)
print(f"Data loaded: {gdsc.shape[0]} samples × {gdsc.shape[1]} columns")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
res_dir = f"results_regression_{timestamp}"
os.makedirs(res_dir, exist_ok=True)
print(f"Results directory: {os.path.abspath(res_dir)}")

id_cols = ["CELL_LINE_NAME", "DRUG_NAME"]
assert all(c in gdsc.columns for c in id_cols + ["LN_IC50"]), "Missing required columns."

y = gdsc["LN_IC50"].values
meta = gdsc[id_cols].copy()
X = gdsc.drop(columns=id_cols + ["LN_IC50"]).apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median(numeric_only=True)).astype(np.float32)
feature_names = X.columns.tolist()
print(f"Feature matrix: {X.shape[0]} × {X.shape[1]}")

# -------- DEFINE MODELS --------
def build_models(use_gpu=True, gpu_id=0):
    xgb_params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if use_gpu:
        xgb_params.update({"device": "cuda", "predictor": "gpu_predictor"})
    else:
        xgb_params.update({"device": "cpu"})

    lgbm_params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "device_type": "gpu" if use_gpu else "cpu",
    }

    cat_params = {
        "iterations": 200,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "loss_function": "RMSE",
        "task_type": "GPU" if use_gpu else "CPU",
    }
    if use_gpu:
        cat_params["devices"] = str(gpu_id)

    models = {
        # "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        # "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=15, n_jobs=-1, random_state=RANDOM_STATE),
        # "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
        #                                               subsample=0.9, max_features=0.3, random_state=RANDOM_STATE),
        # "XGBoost": XGBRegressor(**xgb_params),
        # "LightGBM": LGBMRegressor(**lgbm_params),
        "CatBoost": CatBoostRegressor(**cat_params),
    }
    return models

try:
    regressors = build_models(USE_GPU, GPU_ID)
except Exception as e:
    print(f"[GPU init warning] Fallback to CPU: {e}")
    regressors = build_models(False, GPU_ID)

# -------- CROSS-VALIDATION WITH FOLD METRICS --------
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
results = {}
fold_records = []

for name, model in regressors.items():
    print("\n" + "=" * 90)
    print(f" Training model: {name}")
    print("-" * 90)
    model_start = time.perf_counter()
    fold_mae, fold_mse, fold_rmse, fold_r2 = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_start = time.perf_counter()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        fold_time = time.perf_counter() - fold_start

        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        fold_mae.append(mae); fold_mse.append(mse); fold_rmse.append(rmse); fold_r2.append(r2)
        fold_records.append({"model": name, "fold": fold, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2})
        print(f"Fold {fold}/{N_FOLDS} → MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, Time={fold_time:.1f}s")

    total_model_time = time.perf_counter() - model_start
    results[name] = {
        "MAE_mean": np.mean(fold_mae), "MSE_mean": np.mean(fold_mse),
        "RMSE_mean": np.mean(fold_rmse), "R2_mean": np.mean(fold_r2),
        "Train_min": minutes(total_model_time)
    }

cv_df = pd.DataFrame(results).T.sort_values(by=["RMSE_mean", "MAE_mean", "R2_mean"],
                                            ascending=[True, True, False])
print("\n=== Cross-Validation Summary ===")
print(cv_df.round(4))

# -------- METRIC BAR CHART --------
metrics_plot = ["MAE_mean", "MSE_mean", "RMSE_mean", "R2_mean"]
plt.figure(figsize=(10, 6))
x = np.arange(len(cv_df))
w = 0.2
for i, metric in enumerate(metrics_plot):
    plt.bar(x + (i - 1.5) * w, cv_df[metric].values, width=w, label=metric)
plt.xticks(x, cv_df.index, rotation=15, ha="right")
plt.ylabel("Metric Value")
plt.title("Cross-Validated Mean Metrics by Regressor")
plt.legend()
plt.tight_layout()
metrics_bar_path = os.path.join(res_dir, "metrics_bar_regression.png")
plt.savefig(metrics_bar_path, dpi=150)
plt.close()
print(f"Saved regression metrics bar chart → {metrics_bar_path}")

# -------- BEST MODEL & TRAIN/TEST SPLIT --------
best_name = cv_df.index[0]
best_reg = regressors[best_name]
print(f"\n🏆 Best Regressor: {best_name}")
X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X, y, meta, test_size=0.2, random_state=RANDOM_STATE
)
best_reg.fit(X_train, y_train)

# -------- TEST METRICS --------
y_pred = best_reg.predict(X_test)
mae, mse, rmse, r2 = (mean_absolute_error(y_test, y_pred),
                      mean_squared_error(y_test, y_pred),
                      np.sqrt(mean_squared_error(y_test, y_pred)),
                      r2_score(y_test, y_pred))
print("\n=== Final Test Metrics ===")
print(f"MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

# -------- SHAP ANALYSIS (Task 4a & 4b) --------
print("\nComputing SHAP values for best model...")
background = shap.utils.sample(X_train, 500, random_state=42)
explainer = shap.TreeExplainer(best_reg, data=background, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test)
shap_arr = np.asarray(shap_values)
abs_shap = np.abs(shap_arr)
expected_value = float(np.ravel(np.array(explainer.expected_value))[0])
pos_of = {orig_idx: pos for pos, orig_idx in enumerate(X_test.index)}

# ---- (a) Per-drug Top 10 ----
drug_top10 = {}
print("\n=== (a) Per-Drug Top-10 Features by mean |SHAP| ===")
for drug, orig_idxs in meta_test.groupby("DRUG_NAME").groups.items():
    pos_idxs = [pos_of[i] for i in orig_idxs if i in pos_of]
    if not pos_idxs: 
        continue
    mean_abs = abs_shap[pos_idxs, :].mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:10]
    drug_top10[drug] = [(feature_names[i], float(mean_abs[i])) for i in top_idx]
    print(f"\n{drug}:")
    for f, val in drug_top10[drug]:
        print(f"  {f:30s}  {val:.6f}")
    plt.figure(figsize=(10, 7))
    y_pos = np.arange(10)[::-1]
    plt.barh(y_pos, [v for _, v in drug_top10[drug]][::-1])
    plt.yticks(y_pos, [f for f, _ in drug_top10[drug]][::-1])
    plt.xlabel("Mean |SHAP|")
    plt.title(f"Top 10 Features by mean |SHAP| — {drug}")
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, f"per_drug_top10_{safe_name(drug)}.png"), dpi=150)
    plt.close()

# ---- (b) Least-error sample ----
errors = np.abs(y_pred - y_test)
min_pos = int(np.argmin(errors))
pair = (meta_test.iloc[min_pos]["DRUG_NAME"], meta_test.iloc[min_pos]["CELL_LINE_NAME"])
print(f"\n=== (b) Least-error sample ===")
print(f"Drug={pair[0]}, Cell line={pair[1]}")
print(f"True={y_test[min_pos]:.4f}, Pred={y_pred[min_pos]:.4f}, AbsError={errors[min_pos]:.6f}")
row_shap = shap_arr[min_pos, :]
order = np.argsort(np.abs(row_shap))[::-1][:10]
pair_top10 = [(feature_names[i], float(row_shap[i])) for i in order]
plt.figure(figsize=(10, 7))
y_pos = np.arange(10)[::-1]
plt.barh(y_pos, [v for _, v in pair_top10][::-1], color="#ff7f0e")
plt.yticks(y_pos, [f for f, _ in pair_top10][::-1])
plt.xlabel("SHAP value (signed)")
plt.title(f"Top 10 SHAP — Least-error: {pair[0]} | {pair[1]}")
plt.tight_layout()
plt.savefig(os.path.join(res_dir, f"least_error_top10_{safe_name(pair[0])}_{safe_name(pair[1])}.png"), dpi=150)
plt.close()
print("Saved least-error SHAP bar plot.")

# ---- (c) Clean SHAP Force Plot (Regression, No Feature Names) ----
print("\nGenerating clean SHAP force plot (no feature names) ...")

# Sort by absolute SHAP values for better readability
sorted_idx = np.argsort(np.abs(row_shap))[::-1]
row_shap_sorted = row_shap[sorted_idx]

# --- Generate SHAP force plot (Matplotlib version) ---
plt.figure(figsize=(14, 3.5))
shap.force_plot(
    base_value=expected_value,
    shap_values=row_shap_sorted,
    matplotlib=True,
    show=False
)

ax = plt.gca()
fig = ax.figure

# Remove SHAP’s automatic text annotations ("f(x)" and "base value")
for txt in list(ax.texts):
    t = txt.get_text().strip().lower()
    if "f(x)" in t or "base value" in t:
        txt.set_visible(False)

# Clean layout and add custom title
fig.subplots_adjust(top=0.85, bottom=0.25)
title_str = f"{pair[0]} | {pair[1]}  (Pred={y_pred[min_pos]:.3f}, True={y_test[min_pos]:.3f}, Base={expected_value:.3f})"
fig.text(
    0.02, 0.98,
    title_str,
    ha="left",
    va="top",
    fontsize=11,
    fontweight="bold",
    color="#222222",
)

# Save simplified SHAP force plot
force_path = os.path.join(
    res_dir, f"force_least_error_{safe_name(pair[0])}_{safe_name(pair[1])}_clean.png"
)
fig.savefig(force_path, dpi=200, bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print(f"✅ Saved clean SHAP force plot (no feature names) → {force_path}")

# -------- SAVE CSV SUMMARIES --------
pd.DataFrame({"Metric": ["MAE", "MSE", "RMSE", "R2"],
              "Value": [mae, mse, rmse, r2]}).to_csv(os.path.join(res_dir, "test_metrics.csv"), index=False)
rows = []
for drug, feats in drug_top10.items():
    for fname, val in feats:
        rows.append({"DRUG_NAME": drug, "feature": fname, "mean_abs_shap": val})
pd.DataFrame(rows).to_csv(os.path.join(res_dir, "per_drug_top10_shap.csv"), index=False)
plot_paths = [{"plot_name": os.path.basename(p), "file_path": os.path.abspath(p)}
              for p in sorted(glob.glob(os.path.join(res_dir, "*.png")))]
pd.DataFrame(plot_paths).to_csv(os.path.join(res_dir, "generated_plots.csv"), index=False)

print(f"\nAll results saved in: {os.path.abspath(res_dir)}")
print(f"Total runtime: {minutes(time.perf_counter()-start_total):.2f} min")
