#!/usr/bin/env python3
# ==================================================
# REGRESSION PIPELINE (HW3 Task 3 & 4)
# GPU Acceleration + SHAP + Fold Metrics + Timing
# ==================================================
import os
import re
import json
import time
import glob
import shutil
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

# Single plots directory
os.makedirs("plots", exist_ok=True)

# -------- CONFIG --------
DATA_PATH = "hw3-drug-screening-data.csv"
USE_GPU = True
GPU_ID = 0
N_FOLDS = 5
RANDOM_STATE = 42

# For SHAP decision plot speed/readability
MAX_POINTS_FOR_DECISION = 600

def minutes(sec): return sec / 60.0
def safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(s)).strip('_')

# -------- LOAD DATA --------
start_total = time.perf_counter()
gdsc = pd.read_csv(DATA_PATH)
print(f"Data loaded: {gdsc.shape[0]} samples × {gdsc.shape[1]} columns")

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
    models = {
        # Uncomment any others you want
        # "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        # "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=RANDOM_STATE),
        # "GBM": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.9, max_features=0.3, random_state=RANDOM_STATE),
        # "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", device="cuda" if use_gpu else "cpu"),
        # "LightGBM": LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=64, subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE, device_type="gpu" if use_gpu else "cpu", verbosity=-1),
        "CatBoost": CatBoostRegressor(
            iterations=200, learning_rate=0.05, depth=6,
            random_seed=RANDOM_STATE, verbose=False, loss_function="RMSE",
            task_type="GPU" if use_gpu else "CPU", devices=str(gpu_id)
        ),
    }
    return models

try:
    regressors = build_models(USE_GPU, GPU_ID)
except Exception as e:
    print(f"[ GPU init warning] Fallback to CPU: {e}")
    regressors = build_models(False, GPU_ID)

# -------- CROSS-VALIDATION WITH FOLD METRICS --------
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
results = {}

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

        fold_mae.append(mae)
        fold_mse.append(mse)
        fold_rmse.append(rmse)
        fold_r2.append(r2)

        print(f"Fold {fold}/{N_FOLDS} → MAE={mae:.4f}, MSE={mse:.4f}, "
              f"RMSE={rmse:.4f}, R²={r2:.4f}, Time={fold_time:.1f}s")

    total_model_time = time.perf_counter() - model_start
    results[name] = {
        "MAE_mean": np.mean(fold_mae), "MAE_std": np.std(fold_mae),
        "MSE_mean": np.mean(fold_mse), "RMSE_mean": np.mean(fold_rmse),
        "R2_mean": np.mean(fold_r2), "Train_min": minutes(total_model_time)
    }

    print("-" * 90)
    print(f" Mean across {N_FOLDS} folds: "
          f"MAE={np.mean(fold_mae):.4f}±{np.std(fold_mae):.4f}, "
          f"MSE={np.mean(fold_mse):.4f}, RMSE={np.mean(fold_rmse):.4f}, "
          f"R²={np.mean(fold_r2):.4f}")
    print(f"⏱ Total training time for {name}: {minutes(total_model_time):.2f} min")
    print("=" * 90)

# -------- CV SUMMARY --------
cv_df = pd.DataFrame(results).T.sort_values(
    by=["RMSE_mean", "MAE_mean", "R2_mean"],
    ascending=[True, True, False]
)
print("\n=== Cross-Validation Summary ===")
print(cv_df.round(4))
print(f"\n⏱ Total experiment time: {minutes(time.perf_counter()-start_total):.2f} min")

# -------- METRIC BAR CHART (Dynamic Y-axis scaling: 1.5× max value) --------
metrics_plot = ["MAE_mean", "MSE_mean", "RMSE_mean", "R2_mean"]
plt.figure(figsize=(10, 6))
x = np.arange(len(cv_df))  # each model
w = 0.2

for i, metric in enumerate(metrics_plot):
    offset = (i - 1.5) * w
    plt.bar(x + offset, cv_df[metric].values, width=w, label=metric)

plt.xticks(x, cv_df.index, rotation=15, ha="right")
plt.ylabel("Metric Value")
plt.title("Cross-Validated Mean Metrics by Regressor")
y_max = cv_df[metrics_plot].max().max() * 1.5
plt.ylim(0, y_max)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
plt.tight_layout()
plt.savefig("plots/metrics_bar_regression.png", dpi=150, bbox_inches="tight")
plt.close()
print(" Saved regression metrics bar chart → plots/metrics_bar_regression.png")

# -------- BEST MODEL SELECTION --------
best_name = cv_df.index[0]
best_reg = regressors[best_name]
print(f"\n🏆 Best Regressor: {best_name}")

# -------- FINAL TRAIN/TEST SPLIT --------
X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X, y, meta, test_size=0.2, random_state=RANDOM_STATE
)
best_reg.fit(X_train, y_train)

# -------- TEST METRICS --------
y_pred = best_reg.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n=== Final Test Metrics ===")
print(f"MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

# -------- SHAP ANALYSIS --------
print("\nComputing SHAP values for best model...")
background = shap.utils.sample(X_train, 200, random_state=42)
explainer = shap.TreeExplainer(best_reg, data=background, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test)
shap_arr = np.asarray(shap_values)    # [N_test, F]
abs_shap = np.abs(shap_arr)

# --- map original test indices -> SHAP row positions ---
pos_of = {orig_idx: pos for pos, orig_idx in enumerate(X_test.index)}

# -------- (a) Per-drug Top 10 (safe indexing) --------
drug_top10 = {}
print("\n=== Per-Drug Top-10 Features by mean |SHAP| ===")
for drug, orig_idxs in meta_test.groupby("DRUG_NAME").groups.items():
    pos_idxs = [pos_of[i] for i in orig_idxs if i in pos_of]
    if not pos_idxs:
        continue
    mean_abs = abs_shap[pos_idxs, :].mean(axis=0)   # [F]
    top_idx = np.argsort(mean_abs)[::-1][:10]
    drug_top10[drug] = [(feature_names[i], float(mean_abs[i])) for i in top_idx]

    print(f"\n{drug}:")
    for f, val in drug_top10[drug]:
        print(f"  {f:30s}  {val:.6f}")

# ---- (a) PLOTS: Per-drug Top-10 mean |SHAP| ----
for drug, feats in drug_top10.items():
    if not feats:
        continue
    names, vals = zip(*feats)  # 10 names, 10 values
    plt.figure(figsize=(10, 7))
    y_pos = np.arange(len(names))[::-1]
    plt.barh(y_pos, vals[::-1])
    plt.yticks(y_pos, [names[i] for i in range(len(names)-1, -1, -1)])
    plt.xlabel("Mean |SHAP|")
    plt.title(f"Top 10 Features by mean |SHAP| — {drug}")
    plt.tight_layout()
    out_path = f"plots/per_drug_top10_{safe_name(drug)}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved per-drug top-10 plot: {out_path}")

# -------- (b) Least-error sample (use .iloc for positional) --------
errors = np.abs(y_pred - y_test)
min_pos = int(np.argmin(errors))   # positional inside X_test
pair = (
    meta_test.iloc[min_pos]["DRUG_NAME"],
    meta_test.iloc[min_pos]["CELL_LINE_NAME"]
)
print(f"\nLeast prediction error sample → drug={pair[0]}, cell_line={pair[1]}")
print(f"True LN_IC50={y_test[min_pos]:.4f}, Pred={y_pred[min_pos]:.4f}, AbsError={errors[min_pos]:.6f}")

row_shap = shap_arr[min_pos, :]
order = np.argsort(np.abs(row_shap))[::-1][:10]
pair_top10 = [(feature_names[i], float(row_shap[i])) for i in order]

print("\nTop-10 features for least-error pair (signed SHAP):")
for f, val in pair_top10:
    print(f"  {f:30s}  SHAP={val:.6f}")

# ---- (b) PLOT: Least-error signed SHAP (names left, bigger fig) ----
top_feats  = [feature_names[i] for i in order]
top_shaps  = [float(row_shap[i]) for i in order]
plt.figure(figsize=(10, 7))
y_pos = np.arange(len(top_feats))[::-1]
plt.barh(y_pos, top_shaps[::-1])  # signed SHAP values
plt.yticks(y_pos, [top_feats[i] for i in range(len(top_feats)-1, -1, -1)])
plt.xlabel("SHAP value (signed)")
plt.title(f"Top 10 SHAP — Least-error: {pair[0]} | {pair[1]}")
plt.tight_layout()
out_signed = f"plots/least_error_top10_{safe_name(pair[0])}_{safe_name(pair[1])}.png"
plt.savefig(out_signed, dpi=150)
plt.close()
print(f"Saved least-error signed SHAP plot: {out_signed}")

# -------- GLOBAL & LOCAL SHAP VISUALS WITH LABELS ON LEFT --------
# Global beeswarm (feature names on left)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_arr, X_test, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("plots/summary_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved global SHAP summary beeswarm → plots/summary_beeswarm.png")

# Global decision plot (subsample for speed + ignore_warnings)
n = shap_arr.shape[0]
if n > MAX_POINTS_FOR_DECISION:
    rng = np.random.default_rng(42)
    keep = np.sort(rng.choice(n, size=MAX_POINTS_FOR_DECISION, replace=False))
    shap_arr_plot = shap_arr[keep, :]
else:
    shap_arr_plot = shap_arr

plt.figure(figsize=(14, 8))
shap.decision_plot(
    base_value=explainer.expected_value,   # scalar for regression
    shap_values=shap_arr_plot,
    feature_names=feature_names,
    ignore_warnings=True,                  # don't abort for large N
    show=False
)
plt.title(f"Global SHAP Decision Plot (n={shap_arr_plot.shape[0]})")
plt.tight_layout()
plt.savefig("plots/decision_global.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved global SHAP decision plot → plots/decision_global.png")

# Local waterfall (names on left, clean stacking)
try:
    # Newer SHAP: shap.plots.waterfall with Explanation is preferred, but legacy works widely:
    from shap.plots._waterfall import waterfall_legacy
    plt.figure(figsize=(12, 8))
    waterfall_legacy(
        explainer.expected_value,
        row_shap,
        feature_names=feature_names,
        max_display=20,
        show=False
    )
    plt.title(f"Waterfall — Least-error: {pair[0]} | {pair[1]}")
    plt.tight_layout()
    plt.savefig(f"plots/waterfall_{safe_name(pair[0])}_{safe_name(pair[1])}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved local SHAP waterfall → plots/waterfall_*.png")
except Exception:
    # Fallback to force plot (single-sample) if waterfall not available
    plt.figure(figsize=(12, 4))
    shap.force_plot(
        explainer.expected_value,
        row_shap,
        X_test.iloc[min_pos, :],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.title(f"Force Plot — Least-error: {safe_name(pair[0])} | {safe_name(pair[1])}")
    plt.tight_layout()
    plt.savefig(f"plots/force_{safe_name(pair[0])}_{safe_name(pair[1])}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved local SHAP force plot (fallback).")

# -------- SAVE RESULTS --------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
res_dir = f"results_regression_{timestamp}"
os.makedirs(res_dir, exist_ok=True)

# Summary CSVs
cv_df.to_csv(os.path.join(res_dir, "cv_summary.csv"))
pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2"],
    "Value": [mae, mse, rmse, r2]
}).to_csv(os.path.join(res_dir, "test_metrics.csv"), index=False)

# Save per-drug top-10 SHAP to CSV
rows = []
for drug, feats in drug_top10.items():
    for fname, val in feats:
        rows.append({"DRUG_NAME": drug, "feature": fname, "mean_abs_shap": val})
pd.DataFrame(rows).to_csv(os.path.join(res_dir, "per_drug_top10_shap.csv"), index=False)

# Copy plots & write registry
plot_paths = []
for path in glob.glob(os.path.join("plots", "*.png")):
    dst = os.path.join(res_dir, os.path.basename(path))
    try:
        shutil.copy2(path, dst)
        plot_paths.append({"plot_name": os.path.basename(path), "file_path": os.path.abspath(dst)})
    except Exception:
        pass
pd.DataFrame(plot_paths).to_csv(os.path.join(res_dir, "generated_plots.csv"), index=False)

print(f"\nAll results saved in: {os.path.abspath(res_dir)}")
print(f"Total runtime: {minutes(time.perf_counter()-start_total):.2f} min")
