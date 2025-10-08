#!/usr/bin/env python3
# ==================================================
# REGRESSION PIPELINE (HW3 Task 3 & 4)
# GPU Acceleration + SHAP + Fold Metrics + Timing
# ==================================================
import os
import json
import time
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
os.makedirs("plots", exist_ok=True)

# -------- CONFIG --------
DATA_PATH = "hw3-drug-screening-data.csv"
USE_GPU = True
GPU_ID = 0
N_FOLDS = 5
RANDOM_STATE = 42

def minutes(sec): return sec / 60.0

# -------- LOAD DATA --------
start_total = time.perf_counter()
gdsc = pd.read_csv(DATA_PATH)
print(f"✅ Data loaded: {gdsc.shape[0]} samples × {gdsc.shape[1]} columns")

id_cols = ["CELL_LINE_NAME", "DRUG_NAME"]
assert all(c in gdsc.columns for c in id_cols + ["LN_IC50"]), "Missing required columns."

y = gdsc["LN_IC50"].values
meta = gdsc[id_cols].copy()
X = gdsc.drop(columns=id_cols + ["LN_IC50"]).apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median(numeric_only=True)).astype(np.float32)
feature_names = X.columns.tolist()
print(f"✅ Feature matrix: {X.shape[0]} × {X.shape[1]}")

# -------- DEFINE MODELS --------
def build_models(use_gpu=True, gpu_id=0):
    models = {
        "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,  # halve tree count
            max_depth=15,      # shallower trees
            n_jobs=-1,
            random_state=42
            ),
        "GBM": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.9, max_features=0.3, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE,
            n_jobs=-1, tree_method="hist",
            device="cuda" if use_gpu else "cpu"
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=64,
            subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE,
            device_type="gpu" if use_gpu else "cpu", verbosity=-1
        ),
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
    print(f"[⚠️ GPU init warning] Fallback to CPU: {e}")
    regressors = build_models(False, GPU_ID)

# -------- CROSS-VALIDATION WITH FOLD METRICS --------
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
results = {}

for name, model in regressors.items():
    print("\n" + "=" * 90)
    print(f"🚀 Training model: {name}")
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
    print(f"✅ Mean across {N_FOLDS} folds: "
          f"MAE={np.mean(fold_mae):.4f}±{np.std(fold_mae):.4f}, "
          f"MSE={np.mean(fold_mse):.4f}, RMSE={np.mean(fold_rmse):.4f}, "
          f"R²={np.mean(fold_r2):.4f}")
    print(f"⏱ Total training time for {name}: {minutes(total_model_time):.2f} min")
    print("=" * 90)

# -------- CV SUMMARY --------
cv_df = pd.DataFrame(results).T.sort_values(by=["RMSE_mean", "MAE_mean", "R2_mean"], ascending=[True, True, False])
print("\n=== 📊 Cross-Validation Summary ===")
print(cv_df.round(4))
print(f"\n⏱ Total experiment time: {minutes(time.perf_counter()-start_total):.2f} min")

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

print("\n=== 🧪 Final Test Metrics ===")
print(f"MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

# -------- SHAP ANALYSIS --------
print("\n💡 Computing SHAP values for best model...")
background = shap.utils.sample(X_train, 200, random_state=42)
explainer = shap.TreeExplainer(best_reg, data=background)
shap_values = explainer.shap_values(X_test)
shap_arr = np.asarray(shap_values)
abs_shap = np.abs(shap_arr)

# -------- (a) Per-drug Top 10 --------
drug_top10 = {}
print("\n=== 🔍 Per-Drug Top-10 Features by mean |SHAP| ===")
for drug, idxs in meta_test.groupby("DRUG_NAME").groups.items():
    idxs = list(idxs)
    if len(idxs) == 0:
        continue
    mean_abs = abs_shap[idxs, :].mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:10]
    drug_top10[drug] = [(feature_names[i], float(mean_abs[i])) for i in top_idx]
    print(f"\n{drug}:")
    for f, val in drug_top10[drug]:
        print(f"  {f:30s}  {val:.6f}")

# -------- (b) Least-error sample --------
errors = np.abs(y_pred - y_test)
min_idx = int(np.argmin(errors))
pair = (
    meta_test.loc[min_idx, "DRUG_NAME"],
    meta_test.loc[min_idx, "CELL_LINE_NAME"]
)
print(f"\nLeast prediction error sample → drug={pair[0]}, cell_line={pair[1]}")
print(f"True LN_IC50={y_test[min_idx]:.4f}, Pred={y_pred[min_idx]:.4f}, AbsError={errors[min_idx]:.6f}")

row_shap = shap_arr[min_idx, :]
order = np.argsort(np.abs(row_shap))[::-1][:10]
pair_top10 = [(feature_names[i], float(row_shap[i])) for i in order]

print("\nTop-10 features for least-error pair (signed SHAP):")
for f, val in pair_top10:
    print(f"  {f:30s}  SHAP={val:.6f}")

# -------- FORCE PLOT --------
plt.figure()
shap.force_plot(
    explainer.expected_value,
    row_shap,
    X_test.loc[min_idx, :],
    feature_names=feature_names,
    matplotlib=True,
    show=False
)
plt.title(f"Force Plot — {pair[0]} | {pair[1]}")
plt.tight_layout()
plt.savefig(f"plots/force_{pair[0]}_{pair[1]}.png", dpi=150)
plt.close()
print("💾 Saved SHAP force plot for least-error pair.")

# -------- SAVE RESULTS --------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
res_dir = f"results_regression_{timestamp}"
os.makedirs(res_dir, exist_ok=True)

cv_df.to_csv(os.path.join(res_dir, "cv_summary.csv"))
pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2"],
    "Value": [mae, mse, rmse, r2]
}).to_csv(os.path.join(res_dir, "test_metrics.csv"), index=False)

print(f"\n📁 All results saved in: {os.path.abspath(res_dir)}")
print(f"⏱ Total runtime: {minutes(time.perf_counter()-start_total):.2f} min")
