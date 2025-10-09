# ==============================
# CLASSIFICATION PIPELINE (All Models Use SHAP TreeExplainer)
# ==============================
import os
import sys
import time
import json
import shutil
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Optional: quiet xgboost user warnings about device/predict path, etc.
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# If you keep CuPy for GPU-side XGBoost prediction (optional)
import cupy as cp

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import clone


# -------- CONFIG --------
USE_GPU = True          # GPU acceleration for training (XGB/LGBM/CatBoost)
GPU_ID = 0
TARGET_ID = "TCGA-39-5011-01A"  # Patient ID for SHAP visualization

def minutes(sec):
    return sec / 60.0


# -------- LOAD DATA --------
df = pd.read_csv("lncRNA_5_Cancers.csv")
id_col = df.columns[0]
class_col = df.columns[-1]

sample_ids = df[id_col].astype(str).values
y = df[class_col].astype(str).values
X = df.iloc[:, 1:-1].copy().apply(pd.to_numeric, errors="coerce").astype(np.float32)
X = X.fillna(X.median(numeric_only=True))

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
classes = le.classes_.tolist()
print(f"Classes: {classes}")
print(f"Data: {X.shape[0]} samples × {X.shape[1]} features\n")


# -------- BUILD MODELS --------
def build_models(use_gpu=True, gpu_id=0):
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
            tree_method="hist",                # use 'hist'; GPU enabled via device
            device="cuda" if use_gpu else "cpu"
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=64,
            subsample=0.9, colsample_bytree=0.9, random_state=42,
            device_type="gpu" if use_gpu else "cpu", verbosity=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=100, learning_rate=0.05, depth=6,
            random_seed=42, verbose=False, loss_function="MultiClass",
            task_type="GPU" if use_gpu else "CPU", devices=str(gpu_id)
        ),
        "GBM": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=6, subsample=0.9, max_features=0.2,
            validation_fraction=0.1, n_iter_no_change=10,
            random_state=42
        )
    }
    return models

try:
    models = build_models(USE_GPU, GPU_ID)
except Exception as e:
    print(f"[GPU init warning] Fallback to CPU: {e}")
    models = build_models(False, GPU_ID)


def xgb_predict_labels(model, X_te, n_classes):
    """Keep XGBoost prediction on GPU; fallback to CPU if needed."""
    try:
        booster = model.get_booster()
        cp_X = cp.asarray(X_te.values if hasattr(X_te, "values") else np.asarray(X_te))
        proba = booster.inplace_predict(cp_X)  # on-GPU, returns probabilities for multi
        proba = cp.asnumpy(proba)
        if n_classes > 2:
            return np.argmax(proba, axis=1)
        else:
            return (proba > 0.5).astype(int)
    except Exception:
        # Fallback to standard CPU predict if anything goes wrong
        return model.predict(X_te)


# -------- 5-FOLD CROSS-VALIDATION --------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}  # per-model aggregates
folds_raw = []  # raw per-fold records across all models
overall_start = time.perf_counter()

for name, base_model in models.items():
    print("\n" + "=" * 80)
    print(f"Training model: {name}")
    model_start = time.perf_counter()

    fold_accs, fold_f1s, fold_precisions, fold_recalls, fold_times = [], [], [], [], []

    for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y_enc), start=1):
        print(f"  [{name}] Fold {fold_idx}/5 ...", flush=True)
        fold_start = time.perf_counter()

        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y_enc[tr_idx], y_enc[te_idx]

        model = clone(base_model)
        model.fit(X_tr, y_tr)

        # --- GPU-safe prediction for XGBoost ---
        if name == "XGBoost" and model.get_params().get("device") == "cuda":
            y_pred = xgb_predict_labels(model, X_te, n_classes=len(classes))
        else:
            y_pred = model.predict(X_te)

        # Metrics
        acc  = accuracy_score(y_te, y_pred)
        f1m  = f1_score(y_te, y_pred, average="macro")
        prec = precision_score(y_te, y_pred, average="macro", zero_division=0)
        rec  = recall_score(y_te, y_pred, average="macro", zero_division=0)

        fold_accs.append(acc)
        fold_f1s.append(f1m)
        fold_precisions.append(prec)
        fold_recalls.append(rec)

        fold_time = time.perf_counter() - fold_start
        fold_times.append(fold_time)

        # store raw record for this fold
        folds_raw.append({
            "model": name,
            "fold": fold_idx,
            "accuracy": acc,
            "f1_macro": f1m,
            "precision_macro": prec,
            "recall_macro": rec,
            "fold_minutes": minutes(fold_time),
            "n_train": len(tr_idx),
            "n_test": len(te_idx),
        })

        print(f"     Accuracy={acc:.4f}, F1-macro={f1m:.4f}, "
              f"Precision={prec:.4f}, Recall={rec:.4f}, "
              f"Time={minutes(fold_time):.2f} min")

    model_time = time.perf_counter() - model_start
    results[name] = {
        "accuracy_mean": np.mean(fold_accs),
        "f1_macro_mean": np.mean(fold_f1s),
        "precision_macro_mean": np.mean(fold_precisions),
        "recall_macro_mean": np.mean(fold_recalls),
        "train_minutes": minutes(model_time)
    }

    print(f"{name}: Acc={results[name]['accuracy_mean']:.4f}, "
          f"F1={results[name]['f1_macro_mean']:.4f}, "
          f"Prec={results[name]['precision_macro_mean']:.4f}, "
          f"Rec={results[name]['recall_macro_mean']:.4f}, "
          f"Time={results[name]['train_minutes']:.2f} min")

overall_time = time.perf_counter() - overall_start
res_df = pd.DataFrame(results).T.sort_values(by="f1_macro_mean", ascending=False)
print("\n=== CV Results (mean metrics & timing) ===")
print(res_df)
print(f"\nTotal training time (all models): {minutes(overall_time):.2f} min")

# Ensure plots dir exists
os.makedirs("plots", exist_ok=True)

# Persist CV summary now (also saved again into results folder later)
res_df.to_csv("cv_results.csv", index=True)
print("Saved CV table -> cv_results.csv")


# -------- METRICS BAR CHART --------
plot_df = res_df[["accuracy_mean", "f1_macro_mean", "precision_macro_mean", "recall_macro_mean"]]

plt.figure(figsize=(10, 6))
x = np.arange(len(plot_df))  # models
w = 0.2

plt.bar(x - 1.5*w, plot_df["accuracy_mean"].values,           width=w, label="Accuracy")
plt.bar(x - 0.5*w, plot_df["f1_macro_mean"].values,           width=w, label="F1 (macro)")
plt.bar(x + 0.5*w, plot_df["precision_macro_mean"].values,    width=w, label="Precision (macro)")
plt.bar(x + 1.5*w, plot_df["recall_macro_mean"].values,       width=w, label="Recall (macro)")

plt.xticks(x, plot_df.index, rotation=15, ha="right")
plt.ylabel("Score")
plt.ylim(0, 1.0)
plt.title("Cross-validated Mean Metrics by Model")
plt.legend()
plt.tight_layout()
plt.savefig("plots/metrics_bar.png", dpi=150)
plt.close()
print("Saved metrics bar chart -> plots/metrics_bar.png")


# -------- PICK BEST MODEL --------
best_name = res_df.index[0]
best_clf = models[best_name]
print(f"\nBest model: {best_name}")

# -------- FINAL TRAIN (100% DATA) --------
print(f"\nFitting {best_name} on full dataset...")
t0 = time.perf_counter()
best_clf.fit(X, y_enc)
print(f"Final training took {minutes(time.perf_counter()-t0):.2f} min")


# -------- SHAP FOR BEST MODEL --------
print("\nComputing SHAP values with TreeExplainer (probability, interventional)...")
shap_start = time.perf_counter()

# Provide a background dataset for interventional
background = shap.utils.sample(X, 200, random_state=42)

explainer = shap.TreeExplainer(
    model=best_clf,
    data=background,                            # REQUIRED for interventional
    model_output="probability",
    feature_perturbation="interventional"
)

# Compute SHAP on full X (or subsample for speed)
shap_values = explainer.shap_values(X)
base_values = explainer.expected_value

print(f"SHAP computation took {minutes(time.perf_counter()-shap_start):.2f} min")

# ---------- Normalize SHAP outputs (works for binary & multiclass) ----------
feature_names = X.columns.to_list()  # ensure plain list (not pandas Index)

# sv_by_class -> [C, N, F]
if isinstance(shap_values, list):
    # multiclass typical: list length = n_classes
    sv_by_class = np.stack([np.asarray(s) for s in shap_values], axis=0)
else:
    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        # already [C, N, F]
        sv_by_class = sv
    elif sv.ndim == 2:
        # binary: [N, F] -> add class axis
        sv_by_class = sv[None, ...]
    else:
        raise ValueError(f"Unexpected shap_values shape: {sv.shape}")

# base_per_class -> shape [C]
ev = base_values
if isinstance(ev, (list, tuple, np.ndarray)):
    base_per_class = np.array(ev).reshape(-1)
else:
    base_per_class = np.array([float(ev)])

# If we got a scalar base value but multiple classes, repeat it
n_classes_out = sv_by_class.shape[0]
if base_per_class.size == 1 and n_classes_out > 1:
    base_per_class = np.repeat(base_per_class[0], n_classes_out)


# -------- TOP 10 FEATURES PER CLASS --------
print("\n=== Per-Class Top 10 Features by mean |SHAP| ===")
for cidx, cname in enumerate(classes[:sv_by_class.shape[0]]):
    mask = (y_enc == cidx)
    if not np.any(mask):
        print(f"\n{cname}: [no samples in this class subset]")
        continue

    mean_abs = np.abs(sv_by_class[cidx][mask]).mean(axis=0)  # [F]
    top_idx = np.argsort(mean_abs)[::-1][:10]

    print(f"\n{cname}:")
    for i in top_idx:
<<<<<<< HEAD
        fname = str(feature_names[i])
        val = float(mean_abs[i])
        print(f"  {fname:30s}  {val:.6f}")
=======
        val = mean_abs[i]
        if isinstance(val, np.ndarray):
            val = float(val.mean())  # safely flatten to a scalar
        print(f"  {str(X.columns[i]):30s}  {val:.6f}")
>>>>>>> 06772cda04c107fea7a865d14277a43340614a85

# -------- FORCE PLOTS --------
if TARGET_ID in sample_ids:
    row_idx = np.where(sample_ids == TARGET_ID)[0][0]
else:
    row_idx = 0
    print(f"[Note] {TARGET_ID} not found; using row 0 instead.")

print(f"\nGenerating force plots for {sample_ids[row_idx]} ...")
for cidx, cname in enumerate(classes[:sv_by_class.shape[0]]):
    # safe per-class base value
    base_val = float(base_per_class[cidx]) if base_per_class.ndim > 0 else float(base_per_class)
    shap.force_plot(
        base_val,
        sv_by_class[cidx][row_idx, :],
        X.iloc[row_idx, :],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.title(f"{sample_ids[row_idx]} — {cname}")
    plt.tight_layout()
    out_path = f"plots/force_{sample_ids[row_idx]}_{cname}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

print("\nSaved SHAP force plots for all classes to ./plots/")


# ==============================
# SAVE TRAINING RESULTS (Metrics + Plots) INTO A TIMESTAMPED FOLDER
# ==============================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# 1) Save the overall CV summary table
cv_summary_path = os.path.join(results_dir, "cv_summary.csv")
res_df.to_csv(cv_summary_path, index=True)
print(f"Saved CV summary to {cv_summary_path}")

# 2) Save all per-fold raw scores
folds_df = pd.DataFrame(folds_raw)
folds_csv_path = os.path.join(results_dir, "fold_scores_raw.csv")
folds_df.to_csv(folds_csv_path, index=False)
print(f"Saved per-fold raw metrics to {folds_csv_path}")

# 3) Copy plots into results_dir and record them in a CSV
plot_paths = []
for plot_file in os.listdir("plots"):
    src_path = os.path.join("plots", plot_file)
    if os.path.isfile(src_path):
        dst_path = os.path.join(results_dir, plot_file)
        shutil.copy2(src_path, dst_path)
        plot_paths.append({
            "plot_name": plot_file,
            "file_path": os.path.abspath(dst_path)
        })

plots_csv_path = os.path.join(results_dir, "generated_plots.csv")
pd.DataFrame(plot_paths).to_csv(plots_csv_path, index=False)
print(f"Saved plot file registry to {plots_csv_path}")

# 4) Save a run configuration snapshot
config_snapshot = {
    "timestamp": timestamp,
    "use_gpu": USE_GPU,
    "gpu_id": GPU_ID,
    "best_model": best_name,
    "n_samples": int(X.shape[0]),
    "n_features": int(X.shape[1]),
    "classes": list(classes),
    "target_id": TARGET_ID,
}
with open(os.path.join(results_dir, "run_config.json"), "w") as f:
    json.dump(config_snapshot, f, indent=4)
print(f"Saved run configuration snapshot to {os.path.join(results_dir, 'run_config.json')}")

print(f"\nAll results and plots saved in: {os.path.abspath(results_dir)}")
