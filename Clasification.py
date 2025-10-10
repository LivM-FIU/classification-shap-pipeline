#!/usr/bin/env python3
# ==============================
# CLASSIFICATION PIPELINE (All Models + SHAP Plots)
# ==============================
import os
import time
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Quiet some noisy libs
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# -------- CONFIG --------
USE_GPU = True          # GPU for LightGBM if available
GPU_ID  = 0
TARGET_ID = "TCGA-39-5011-01A"  # Patient ID for force plots
N_FOLDS = 5
RANDOM_STATE = 42
TOPK = 10               # top features to show in bars
SHAP_SUBSAMPLE = 8000   # samples to compute SHAP on
SHAP_BACKGROUND = 800   # background size for interventional explainer

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
feature_names = X.columns.tolist()

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
classes = le.classes_.tolist()
print(f"Classes: {classes}")
print(f"Data: {X.shape[0]} samples × {X.shape[1]} features\n")

# -------- RESULTS DIRECTORY --------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"classification_results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
print(f"Results directory: {os.path.abspath(results_dir)}\n")

# -------- BUILD MODELS --------
def build_models(use_gpu=True, gpu_id=0):
    models = {
        # --- 1. Decision Tree ---
        "DecisionTree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=None,
            random_state=RANDOM_STATE
        ),

        # --- 2. Random Forest ---
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        # --- 3. Gradient Boosting Machine (GBM) ---
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            max_features=0.8,
            random_state=RANDOM_STATE
        ),

        # --- 4. XGBoost ---
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist" if not use_gpu else "gpu_hist",
            predictor="gpu_predictor" if use_gpu else "auto",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        # --- 5. LightGBM ---
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            device_type="gpu" if use_gpu else "cpu",
            verbosity=-1
        ),

        # --- 6. CatBoost ---
        "CatBoost": CatBoostClassifier(
            iterations=100,
            learning_rate=0.05,
            depth=6,
            random_seed=RANDOM_STATE,
            verbose=False,
            loss_function="MultiClass",
            task_type="GPU" if use_gpu else "CPU",
            devices=str(gpu_id) if use_gpu else None
        ),
    }

    return models

try:
    models = build_models(USE_GPU, GPU_ID)
except Exception as e:
    print(f"[GPU init warning] Fallback to CPU: {e}")
    models = build_models(False, GPU_ID)

# -------- 5-FOLD CROSS-VALIDATION --------
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
results = {}
overall_start = time.perf_counter()

for name, base_model in models.items():
    print("\n" + "=" * 90)
    print(f"Training model: {name}")
    model_start = time.perf_counter()

    fold_accs, fold_f1s, fold_precisions, fold_recalls = [], [], [], []

    for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y_enc), start=1):
        print(f"  [{name}] Fold {fold_idx}/{N_FOLDS} ...")
        fold_start = time.perf_counter()

        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y_enc[tr_idx], y_enc[te_idx]

        model = clone(base_model)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        acc  = accuracy_score(y_te, y_pred)
        f1m  = f1_score(y_te, y_pred, average="macro")
        prec = precision_score(y_te, y_pred, average="macro", zero_division=0)
        rec  = recall_score(y_te, y_pred, average="macro", zero_division=0)

        fold_accs.append(acc)
        fold_f1s.append(f1m)
        fold_precisions.append(prec)
        fold_recalls.append(rec)

        fold_time = time.perf_counter() - fold_start
        print(f"     Acc={acc:.4f}, F1={f1m:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, Time={minutes(fold_time):.2f}m")

    model_time = time.perf_counter() - model_start
    results[name] = {
        "accuracy_mean": np.mean(fold_accs),
        "f1_macro_mean": np.mean(fold_f1s),
        "precision_macro_mean": np.mean(fold_precisions),
        "recall_macro_mean": np.mean(fold_recalls),
        "train_minutes": minutes(model_time)
    }

overall_time = time.perf_counter() - overall_start
res_df = pd.DataFrame(results).T.sort_values(by="f1_macro_mean", ascending=False)
print("\n=== CV Results ===")
print(res_df)
print(f"\nTotal training time: {minutes(overall_time):.2f} min")

# -------- METRICS BAR CHART --------
plt.figure(figsize=(10, 6))
x = np.arange(len(res_df))
w = 0.2
plt.bar(x - 1.5*w, res_df["accuracy_mean"], width=w, label="Accuracy")
plt.bar(x - 0.5*w, res_df["f1_macro_mean"], width=w, label="F1")
plt.bar(x + 0.5*w, res_df["precision_macro_mean"], width=w, label="Precision")
plt.bar(x + 1.5*w, res_df["recall_macro_mean"], width=w, label="Recall")
plt.xticks(x, res_df.index, rotation=15, ha="right")
plt.ylabel("Score")
plt.title("Cross-Validated Mean Metrics")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "metrics_bar.png"), dpi=150)
plt.close()

# -------- TRAIN BEST MODEL --------
best_name = res_df.index[0]
best_clf = models[best_name]
print(f"\nBest model: {best_name}")
t0 = time.perf_counter()
best_clf.fit(X, y_enc)
print(f"Training took {minutes(time.perf_counter()-t0):.2f} min")

# ========== SHAP HELPERS ==========
def normalize_shap_outputs(shap_values, n_classes):
    """
    Return sv_by_class with shape [C, N, F] regardless of SHAP return format.
    Acceptable inputs:
      - list of length C, each [N, F]
      - ndarray [N, F, C]
      - ndarray [C, N, F]
      - ndarray [N, F] (binary-like -> single class)
    """
    if isinstance(shap_values, list):
        arrs = [np.asarray(a) for a in shap_values]  # each [N,F]
        # sanity: all same shape
        N, F = arrs[0].shape
        sv_by_class = np.stack(arrs, axis=0)  # [C,N,F]
        return sv_by_class

    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # Could be [N,F,C] or [C,N,F]
        if arr.shape[-1] == n_classes:     # [N,F,C] -> [C,N,F]
            return np.moveaxis(arr, -1, 0)
        elif arr.shape[0] == n_classes:    # already [C,N,F]
            return arr
        else:
            raise ValueError(f"Unexpected 3D SHAP shape {arr.shape} for {n_classes} classes")
    elif arr.ndim == 2:
        # [N,F] -> single class
        return arr[None, ...]              # [1,N,F]
    else:
        raise ValueError(f"Unsupported SHAP shape: {arr.shape}")

def get_base_values(expected_value, n_classes):
    """Return base_values as shape [C]."""
    bv = np.asarray(expected_value)
    if bv.ndim == 0:
        return np.repeat(float(bv), n_classes) if n_classes > 1 else np.array([float(bv)])
    if bv.ndim == 1:
        if len(bv) == 1 and n_classes > 1:
            return np.repeat(float(bv[0]), n_classes)
        if len(bv) == n_classes:
            return bv.astype(float)
    raise ValueError(f"Unexpected expected_value shape: {bv.shape}")

# -------- SHAP COMPUTATION (Full Dataset) --------
print("\nComputing SHAP values using the entire dataset ...")
shap_start = time.perf_counter()

# Use the entire dataset for SHAP
shap_X = X.copy()
shap_y = y_enc.copy()

# Use a moderate background (too large will slow down computation)
bg_n = min(1000, len(shap_X))  # cap at 1000 for performance
background = shap.utils.sample(shap_X, bg_n, random_state=RANDOM_STATE)

explainer = shap.TreeExplainer(
    best_clf,
    data=background,
    feature_perturbation="interventional",
    model_output="probability"
)
raw_shap = explainer.shap_values(shap_X)
sv_by_class = normalize_shap_outputs(raw_shap, n_classes=len(classes))  # [C, N, F]
base_values = get_base_values(explainer.expected_value, n_classes=len(classes))

print(f"SHAP computation took {minutes(time.perf_counter() - shap_start):.2f} min")

# Align feature names
F_out = sv_by_class.shape[-1]
if len(feature_names) != F_out:
    feature_names = feature_names[:F_out] + [f"shap_feature_{i}" for i in range(len(feature_names), F_out)]

# -------- CLASS-WISE |SHAP| --------
class_feature_means = {}
class_counts = {}

for cidx, cname in enumerate(classes[:sv_by_class.shape[0]]):
    mask = (shap_y == cidx)
    n_samples_class = int(mask.sum())
    class_counts[cname] = n_samples_class
    print(f"  → {cname}: {n_samples_class} samples used for SHAP summary")

    if n_samples_class == 0:
        continue

    mean_abs = np.abs(sv_by_class[cidx][mask, :]).mean(axis=0)
    class_feature_means[cname] = pd.Series(mean_abs, index=feature_names)

if not class_feature_means:
    print("[Warning] No class SHAP summaries were computed.")
else:
    class_order = [c for c in classes if c in class_feature_means]
    class_df = pd.DataFrame({c: class_feature_means[c] for c in class_order})
    class_df["cohort_mean_abs_shap"] = class_df.mean(axis=1)

    # --- Top Features ---
    top_feats = class_df["cohort_mean_abs_shap"].nlargest(TOPK).index
    top_df = class_df.loc[top_feats].sort_values("cohort_mean_abs_shap", ascending=True)

    # --- Cohort-wide Bar Plot ---
    plt.figure(figsize=(12, 7))
    y_pos = np.arange(len(top_df))
    left = np.zeros(len(top_df))
    cmap = plt.cm.get_cmap("tab10", len(class_order))

    for i, cname in enumerate(class_order):
        contrib = top_df[cname].values
        plt.barh(y_pos, contrib, left=left, color=cmap(i), edgecolor="black",
                 label=f"{cname} (n={class_counts.get(cname, 0)})")
        left += contrib

    plt.yticks(y_pos, top_df.index)
    plt.xlabel("Mean |SHAP|")
    plt.title("Cohort-wise Feature Importance (stacked by class)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cohort_feature_importance.png"), dpi=150)
    plt.close()

    # --- Per-Class Bar Plots ---
    for i, cname in enumerate(class_order):
        series = class_feature_means[cname].nlargest(TOPK)
        plt.figure(figsize=(10, 6))
        pos = np.arange(len(series))
        plt.barh(pos, series.values[::-1], color=cmap(i), edgecolor="black")
        plt.yticks(pos, series.index[::-1])
        plt.xlabel("Mean |SHAP|")
        plt.title(f"Top {TOPK} Features — {cname} (n={class_counts[cname]})")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"feature_importance_{cname.replace(' ', '_')}.png"), dpi=150)
        plt.close()


    # per-class bars
    for i, cname in enumerate(class_order):
        series = class_feature_means[cname].nlargest(TOPK)
        plt.figure(figsize=(10, 6))
        pos = np.arange(len(series))
        plt.barh(pos, series.values[::-1], color=cmap(i), edgecolor="black")
        plt.yticks(pos, series.index[::-1])
        plt.xlabel("Mean |SHAP|")
        plt.title(f"Top {TOPK} Features — {cname} (n={class_counts[cname]})")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"feature_importance_{cname.replace(' ', '_')}.png"), dpi=150)
        plt.close()

# -------- FORCE PLOTS (TARGET_ID) --------
# We compute SHAP for the exact target row to avoid mismatch with shap_X
if TARGET_ID in sample_ids:
    row_idx = np.where(sample_ids == TARGET_ID)[0][0]
else:
    row_idx = 0
    print(f"[Note] {TARGET_ID} not found; using first sample ({sample_ids[0]}).")

target_row = X.iloc[[row_idx]]
# reuse the same explainer (interventional + probability)
target_shap_raw = explainer.shap_values(target_row)   # list or array
target_sv_by_class = normalize_shap_outputs(target_shap_raw, n_classes=len(classes))  # [C,1,F]
target_sv_by_class = target_sv_by_class[:, 0, :]  # -> [C,F]
try:
    target_proba = best_clf.predict_proba(target_row)[0]
except Exception:
    target_proba = None

for cidx, cname in enumerate(classes[:target_sv_by_class.shape[0]]):
    base_val = float(base_values[cidx])
    plt.figure(figsize=(12, 8))  # taller figure for better spacing
    shap.force_plot(
        base_val,
        target_sv_by_class[cidx, :],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )

    # Clean unwanted text overlays ("f(x)", "base value")
    ax = plt.gca()
    fig = ax.figure
    for txt in list(ax.texts):
        t = txt.get_text().strip().lower()
        if "f(x)" in t or "base value" in t:
            txt.set_visible(False)

    # Adjust layout for more headroom
    fig.subplots_adjust(top=0.80)

    # Title placed completely OUTSIDE the plot (top-left of figure)
    title_str = f"{sample_ids[row_idx]} — {cname}"
    fig.text(
        0.02, 0.98,                  # coordinates: left/top margin (0–1)
        title_str,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#222222",
    )

    # Save and close properly
    out_path = os.path.join(results_dir, f"force_{sample_ids[row_idx]}_{cname}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    print(f"Saved force plot → {out_path}")

# -------- SAVE CONFIG --------
cfg = {
    "timestamp": timestamp,
    "use_gpu": USE_GPU,
    "best_model": best_name,
    "n_samples": int(X.shape[0]),
    "n_features": int(X.shape[1]),
    "classes": classes,
    "target_id": TARGET_ID
}

print(f"\nAll results and plots saved")
