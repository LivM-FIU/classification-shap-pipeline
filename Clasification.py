# ==============================
# CLASSIFICATION PIPELINE (All Models Use SHAP TreeExplainer)
# ==============================
import os, sys

import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

import cupy as cp
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import clone

# -------- CONFIG --------
USE_GPU = True          # GPU acceleration for training (XGB/LGBM/CatBoost)
GPU_ID = 0
TARGET_ID = "TCGA-39-5011-01A"  # Patient ID for SHAP visualization

def minutes(sec): return sec / 60.0


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
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
            tree_method="hist",                # GPU hist is deprecated
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
        "GBM": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
            max_depth=6, subsample=0.9, max_features=0.2,
            validation_fraction=0.1, n_iter_no_change=10,
            random_state=42)
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
        proba = booster.inplace_predict(cp_X, predict_type="value")  # on-GPU
        proba = cp.asnumpy(proba)
        if n_classes > 2:
            return np.argmax(proba, axis=1)
        else:
            return (proba > 0.5).astype(int)
    except Exception:
        # if CuPy not available or any issue, fall back
        return model.predict(X_te)
    
# -------- 5-FOLD CROSS-VALIDATION --------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
overall_start = time.perf_counter()

for name, base_model in models.items():
    print("\n" + "=" * 80)
    print(f"▶ Training model: {name}")
    model_start = time.perf_counter()

    fold_accs, fold_f1s, fold_times = [], [], []

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
        acc, f1m = accuracy_score(y_te, y_pred), f1_score(y_te, y_pred, average="macro")
        fold_accs.append(acc)
        fold_f1s.append(f1m)

        fold_time = time.perf_counter() - fold_start
        fold_times.append(fold_time)
        print(f"     Accuracy={acc:.4f}, F1-macro={f1m:.4f}, Time={minutes(fold_time):.2f} min")

    model_time = time.perf_counter() - model_start
    results[name] = {
        "accuracy_mean": np.mean(fold_accs),
        "f1_macro_mean": np.mean(fold_f1s),
        "train_minutes": minutes(model_time)
    }

    print(f"⏱ {name}: mean Accuracy={results[name]['accuracy_mean']:.4f}, "
          f"F1-macro={results[name]['f1_macro_mean']:.4f}, "
          f"Training time={results[name]['train_minutes']:.2f} min")

overall_time = time.perf_counter() - overall_start
res_df = pd.DataFrame(results).T.sort_values(by="f1_macro_mean", ascending=False)
print("\n=== CV Results (mean metrics & timing) ===")
print(res_df)
print(f"\nTotal training time (all models): {minutes(overall_time):.2f} min")

# Pick best model
best_name = res_df.index[0]
best_clf = models[best_name]
print(f"\n Best model: {best_name}")

# -------- FINAL TRAIN (100% DATA) --------
print(f"\n Fitting {best_name} on full dataset...")
t0 = time.perf_counter()
best_clf.fit(X, y_enc)
print(f" Final training took {minutes(time.perf_counter()-t0):.2f} min")

# -------- SHAP FOR ALL MODELS --------
print("\n Computing SHAP values with TreeExplainer for all models (uniform method)...")
shap_start = time.perf_counter()

# Passing the model lets TreeExplainer trace how each feature influences output.
explainer = shap.TreeExplainer(best_clf, model_output="probability", feature_perturbation='interventional')
shap_values = explainer.shap_values(X)
base_values = explainer.expected_value

print(f" SHAP computation took {(time.perf_counter()-shap_start)/60:.2f} min")

# Normalize to consistent shape
if isinstance(shap_values, list):
    sv_by_class = np.stack(shap_values, axis=0)  # [C, N, F]
    base_per_class = np.array(base_values)
else:
    sv_by_class = np.expand_dims(np.asarray(shap_values), 0)
    base_per_class = np.array([base_values]) if np.isscalar(base_values) else np.mean(base_values)

# -------- TOP 10 FEATURES PER CLASS --------
print("\n=== Per-Class Top 10 Features by mean |SHAP| ===")
for cidx, cname in enumerate(classes[:sv_by_class.shape[0]]):
    mask = (y_enc == cidx)
    mean_abs = np.abs(sv_by_class[cidx][mask]).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:10]
    print(f"\n{cname}:")
    for i in top_idx:
        print(f"  {X.columns[i]:30s}  {mean_abs[i]:.6f}")

# -------- FORCE PLOTS --------
if TARGET_ID in sample_ids:
    row_idx = np.where(sample_ids == TARGET_ID)[0][0]
else:
    row_idx = 0
    print(f"[Note] {TARGET_ID} not found; using row 0 instead.")

os.makedirs("plots", exist_ok=True)
print(f"\n Generating force plots for {sample_ids[row_idx]} ...")

for cidx, cname in enumerate(classes[:sv_by_class.shape[0]]):
    shap.force_plot(
        base_per_class[cidx],
        sv_by_class[cidx][row_idx, :],
        X.iloc[row_idx, :],
        feature_names=X.columns,
        matplotlib=True,
        show=False
    )
    plt.title(f"{sample_ids[row_idx]} — {cname}")
    plt.tight_layout()
    plt.savefig(f"plots/force_{sample_ids[row_idx]}_{cname}.png", dpi=150)
    plt.close()

print("\n Saved SHAP force plots for all classes to ./plots/")
