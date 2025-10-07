# ==============================
# CLASSIFICATION PIPELINE (GPU + progress + timing + fast SHAP)
# Schema: first column = ID, last column = class, middle = features
# ==============================
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import clone
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# --------- Config ----------
USE_GPU = True        # flip False to force CPU
GPU_ID  = 0           # pick your device
TARGET_ID = "TCGA-39-5011-01A"   # Patient for force plots

def minutes(sec: float) -> float:
    return sec / 60.0

# --------- Load data (FIRST COL = ID, LAST COL = CLASS) ----------
df = pd.read_csv("lncRNA_5_Cancers.csv")
id_col = df.columns[0]
class_col = df.columns[-1]

sample_ids = df[id_col].astype(str).values
y = df[class_col].astype(str).values
X = df.iloc[:, 1:-1].copy()

# Ensure numeric; cast to float32 (friendlier to GPU VRAM)
X = X.apply(pd.to_numeric, errors="coerce").astype(np.float32)
if X.isna().any().any():
    X = X.fillna(X.median(numeric_only=True))

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
classes = le.classes_.tolist()
print(f"Classes: {classes}")
print(f"Data: {X.shape[0]} samples × {X.shape[1]} features\n")

# --------- Build models (GPU where possible with fallback) ----------
def build_models(use_gpu=True, gpu_id=0):
    models = {}
    # CPU-only sklearn
    models["DecisionTree"] = DecisionTreeClassifier(random_state=42)
    models["RandomForest"] = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    models["GBM"] = GradientBoostingClassifier(random_state=42)

    # XGBoost
    xgb_params = dict(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        n_jobs=-1, eval_metric="mlogloss", tree_method="hist"
    )
    if use_gpu:
        xgb_params.update(tree_method="gpu_hist", predictor="gpu_predictor", gpu_id=gpu_id)
    models["XGBoost"] = XGBClassifier(**xgb_params)

    # LightGBM
    lgb_params = dict(
        n_estimators=600, learning_rate=0.05, num_leaves=64,
        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
    )
    if use_gpu:
        # Requires GPU-enabled LightGBM build; will raise if not available
        lgb_params.update(device_type="gpu")
    models["LightGBM"] = LGBMClassifier(**lgb_params)

    # CatBoost
    cb_params = dict(
        iterations=600, learning_rate=0.05, depth=6,
        random_seed=42, verbose=False, loss_function="MultiClass"
    )
    if use_gpu:
        cb_params.update(task_type="GPU", devices=str(gpu_id))
    models["CatBoost"] = CatBoostClassifier(**cb_params)

    return models

try:
    models = build_models(USE_GPU, GPU_ID)
except Exception as e:
    print("[GPU init warning] Falling back to CPU for all models:", e)
    models = build_models(False, GPU_ID)

# --------- 5-Fold CV with detailed logging + timing ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
overall_start = time.perf_counter()

for name, base_model in models.items():
    print("\n" + "="*86)
    print(f" Training model: {name}")
    model_start = time.perf_counter()

    fold_accs, fold_f1s, fold_times = [], [], []

    for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y_enc), start=1):
        fold_label = f"[{name}] Fold {fold_idx}/5"
        fold_start = time.perf_counter()

        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y_enc[tr_idx], y_enc[te_idx]

        model = clone(base_model)
        print(f"{fold_label}: fit on {X_tr.shape[0]} train / validate on {X_te.shape[0]} …", flush=True)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        f1m = f1_score(y_te, y_pred, average="macro")

        fold_time = time.perf_counter() - fold_start
        fold_accs.append(acc); fold_f1s.append(f1m); fold_times.append(fold_time)
        print(f"{fold_label}: Accuracy={acc:.4f}  F1-macro={f1m:.4f}  (elapsed {minutes(fold_time):.2f} min)")

    model_time = time.perf_counter() - model_start
    results[name] = {
        "accuracy_mean": float(np.mean(fold_accs)),
        "accuracy_std": float(np.std(fold_accs)),
        "f1_macro_mean": float(np.mean(fold_f1s)),
        "f1_macro_std": float(np.std(fold_f1s)),
        "train_minutes": float(minutes(model_time)),
        "fold_minutes": [float(minutes(t)) for t in fold_times],
    }

    print(f" {name}: mean Accuracy={results[name]['accuracy_mean']:.4f} ± {results[name]['accuracy_std']:.4f}, "
          f"mean F1-macro={results[name]['f1_macro_mean']:.4f} ± {results[name]['f1_macro_std']:.4f}")
    print(f" {name}: total training time = {results[name]['train_minutes']:.2f} minutes "
          f"(per-fold: {[f'{m:.2f}' for m in results[name]['fold_minutes']]} min)")

overall_time = time.perf_counter() - overall_start
res_df = pd.DataFrame(results).T.sort_values(by=["f1_macro_mean", "accuracy_mean"], ascending=False)
print("\n" + "="*86)
print("=== Classification CV Results (mean ± std) with timing ===")
print(res_df[["accuracy_mean","accuracy_std","f1_macro_mean","f1_macro_std","train_minutes"]])
print(f"\n Total wall-clock time (all models): {minutes(overall_time):.2f} minutes")

best_clf_name = res_df.index[0]
best_clf = models[best_clf_name]
print(f"\n Best classifier: {best_clf_name}")

# --------- Final fit on full data (timed) ----------
final_start = time.perf_counter()
print(f"\n Fitting best model ({best_clf_name}) on FULL dataset ({X.shape[0]}×{X.shape[1]}) …")
best_clf.fit(X, y_enc)
final_minutes = minutes(time.perf_counter() - final_start)
print(f" Final full-data training time ({best_clf_name}): {final_minutes:.2f} minutes")

# --------- SHAP / Contributions (GPU-friendly native first; fallback to TreeExplainer) ----------
feature_names = X.columns.tolist()
n_features = X.shape[1]
n_classes = len(classes)

def get_native_contribs(model, X_df):
    """Return SHAP-like contributions and per-class base values from native APIs when possible.
       Output shape standardized to [n_classes, n_samples, n_features+1] (last col = bias)."""
    name = type(model).__name__.lower()
    try:
        if "xgb" in name:
            # returns [n_samples, (n_classes or 1)*(n_features+1)]
            sv = model.predict(X_df, pred_contribs=True)
            sv = np.asarray(sv)
            if n_classes > 2:  # multiclass -> flatten per class
                sv = sv.reshape(sv.shape[0], n_classes, n_features + 1)   # [N, C, F+1]
                sv = np.transpose(sv, (1, 0, 2))                          # [C, N, F+1]
            else:
                sv = sv[None, ...]  # [1, N, F+1]
            return sv
        if "lgbm" in name:
            sv = model.predict(X_df, pred_contrib=True)
            sv = np.asarray(sv)
            if n_classes > 2:
                sv = sv.reshape(sv.shape[0], n_classes, n_features + 1)   # [N, C, F+1]
                sv = np.transpose(sv, (1, 0, 2))                          # [C, N, F+1]
            else:
                sv = sv[None, ...]
            return sv
        if "catboost" in name:
            from catboost import Pool
            pool = Pool(X_df, label=None)
            sv = model.get_feature_importance(pool, type="ShapValues")     # [N, (F+1)*C] or [N, F+1]
            sv = np.asarray(sv)
            if n_classes > 2:
                sv = sv.reshape(sv.shape[0], n_classes, n_features + 1)   # [N, C, F+1]
                sv = np.transpose(sv, (1, 0, 2))                          # [C, N, F+1]
            else:
                sv = sv[None, ...]
            return sv
    except Exception as e:
        print(f"[Contribs native path unavailable for {type(model).__name__}] -> {e}")
    return None  # signal fallback

def get_treeexplainer_shap(model, X_df):
    import shap
    exp = shap.TreeExplainer(model)
    sv = exp.shap_values(X_df)  # list length n_classes for multiclass
    base = exp.expected_value
    # Standardize to [C, N, F] (no bias for TreeExplainer)
    if isinstance(sv, list):
        sv = np.stack([np.asarray(s) for s in sv], axis=0)  # [C, N, F]
    else:
        sv = np.asarray(sv)[None, ...]                      # [1, N, F]
    return sv, base

print("\n Computing feature contributions (native GPU path if available)…")
sv_native = get_native_contribs(best_clf, X)
if sv_native is not None:
    # native gives [C, N, F+1] where last col is bias; per-class base values = bias for each sample
    sv_by_class = sv_native[:, :, :-1]     # drop bias for feature importances
    base_values = sv_native[:, :, -1].mean(axis=1)  # avg bias per class (scalar for plotting)
else:
    # fallback to TreeExplainer
    import shap
    shap.initjs()
    sv_by_class, base_values = get_treeexplainer_shap(best_clf, X)  # [C,N,F], base list/float

# --------- Per-class Top-10 features by mean |SHAP| (Task 2a)
print("\n=== Per-class Top-10 Features by mean |contrib| ===")
for cidx, cname in enumerate(classes):
    mask = (y_enc == cidx)
    sv_abs = np.abs(sv_by_class[cidx])  # [N, F]
    mean_abs = sv_abs[mask].mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:10]
    print(f"\n{cname}:")
    for i in top_idx:
        print(f"  {feature_names[i]:30s}  {float(mean_abs[i]):.6f}")

# --------- Force plots for TARGET_ID (Task 2b)
# Get row index
if TARGET_ID in set(sample_ids):
    row_idx = int(np.where(sample_ids == TARGET_ID)[0][0])
else:
    row_idx = 0
    print(f"\n[Note] Could not find {TARGET_ID}; using index 0 for demo.")
X_row = X.iloc[row_idx, :].values

# Ensure output dir
os.makedirs("plots", exist_ok=True)

print(f"\n Saving per-class force plots for sample #{row_idx} (ID={sample_ids[row_idx]}) …")
for cidx, cname in enumerate(classes):
    # choose base for this class
    if isinstance(base_values, (list, tuple, np.ndarray)):
        base_c = base_values[cidx] if np.ndim(base_values) == 1 else float(np.mean(base_values[cidx]))
    else:
        base_c = float(base_values)
    # shap.force_plot expects: base value (scalar), shap values (F,), feature values (F,)
    try:
        import shap
        shap.force_plot(
            base_c,
            sv_by_class[cidx][row_idx, :],
            X_row,
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.title(f"Force Plot — {sample_ids[row_idx]} — Class {cname}")
        plt.tight_layout()
        plt.savefig(f"plots/force_{sample_ids[row_idx]}_{cname}.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"[force plot warning] {cname}: {e}")
print("✅ Saved force plots to ./plots/")
