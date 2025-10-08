# -----------------------------
# REGRESSION PIPELINE (HW3 Task 3 & 4) - Improved
# -----------------------------
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ---------- Load data ----------
gdsc = pd.read_csv("hw3-drug-screening-data.csv")  # Update path if needed

id_cols = ["CELL_LINE_NAME", "DRUG_NAME"]
assert all(c in gdsc.columns for c in id_cols + ["LN_IC50"]), "Missing required columns."

y = gdsc["LN_IC50"].values
meta = gdsc[id_cols].copy()
X = gdsc.drop(columns=id_cols + ["LN_IC50"])
feature_names = X.columns.tolist()

# ---------- Define models ----------
regressors = {
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
    "GBM": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1, tree_method="hist"
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=64,
        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
    ),
    "CatBoost": CatBoostRegressor(
        iterations=200, learning_rate=0.05, depth=6,
        random_seed=42, verbose=False, loss_function="RMSE"
    ),
}

# ---------- CV & Model selection ----------
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "MAE": make_scorer(mean_absolute_error),
    "MSE": make_scorer(mean_squared_error),
    "R2": make_scorer(r2_score),
}

reg_results = {}
for name, model in regressors.items():
    cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    mae = np.mean(cv_res["test_MAE"])
    mse = np.mean(cv_res["test_MSE"])
    rmse = np.sqrt(mse)
    r2 = np.mean(cv_res["test_R2"])
    reg_results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

reg_df = pd.DataFrame(reg_results).T.sort_values(by=["RMSE", "MAE", "R2"], ascending=[True, True, False])
print("\n=== Regression CV Results (mean metrics) ===")
print(reg_df)

best_reg_name = reg_df.index[0]
best_reg = regressors[best_reg_name]
print(f"\nBest regressor: {best_reg_name}")

# ---------- Train-test split + final fit ----------
# keep alignment between X_test rows and meta_test rows by resetting index
X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X, y, meta, test_size=0.2, random_state=42
)
# reset indices so positions 0..N_test-1 match across X_test, y_test, meta_test
X_test = X_test.reset_index(drop=True)
y_test = pd.Series(y_test).reset_index(drop=True).values
meta_test = meta_test.reset_index(drop=True)

best_reg.fit(X_train, y_train)

# ---------- SHAP on full X_test (needed for per-drug grouping) ----------
# choose explainer for tree models
if best_reg_name in ["DecisionTree", "RandomForest", "GBM", "XGBoost", "LightGBM", "CatBoost"]:
    explainer_r = shap.TreeExplainer(best_reg)
else:
    explainer_r = shap.Explainer(best_reg, X_train)

# compute SHAP on the full test set (so grouping by drug uses exact test indices)
# If X_test is large and you worry about runtime, you can still compute for full X_test but may subsample per-drug later.
shap_values_r = explainer_r.shap_values(X_test) 
# Ensure numpy array shape (N, F)
shap_arr = np.asarray(shap_values_r)
abs_shap = np.abs(shap_arr)  # shape: (N, F)

# ---------- Task 4a: Per-drug Top-10 by mean |SHAP| (on X_test) ----------
drug_top10 = {}
# meta_test has index 0..N_test-1 now
groups = meta_test.groupby("drug").groups
for drug_name, idxs in groups.items():
    idxs = list(idxs)  # positions inside X_test / abs_shap
    if len(idxs) == 0:
        continue
    mean_abs = abs_shap[idxs, :].mean(axis=0)   # mean across samples of that drug
    top_idx = np.argsort(mean_abs)[::-1][:10]
    drug_top10[drug_name] = [(X.columns[i], float(mean_abs[i])) for i in top_idx]

# Print per-drug top-10
print("\n=== Per-drug Top-10 Features by mean |SHAP| (on test set) ===")
for d, feats in drug_top10.items():
    print(f"\n{d}:")
    for f, val in feats:
        print(f"  {f:30s}  {val:.6f}")

# ---------- Task 4b: least-error drug-cell pair and its top-10 ----------
y_pred = best_reg.predict(X_test)
errors = np.abs(y_pred - y_test)
min_idx = int(np.argmin(errors))   # index into X_test / meta_test
pair = (meta_test.loc[min_idx, "drug"], meta_test.loc[min_idx, "cell_line"])
print(f"\nLeast prediction error sample: drug={pair[0]}, cell_line={pair[1]}")
print(f"True LN_IC50={y_test[min_idx]:.4f}, Pred={y_pred[min_idx]:.4f}, AbsError={errors[min_idx]:.6f}")

row_shap = shap_arr[min_idx, :]     # signed shap values for that sample
order = np.argsort(np.abs(row_shap))[::-1][:10]
pair_top10 = [(X.columns[i], float(row_shap[i])) for i in order]

print("\nTop-10 features for least-error pair (signed SHAP):")
for f, val in pair_top10:
    print(f"  {f:30s}  SHAP={val:.6f}")

# optional: force plot (matplotlib)
shap.force_plot(
    explainer_r.expected_value,
    row_shap,
    X_test.loc[min_idx, :],
    feature_names=X.columns.tolist(),
    matplotlib=True,
    show=False
)
plt.title(f"Force Plot — {pair[0]} | {pair[1]}")
plt.tight_layout()
plt.savefig(f"force_{pair[0]}_{pair[1]}.png", dpi=150)
plt.close()
print("Saved regression force plot for least-error pair.")