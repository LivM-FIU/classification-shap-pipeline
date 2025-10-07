# -----------------------------
# REGRESSION PIPELINE (HW3 Task 3 & 4)
# -----------------------------
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import shap
import matplotlib.pyplot as plt

# ---------- Load data ----------
# If downloaded from Google Drive, load the CSV path here:
# The dataset should include columns: ["cell_line", "drug", "LN_IC50", <gene features...>]
gdsc = pd.read_csv("GDSC2_13drugs_LN_IC50.csv")  # rename to your local filename

id_cols = ["cell_line", "drug"]
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
        n_estimators=600, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1,
        tree_method="hist"
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=800, learning_rate=0.05, num_leaves=64,
        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
    ),
    "CatBoost": CatBoostRegressor(
        iterations=800, learning_rate=0.05, depth=6,
        random_seed=42, verbose=False, loss_function="RMSE"
    ),
}

# ---------- CV & Model selection (Task 3) ----------
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
    "mse": make_scorer(mean_squared_error, greater_is_better=False),
    # RMSE we’ll compute from MSE later
    "r2": make_scorer(r2_score)
}

reg_results = {}
for name, model in regressors.items():
    cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    mae = -np.mean(cv_res["test_mae"])
    mse = -np.mean(cv_res["test_mse"])
    rmse = np.sqrt(mse)
    r2 = np.mean(cv_res["test_r2"])
    reg_results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

reg_df = pd.DataFrame(reg_results).T.sort_values(by=["RMSE","MAE","R2"], ascending=[True,True,False])
print("\n=== Regression CV Results (mean metrics) ===")
print(reg_df)

best_reg_name = reg_df.index[0]
best_reg = regressors[best_reg_name]
print(f"\nBest regressor: {best_reg_name}")

# Fit best regressor on train; keep a test split for SHAP & error analysis
X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X, y, meta, test_size=0.2, random_state=42
)
best_reg.fit(X_train, y_train)

# ---------- SHAP: Global (per-drug Top-10) (Task 4a) ----------
# TreeExplainer works well for these models
explainer_r = shap.TreeExplainer(best_reg)
shap_values_r = explainer_r.shap_values(X_test)  # shape: [n_samples, n_features]
abs_shap = np.abs(shap_values_r)

# For each drug, compute mean |SHAP| across its rows, then top-10 features
drug_top10 = {}
for drug_name, idxs in meta_test.groupby("drug").groups.items():
    idxs = list(idxs)
    if len(idxs) == 0:
        continue
    mean_abs = abs_shap[idxs, :].mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:10]
    drug_top10[drug_name] = [(feature_names[i], float(mean_abs[i])) for i in top_idx]

print("\n=== Per-drug Top-10 Features by mean |SHAP| (on test set) ===")
for d, feats in drug_top10.items():
    print(f"\n{d}:")
    for f, val in feats:
        print(f"  {f:30s}  {val:.6f}")

# ---------- Least prediction error pair & its top-10 features (Task 4b) ----------
y_pred = best_reg.predict(X_test)
errors = np.abs(y_pred - y_test)
min_idx = int(np.argmin(errors))
pair = (meta_test.iloc[min_idx]["drug"], meta_test.iloc[min_idx]["cell_line"])
print(f"\nLeast prediction error sample: drug={pair[0]}, cell_line={pair[1]}")
print(f"True LN_IC50={y_test[min_idx]:.4f}, Pred={y_pred[min_idx]:.4f}, AbsError={errors[min_idx]:.6f}")

row_shap = shap_values_r[min_idx, :]
order = np.argsort(np.abs(row_shap))[::-1][:10]
pair_top10 = [(feature_names[i], float(row_shap[i])) for i in order]

print("\nTop-10 features for least-error pair (signed SHAP):")
for f, val in pair_top10:
    print(f"  {f:30s}  SHAP={val:.6f}")

# Optional: a force plot image for this pair
shap.force_plot(
    explainer_r.expected_value,
    row_shap,
    X_test.iloc[min_idx, :],
    feature_names=feature_names,
    matplotlib=True,
    show=False
)
plt.title(f"Force Plot — {pair[0]} | {pair[1]}")
plt.tight_layout()
plt.savefig(f"force_{pair[0]}_{pair[1]}.png", dpi=150)
plt.close()
print("Saved regression force plot for least-error pair.")
