# 🧠 ML-MULTI-ALGO: Tree-Based Models with SHAP Interpretation

This project trains multiple **tree-based machine learning models** (Decision Trees, Random Forests, Gradient Boosting, XGBoost, LightGBM, and CatBoost) for **classification and regression**, and uses **SHAP** to interpret model predictions.

## 🚀 Features
- Automated training and cross-validation across multiple models
- GPU acceleration for XGBoost, LightGBM, and CatBoost
- SHAP explainability (global + per-class feature importance)
- Force plots saved as PNGs for interpretability

---

## 📦 Installation

Clone this repository:

```bash
git clone https://github.com/<your-username>/ML-MULTI-ALGO.git
cd ML-MULTI-ALGO

# Create virtual environment
python -m venv venv
# Activate it
venv\Scripts\activate      # (Windows)
# or
source venv/bin/activate   # (Mac/Linux)

# Install required dependencies
pip install -r requirements.txt
