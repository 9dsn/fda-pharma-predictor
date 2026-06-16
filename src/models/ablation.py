"""Dropping briefing char count since log regressions depends heavily on this"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix
import shap
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = "data/processed/feature_matrix.csv"
METADATA_COLS = ["source", "meeting_date", "question_text"]
TARGET_COL = "outcome"
VALID_LABELS = ["yes", "no"]

# whats being tested
ABLATION_FEATURE = "briefing_char_count"

# how much the accuract drops is acceptable, before its red flag
DROP_THRESHOLD = 0.10  # 10 percentage points

#─────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────
def load_data(drop_feature=None):
    """loading the feature matrix"""
    df = pd.read_csv(DATA_PATH)
    clean_df = df[df[TARGET_COL].isin(VALID_LABELS)].copy()
 
    X = clean_df.drop(columns=METADATA_COLS + [TARGET_COL])
    y = clean_df[TARGET_COL].reset_index(drop=True)
 
    if drop_feature and drop_feature in X.columns:
        X = X.drop(columns=[drop_feature])
        print(f"  [ablation] dropped '{drop_feature}' → {X.shape[1]} features remaining")
    else:
        print(f"  [full] using all {X.shape[1]} features")
 
    return X, y


# ─────────────────────────────────────────────────────────
# LOOCV
# ─────────────────────────────────────────────────────────
def run_loocv(X, y, model_name):
    """Leave-One-Out Cross-Validation"""
    loo = LeaveOneOut()
    X_arr, y_arr = X.values, y.values
    y_true, y_pred, y_proba = [], [], []
 
    for train_idx, test_idx in loo.split(X_arr):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]
 
        if model_name == "logistic_regression":
            # LR needs scaling: each feature must have mean≈0, std≈1
            # We fit the scaler ONLY on training data to avoid data leakage
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)  # use train's mean/std on test
            model = LogisticRegression(penalty="l2", max_iter=1000, random_state=42)
 
        elif model_name == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
 
        model.fit(X_train, y_train)
        y_pred.append(model.predict(X_test)[0])
 
        pos_idx = list(model.classes_).index("yes")
        y_proba.append(model.predict_proba(X_test)[0, pos_idx])
        y_true.append(y_test[0])
 
    return {"y_true": y_true, "y_pred": y_pred, "y_proba": y_proba}


# ─────────────────────────────────────────────────────────
# SHAP on ablated model (to see what fills the vacuum)
# ─────────────────────────────────────────────────────────
def compute_shap_importance(X, y, model_name):
    """fit on full dataset, then compute mean |SHAP| per feature. (finds features that model leans on now)"""
    feature_names = list(X.columns)
    X_arr = X.values
 
    if model_name == "logistic_regression":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)
        model = LogisticRegression(penalty="l2", max_iter=1000, random_state=42)
        model.fit(X_scaled, y)
        explainer = shap.LinearExplainer(model, X_scaled)
        sv = explainer(X_scaled)
    else:
        X_scaled = X_arr
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        explainer = shap.TreeExplainer(model)
        sv = explainer(X_scaled)
 
    # Handle 3D output (binary classification returns shape: n x features x classes)
    if sv.values.ndim == 3:
        sv = sv[..., 1]  # slice the "yes" class
 
    mean_abs = np.abs(sv.values).mean(axis=0)
    ranked = sorted(zip(feature_names, mean_abs), key=lambda x: -x[1])
    return ranked


# ─────────────────────────────────────────────────────────
# Run one full comparison for a given model
# ─────────────────────────────────────────────────────────
def run_comparison(model_name):
    print(f"\n{'═'*55}")
    print(f"  {model_name.upper().replace('_', ' ')}")
    print(f"{'═'*55}")
 
    #this is the full model
    print("\n[1/2] Full model (all features)")
    X_full, y = load_data(drop_feature=None)
    results_full = run_loocv(X_full, y, model_name)
    acc_full = accuracy_score(results_full["y_true"], results_full["y_pred"])
    cm_full = confusion_matrix(results_full["y_true"], results_full["y_pred"])
    print(f"  Accuracy: {acc_full:.2%}  ({int(acc_full*len(y))}/{len(y)})")
    print(f"  Confusion matrix:\n{cm_full}")
 
    # ablated model
    print(f"\n[2/2] Ablated model (dropping '{ABLATION_FEATURE}')")
    X_abl, y = load_data(drop_feature=ABLATION_FEATURE)
    results_abl = run_loocv(X_abl, y, model_name)
    acc_abl = accuracy_score(results_abl["y_true"], results_abl["y_pred"])
    cm_abl = confusion_matrix(results_abl["y_true"], results_abl["y_pred"])
    print(f"  Accuracy: {acc_abl:.2%}  ({int(acc_abl*len(y))}/{len(y)})")
    print(f"  Confusion matrix:\n{cm_abl}")
 
    # delta
    delta = acc_full - acc_abl
    print(f"\n  Accuracy drop: {delta:.2%} ({acc_full:.2%} → {acc_abl:.2%})")
 
    # shap on the ablated model
    print(f"\n  SHAP importances (ablated model, top 8):")
    shap_ranked = compute_shap_importance(X_abl, y, model_name)
    for feat, val in shap_ranked[:8]:
        bar = "█" * int(val * 50)
        print(f"    {feat:<30} {val:.4f}  {bar}")
 
    return delta, acc_full, acc_abl
 

# ─────────────────────────────────────────────────────────
# Verdict
# ─────────────────────────────────────────────────────────
def print_verdict(results_by_model):
    """summarize the findings"""
    print(f"\n{'═'*55}")
    print("  VERDICT")
    print(f"{'═'*55}\n")
 
    any_problem = False
    for model_name, (delta, acc_full, acc_abl) in results_by_model.items():
        label = model_name.replace("_", " ").upper()
        status = "PROBLEM" if delta > DROP_THRESHOLD else "OK"
        print(f"  {label}")
        print(f"    Full: {acc_full:.2%}  →  Ablated: {acc_abl:.2%}  (drop: {delta:.2%})  {status}\n")
        if delta > DROP_THRESHOLD:
            any_problem = True
 
    if any_problem:
        print("At least one model collapsed without briefing_char_count.")
    else:
        print("Both models retain meaningful accuracy without briefing_char_count.")

def main():
    print("\nPhase 7 — Ablation Study")
    results = {}
    for model_name in ["logistic_regression", "random_forest"]:
        delta, acc_full, acc_abl = run_comparison(model_name)
        results[model_name] = (delta, acc_full, acc_abl)
 
    print_verdict(results)
 
 
if __name__ == "__main__":
    main()