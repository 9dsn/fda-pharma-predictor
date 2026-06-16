"""ML Model Training, LOOCV Evaluation, and SHAP Explainability

This module trains two classifiers on FDA AdCom briefing text features,
evaluates them with Leave-One-Out Cross-Validation (LOOCV), and computes
SHAP values on a final full-dataset fit for model explainability"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import shap
import warnings

warnings.filterwarnings("ignore")  # suppress sklearn convergence noise during LOOCV

DATA_PATH = "data/processed/feature_matrix.csv"

METADATA_COLS = ["source", "meeting_date", "question_text"]

TARGET_COL = "outcome"

#dropped the tie
VALID_LABELS = ["yes", "no"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """load the feature matrix CSV, drop metadata columns and 'tie' rows,
    and return (X, y) ready for scikit-learn."""

    df = pd.read_csv(DATA_PATH)
    clean_df = df[df[TARGET_COL].isin(VALID_LABELS)]

    X = clean_df.drop(columns=METADATA_COLS + [TARGET_COL]).copy()
    y = clean_df[TARGET_COL].reset_index(drop=True)

    print(f"loaded X: {X.shape}  y distribution: {y.value_counts().to_dict()}")
    return X, y


def run_loocv(X: pd.DataFrame, y: pd.Series, model_name: str) -> dict:
    """Run Leave-One-Out Cross-Validation for a single model type"""
    loo = LeaveOneOut()

    y_true = []
    y_pred = []
    y_pred_proba = []

    # converting it to numpy for clean indexing inside the loop
    X_arr = X.values
    y_arr = y.values

    # Iterate: each iteration gives train indices and one test index.
    for train_idx, test_idx in loo.split(X_arr):

        # slicing
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        if model_name == "logistic_regression":
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            model = LogisticRegression(
                penalty="l2",
                max_iter=1000,
                random_state=42  
            )
        elif model_name == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

        model.fit(X_train, y_train)
        y_pred.append(model.predict(X_test)[0])
            # probability for "yes"
        pos_index = list(model.classes_).index("yes")
        y_pred_proba.append(model.predict_proba(X_test)[0, pos_index])
            #append the true label
        y_true.append(y_test[0])

    return {
        "model_name": model_name,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_results(results: dict) -> None:
    """Print evaluation summary for one model's LOOCV results"""
    model_name = results["model_name"]
    y_true = results["y_true"]
    y_pred = results["y_pred"]

    print(f"\n{'='*50}")
    print(f"  Results: {model_name.upper()}")
    print(f"{'='*50}")

    # accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy {acc:.2%} ({sum(t == p for t, p in zip(y_true, y_pred))} / {len(y_true)} correct)")
    # confusion matrix
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_true, y_pred))

    # full classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


# ---------------------------------------------------------------------------
# Final Full-Dataset Fit
# ---------------------------------------------------------------------------
def fit_final_model(X: pd.DataFrame, y: pd.Series, model_name: str):
    """Fit the chosen model on the ENTIRE dataset (no train/test split)"""
    feature_names = list(X.columns)
    X_arr = X.values

    # scale if log reg
    if model_name == "logistic_regression":
        scaler = StandardScaler()
        scaler.fit(X_arr)
        X_scaled = scaler.transform(X_arr)  # fit+transform in one step
    else:
        X_scaled = X_arr

    # Instantiate model
    if model_name == "logistic_regression":
        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=1000,
            random_state=42
        )
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    model.fit(X_scaled, y)

    return model, X_scaled, feature_names

# ---------------------------------------------------------------------------
# Step 5: SHAP Explainability
# ---------------------------------------------------------------------------
def compute_shap(model, X_scaled: np.ndarray, feature_names: list, model_name: str):
    """compute and display SHAP values for every row in the dataset.

    SHAP (SHapley Additive exPlanations) answers the question:
    "For this specific prediction, how much did each feature push the
    probability up or down compared to the average prediction?"""

    print(f"\nComputing SHAP values for {model_name}...")

    # instantiate the correct explainer
    if model_name == "logistic_regression":
        explainer = shap.LinearExplainer(model, X_scaled)
    elif model_name == "random_forest":
        explainer = shap.TreeExplainer(model)

    # Compute SHAP values
    shap_values = explainer(X_scaled)

    # Handle binary classification shape: (n_samples, n_features, n_classes) → slice class 1 ("yes")
    if shap_values.values.ndim == 3:
        shap_values = shap_values[..., 1]  # keep only the "yes" class

    # Print mean absolute SHAP value per feature (importance ranking)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    print("\nMean |SHAP| per feature (higher = more influential):")
    for name, val in sorted(zip(feature_names, mean_abs_shap), key=lambda x: -x[1]):
        print(f"  {name:<30} {val:.4f}")

    # summary plot
    shap.summary_plot(shap_values, features=X_scaled, feature_names=feature_names)


# ---------------------------------------------------------------------------
# Orchestrator / Entry Point
# ---------------------------------------------------------------------------
def main():
    # Step 1 — Load
    X, y = load_data(DATA_PATH)

    # Step 2 & 3 — LOOCV + Evaluate
    for model_name in ["logistic_regression", "random_forest"]:
        results = run_loocv(X, y, model_name)
        evaluate_results(results)

    # Step 4 & 5 — Full-dataset fit + SHAP
    for model_name in ["logistic_regression", "random_forest"]:
        model, X_scaled, feature_names = fit_final_model(X, y, model_name)
        compute_shap(model, X_scaled, feature_names, model_name)


if __name__ == "__main__":
    main()