"""Train and persist the deployment model for the FDA AdCom Vote Predictor.

Run order:
1. pip install streamlit pdfplumber shap joblib scikit-learn --break-system-packages
2. PYTHONPATH=. python train_and_save.py        # generates saved_model/
3. streamlit run app.py                          # test locally at localhost:8501
4. Push to Hugging Face Space: app.py, requirements.txt, README.md, saved_model/
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut


DATA_PATH = Path("data/processed/feature_matrix.csv")
OUTPUT_DIR = Path("saved_model")
METADATA_COLS = ["source", "meeting_date", "question_text", "outcome"]
DROP_COLS = ["briefing_char_count"]
VALID_LABELS = ["yes", "no"]
RANDOM_STATE = 42


def load_feature_matrix(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the processed feature matrix and keep only modeled vote outcomes.

    The deployment model should match the validated binary setup, so rows with
    non-yes/no outcomes are excluded before feature selection or fitting.
    """
    df = pd.read_csv(path)
    return df[df["outcome"].isin(VALID_LABELS)].copy().reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build X/y while dropping leakage, metadata, and unusable features.

    `briefing_char_count` was removed by ablation, and zero-variance columns
    cannot help the forest or future inference, so they are dropped here and
    the resulting ordered feature list is saved for the Streamlit app.
    """
    feature_df = df.drop(columns=METADATA_COLS + DROP_COLS, errors="ignore").copy()
    zero_variance_cols = [
        col for col in feature_df.columns if feature_df[col].nunique(dropna=False) <= 1
    ]
    X = feature_df.drop(columns=zero_variance_cols)
    y = df["outcome"].copy()
    return X, y


def make_random_forest() -> RandomForestClassifier:
    """Create the exact Random Forest used for deployment and LOOCV scoring."""
    return RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)


def run_loocv_probabilities(X: pd.DataFrame, y: pd.Series) -> tuple[list[float], list[str]]:
    """Generate out-of-fold yes probabilities with Leave-One-Out CV.

    Each row is predicted only by a model that did not train on that row, which
    keeps the historical probabilities honest for the app's results tab.
    """
    loo = LeaveOneOut()
    probabilities: list[float] = []
    predictions: list[str] = []

    X_values = X.values
    y_values = y.values

    for train_idx, test_idx in loo.split(X_values):
        model = make_random_forest()
        model.fit(X_values[train_idx], y_values[train_idx])
        yes_idx = list(model.classes_).index("yes")
        probabilities.append(float(model.predict_proba(X_values[test_idx])[0, yes_idx]))
        predictions.append(str(model.predict(X_values[test_idx])[0]))

    return probabilities, predictions


def train_final_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """Fit the deployable Random Forest on all available labeled rows.

    Cross-validation estimates historical performance, but the app should use
    every validated meeting to train the final production artifact.
    """
    model = make_random_forest()
    model.fit(X, y)
    return model


def build_historical_results(
    df: pd.DataFrame, probabilities: list[float], predictions: list[str]
) -> pd.DataFrame:
    """Create the compact historical table consumed by the Streamlit app."""
    historical = df[["meeting_date", "question_text", "outcome"]].copy()
    historical["prob_yes"] = probabilities
    historical["predicted"] = predictions
    historical["correct"] = historical["outcome"] == historical["predicted"]
    return historical


def save_artifacts(
    model: RandomForestClassifier,
    feature_names: list[str],
    historical: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> list[Path]:
    """Persist the trained model, feature order, and historical predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.pkl"
    features_path = output_dir / "feature_names.pkl"
    historical_path = output_dir / "historical.csv"

    joblib.dump(model, model_path)
    joblib.dump(feature_names, features_path)
    historical.to_csv(historical_path, index=False)

    return [model_path, features_path, historical_path]


def format_size(path: Path) -> str:
    """Return a human-readable artifact size for the success message."""
    size = path.stat().st_size
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def main() -> None:
    """Run the full deployment training workflow and report saved artifacts."""
    df = load_feature_matrix()
    X, y = split_features_target(df)
    probabilities, predictions = run_loocv_probabilities(X, y)
    model = train_final_model(X, y)
    historical = build_historical_results(df, probabilities, predictions)
    saved_paths = save_artifacts(model, list(X.columns), historical)

    print("\nDeployment model training complete.")
    print(f"Rows trained: {len(df)}")
    print(f"Features saved: {list(X.columns)}")
    print("Saved artifacts:")
    for path in saved_paths:
        print(f"  - {path} ({format_size(path)})")


if __name__ == "__main__":
    main()
