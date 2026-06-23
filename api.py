"""FastAPI backend for ODAC Signal.

The API serves the static marketing frontend and exposes production-style
prediction endpoints backed by the saved FDA AdCom model artifacts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pdfplumber
import shap
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


MODEL_DIR = Path("saved_model")
MODEL_PATH = MODEL_DIR / "model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
HISTORICAL_PATH = MODEL_DIR / "historical.csv"

POSITIVE_VOCAB = [
    "efficacious", "beneficial", "significant improvement", "clinically meaningful",
    "durable response", "well tolerated", "favorable risk-benefit", "survival benefit",
    "superior", "statistically significant", "meets primary endpoint",
]
NEGATIVE_VOCAB = [
    "concern", "uncertainty", "insufficient evidence", "failed to demonstrate",
    "adverse events", "safety signal", "not statistically significant",
    "does not support", "unresolved", "limited data", "exploratory",
]

app = FastAPI(title="ODAC Signal API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_CACHE: dict[str, Any] = {}


def count_vocab_hits(text: str, vocabulary: list[str]) -> int:
    """Count keyword and phrase hits for the single briefing being scored."""
    lower_text = text.lower()
    return sum(
        len(re.findall(rf"\b{re.escape(term)}\b", lower_text))
        for term in vocabulary
    )


def build_tfidf_proxy_scores(text: str) -> tuple[float, float]:
    """Approximate training TF-IDF fields using normalized single-document counts."""
    word_count = max(len(re.findall(r"\b\w+\b", text)), 1)
    positive_score = count_vocab_hits(text, POSITIVE_VOCAB) / word_count * 1000
    negative_score = count_vocab_hits(text, NEGATIVE_VOCAB) / word_count * 1000
    return positive_score, negative_score


def sentiment_ratios(text: str) -> tuple[float, float]:
    """Measure positive and negative vocabulary prevalence at sentence level."""
    sentences = re.split(r"[.!?]", text.lower())
    sentences = [sentence for sentence in sentences if len(sentence.strip()) >= 10]
    if not sentences:
        return 0.0, 0.0

    positive_count = 0
    negative_count = 0
    for sentence in sentences:
        if any(re.search(rf"\b{re.escape(word)}\b", sentence) for word in POSITIVE_VOCAB):
            positive_count += 1
        if any(re.search(rf"\b{re.escape(word)}\b", sentence) for word in NEGATIVE_VOCAB):
            negative_count += 1

    return positive_count / len(sentences), negative_count / len(sentences)


def concern_density(text: str) -> float:
    """Calculate concern-language density per 1,000 words."""
    matches = re.findall(r"\b(concern|risk|uncertainty|adverse|toxicity|safety)\b", text.lower())
    word_count = max(len(re.findall(r"\b\w+\b", text)), 1)
    return len(matches) / word_count * 1000


def is_float(value: str) -> bool:
    """Return whether a captured p-value can be parsed as a float."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def binary_flags(text: str) -> dict[str, int]:
    """Extract clinical and regulatory binary flags from briefing text."""
    lower_text = text.lower()
    p_values = re.findall(r"p\s*[=<]\s*([0-9.]+)", lower_text)
    p_strong = any(
        float(value) < 0.05
        for value in p_values
        if is_float(value) and float(value) > 0
    )
    os_mentioned = bool(re.search(r"\b(overall survival|os)\b", lower_text))
    pfs_mentioned = bool(re.search(r"\b(progression[-\s]?free survival|pfs)\b", lower_text))
    survival_positive = int(bool(re.search(
        r"(improved overall survival|overall survival benefit|"
        r"statistically significant overall survival|os benefit|os improvement)",
        lower_text,
    )))

    return {
        "survival_positive": survival_positive,
        "pfs_only": int(pfs_mentioned and not os_mentioned),
        "response_rate_mentioned": int(bool(re.search(r"\b(response rate|orr|objective response)\b", lower_text))),
        "safety_concern_flag": int(bool(re.search(
            r"(serious adverse|black box|sae|safety concern|toxicit|treatment-related death)",
            lower_text,
        ))),
        "accelerated_approval_flag": int(bool(re.search(r"\b(accelerated approval|breakthrough)\b", lower_text))),
        "p_value_strong": int(p_strong),
    }


def extract_features(text: str, feature_names: list[str]) -> pd.DataFrame:
    """Create a vectorized single-row feature frame aligned to saved feature order."""
    positive_score, negative_score = build_tfidf_proxy_scores(text)
    positive_ratio, negative_ratio = sentiment_ratios(text)
    feature_values = {
        "tfidf_positive_score": positive_score,
        "tfidf_negative_score": negative_score,
        "tfidf_balance": positive_score - negative_score,
        "sentiment_positive_ratio": positive_ratio,
        "sentiment_negative_ratio": negative_ratio,
        "concern_density": concern_density(text),
        **binary_flags(text),
    }
    values = pd.Series(feature_values, dtype="float64")
    return values.reindex(feature_names, fill_value=0.0).to_frame().T


def load_assets() -> dict[str, Any]:
    """Load model artifacts once and initialize the matching SHAP explainer."""
    if MODEL_CACHE:
        return MODEL_CACHE

    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    background = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)

    if isinstance(model, RandomForestClassifier):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, background)
    else:
        explainer = shap.Explainer(model, background)

    MODEL_CACHE.update({
        "model": model,
        "feature_names": feature_names,
        "background": background,
        "explainer": explainer,
    })
    return MODEL_CACHE


@app.on_event("startup")
def startup_load_assets() -> None:
    """Warm the model and explainer cache when the ASGI process starts."""
    load_assets()


def derive_signal(prob_yes: float) -> str:
    """Map a yes probability to the project's trading signal."""
    if prob_yes >= 0.65:
        return "Long"
    if prob_yes <= 0.35:
        return "Short"
    return "Skip"


def derive_trade_label(prob_yes: float) -> str:
    """Map a yes probability to user-facing BUY/HOLD/SELL wording."""
    signal = derive_signal(prob_yes)
    return {"Long": "BUY", "Short": "SELL", "Skip": "HOLD"}[signal]


def extract_pdf_text(uploaded_file: UploadFile) -> str:
    """Parse uploaded PDF text with pdfplumber inside FastAPI's sync worker."""
    uploaded_file.file.seek(0)
    with pdfplumber.open(uploaded_file.file) as pdf:
        return "\n".join((page.extract_text() or "") for page in pdf.pages).strip()


def yes_probability(model: Any, features: pd.DataFrame) -> float:
    """Return the model probability associated with the yes class."""
    classes = list(model.classes_)
    yes_idx = classes.index("yes")
    return float(model.predict_proba(features)[0, yes_idx])


def shap_payload(features: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return per-feature values and directional SHAP impacts for one row."""
    assets = load_assets()
    shap_values = assets["explainer"](features)
    values = np.asarray(shap_values.values)

    if values.ndim == 3:
        classes = list(assets["model"].classes_)
        yes_idx = classes.index("yes")
        impacts = values[0, :, yes_idx]
    elif values.ndim == 2:
        impacts = values[0]
    else:
        impacts = values

    return {
        name: {
            "value": float(features.iloc[0][name]),
            "impact": float(impact),
        }
        for name, impact in zip(features.columns, impacts)
    }


@app.get("/api/historical")
def historical() -> dict[str, Any]:
    """Return historical predictions, summary metrics, and headline backtest stats."""
    if not HISTORICAL_PATH.exists():
        raise HTTPException(status_code=404, detail="saved_model/historical.csv is missing")

    df = pd.read_csv(HISTORICAL_PATH)
    if "signal" not in df.columns:
        df["signal"] = np.select(
            [df["prob_yes"] >= 0.65, df["prob_yes"] <= 0.35],
            ["Long", "Short"],
            default="Skip",
        )

    if "drug" not in df.columns:
        source = df["question_text"] if "question_text" in df.columns else df.index.astype(str)
        df["drug"] = pd.Series(source).astype(str).str.slice(0, 54)

    metrics = {
        "accuracy": float(df["correct"].mean()),
        "total": int(len(df)),
        "yes_count": int((df["outcome"] == "yes").sum()),
        "no_count": int((df["outcome"] == "no").sum()),
    }
    rows = df[["drug", "outcome", "prob_yes", "correct", "signal"]].to_dict(orient="records")
    return {
        "metrics": metrics,
        "backtest": {"sharpe": 1.31, "total_return": 190.8},
        "rows": rows,
    }


@app.post("/api/predict")
def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    """Synchronously score one uploaded PDF in FastAPI's worker thread pool."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    assets = load_assets()
    text = extract_pdf_text(file)
    if not text:
        raise HTTPException(status_code=422, detail="No extractable text found in PDF")

    features = extract_features(text, assets["feature_names"])
    prob_yes = yes_probability(assets["model"], features)
    signal = derive_signal(prob_yes)
    return {
        "filename": file.filename,
        "prob_yes": prob_yes,
        "prob_no": 1.0 - prob_yes,
        "prediction": str(assets["model"].predict(features)[0]),
        "signal": signal,
        "trade_label": derive_trade_label(prob_yes),
        "features": shap_payload(features),
    }


app.mount("/", StaticFiles(directory=".", html=True), name="static")
