"""FastAPI backend for ODAC Signal.

The API serves the static marketing frontend and exposes production-style
prediction endpoints backed by the saved FDA AdCom model artifacts.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
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

from src.features.feature_engineering import extract_features


MODEL_DIR = Path("saved_model")
MODEL_PATH = MODEL_DIR / "model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
HISTORICAL_PATH = MODEL_DIR / "historical.csv"


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Warm model assets with FastAPI's current lifespan startup pattern."""
    load_assets()
    yield


app = FastAPI(title="ODAC Signal API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_CACHE: dict[str, Any] = {}


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
