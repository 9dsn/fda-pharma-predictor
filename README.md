---
title: FDA AdCom Vote Predictor
emoji: 💊
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# FDA AdCom Vote Predictor

This Streamlit app predicts FDA Oncologic Drugs Advisory Committee vote outcomes from public briefing PDFs. It extracts clinical and NLP features, runs a saved Random Forest model, explains each prediction with SHAP, and summarizes the historical financial backtest used to sanity-check the signal.

## Methodology

| Step | Method |
|---|---|
| Validation | Leave-One-Out Cross-Validation on 25 historical meetings |
| Model | Random Forest classifier with 100 estimators |
| Explainability | SHAP TreeExplainer waterfall plots |
| Financial test | Walk-forward event-driven backtest |
| Benchmark | XBI-adjusted stock reaction around ODAC events |

## Backtest Results

| Universe | Trades | Sharpe | Hit Rate | Total Return | Max Drawdown |
|---|---:|---:|---:|---:|---:|
| Small-cap only | 5 | 1.31 | 80.0% | 190.8% | -1.1% |
| All tradeable | 13 | 0.92 | 46.2% | 191.0% | -2.1% |

Important caveat: the small-cap result is based on only n=5 trades, so it is indicative rather than statistically significant. This is research tooling, not investment advice.

## Tech Stack

Streamlit, pandas, NumPy, scikit-learn, SHAP, pdfplumber, matplotlib, and joblib.


# ODAC Signal — FDA AdCom Vote Predictor

> Predicts FDA Oncologic Drugs Advisory Committee vote outcomes from briefing PDFs using NLP + Random Forest. Returns P(YES), BUY/HOLD/SELL signal, and SHAP explanations.

**[Live App]([https://your-app.onrender.com)](https://odac-signal.onrender.com/)** · **[Streamlit Demo](https://huggingface.co/spaces/ds-n-sd/odac-signal)**

## Deployment Files
## What it does
...
- `app.py` loads `saved_model/model.pkl` and `saved_model/feature_names.pkl`
- `train_and_save.py` creates the saved model artifacts from `data/processed/feature_matrix.csv`
- `saved_model/historical.csv` powers the historical results tab
- `requirements.txt` lists the Hugging Face Space runtime dependencies
