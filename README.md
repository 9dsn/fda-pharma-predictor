# FDA ODAC Vote Predictor

Predicting FDA Oncologic Drugs Advisory Committee vote outcomes from public briefing
documents, then checking whether the signal lines up with biotech stock moves.

This started as a messy public-data problem: ODAC meeting minutes and briefing docs are
mostly PDFs spread across FDA.gov and Wayback Machine. The project turns that into a
small modeling pipeline: scrape PDFs, extract vote records, join briefing text, engineer
clinical/NLP features, train classifiers, and run an event-driven backtest.

## Current Results

- 28 extracted ODAC vote rows from 2020-2026
- 25 rows with usable briefing text
- 16-column feature matrix with sentiment, concern density, survival/PFS, accelerated
  approval, p-value, and TF-IDF features
- LOOCV model evaluation with logistic regression and random forest
- SHAP feature-importance pass plus a `briefing_char_count` ablation check
- Backtest using out-of-fold random forest probabilities and ticker mappings

Backtest headline:

| Universe | Trades | Total Return | Sharpe | Hit Rate | Max Drawdown |
|---|---:|---:|---:|---:|---:|
| Small-cap only | 5 | 190.82% | 1.312 | 80.0% | -1.08% |
| All tradeable | 13 | 190.98% | 0.924 | 46.2% | -2.12% |

Important: this is a tiny dataset, so the backtest is more of a proof-of-signal than a
tradable strategy. The results are interesting, not conclusive.

## Pipeline

```text
FDA.gov / Wayback Machine
        -> minutes + briefing PDFs
        -> structured vote extraction
        -> vote + briefing-text dataset
        -> clinical/NLP feature matrix
        -> LOOCV classifiers + SHAP
        -> event-driven stock backtest
```

## Usage

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Scrape ODAC minutes:

```bash
python scripts/scrape_all.py
```

Extract vote records:

```bash
python scripts/extract_votes.py
```

Build the joined vote + briefing dataset:

```bash
python scripts/build_dataset.py
```

Build model features:

```bash
python scripts/build_features.py
```

Train/evaluate models and save LOOCV probabilities:

```bash
python -m src.models.train_evaluate
```

Run the backtest:

```bash
python -m src.evaluation.backtest
```

## Key Outputs

```text
data/processed/votes.csv
data/processed/vote_briefing_dataset.csv
data/processed/feature_matrix.csv
data/processed/loocv_probabilities.csv
data/processed/ticker_map.csv
data/processed/backtest_trades_smallcap.csv
data/processed/backtest_trades_all.csv
data/processed/backtest_summary.json
data/processed/plots/equity_smallcap.png
data/processed/plots/equity_all.png
```

## Repo Structure

```text
src/scraping/      FDA and Wayback scraping
src/parsing/       PDF text and vote extraction
src/features/      dataset join + feature engineering
src/models/        LOOCV training, SHAP, ablation
src/evaluation/    stock-event backtest
scripts/           CLI entry points
docs/              methodology notes
data/processed/    generated datasets and backtest outputs
```

## Limitations

- ODAC only, so sample size is small.
- Some old FDA/Wayback briefing PDFs are missing, blocked, or corrupted.
- Ticker mapping is manual and should be audited before any serious use.
- The backtest uses a simple event window and XBI adjustment; it does not model slippage,
  borrow costs, liquidity, or position sizing.
- This is research code, not financial advice or a production trading system.

## License

MIT
