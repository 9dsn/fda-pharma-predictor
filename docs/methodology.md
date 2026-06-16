# Methodology

## Goal

Predict FDA ODAC vote outcomes from public briefing documents, then check whether the
model signal lines up with biotech stock moves around the meeting.

This is research code, not a production trading system.

## Scope

- Committee: ODAC only
- Years: 2020-2026
- Main sources: FDA.gov and Wayback Machine
- Final structured votes: 28 rows
- Usable briefing-text rows: 25
- Feature matrix: 16 columns

## Data Sources

- Live FDA pages for newer meetings
- Wayback Machine for older meetings that moved off the live FDA site
- Minutes PDFs for vote questions and vote counts
- Briefing PDFs for model features
- Yahoo Finance through `yfinance` for event-window stock prices

## Pipeline

```text
FDA / Wayback pages
-> meeting PDFs
-> structured vote extraction
-> vote + briefing text dataset
-> NLP / clinical features
-> LOOCV model predictions
-> event-driven stock backtest
```

## Vote Extraction

The vote extractor looks for vote sections in the meeting minutes and pulls:

- question number
- question text
- yes / no / abstain counts
- total votes
- outcome
- source meeting slug
- meeting date

The vote format is usually close to:

```text
N. VOTE: <question>? Vote Result: Yes: N No: N Abstain: N
```

The parser also keeps sanity checks for vote counts so bad rows are easier to catch.

## Briefing Join

Each vote row is joined to the briefing text for the same meeting source. If multiple
briefing PDFs exist for one meeting, their extracted text is concatenated.

Some old FDA / Wayback files are missing, blocked, or corrupted, so not every vote row has
usable briefing text. I treat that as a source-data limitation instead of forcing fake data.

## Features

The feature matrix keeps metadata plus the target outcome, then adds simple features from
the briefing text:

- positive / negative TF-IDF clinical language scores
- TF-IDF balance
- positive / negative sentence ratios
- concern density
- briefing character count
- survival-positive flag
- PFS-only flag
- safety concern flag
- accelerated approval flag
- strong p-value flag

I kept the features interpretable on purpose because the dataset is too small for a deep
learning approach.

## Modeling

I used leave-one-out cross-validation because the dataset is tiny. A normal train/test
split would be too unstable.

Models tested:

- logistic regression
- random forest

The random forest LOOCV probabilities are saved and used in the backtest. That matters
because each probability is out-of-fold, so the backtest is not using predictions from a
model that trained on the same row.

## Explainability

I used SHAP to inspect which features the models lean on.

I also ran an ablation study dropping `briefing_char_count`, because document length can
be a sneaky shortcut feature. The goal was to check whether the model still had signal
without relying mainly on briefing length.

## Backtest

The backtest joins LOOCV probabilities to a manual ticker map.

Rules:

- long if `prob_yes >= 0.65`
- short if `prob_yes <= 0.35`
- skip middle-confidence predictions
- compare stock move against XBI over the event window
- run both small-cap-only and all-tradeable universes

Current result:

| Universe | Trades | Total Return | Sharpe | Hit Rate | Max Drawdown |
|---|---:|---:|---:|---:|---:|
| Small-cap only | 5 | 190.82% | 1.312 | 80.0% | -1.08% |
| All tradeable | 13 | 190.98% | 0.924 | 46.2% | -2.12% |

These results are interesting, but the sample is way too small to call this a real
trading strategy.

## Known Limitations

- ODAC only, so the sample size is small.
- Some PDFs are missing, blocked, or corrupted.
- Ticker mapping is manual and should be audited.
- The backtest does not include slippage, borrow costs, liquidity, or position sizing.
- The model could still be learning document/source artifacts.
- More committees and more years are needed before making strong claims.

## Next If I Kept Going

- Expand to more FDA advisory committees.
- Add final FDA approval outcome labels.
- Automate sponsor-company and ticker mapping.
- Add parser and feature tests.
- Add a PDF coverage report.
- Make the backtest more realistic.
