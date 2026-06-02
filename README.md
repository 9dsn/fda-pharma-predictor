# FDA ODAC Advisory Committee Vote Predictor

A pipeline to predict FDA Oncologic Drugs Advisory Committee (ODAC) vote outcomes from
publicly available briefing documents, and validate those predictions as a leading
indicator for drug approval decisions and biotech stock moves.

Right now the scraper is fully built and pulling historical Minutes PDFs from 2020 to
present. Upcoming phases will extract vote records from those PDFs, engineer NLP features
from the briefing documents, and eventually train a classifier to predict outcomes before
a committee meets.

---

## Status

| Phase | Description | Status |
|---|---|---|
| 1 | Repo setup, data planning | Done |
| 2 | ODAC minutes scraper (2020–present) | **Done** |
| 3 | Vote extraction + structured dataset | In progress |
| 4 | NLP features from briefing docs | Upcoming |
| 5 | Model training + evaluation | Upcoming |
| 6 | Stock/approval correlation analysis | Upcoming |

---

## Motivation

FDA Advisory Committee meetings are one of the few moments where a drug's fate is debated
publicly — expert panelists vote yes/no on approval questions and those votes are a strong
signal for what the FDA ultimately decides. For biotech stocks, an AdCom vote can move a
company's price by 30–60% in a single day.

Most of this information is buried in PDFs on FDA.gov. The goal here is to surface it
systematically: scrape the historical record, extract structured vote data, and eventually
build a model that can predict a committee's likely outcome from the briefing documents
released ahead of the meeting. If that signal is predictive, it's a useful leading
indicator for both approval timelines and market reactions.

---

## Pipeline

```text
FDA.gov / Wayback Machine
        │
        ▼
  Phase 2: Scraper
  (Minutes PDFs → data/raw/)
        │
        ▼
  Phase 3: Vote Extractor          ← you are here
  (PDFs → structured vote records)
        │
        ▼
  Phase 4: Feature Engineering
  (briefing docs → NLP features)
        │
        ▼
  Phase 5: Model
  (features → vote prediction)
        │
        ▼
  Phase 6: Validation
  (predictions vs. approvals/stock moves)
```

---

## Setup

```bash
git clone https://github.com/9dsn/fda-pharma-predictor
cd fda-pharma-predictor

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -e .

cp .env.example .env           # fill in any keys if needed
```

Requires Python 3.10+.

---

## Usage

Run the scraper for all years (2020–2026 by default):

```bash
python scripts/scrape_all.py
```

Scrape specific years only:

```bash
python scripts/scrape_all.py --years 2022 2023 2024
```

PDFs are saved to `data/raw/` (gitignored). Already-downloaded files are skipped
automatically, so re-running is safe.

The scraper uses live FDA.gov for 2023–present and the Wayback Machine for 2020–2022,
where older meeting pages have been archived off the live site.

---

## Repo Structure

```text
fda-pharma-predictor/
├── src/
│   ├── scraping/       # Phase 2: ODAC minutes scraper
│   ├── parsing/        # Phase 3: vote extraction (upcoming)
│   ├── features/       # Phase 4: NLP feature engineering (upcoming)
│   ├── models/         # Phase 5: classifier (upcoming)
│   └── evaluation/     # Phase 6: validation (upcoming)
├── scripts/
│   └── scrape_all.py   # CLI entry point for the scraper
├── data/
│   ├── raw/            # downloaded PDFs (gitignored)
│   ├── interim/        # extracted vote records (gitignored)
│   └── processed/      # model-ready features (gitignored)
├── docs/
│   └── methodology.md  # decisions, data sources, known issues
├── notebooks/          # exploration notebooks
├── tests/
└── pyproject.toml
```

---

## Methodology

Design decisions, data sources, known bugs fixed, and TODOs are documented in
[docs/methodology.md](docs/methodology.md).

---

## Roadmap

- [ ] **Phase 3** — parse vote records from Minutes PDFs using pdfplumber; build a
  structured dataset of `(meeting_date, drug, question, yes, no, abstain)`
- [ ] **Phase 4** — scrape briefing documents; extract NLP features (sentiment, statistical
  claim density, label language)
- [ ] **Phase 5** — train a classifier; time-based train/test split to avoid leakage;
  calibration analysis
- [ ] **Phase 6** — correlate predictions against actual FDA approval decisions and stock
  price moves around AdCom dates
- [ ] Expand beyond ODAC to PCNS, BRUDAC if dataset is too small

---

## License

MIT
