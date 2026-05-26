## 2026/05/18 — Phase 1 Data Inventory

### FDA's archive policy
- FDA keeps the last 3 ish years of ODAC meetings on live FDA.gov
  - Currently the years 2023, 2024, 2025, 2026 are live
- Older meetings (before 2022) are accessed via Wayback Machine (they are archived)
- single entry point for our scraper is:
  https://www.fda.gov/advisory-committees/human-drug-advisory-committees/oncologic-drugs-advisory-committee

### Available data range
- Live FDA: 2023–2026 (~4 years)
- Wayback (new format): 2016–2022 (~7 years)
- Wayback (old format): 2009–2015 (~7 years, harder to parse)
- Pre-2009: also accessible, likely not worth the effort
- **Total realistic training range: 2020–present (around 6 years)**

  ### Extraction rule
Regex pattern (conceptually):
- Find text/lines starting with `N. VOTE:` where N is a number
- Capture the question text up to the `?`
- On the next line, find `Vote Result: Yes: <int> No: <int> Abstain: <int>`
- Critical: distinguish `VOTE:` from `DISCUSSION:` — minutes contain both, only `VOTE:` lines are labels

### PDF URL pattern
- All FDA-hosted PDFs use: `https://www.fda.gov/media/<NUMBER>/download`
- On Wayback, prefix with: `https://web.archive.org/web/<TIMESTAMP>/`
- The Minutes PDF link is found on the meeting page; we identify it by document title containing "Minutes"

### Recon complete
Ready to begin scraper implementation.
- Next: write src/scraping/odac_scraper.py

## 2026-05-25 — Phase 2 mostly complete
- Scraper works end-to-end on live FDA (2023-2026)
- 13 PDFs in data/raw/, one expected failure (the one in April 2026 due to Minutes not being posted)
- Remaining: Wayback (2020–2022) + scripts/scrape_all.py entry point
