# Methodology

## Project
- Predict FDA Advisory Committee voting outcomes from briefing documents
- Validate predictions as leading indicator for approval and stock reactions
- Training range: 2020–present, ODAC (Oncologic Drugs Advisory Committee)

## Data sources
- Live FDA: 2023–2026
- Wayback Machine: 2020–2022 (older meetings archived off live FDA)
- Single entry point: https://www.fda.gov/advisory-committees/human-drug-advisory-committees/oncologic-drugs-advisory-committee

## Vote format (confirmed across 2020/2021/2022)
- Pattern: `N. VOTE: <question>?  Vote Result: Yes: N No: N Abstain: N`
- Found inside the "Minutes" PDF for each meeting

## PDF URL pattern
- FDA-hosted PDFs: `https://www.fda.gov/media/<NUMBER>/download`
- On Wayback: prefix with `https://web.archive.org/web/<TIMESTAMP>/`

## Data model
- meeting_date, drug, question_number → (yes, no, abstain)

## Phase 2 bugs found and fixed
- FDA blocks default User-Agent (`python-requests/...`) so i had to add browser UA header
- `find_meeting_urls` initially assumed all relative URLs were fda.gov so it broke on Wayback. Added `base_url` parameter with FDA default to fix
- Some meetings have no Minutes PDF posted yet so i handled with try/except in crawler loop

## Decisions
- Hardcoded Wayback year URLs rather than scraping dynamically since they're static archive snapshots, can change later
- Scope: ODAC only for now since it keeps data clean but it can be expandable later
- The data not committed to git (.gitignored data/raw/), so scraper is a reproducible
