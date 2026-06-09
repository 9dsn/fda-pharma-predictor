"""building the joint dataset: votes.csv + briefing text"""

import re
from pathlib import Path
import pandas as pd
from src.parsing.vote_extractor import extract_pdf_text

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
}

def parse_meeting_date(slug):
    """parse meeting date from slug"""
    slug_lower = slug.lower()

    month_pattern = "|".join(MONTHS.keys())
    pattern = re.compile(rf"({month_pattern})-(\d+)(?:-\d+)?-(\d{{4}})")

    match = pattern.search(slug_lower)
    if not match:
        return None  # date could not be found

    name = match.group(1)  # already lowercase
    day = int(match.group(2))
    year = int(match.group(3))
    num = MONTHS[name]

    return f"{year:04d}-{num:02d}-{day:02d}"


def extract_briefing_text(meeting_slug, briefings_dir="data/raw/briefings"):
    """extracting and concatenating text from all the briefing PDFs for one meeting"""
    folder = Path(briefings_dir) / meeting_slug

    if not folder.exists():
        return None

    briefing_texts = []
    for pdf_path in sorted(folder.glob("*.pdf")):
        try:
            briefing_texts.append(extract_pdf_text(pdf_path))
        except Exception as e:
            print(f"failed briefing text extraction: {pdf_path}: {e}")

    if not briefing_texts:
        return None

    return "\n\n--- NEXT BRIEFING ---\n\n".join(briefing_texts)


def build_vote_briefing_dataset(
    votes_csv="data/processed/votes.csv",
    output_csv="data/processed/vote_briefing_dataset.csv",
    briefings_dir="data/raw/briefings",
):
    """joining the structured vote records with extracted briefing text"""
    df = pd.read_csv(votes_csv)

    if "meeting_date" not in df.columns:
        df["meeting_date"] = df["source"].map(parse_meeting_date)

    briefing_cache = {}

    def get_briefing_text(source):
        if source not in briefing_cache:
            briefing_cache[source] = extract_briefing_text(source, briefings_dir=briefings_dir)
        return briefing_cache[source]

    df["briefing_text"] = df["source"].map(get_briefing_text)
    df["has_briefing_text"] = df["briefing_text"].notna()
    df["briefing_char_count"] = df["briefing_text"].fillna("").str.len()

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    missing_count = int((~df["has_briefing_text"]).sum())
    print(f"Built dataset: {len(df)} rows, {missing_count} rows missing briefing text, wrote {output_csv}")
    return df
