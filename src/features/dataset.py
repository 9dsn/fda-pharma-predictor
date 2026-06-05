"""building the joint dataset: votes.csv + briefing text"""

import re
from pathlib import Path

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