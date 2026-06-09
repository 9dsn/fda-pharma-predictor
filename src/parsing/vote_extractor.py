"""extracting vote data from the FDA odac Minutes PDF"""

import re
import pdfplumber
from pathlib import Path
import pandas as pd

VOTE_PATTERN = re.compile(r"(\d+)\.\s*VOTE:\s*((?:(?!VOTE:).)+?\?).{0,200}?Vote Results?:\s*Yes:\s*(\d+)\s*No:\s*(\d+)\s*Abstain:\s*(\d+)",
    re.DOTALL,)
MIN_EXPECTED_PANEL_SIZE = 8
MAX_EXPECTED_PANEL_SIZE = 22


def normalize_question_text(question_text):
    """Normalize extracted question text for stable comparisons."""
    return re.sub(r"\s+", " ", question_text).strip()


def vote_outcome(yes, no, abstain):
    """Return the simple vote winner."""
    if yes > no:
        return "yes"
    if no > yes:
        return "no"
    return "tie"


def vote_count_sane(total_votes):
    """Flag panel sizes that are in the expected ODAC range."""
    return MIN_EXPECTED_PANEL_SIZE <= total_votes <= MAX_EXPECTED_PANEL_SIZE

def extract_pdf_text(pdf_path):
    """open pdf and returns full text as string"""
    with pdfplumber.open(pdf_path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
        full_text = "\n".join(pages)
    return full_text


def extract_votes(text):
    """vote regex and returns a list of vote dicts"""
    matches = VOTE_PATTERN.findall(text)

    vote_list = []

    for m in matches:
        question_num, question_text, yes, no, abstain = m
        yes = int(yes)
        no = int(no)
        abstain = int(abstain)
        total_votes = yes + no + abstain
        vote_dict = {
            "question_number": int(question_num),
            "question_text": normalize_question_text(question_text),
            "yes": yes,
            "no": no,
            "abstain": abstain,
            "total_votes": total_votes,
            "outcome": vote_outcome(yes, no, abstain),
            "vote_count_sane": vote_count_sane(total_votes),
        }
        vote_list.append(vote_dict)
    return vote_list


def process_one_pdf(pdf_path):
    """extracting all votes from one pdf"""
    from src.features.dataset import parse_meeting_date

    text = extract_pdf_text(pdf_path)
    vote_list = extract_votes(text)
    
    source = Path(pdf_path).stem
    for vd in vote_list:
        vd["source"] = source
        vd["meeting_date"] = parse_meeting_date(source)

    return vote_list

def process_all_pdfs(input_dir="data/raw", output_csv="data/processed/votes.csv", deduplicate=True):
    """run on every pdf in the input directory, then write to output csv"""
    all_pdfs = sorted(Path(input_dir).glob("*.pdf"))
    all_votes = []
    success = 0
    
    for pdf in all_pdfs:
        try:
            pdf_vote = process_one_pdf(pdf)
            all_votes.extend(pdf_vote)
            success += 1
        except Exception as e:
            print(f"failed: {pdf}: {e}")
    
    df = pd.DataFrame(all_votes)

    if deduplicate and not df.empty:
        df["_normalized_question_text"] = df["question_text"].map(normalize_question_text).str.lower()
        dedupe_columns = [
            "meeting_date",
            "question_number",
            "_normalized_question_text",
            "yes",
            "no",
            "abstain",
        ]
        before_count = len(df)
        df = df.drop_duplicates(subset=dedupe_columns).drop(columns=["_normalized_question_text"])
        duplicate_count = before_count - len(df)
    else:
        duplicate_count = 0

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True) # creates data/processed/ if its not existent
    df.to_csv(output_csv, index=False)

    print(f"Extracted: {len(df)} votes from {success} PDFs, removed {duplicate_count} duplicates, wrote {output_csv}")
    return len(df)
