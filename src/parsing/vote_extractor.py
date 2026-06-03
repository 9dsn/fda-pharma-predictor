"""extracting vote data from the FDA odac Minutes PDF"""

import re
import pdfplumber
from pathlib import Path
import pandas as pd

VOTE_PATTERN = re.compile(r"(\d+)\.\s*VOTE:\s*((?:(?!VOTE:).)+?\?).{0,200}?Vote Results?:\s*Yes:\s*(\d+)\s*No:\s*(\d+)\s*Abstain:\s*(\d+)",
    re.DOTALL,)

def extract_pdf_text(pdf_path):
    """open pdf and returns full text as string"""
    with pdfplumber.open(pdf_path) as pdf:
        pages = [page.extract_text() for page in pdf.pages]
        full_text = "\n".join(pages)
    return full_text


def extract_votes(text):
    """vote regex and returns a list of vote dicts"""
    matches = VOTE_PATTERN.findall(text)

    vote_list = []

    for m in matches:
        question_num, question_text, yes, no, abstain = m
        vote_dict = {
            "question_number": int(question_num),
            "question_text": question_text.strip().replace("\n", " "),
            "yes": int(yes),
            "no": int(no),
            "abstain": int(abstain)
        }
        vote_list.append(vote_dict)
    return vote_list


def process_one_pdf(pdf_path):
    """extracting all votes from one pdf"""
    text = extract_pdf_text(pdf_path)
    vote_list = extract_votes(text)
    
    source = Path(pdf_path).stem
    for vd in vote_list:
        vd["source"] = source

    return vote_list

def process_all_pdfs(input_dir="data/raw", output_csv="data/processed/votes.csv"):
    """run on every pdf in the input directory, then write to output csv"""
    all_pdfs = Path(input_dir).glob("*.pdf")
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
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True) # creates data/processed/ if its not existent
    df.to_csv(output_csv, index=False)

    print(f"Extracted: {len(all_votes)} votes from {success} PDFs, wrote {output_csv}")
    return len(all_votes)

