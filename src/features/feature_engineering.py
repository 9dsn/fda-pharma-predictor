"""this performs FE on the vote briefing dataset. extracts relevant features from the raw data, (e.g., word counts, sentiment scores) and metadata (e.g., vote outcome, date)
Input:  data/processed/vote_briefing_dataset.csv
Output: data/processed/feature_matrix.csv  (25 usable rows x 16 cols: 12 features + 4 metadata/target)
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


# vocab for TF-IDF 
POSITIVE_VOCAB = [
    "efficacious", "beneficial", "significant improvement",
    "clinically meaningful", "durable response", "well tolerated",
    "favorable risk-benefit", "survival benefit", "superior",
    "statistically significant", "meets primary endpoint"
]
NEGATIVE_VOCAB = [
    "concern", "uncertainty", "insufficient evidence",
    "failed to demonstrate", "adverse events", "safety signal",
    "not statistically significant", "does not support",
    "unresolved", "limited data", "exploratory"
]


def build_tfidf_scores(texts: list[str]):
    """"fit a TF IDF vectorizer on all the briefing texts
    then it returns (positive scores, negative scores) one float per document"""

    vectorizer = TfidfVectorizer(vocabulary=POSITIVE_VOCAB + NEGATIVE_VOCAB, sublinear_tf=True, ngram_range=(1,3))
    tfidf_matrix = vectorizer.fit_transform(texts)
    vocab = vectorizer.vocabulary_
    positive_indices = [vocab[word] for word in POSITIVE_VOCAB if word in vocab]
    negative_indices = [vocab[word] for word in NEGATIVE_VOCAB if word in vocab]

    #summing the matrix rows for pos terms
    if positive_indices:
        positive_scores = np.asarray(tfidf_matrix[:, positive_indices].sum(axis=1)).flatten()
    else:
        positive_scores = np.zeros(len(texts))

    # negative terms
    if negative_indices:
        negative_scores = np.asarray(tfidf_matrix[:, negative_indices].sum(axis=1)).flatten()
    else:
        negative_scores = np.zeros(len(texts))
    
    # return as a tuple
    return positive_scores, negative_scores


def sentiment_ratios(text):
    """"split text into sentences, counts pos/neg keyword hits"""

    sentences = re.split(r'[.!?]', text.lower())
    sentences = [s for s in sentences if len(s.strip()) >= 10] # filter out very short sentences
    
    if not sentences:
        return 0.0, 0.0
    
    n = len(sentences)
    
    pos_count = 0
    neg_count = 0

    for sentence in sentences:
        # using \b to prevent partial word matches
        if any(re.search(rf"\b{re.escape(word)}\b", sentence) for word in POSITIVE_VOCAB):
            pos_count += 1
        if any(re.search(rf'\b{re.escape(word)}\b', sentence) for word in NEGATIVE_VOCAB):
            neg_count += 1
        
    return (pos_count / n), (neg_count / n)


def concern_density(text):
    """"measures concerned tone of the document per 1,000 words"""
    pattern = r"\b(concern|risk|uncertainty|adverse|toxicity|safety)\b" # inclues boundaries, EX: not to confuse with 'briskly
    matches = re.findall(pattern, text.lower())
    word_count = len(re.findall(r"\b\w+\b", text))
    return (len(matches) / max(word_count, 1)) * 1000 # prevent div by zero


def binary_flags(text):
    """returns dict of all binary featues"""
    t = text.lower()

    # capture p values and check if any (<0.05)
    p_vals = re.findall(r'p\s*[=<]\s*([0-9.]+)', t)
    p_strong = any (
        float(v) < 0.05 
        for v in p_vals
        if _is_float(v) and float(v) > 0
    )

    # overall survival
    os_mentioned = bool(
    re.search(r'\b(overall survival|os)\b', t))

    survival_positive = int(bool(
        re.search(
            r'(improved overall survival|overall survival benefit|'
            r'statistically significant overall survival|'
            r'os benefit|os improvement)',
            t
        )
    ))

    # progression free survival
    pfs_mentioned = bool(
        re.search(r'\b(progression[-\s]?free survival|pfs)\b', t)
    )
    return {
        # clincial drug quality
        "survival_positive": survival_positive,
        "pfs_only": int(pfs_mentioned and not os_mentioned),
        # specific concerns and regulatory flags
        "safety_concern_flag": int(bool(re.search(r'(serious adverse|black box|sae|'
                                                  r'safety concern|toxicit|treatment-related death)', t))),
        "accelerated_approval_flag": int(bool(re.search(r'\b(accelerated approval|breakthrough)\b', t))),
        "p_value_strong": int(p_strong),
    }


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    

# MAIN BUILD FUNCTION
def build_feature_matrix(
        dataset_path = "data/processed/vote_briefing_dataset.csv",
        output_path = "data/processed/feature_matrix.csv"):
    """loads joined data, filters to usable docs, extracts fts, saves final matrix for modeling"""

    # load dataset
    df = pd.read_csv(dataset_path)

    # filter to rows with briefing text
    df_clean = df[df["has_briefing_text"] == True].copy().reset_index(drop=True)

    # pull out the raw txt into python list
    texts = df_clean["briefing_text"].fillna("").tolist()

    # generate the TF-IDF array
    pos_scores, neg_scores = build_tfidf_scores(texts)
    
    # loop thorugh every row to extract row level fts
    rows = []
    for r, row in df_clean.iterrows():
        text = row["briefing_text"] or ""

        # calling the row level func
        pos_r, neg_r = sentiment_ratios(text)
        flags = binary_flags(text)

        # combine all features into one dict
        row_dict = {
            # meta data cols
            "source": row["source"],
            "meeting_date": row["meeting_date"],
            "question_text": row["question_text"],
            "outcome": row["outcome"], # target variable

            #numeric features
            "tfidf_positive_score": pos_scores[r],
            "tfidf_negative_score": neg_scores[r],
            "tfidf_balance": pos_scores[r] - neg_scores[r],
            "sentiment_positive_ratio": pos_r,
            "sentiment_negative_ratio": neg_r,
            "concern_density": concern_density(text),
            "briefing_char_count": row["briefing_char_count"],

            # unpacking binary flags
            **flags
        }

        rows.append(row_dict)

    # comvert list into clean pandas dataframe
    feature_df = pd.DataFrame(rows)

    # write out to disk
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)

    print(f"Feature matrix saved to {output_path}")
    print(f"Final matrix shape: {feature_df.shape}")
    return feature_df
