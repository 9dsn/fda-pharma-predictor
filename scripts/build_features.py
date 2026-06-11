"""CLI for building the final feature matrix for modeling"""

import argparse
from src.features.feature_engineering import build_feature_matrix

def main():
    parser = argparse.ArgumentParser(description="build feature matrix from briefing text")
    parser.add_argument("--dataset", default="data/processed/vote_briefing_dataset.csv")
    parser.add_argument("--output", default="data/processed/feature_matrix.csv")
    args = parser.parse_args()
    df = build_feature_matrix(args.dataset, args.output)
    print(df[["outcome", "tfidf_positive_score", "safety_concern_flag",
              "survival_benefit_mentioned"]].to_string())

if __name__ == "__main__":
    main()