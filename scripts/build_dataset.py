"""Build the joined vote + briefing-text dataset."""

import argparse

from src.features.dataset import build_vote_briefing_dataset


def main():
    parser = argparse.ArgumentParser(description="Build vote records joined with briefing text")
    parser.add_argument(
        "--votes-csv",
        default="data/processed/votes.csv",
        help="Structured vote CSV produced by scripts/extract_votes.py",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/vote_briefing_dataset.csv",
        help="Path to write the joined dataset",
    )
    parser.add_argument(
        "--briefings-dir",
        default="data/raw/briefings",
        help="Directory containing per-meeting briefing PDF folders",
    )
    args = parser.parse_args()

    build_vote_briefing_dataset(
        votes_csv=args.votes_csv,
        output_csv=args.output_csv,
        briefings_dir=args.briefings_dir,
    )


if __name__ == "__main__":
    main()
