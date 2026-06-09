"""Run vote extraction from Minutes PDFs into a structured CSV."""

import argparse

from src.parsing.vote_extractor import process_all_pdfs


def main():
    parser = argparse.ArgumentParser(description="Extract ODAC vote records from Minutes PDFs")
    parser.add_argument(
        "--input-dir",
        default="data/raw",
        help="Directory containing Minutes PDF files",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/votes.csv",
        help="Path to write extracted vote records",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep duplicate vote rows instead of dropping them",
    )
    args = parser.parse_args()

    process_all_pdfs(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        deduplicate=not args.no_dedupe,
    )


if __name__ == "__main__":
    main()
