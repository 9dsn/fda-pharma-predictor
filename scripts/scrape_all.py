"""running the full odac scraper for year ranges"""
import argparse
from src.scraping.odac_scraper import crawl_all_years

def main():
    parser = argparse.ArgumentParser(description="Scrape FDA odac minutes pdf in meetings")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+", #it accepst one or more vals
        default=[2020, 2021, 2022, 2023, 2024, 2025, 2026],
    )
    args = parser.parse_args()

    print(f"scraping for years: {args.years}")
    scraped_paths = crawl_all_years(args.years)

    print(f"scraping complete, total collected: {len(scraped_paths)}")

if __name__ == "__main__":
    main()