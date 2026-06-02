"""Scrape FDA ODAC advisory committee meeting pages."""

import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time

WAYBACK_YEAR_URLS = {
    2020: "https://web.archive.org/web/20201205131521/https://www.fda.gov/advisory-committees/oncologic-drugs-advisory-committee/2020-meeting-materials-oncologic-drugs-advisory-committee",
    2021: "https://web.archive.org/web/20230723092033/https://www.fda.gov/advisory-committees/oncologic-drugs-advisory-committee/2021-meeting-materials-oncologic-drugs-advisory-committee",
    2022: "https://web.archive.org/web/20230326222702/https://www.fda.gov/advisory-committees/oncologic-drugs-advisory-committee/2022-meeting-materials-oncologic-drugs-advisory-committee",
}

def fetch_meeting_page(url):
    """Fetch the HTML of an FDA meeting page."""

    #had to add this due to bot detection, exception error
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status() #throw an exception if there's an error
    return response.text


def find_minutes_pdf_url(html, base_url="https://www.fda.gov"):
    """finds the URL of the minutes pdf in a meeting page"""
    soup = BeautifulSoup(html, "html.parser")

    # this finds every link where its visiblly says Minutes
    minutes_links = []
    for link in soup.find_all("a"):
        if "Minutes" in link.get_text():
            minutes_links.append(link)

    # should be only 1
    if len(minutes_links) == 0:
        raise ValueError("no Minutes link found")
    if len(minutes_links) > 1:
        raise ValueError(f"expected 1 Minutes link, but found {len(minutes_links)}")

    # this gets the URL from the link's href attribute
    relative_url = minutes_links[0].get("href")

    # the URL is relative (/media/...) so making it valid
    if relative_url.startswith("/"):
        return base_url + relative_url
    return relative_url


def download_pdf(url, dest_path):
    """download a PDF from a URL and save it to disk"""

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers, timeout=60) #increased timeout
    response.raise_for_status()
    
    with open(dest_path, "wb") as f: #wb, write bytes
        f.write(response.content)


def scrape_one_meeting(meeting_url, output_dir="data/raw", base_url="https://www.fda.gov"):
    """Scraping 1 ODAC meeting: fetch page, find Minutes PDF, download it
    then it returns the path where the PDF was saved"""
    
    html = fetch_meeting_page(meeting_url)

    pdf_url = find_minutes_pdf_url(html, base_url=base_url)

    # builds filename
    slug = meeting_url.rstrip("/").split("/")[-1]
    dest_path = Path(output_dir) / f"{slug}.pdf"

    if dest_path.exists():
        print(f"skipping {slug}, already had")
        return dest_path

    download_pdf(pdf_url, dest_path)

    return dest_path


def find_meeting_urls(year_html, base_url="https://www.fda.gov"):
    """finding all meeting page URLs on an ODAC year listing page"""
    soup = BeautifulSoup(year_html, "html.parser")

    meeting_urls = []
    for link in soup.find_all("a"):
        href = link.get("href")

        if not href: # skip links with no href,
            continue 
        if "/advisory-committee-calendar/" not in href: # if href is not a meeting page
            continue

        # converting relative URLs to real/valid linkss
        if href.startswith("/"):
            full_url = base_url + href
        else:
            full_url = href

        meeting_urls.append(full_url)

    return meeting_urls


def crawl_year(year_url, base_url="https://www.fda.gov"):
    """craw every meeting on a year listing page then downloading Minutes PDF"""

    html_data = fetch_meeting_page(year_url)

    meeting_urls = find_meeting_urls(html_data, base_url=base_url)

    dest_paths = []

    for meeting in meeting_urls:
        try:
            print(f"currently scraping: {meeting}")

            temp_path = scrape_one_meeting(meeting, base_url=base_url)
            dest_paths.append(temp_path)

            time.sleep(1) # to give some time between loop iteration

        except Exception as e:
            print(f"failed: {meeting}: {e}")
    return dest_paths


def crawl_all_years(years):
    """crawl every meeting across the given years and download all the Minutes PDF"""
    all_paths = []

    for year in years:
        print(f"crawl year {year}")

        if year in WAYBACK_YEAR_URLS:
            year_url = WAYBACK_YEAR_URLS[year]
            base_url = "https://web.archive.org"
        else:
            year_url = f"https://www.fda.gov/advisory-committees/oncologic-drugs-advisory-committee/{year}-meeting-materials-oncologic-drugs-advisory-committee"
            base_url = "https://www.fda.gov"

        try:
            current_year_paths = crawl_year(year_url, base_url=base_url)
            all_paths.extend(current_year_paths)
            time.sleep(1)

        except Exception as e:
            print(f"failed: {year}: {e}")
    
    return all_paths