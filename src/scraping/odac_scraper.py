"""Scrape FDA ODAC advisory committee meeting pages."""

import requests
from bs4 import BeautifulSoup

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

def find_minutes_pdf_url(html):
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
        return "https://www.fda.gov" + relative_url
    return relative_url