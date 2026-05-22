"""Scrape FDA ODAC advisory committee meeting pages."""

import requests

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