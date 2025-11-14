# task1_scraper.py
"""
Amazon.in laptop scraper
Saves a CSV with columns: timestamp, image_url, title, rating, price, result_type (Ad/Organic), product_url
Usage: python task1_scraper.py "laptop"  (or any search term)
"""
import requests
from bs4 import BeautifulSoup
import csv
import time
import sys
from urllib.parse import urljoin
from datetime import datetime
import random

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/116.0 Safari/537.36",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8"
}

BASE = "https://www.amazon.in"

def fetch_search_page(query, page=1):
    q = query.replace(" ", "+")
    url = f"{BASE}/s?k={q}&page={page}"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text

def parse_results(html):
    soup = BeautifulSoup(html, "html.parser")
    # Each result is often in div with data-component-type="s-search-result"
    results = []
    for div in soup.select('div[data-component-type="s-search-result"]'):
        try:
            # Check for "Sponsored" badge
            sponsored = False
            # Look for common sponsored badges
            if div.select_one("span.s-label-popover-default") or div.select_one("span[data-component-type='sp-sponsored-result']"):
                sponsored = True
            result_type = "Ad" if sponsored else "Organic"

            # Title and product URL
            a = div.select_one("h2 a.a-link-normal")
            title = a.get_text(strip=True) if a else None
            product_url = urljoin(BASE, a['href']) if a and a.has_attr('href') else None

            # Image
            img = div.select_one("img.s-image")
            image_url = img['src'] if img and img.has_attr('src') else None

            # Price - Amazon uses span.a-price-whole and a-price-fraction
            price_whole = div.select_one("span.a-price-whole")
            price_frac = div.select_one("span.a-price-fraction")
            if price_whole:
                price = price_whole.get_text(strip=True)
                if price_frac:
                    price += price_frac.get_text(strip=True)
            else:
                # fallback
                price = None

            # Rating
            rating = None
            rtag = div.select_one("span.aok-inline-block span.a-icon-alt")
            if rtag:
                rating = rtag.get_text(strip=True)

            results.append({
                "image_url": image_url,
                "title": title,
                "rating": rating,
                "price": price,
                "result_type": result_type,
                "product_url": product_url
            })
        except Exception:
            continue
    return results

def save_csv(rows, filename):
    keys = ["timestamp", "image_url", "title", "rating", "price", "result_type", "product_url"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def main():
    query = "laptop"
    if len(sys.argv) > 1:
        query = sys.argv[1]
    all_rows = []
    pages_to_fetch = 2  # change as needed; be polite
    for p in range(1, pages_to_fetch + 1):
        try:
            html = fetch_search_page(query, page=p)
            results = parse_results(html)
            ts = datetime.utcnow().isoformat()
            for r in results:
                r["timestamp"] = ts
                all_rows.append(r)
            # polite sleep
            time.sleep(random.uniform(1.5, 3.0))
        except Exception as e:
            print("Error fetching/parsing page:", e)
            break

    if not all_rows:
        print("No results found. Page structure may have changed.")
        return

    outname = f"laptops_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_csv(all_rows, outname)
    print(f"Saved {len(all_rows)} results to {outname}")

if __name__ == "__main__":
    main()
