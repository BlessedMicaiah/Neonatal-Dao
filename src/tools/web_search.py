import requests
from bs4 import BeautifulSoup
from typing import List


USER_AGENT = "Mozilla/5.0 (compatible; NeonatalDaoBot/1.0; +https://example.com/bot)"
HEADERS = {"User-Agent": USER_AGENT}


def google_search(query: str, num_results: int = 5) -> List[str]:
    """Very simple Google search scraper (for demonstration only)."""
    url = f"https://www.google.com/search?q={query}&num={num_results}"
    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    links = []
    for g in soup.select("div.g"):
        link = g.find("a", href=True)
        if link and link["href"].startswith("http"):
            links.append(link["href"])
        if len(links) >= num_results:
            break
    return links


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Simple Google Web Search")
    parser.add_argument("query", type=str)
    args = parser.parse_args()
    results = google_search(args.query)
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
