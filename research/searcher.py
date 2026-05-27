"""
research/searcher.py
NEXUS Web Searcher — finds pages on the internet, no API key required.

Uses DuckDuckGo's HTML search (no account needed) with a requests+BeautifulSoup
fallback. Returns structured SearchResult objects for the summarizer.

Pipeline:
    research/searcher.py    ← YOU ARE HERE
           ↓
    research/fetcher.py
           ↓
    research/summarizer.py
           ↓
    research/memory.py

Usage:
    from research.searcher import WebSearcher
    s = WebSearcher()
    results = s.search("buffer overflow exploit mitigation")
    for r in results:
        print(r.title, r.url)
"""

from __future__ import annotations

import logging
import time
import re
import random
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("nexus.research.searcher")

# ── lazy imports ──────────────────────────────────────────────
try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

DEFAULT_MAX_RESULTS = 5
REQUEST_TIMEOUT     = 10      # seconds
REQUEST_DELAY       = 1.0     # seconds between requests (polite crawling)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

DDGO_URL    = "https://html.duckduckgo.com/html/"
GOOGLE_URL  = "https://www.google.com/search"
BING_URL    = "https://www.bing.com/search"
WIKI_API    = "https://en.wikipedia.org/w/api.php"


# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """One search result entry."""
    title:   str
    url:     str
    snippet: str = ""
    rank:    int = 0

    def __str__(self) -> str:
        return f"[{self.rank}] {self.title}\n    {self.url}\n    {self.snippet[:120]}..."

    def is_valid(self) -> bool:
        return bool(self.url and self.url.startswith("http"))


@dataclass
class SearchResponse:
    """Full response from a search query."""
    query:   str
    results: list[SearchResult] = field(default_factory=list)
    source:  str = ""
    elapsed: float = 0.0
    error:   str = ""

    @property
    def success(self) -> bool:
        return len(self.results) > 0 and not self.error

    @property
    def urls(self) -> list[str]:
        return [r.url for r in self.results]

    def __str__(self) -> str:
        if not self.success:
            return f"SearchResponse(FAILED: {self.error})"
        return (
            f"SearchResponse(query={self.query!r}, "
            f"results={len(self.results)}, source={self.source!r})"
        )


# ─────────────────────────────────────────────────────────────
#  WEB SEARCHER
# ─────────────────────────────────────────────────────────────

class WebSearcher:
    """
    Searches the web using DuckDuckGo HTML (no API key, no rate limits).
    Falls back to a simple Google scrape if DDG fails.

    Usage:
        searcher = WebSearcher()
        response = searcher.search("python buffer overflow ctf")
        for r in response.results:
            print(r.title, r.url)
    """

    def __init__(self, timeout: int = REQUEST_TIMEOUT, delay: float = REQUEST_DELAY):
        if not _REQUESTS_AVAILABLE:
            raise ImportError("requests not installed. Run: pip install requests")
        if not _BS4_AVAILABLE:
            raise ImportError("beautifulsoup4 not installed. Run: pip install beautifulsoup4")

        self.timeout = timeout
        self.delay   = delay
        self._session = requests.Session()
        self._session.headers.update(HEADERS)
        log.info("WebSearcher ready.")

    # ── public API ────────────────────────────────────────────

    def search(
        self,
        query: str,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> SearchResponse:
        """
        Search the web and return ranked results.

        Parameters
        ----------
        query       : search query string
        max_results : maximum number of results to return

        Returns
        -------
        SearchResponse
        """
        if not query or not query.strip():
            return SearchResponse(query="", error="Empty query.")

        query = query.strip()
        log.info("Searching: %r (max=%d)", query, max_results)

        t0 = time.time()

        # Try DuckDuckGo first
        response = self._search_ddg(query, max_results)

        # Fall back to Google
        if not response.success:
            log.warning("DDG returned no results — trying Google.")
            response = self._search_google(query, max_results)

        # Fall back to Bing
        if not response.success:
            log.warning("Google returned no results — trying Bing.")
            response = self._search_bing(query, max_results)

        # Last resort: Wikipedia
        if not response.success:
            log.warning("Bing returned no results — trying Wikipedia.")
            response = self._search_wikipedia(query, max_results)

        response.elapsed = time.time() - t0
        log.info(
            "Search complete: %d results in %.2fs via %s",
            len(response.results), response.elapsed, response.source,
        )
        return response

    def quick_search(self, query: str) -> list[str]:
        """Return just a list of URLs for a query."""
        return self.search(query).urls

    # ── DuckDuckGo backend ────────────────────────────────────

    def _search_ddg(self, query: str, max_results: int) -> SearchResponse:
        """Search via DuckDuckGo's HTML interface."""
        try:
            resp = self._session.post(
                DDGO_URL,
                data={"q": query, "b": "", "kl": ""},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except Exception as e:
            log.warning("DDG request failed: %s", e)
            return SearchResponse(query=query, error=str(e))

        soup    = BeautifulSoup(resp.text, "html.parser")
        results = []

        for i, result in enumerate(soup.select(".result__body")[:max_results]):
            title_tag   = result.select_one(".result__title a")
            snippet_tag = result.select_one(".result__snippet")

            if not title_tag:
                continue

            title   = title_tag.get_text(strip=True)
            url     = title_tag.get("href", "")
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

            # DDG sometimes uses redirect URLs — extract real URL
            url = self._clean_ddg_url(url)

            if not url.startswith("http"):
                continue

            results.append(SearchResult(
                title   = title,
                url     = url,
                snippet = snippet,
                rank    = i + 1,
            ))

        if not results:
            return SearchResponse(query=query, error="No results from DDG.")

        return SearchResponse(query=query, results=results, source="duckduckgo")

    def _clean_ddg_url(self, url: str) -> str:
        """Extract the real URL from DDG's redirect format."""
        if "uddg=" in url:
            try:
                from urllib.parse import unquote, parse_qs, urlparse
                parsed = urlparse(url)
                params = parse_qs(parsed.query)
                if "uddg" in params:
                    return unquote(params["uddg"][0])
            except Exception:
                pass
        return url

    # ── Google fallback backend ───────────────────────────────

    def _search_google(self, query: str, max_results: int) -> SearchResponse:
        """Scrape Google search results as fallback."""
        try:
            params = {"q": query, "num": max_results + 2, "hl": "en"}
            resp   = self._session.get(
                GOOGLE_URL, params=params, timeout=self.timeout
            )
            resp.raise_for_status()
        except Exception as e:
            log.warning("Google request failed: %s", e)
            return SearchResponse(query=query, error=str(e))

        soup    = BeautifulSoup(resp.text, "html.parser")
        results = []
        rank    = 1

        for div in soup.select("div.g")[:max_results]:
            a_tag       = div.select_one("a[href]")
            title_tag   = div.select_one("h3")
            snippet_tag = div.select_one(".VwiC3b, .IsZvec, span[data-ved]")

            if not a_tag or not title_tag:
                continue

            url     = a_tag.get("href", "")
            title   = title_tag.get_text(strip=True)
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

            if not url.startswith("http"):
                continue

            results.append(SearchResult(
                title=title, url=url, snippet=snippet, rank=rank
            ))
            rank += 1

        if not results:
            return SearchResponse(query=query, error="No results from Google.")

        return SearchResponse(query=query, results=results, source="google")


    # ── Bing fallback backend ─────────────────────────────────

    def _search_bing(self, query: str, max_results: int) -> SearchResponse:
        """Scrape Bing search results."""
        try:
            params = {"q": query, "count": max_results + 2}
            headers = dict(self._session.headers)
            headers["User-Agent"] = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
            resp = self._session.get(
                BING_URL, params=params, headers=headers, timeout=self.timeout
            )
            resp.raise_for_status()
        except Exception as e:
            log.warning("Bing request failed: %s", e)
            return SearchResponse(query=query, error=str(e))

        soup    = BeautifulSoup(resp.text, "html.parser")
        results = []
        rank    = 1

        for li in soup.select("li.b_algo")[:max_results]:
            a_tag       = li.select_one("h2 a")
            snippet_tag = li.select_one(".b_caption p, p.b_lineclamp2, .b_algoSlug")
            if not a_tag:
                continue
            url     = a_tag.get("href", "")
            title   = a_tag.get_text(strip=True)
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
            if not url.startswith("http"):
                continue
            results.append(SearchResult(title=title, url=url, snippet=snippet, rank=rank))
            rank += 1

        if not results:
            return SearchResponse(query=query, error="No results from Bing.")
        return SearchResponse(query=query, results=results, source="bing")

    # ── Wikipedia fallback ────────────────────────────────────

    def _search_wikipedia(self, query: str, max_results: int) -> SearchResponse:
        """Query Wikipedia's opensearch API — always works, no bot-detection."""
        try:
            # Opensearch for title matches
            resp = self._session.get(
                WIKI_API,
                params={
                    "action": "opensearch",
                    "search": query,
                    "limit": max_results,
                    "namespace": 0,
                    "format": "json",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            titles   = data[1] if len(data) > 1 else []
            snippets = data[2] if len(data) > 2 else []
            urls     = data[3] if len(data) > 3 else []

            results = []
            for i, (title, snippet, url) in enumerate(zip(titles, snippets, urls)):
                results.append(SearchResult(
                    title=title, url=url,
                    snippet=snippet[:300], rank=i + 1,
                ))

            if not results:
                return SearchResponse(query=query, error="No results from Wikipedia.")
            log.info("Wikipedia returned %d results for %r", len(results), query)
            return SearchResponse(query=query, results=results, source="wikipedia")
        except Exception as e:
            log.warning("Wikipedia search failed: %s", e)
            return SearchResponse(query=query, error=str(e))


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "NEXUS AI assistant python"
    print(f"\n  Searching: {query!r}\n")

    searcher = WebSearcher()
    response = searcher.search(query, max_results=5)

    if response.success:
        for r in response.results:
            print(f"\n  [{r.rank}] {r.title}")
            print(f"       {r.url}")
            print(f"       {r.snippet[:120]}")
    else:
        print(f"  Error: {response.error}")

    print(f"\n  Source : {response.source}")
    print(f"  Time   : {response.elapsed:.2f}s\n")