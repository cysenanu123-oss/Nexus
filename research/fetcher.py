"""
research/fetcher.py
NEXUS Page Fetcher — downloads web pages and extracts clean text content.

Strips HTML, navigation, ads, and boilerplate. Returns clean article text
that the summarizer can actually use.

Pipeline:
    research/searcher.py
           ↓
    research/fetcher.py    ← YOU ARE HERE
           ↓
    research/summarizer.py
           ↓
    research/memory.py

Usage:
    from research.fetcher import PageFetcher
    f = PageFetcher()
    page = f.fetch("https://en.wikipedia.org/wiki/Buffer_overflow")
    print(page.text[:500])
"""

from __future__ import annotations

import logging
import time
import re
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("nexus.research.fetcher")

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

try:
    from bs4 import BeautifulSoup, Comment
    _BS4_OK = True
except ImportError:
    _BS4_OK = False


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

FETCH_TIMEOUT     = 12          # seconds
MAX_TEXT_CHARS    = 12_000      # truncate page text beyond this
MIN_TEXT_CHARS    = 100         # ignore pages with less than this

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Tags to strip entirely (no content needed)
TAGS_TO_STRIP = {
    "script", "style", "noscript", "nav", "footer", "header",
    "aside", "form", "button", "input", "select", "textarea",
    "iframe", "advertisement", "figure", "figcaption",
    "cookie", "popup", "banner", "sidebar",
}

# CSS classes/IDs that typically contain junk
JUNK_PATTERNS = re.compile(
    r"(nav|menu|footer|header|sidebar|cookie|banner|ad-|ads-|advert|popup|"
    r"subscribe|newsletter|social|share|comment|related|recommend)",
    re.I,
)


# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class FetchedPage:
    """Cleaned content from one web page."""
    url:        str
    title:      str   = ""
    text:       str   = ""
    word_count: int   = 0
    status:     int   = 0
    elapsed:    float = 0.0
    error:      str   = ""

    @property
    def success(self) -> bool:
        return bool(self.text) and not self.error

    @property
    def snippet(self) -> str:
        """First 300 characters of text."""
        return self.text[:300].strip()

    def __str__(self) -> str:
        if self.error:
            return f"FetchedPage(FAILED: {self.error})"
        return (
            f"FetchedPage(url={self.url!r}, "
            f"words={self.word_count}, "
            f"status={self.status})"
        )


# ─────────────────────────────────────────────────────────────
#  PAGE FETCHER
# ─────────────────────────────────────────────────────────────

class PageFetcher:
    """
    Downloads web pages and extracts clean readable text.

    Handles:
        - HTML stripping and boilerplate removal
        - Character encoding issues
        - Timeouts and error recovery
        - Multiple URLs in batch

    Usage:
        fetcher = PageFetcher()
        page = fetcher.fetch("https://docs.python.org/3/library/subprocess.html")
        print(page.text)

        # Batch fetch
        pages = fetcher.fetch_many(["https://...", "https://..."])
    """

    def __init__(self, timeout: int = FETCH_TIMEOUT, max_chars: int = MAX_TEXT_CHARS):
        if not _REQUESTS_OK:
            raise ImportError("requests not installed. Run: pip install requests")
        if not _BS4_OK:
            raise ImportError("beautifulsoup4 not installed. Run: pip install beautifulsoup4")

        self.timeout   = timeout
        self.max_chars = max_chars

        self._session  = requests.Session()
        self._session.headers.update(HEADERS)
        log.info("PageFetcher ready.")

    # ── public API ────────────────────────────────────────────

    def fetch(self, url: str) -> FetchedPage:
        """
        Download and clean one page.

        Returns FetchedPage — always (check .success for errors).
        """
        log.info("Fetching: %s", url)
        t0 = time.time()

        try:
            resp = self._session.get(url, timeout=self.timeout, allow_redirects=True)
            resp.raise_for_status()
            status = resp.status_code
        except requests.exceptions.Timeout:
            return FetchedPage(url=url, error=f"Timeout after {self.timeout}s")
        except requests.exceptions.HTTPError as e:
            return FetchedPage(url=url, status=e.response.status_code, error=str(e))
        except Exception as e:
            return FetchedPage(url=url, error=str(e))

        # Parse
        try:
            page = self._parse(resp.text, url)
            page.status  = status
            page.elapsed = time.time() - t0
        except Exception as e:
            log.error("Parse error for %s: %s", url, e)
            page = FetchedPage(url=url, status=status, error=f"Parse error: {e}")

        if page.success:
            log.info(
                "Fetched %s — %d words in %.2fs",
                url, page.word_count, page.elapsed,
            )
        else:
            log.warning("Fetch yielded no usable content: %s", url)

        return page

    def fetch_many(
        self,
        urls: list[str],
        delay: float = 0.8,
        max_pages: int = 5,
    ) -> list[FetchedPage]:
        """
        Fetch multiple pages with a polite delay.

        Parameters
        ----------
        urls      : list of URLs
        delay     : seconds between requests
        max_pages : cap on total pages fetched

        Returns
        -------
        List of FetchedPage (successful fetches only)
        """
        pages = []
        for url in urls[:max_pages]:
            page = self.fetch(url)
            if page.success:
                pages.append(page)
            if delay > 0:
                time.sleep(delay)
        return pages

    # ── HTML parsing & cleaning ───────────────────────────────

    def _parse(self, html: str, url: str) -> FetchedPage:
        """Extract clean text from raw HTML."""
        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title_tag = soup.find("title")
        title     = title_tag.get_text(strip=True) if title_tag else ""

        # Remove HTML comments
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()

        # Remove junk tags entirely
        for tag_name in TAGS_TO_STRIP:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        SAFE_TAGS = {"html", "body", "main", "article"}

        # Remove elements with junk class/id patterns
        for tag in soup.find_all(True):
            if tag.name in SAFE_TAGS:
                continue
            if tag.attrs is None:
                continue
            
            c = tag.get("class", [])
            classes = " ".join(c) if isinstance(c, list) else str(c)
            
            tag_id = tag.get("id", "")
            tag_id = " ".join(tag_id) if isinstance(tag_id, list) else str(tag_id)
            
            combined = (classes + " " + tag_id).lower()
            if "content" in combined or "main" in combined:
                continue
                
            if JUNK_PATTERNS.search(classes) or JUNK_PATTERNS.search(tag_id):
                tag.decompose()

        # Try to find the main content area
        content = self._find_main_content(soup)

        # Extract text
        raw_text = content.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        text = self._clean_text(raw_text)

        # Truncate
        text = text[:self.max_chars]

        if len(text) < MIN_TEXT_CHARS:
            return FetchedPage(url=url, title=title, error="Page too short or no content")

        return FetchedPage(
            url        = url,
            title      = title,
            text       = text,
            word_count = len(text.split()),
        )

    def _find_main_content(self, soup: "BeautifulSoup"):
        """Try to locate the main content container."""
        # Priority order: semantic HTML5 > common class patterns > body
        selectors = [
            "article",
            "main",
            "[role='main']",
            ".article-body",
            ".post-content",
            ".entry-content",
            ".content-body",
            "#content",
            "#main",
            ".main-content",
        ]
        for sel in selectors:
            tag = soup.select_one(sel)
            if tag and len(tag.get_text(strip=True)) > MIN_TEXT_CHARS:
                return tag

        # Fall back to body
        body = soup.find("body")
        return body if body else soup

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace and remove garbage lines."""
        lines = []
        for line in text.splitlines():
            line = line.strip()
            # Skip very short lines (navigation fragments, single words)
            if len(line) < 4:
                continue
            # Skip lines that are just numbers or symbols
            if re.match(r"^[\d\s\W]{1,10}$", line):
                continue
            lines.append(line)

        # Collapse multiple blank lines
        result = "\n".join(lines)
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    url = sys.argv[1] if len(sys.argv) > 1 else "https://en.wikipedia.org/wiki/Buffer_overflow"
    print(f"\n  Fetching: {url}\n")

    fetcher = PageFetcher()
    page    = fetcher.fetch(url)

    if page.success:
        print(f"  Title  : {page.title}")
        print(f"  Words  : {page.word_count}")
        print(f"  Time   : {page.elapsed:.2f}s")
        print(f"\n  --- First 600 chars ---\n")
        print(f"  {page.text[:600]}\n")
    else:
        print(f"  Error: {page.error}\n")