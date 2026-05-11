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
    page = f.fetch("https://owasp.org/www-community/attacks/SQL_Injection")
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

FETCH_TIMEOUT     = 15          # seconds
MAX_TEXT_CHARS    = 12_000      # truncate page text beyond this
MIN_TEXT_CHARS    = 80          # lowered — some good pages are terse

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Tags that are definitely junk — safe to remove entirely
TAGS_TO_STRIP = {
    "script", "style", "noscript", "iframe",
    "button", "input", "select", "textarea",
}

# Class/ID patterns for clearly junk containers
# Kept intentionally narrow — don't over-strip
JUNK_PATTERNS = re.compile(
    r"\b(cookie-?banner|cookie-?notice|gdpr|newsletter-?signup|"
    r"social-?share|share-?buttons?|modal|overlay|"
    r"ads?-?container|advertisement|sidebar-?ad)\b",
    re.I,
)

# Selectors tried in order for main content — most specific first
CONTENT_SELECTORS = [
    "article",
    "main",
    "[role='main']",
    ".post-body",
    ".post-content",
    ".article-body",
    ".article-content",
    ".entry-content",
    ".content-body",
    ".page-content",
    ".wiki-content",
    ".markdown-body",        # GitHub
    ".prose",                # Tailwind/OWASP
    "#content article",
    "#content",
    "#main-content",
    "#main",
    ".main-content",
    ".container article",
    ".container",
    "body",                  # absolute fallback
]


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

    Strategy:
        1. Strip definitely-junk tags (scripts, styles, iframes)
        2. Remove clearly junk containers by class/ID pattern
        3. Walk CONTENT_SELECTORS to find the richest content block
        4. If nothing qualifies, use full body text (no false negatives)
        5. Clean whitespace and remove navigation fragments

    Usage:
        fetcher = PageFetcher()
        page = fetcher.fetch("https://owasp.org/www-community/attacks/SQL_Injection")
        print(page.text)

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
        """Download and clean one page."""
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
            log.warning("Fetch yielded no usable content: %s  error=%r", url, page.error)

        return page

    def fetch_many(
        self,
        urls: list[str],
        delay: float = 0.8,
        max_pages: int = 5,
    ) -> list[FetchedPage]:
        """Fetch multiple pages with a polite delay. Returns successful fetches only."""
        pages = []
        for url in urls[:max_pages]:
            page = self.fetch(url)
            if page.success:
                pages.append(page)
            if delay > 0:
                time.sleep(delay)
        return pages

    # ── HTML parsing ──────────────────────────────────────────

    def _parse(self, html: str, url: str) -> FetchedPage:
        """Extract clean text from raw HTML."""
        soup = BeautifulSoup(html, "html.parser")

        # Title
        title_tag = soup.find("title")
        title     = title_tag.get_text(strip=True) if title_tag else ""

        # 1. Remove HTML comments
        for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
            c.extract()

        # 2. Strip definitely-junk tags (non-content)
        for tag_name in TAGS_TO_STRIP:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # 3. Remove containers whose class/id screams "junk"
        #    (narrow pattern — only obvious ad/cookie/modal wrappers)
        for tag in soup.find_all(True):
            attrs = " ".join(
                tag.get("class", []) + [tag.get("id", "")]
            )
            if JUNK_PATTERNS.search(attrs):
                tag.decompose()

        # 4. Find best content block
        content_tag = self._find_main_content(soup)
        raw_text    = content_tag.get_text(separator="\n", strip=True)

        # 5. Clean the text
        text = self._clean_text(raw_text)
        text = text[:self.max_chars]

        if len(text.strip()) < MIN_TEXT_CHARS:
            # Last resort: pull ALL text from soup (ignore structure entirely)
            text = self._clean_text(soup.get_text(separator="\n", strip=True))
            text = text[:self.max_chars]

        if len(text.strip()) < MIN_TEXT_CHARS:
            return FetchedPage(url=url, title=title, error="Page too short or no content")

        return FetchedPage(
            url        = url,
            title      = title,
            text       = text,
            word_count = len(text.split()),
        )

    def _find_main_content(self, soup: "BeautifulSoup"):
        """
        Walk CONTENT_SELECTORS and return the first tag that contains
        a meaningful amount of text. Falls back to <body>, then full soup.
        """
        for selector in CONTENT_SELECTORS:
            try:
                tag = soup.select_one(selector)
            except Exception:
                continue

            if tag is None:
                continue

            text_len = len(tag.get_text(strip=True))
            if text_len >= MIN_TEXT_CHARS:
                log.debug("Content selector matched: %r (%d chars)", selector, text_len)
                return tag

        # Nothing matched with sufficient text — use full soup
        log.debug("No content selector matched — using full document.")
        return soup

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace and drop navigation-style fragments."""
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Skip lines that are purely numeric/symbolic (page numbers, bullets)
            if re.match(r"^[\d\s\W]{1,6}$", line):
                continue
            # Skip very short lines that look like menu items (< 3 words)
            words = line.split()
            if len(words) < 3 and len(line) < 20:
                # Keep if it looks like a heading (title-case or ALL CAPS)
                if not (line.istitle() or line.isupper()):
                    continue
            lines.append(line)

        result = "\n".join(lines)
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    url = sys.argv[1] if len(sys.argv) > 1 else \
        "https://owasp.org/www-community/attacks/SQL_Injection"
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