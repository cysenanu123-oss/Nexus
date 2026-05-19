"""
cyber/intel.py
NEXUS Cyber Intelligence — news feeds, CVE lookups, exploit search.

Pulls from:
  - NVD (NIST CVE database)  — JSON API, no key needed
  - Exploit-DB               — web scrape / searchsploit CLI
  - RSS feeds                — The Hacker News, Krebs, BleepingComputer, SANS ISC
  - GitHub Security Advisories — REST API, no key needed
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

log = logging.getLogger("nexus.cyber.intel")

# ── RSS feeds (no auth needed) ────────────────────────────────
CYBER_FEEDS = {
    "hackernews": "https://feeds.feedburner.com/TheHackersNews",
    "krebs":      "https://krebsonsecurity.com/feed/",
    "bleeping":   "https://www.bleepingcomputer.com/feed/",
    "sans":       "https://isc.sans.edu/rssfeed_full.xml",
    "cisa":       "https://www.cisa.gov/cybersecurity-advisories/all.xml",
    "exploit_db": "https://www.exploit-db.com/rss.xml",
}

NVD_API     = "https://services.nvd.nist.gov/rest/json/cves/2.0"
GHSA_API    = "https://api.github.com/advisories"
EXPLOITDB   = "https://www.exploit-db.com/search"

_CACHE_DIR  = Path(__file__).parent.parent / "data" / "intel_cache"


class CyberIntel:
    """
    Cyber intelligence: latest news, CVE details, exploit search.

    All methods are narrated via StreamOutput so NEXUS tells you
    exactly what it's searching and where.
    """

    def __init__(self):
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            from core.stream_output import get_output
            self._out = get_output()
        except ImportError:
            self._out = None

    def _say(self, method: str, *args):
        if not self._out:
            return
        getattr(self._out, method, self._out.thinking)(*args)

    # ── News ──────────────────────────────────────────────────

    def latest_news(self, limit: int = 8, source: str = "all") -> str:
        """
        Pull latest cybersecurity headlines from RSS feeds.

        Args:
            limit:  max articles to return
            source: "all" | "hackernews" | "krebs" | "bleeping" | "sans" | "cisa"
        """
        try:
            import feedparser
        except ImportError:
            return "[!] feedparser not installed: pip install feedparser"

        feeds = (
            {source: CYBER_FEEDS[source]}
            if source in CYBER_FEEDS
            else CYBER_FEEDS
        )

        articles: list[dict] = []
        for name, url in feeds.items():
            self._say("searching", f"RSS: {name} ({url})")
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:max(2, limit // len(feeds))]:
                    pub = entry.get("published", "")
                    date = self._parse_date(pub)
                    articles.append({
                        "title":  entry.get("title", "Untitled")[:120],
                        "source": name,
                        "date":   date,
                        "link":   entry.get("link", ""),
                        "summary": self._strip_html(entry.get("summary", ""))[:200],
                    })
            except Exception as exc:
                log.debug("Feed %s failed: %s", name, exc)

        articles.sort(key=lambda x: x["date"], reverse=True)

        if not articles:
            return "  [!] No news articles retrieved. Check internet connection."

        lines = [
            f"\n{'═'*60}",
            "  NEXUS CYBER INTELLIGENCE — Latest News",
            f"{'═'*60}\n",
        ]
        for a in articles[:limit]:
            lines.append(f"  [{a['source'].upper():<12}] {a['date']}")
            lines.append(f"  {a['title']}")
            if a["summary"]:
                lines.append(f"  {a['summary']}")
            lines.append(f"  → {a['link']}")
            lines.append("")

        self._say("done", f"Retrieved {len(articles[:limit])} articles")
        return "\n".join(lines)

    # ── CVE Lookup ────────────────────────────────────────────

    def cve_lookup(self, cve_id: str) -> str:
        """
        Look up a specific CVE from the NVD database.

        Args:
            cve_id: e.g. "CVE-2024-1234" or "2024-1234"
        """
        if not cve_id.upper().startswith("CVE-"):
            cve_id = "CVE-" + cve_id

        self._say("searching", f"NVD database: {cve_id}")

        try:
            import requests
            resp = requests.get(
                NVD_API,
                params={"cveId": cve_id.upper()},
                timeout=15,
                headers={"User-Agent": "NEXUS-AI/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return f"[!] NVD lookup failed: {exc}"

        vulns = data.get("vulnerabilities", [])
        if not vulns:
            return f"  [!] {cve_id} not found in NVD database."

        vuln = vulns[0]["cve"]
        cve_data = vuln.get("id", cve_id)
        desc_list = vuln.get("descriptions", [])
        desc = next((d["value"] for d in desc_list if d["lang"] == "en"), "No description.")

        metrics = vuln.get("metrics", {})
        cvss_v3 = metrics.get("cvssMetricV31", [{}])[0]
        cvss_data = cvss_v3.get("cvssData", {})
        score     = cvss_data.get("baseScore", "N/A")
        severity  = cvss_data.get("baseSeverity", "N/A")
        vector    = cvss_data.get("vectorString", "N/A")

        refs = vuln.get("references", [])[:5]
        ref_lines = [f"  → {r.get('url', '')}" for r in refs]

        published = vuln.get("published", "?")[:10]
        modified  = vuln.get("lastModified", "?")[:10]

        self._say("done", f"{cve_id} — CVSS {score} ({severity})")

        return (
            f"\n{'═'*60}\n"
            f"  {cve_data}\n"
            f"{'═'*60}\n"
            f"  Score    : {score} ({severity})\n"
            f"  Vector   : {vector}\n"
            f"  Published: {published}  |  Modified: {modified}\n\n"
            f"  Description:\n"
            f"  {desc[:600]}\n\n"
            f"  References:\n"
            + "\n".join(ref_lines)
            + f"\n{'═'*60}\n"
        )

    def cve_search(self, keyword: str, limit: int = 5) -> str:
        """
        Search NVD for recent CVEs matching a keyword.

        Args:
            keyword: e.g. "apache", "buffer overflow", "sql injection"
            limit:   max results
        """
        self._say("searching", f"NVD CVE search: {keyword!r}")

        try:
            import requests
            resp = requests.get(
                NVD_API,
                params={
                    "keywordSearch": keyword,
                    "resultsPerPage": limit,
                    "startIndex":     0,
                },
                timeout=20,
                headers={"User-Agent": "NEXUS-AI/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return f"[!] NVD search failed: {exc}"

        vulns = data.get("vulnerabilities", [])
        if not vulns:
            return f"  No CVEs found for: {keyword!r}"

        lines = [
            f"\n{'═'*60}",
            f"  CVE SEARCH: {keyword!r}  ({data.get('totalResults', 0)} total results)",
            f"{'═'*60}\n",
        ]
        for item in vulns[:limit]:
            cve  = item["cve"]
            cid  = cve.get("id", "?")
            desc_list = cve.get("descriptions", [])
            desc = next((d["value"] for d in desc_list if d["lang"] == "en"), "")[:200]
            pub  = cve.get("published", "?")[:10]

            metrics = cve.get("metrics", {})
            cvss_v3 = metrics.get("cvssMetricV31", [{}])[0]
            score   = cvss_v3.get("cvssData", {}).get("baseScore", "N/A")

            lines.append(f"  [{cid}]  CVSS {score}  ({pub})")
            lines.append(f"  {desc}\n")

        self._say("done", f"Found {len(vulns)} CVEs for {keyword!r}")
        return "\n".join(lines)

    # ── Exploit Search ────────────────────────────────────────

    def exploit_search(self, query: str, limit: int = 10) -> str:
        """
        Search for exploits using searchsploit (Exploit-DB CLI) or web fallback.

        Args:
            query: e.g. "apache 2.4", "vsftpd", "smb"
            limit: max results
        """
        self._say("searching", f"Exploit search: {query!r}")

        # Try searchsploit first (best option)
        result = self._searchsploit(query, limit)
        if result:
            self._say("done", f"Exploit search complete for {query!r}")
            return result

        # Web fallback
        return self._exploitdb_web(query, limit)

    def _searchsploit(self, query: str, limit: int) -> Optional[str]:
        """Run searchsploit CLI if available."""
        try:
            proc = subprocess.run(
                ["searchsploit", "--json", query],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode != 0:
                return None
            data = json.loads(proc.stdout)
            matches = data.get("RESULTS_EXPLOIT", []) + data.get("RESULTS_SHELLCODE", [])
            if not matches:
                return f"  No exploits found via searchsploit for: {query!r}"

            lines = [
                f"\n{'═'*60}",
                f"  EXPLOIT-DB RESULTS: {query!r}",
                f"{'═'*60}\n",
                f"  {'EDB-ID':<8} {'PLATFORM':<12} {'TYPE':<12} TITLE",
                f"  {'─'*56}",
            ]
            for m in matches[:limit]:
                eid   = m.get("EDB-ID", "?")
                title = m.get("Title", "?")[:50]
                plat  = m.get("Platform", "?")[:10]
                mtype = m.get("Type", "?")[:10]
                path  = m.get("Path", "")
                lines.append(f"  {eid:<8} {plat:<12} {mtype:<12} {title}")
                if path:
                    lines.append(f"          Path: {path}")
            lines.append(f"\n  {'─'*56}")
            lines.append(f"  searchsploit -x <edb-id>   to view exploit")
            lines.append(f"  searchsploit -m <edb-id>   to copy to current dir")
            return "\n".join(lines)

        except (FileNotFoundError, json.JSONDecodeError, subprocess.TimeoutExpired):
            return None

    def _exploitdb_web(self, query: str, limit: int) -> str:
        """Scrape Exploit-DB web search as fallback."""
        try:
            import requests
            from bs4 import BeautifulSoup
            resp = requests.get(
                EXPLOITDB,
                params={"q": query, "type": "papers", "platform": ""},
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                    "Accept": "text/html",
                },
                timeout=15,
            )
            if resp.status_code != 200:
                return f"  [!] Exploit-DB web search returned HTTP {resp.status_code}"

            soup = BeautifulSoup(resp.text, "html.parser")
            rows = soup.select("table#exploits-table tbody tr")
            if not rows:
                return f"  No exploits found on Exploit-DB for: {query!r}"

            lines = [
                f"\n{'═'*60}",
                f"  EXPLOIT-DB WEB RESULTS: {query!r}",
                f"{'═'*60}\n",
            ]
            for row in rows[:limit]:
                cols = row.find_all("td")
                if len(cols) < 4:
                    continue
                title = cols[2].get_text(strip=True)[:55]
                eid   = cols[0].get_text(strip=True)
                pform = cols[4].get_text(strip=True) if len(cols) > 4 else "?"
                lines.append(f"  EDB-{eid:<8} [{pform:<8}]  {title}")
            lines.append(f"\n  → https://www.exploit-db.com/search?q={quote_plus(query)}")
            return "\n".join(lines)

        except Exception as exc:
            return f"  [!] Exploit-DB web search failed: {exc}"

    def exploit_download(self, edb_id: str, dest_dir: str = "data/exploits") -> str:
        """
        Download an exploit from Exploit-DB by EDB ID.

        Args:
            edb_id:   Exploit-DB ID (numeric, e.g. "47887")
            dest_dir: where to save (relative to project root or absolute)
        """
        edb_id = edb_id.strip().lstrip("EDB-").lstrip("edb-")
        dest = Path(dest_dir)
        if not dest.is_absolute():
            dest = Path(__file__).parent.parent / dest_dir
        dest.mkdir(parents=True, exist_ok=True)

        # Try searchsploit -m first
        self._say("searching", f"Downloading exploit EDB-{edb_id}")
        try:
            proc = subprocess.run(
                ["searchsploit", "-m", edb_id],
                capture_output=True, text=True, timeout=30,
                cwd=str(dest),
            )
            if proc.returncode == 0:
                files = list(dest.glob(f"*{edb_id}*"))
                if files:
                    self._say("done", f"Saved to {files[0]}")
                    return f"  ✓ Downloaded EDB-{edb_id} → {files[0]}"
        except FileNotFoundError:
            pass

        # HTTP fallback
        try:
            import requests
            for ext in (".py", ".rb", ".c", ".sh", ".pl", ".txt", ".php", ".ps1"):
                url = f"https://www.exploit-db.com/download/{edb_id}"
                resp = requests.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=20,
                    allow_redirects=True,
                )
                if resp.status_code == 200:
                    fname = dest / f"EDB-{edb_id}{ext}"
                    fname.write_bytes(resp.content)
                    self._say("done", f"Saved {fname.name}")
                    return f"  ✓ Downloaded EDB-{edb_id} → {fname}"
        except Exception as exc:
            return f"  [!] Download failed: {exc}"

        return f"  [!] Could not download EDB-{edb_id}. Try: searchsploit -m {edb_id}"

    # ── GitHub Security Advisories ────────────────────────────

    def ghsa_search(self, ecosystem: str = "", severity: str = "critical") -> str:
        """
        Fetch recent GitHub Security Advisories.

        Args:
            ecosystem: "pip" | "npm" | "maven" | "go" | "" (all)
            severity:  "critical" | "high" | "medium" | "low"
        """
        self._say("searching", f"GitHub Security Advisories (ecosystem={ecosystem or 'all'}, severity={severity})")

        try:
            import requests
            params = {"per_page": 10, "severity": severity}
            if ecosystem:
                params["ecosystem"] = ecosystem
            resp = requests.get(
                GHSA_API,
                params=params,
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "NEXUS-AI/1.0",
                },
                timeout=15,
            )
            resp.raise_for_status()
            advisories = resp.json()
        except Exception as exc:
            return f"[!] GHSA lookup failed: {exc}"

        if not advisories:
            return "  No GitHub advisories found."

        lines = [
            f"\n{'═'*60}",
            f"  GITHUB SECURITY ADVISORIES ({severity.upper()})",
            f"{'═'*60}\n",
        ]
        for a in advisories:
            ghsa_id  = a.get("ghsa_id", "?")
            summary  = a.get("summary", "?")[:80]
            svty     = a.get("severity", "?").upper()
            pub      = (a.get("published_at", "?") or "?")[:10]
            cvss     = a.get("cvss", {}).get("score", "N/A") if a.get("cvss") else "N/A"
            cve_list = [c.get("value") for c in a.get("identifiers", []) if c.get("type") == "CVE"]
            cves     = ", ".join(cve_list) or "—"
            lines.append(f"  [{svty:<8}] {ghsa_id}  CVSS {cvss}  ({pub})")
            lines.append(f"  {summary}")
            lines.append(f"  CVEs: {cves}\n")

        self._say("done", f"Retrieved {len(advisories)} advisories")
        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _strip_html(text: str) -> str:
        return re.sub(r"<[^>]+>", " ", text).strip()

    @staticmethod
    def _parse_date(pub_str: str) -> str:
        if not pub_str:
            return "?"
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(pub_str)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
        return pub_str[:10]
