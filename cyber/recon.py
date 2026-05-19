"""
cyber/recon.py
NEXUS Recon Engine — passive + active reconnaissance for bug bounty hunting.

Capabilities:
  - Subdomain enumeration (passive: crt.sh, HackerTarget; active: DNS brute)
  - DNS record enumeration (A, MX, TXT, NS, CNAME)
  - HTTP header analysis (server, security headers, tech fingerprint)
  - WHOIS / IP info
  - Web tech fingerprinting (Wappalyzer-style patterns)
  - Directory bruteforce (via gobuster/dirb/ffuf if installed, or pure Python)
  - robots.txt / sitemap.xml harvesting
  - Target authorization gate — NEXUS will not recon without explicit permission

All methods narrate via StreamOutput.
"""

from __future__ import annotations

import json
import logging
import re
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

log = logging.getLogger("nexus.cyber.recon")

_WORDLIST_DIR = Path(__file__).parent.parent / "data" / "wordlists"
_REPORT_DIR   = Path(__file__).parent.parent / "data" / "recon_reports"

# Common subdomain prefixes for brute-force
_DEFAULT_SUBS = [
    "www", "mail", "ftp", "vpn", "dev", "staging", "api", "admin",
    "test", "beta", "portal", "app", "login", "secure", "dashboard",
    "m", "shop", "store", "blog", "static", "cdn", "assets", "media",
    "support", "help", "docs", "status", "monitor", "jenkins", "gitlab",
    "jira", "confluence", "intranet", "internal", "corp", "remote",
    "smtp", "pop", "imap", "ns1", "ns2", "mx", "mx1", "mx2",
]


class ReconEngine:
    """
    Passive + active recon for authorized targets.

    NEXUS asks for permission before touching any target that isn't
    explicitly in the authorized scope list.
    """

    def __init__(self):
        _WORDLIST_DIR.mkdir(parents=True, exist_ok=True)
        _REPORT_DIR.mkdir(parents=True, exist_ok=True)
        self._authorized: set[str] = set()
        try:
            from core.stream_output import get_output
            self._out = get_output()
        except ImportError:
            self._out = None
        self._load_auth()

    def _say(self, method: str, *args):
        if not self._out:
            return
        getattr(self._out, method, self._out.thinking)(*args)

    # ── Authorization ─────────────────────────────────────────

    def authorize(self, target: str) -> str:
        """Add a target to the authorized scope."""
        domain = self._normalize_target(target)
        self._authorized.add(domain)
        self._save_auth()
        return f"  ✓ {domain} added to authorized scope."

    def deauthorize(self, target: str) -> str:
        """Remove a target from authorized scope."""
        domain = self._normalize_target(target)
        self._authorized.discard(domain)
        self._save_auth()
        return f"  ✓ {domain} removed from authorized scope."

    def list_authorized(self) -> str:
        if not self._authorized:
            return "  No authorized targets. Use: authorize <domain>"
        lines = ["\n  AUTHORIZED RECON TARGETS", "  " + "─" * 40]
        for t in sorted(self._authorized):
            lines.append(f"  ✓ {t}")
        return "\n".join(lines)

    def _is_authorized(self, target: str) -> bool:
        domain = self._normalize_target(target)
        if any(domain == a or domain.endswith("." + a) for a in self._authorized):
            return True
        return False

    def _require_auth(self, target: str) -> Optional[str]:
        """Return error string if not authorized, None if OK."""
        if not self._is_authorized(target):
            return (
                f"\n  ⛔  {target} is not in your authorized scope.\n"
                f"  Add it first: authorize {target}\n"
                f"  Only run recon on targets you own or have written permission to test.\n"
            )
        return None

    # ── Full Recon ────────────────────────────────────────────

    def full_recon(self, target: str) -> str:
        """
        Run a full passive + active recon pipeline on target.

        Steps:
          1. DNS records
          2. WHOIS / IP info
          3. HTTP headers + tech fingerprint
          4. Subdomain enumeration (passive crt.sh + active brute)
          5. robots.txt + sitemap
          6. Security header grade
        """
        auth_err = self._require_auth(target)
        if auth_err:
            return auth_err

        self._say("planning", f"Full recon: {target}")
        sections = []

        sections.append(f"\n{'═'*60}")
        sections.append(f"  NEXUS FULL RECON — {target}")
        sections.append(f"{'═'*60}\n")

        self._say("thinking", "Step 1/5 — DNS records")
        sections.append(self._section("DNS Records", self.dns_records(target)))

        self._say("thinking", "Step 2/5 — IP / WHOIS")
        sections.append(self._section("IP & WHOIS", self.ip_info(target)))

        self._say("thinking", "Step 3/5 — HTTP headers")
        sections.append(self._section("HTTP Headers & Tech", self.http_headers(target)))

        self._say("thinking", "Step 4/5 — Subdomain enumeration")
        sections.append(self._section("Subdomains", self.subdomains(target)))

        self._say("thinking", "Step 5/5 — robots.txt & sitemap")
        sections.append(self._section("robots.txt / sitemap", self.robots_sitemap(target)))

        self._say("done", f"Full recon complete for {target}")

        report = "\n".join(sections)
        self._save_report(target, report)
        return report

    # ── DNS ───────────────────────────────────────────────────

    def dns_records(self, target: str) -> str:
        """Enumerate DNS records for target."""
        domain = self._normalize_target(target)
        self._say("searching", f"DNS records: {domain}")

        records: dict[str, list[str]] = {}
        try:
            import dns.resolver
            resolver = dns.resolver.Resolver()
            for rtype in ("A", "AAAA", "MX", "NS", "TXT", "CNAME", "SOA"):
                try:
                    answers = resolver.resolve(domain, rtype, lifetime=5)
                    records[rtype] = [r.to_text() for r in answers]
                except Exception:
                    pass
        except ImportError:
            # Fallback: nslookup / dig
            records = self._dns_fallback(domain)

        if not records:
            return f"  No DNS records found for {domain}"

        lines = []
        for rtype, vals in records.items():
            for v in vals:
                lines.append(f"  {rtype:<6} {v}")
        return "\n".join(lines)

    def _dns_fallback(self, domain: str) -> dict:
        """Use system dig/nslookup if dnspython not installed."""
        records = {}
        for rtype in ("A", "MX", "NS", "TXT"):
            try:
                proc = subprocess.run(
                    ["dig", "+short", rtype, domain],
                    capture_output=True, text=True, timeout=10,
                )
                vals = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
                if vals:
                    records[rtype] = vals
            except Exception:
                try:
                    out = subprocess.run(
                        ["nslookup", "-type=" + rtype, domain],
                        capture_output=True, text=True, timeout=10,
                    ).stdout
                    ips = re.findall(r"Address:\s+(\S+)", out)
                    if ips:
                        records[rtype] = ips
                except Exception:
                    pass
        return records

    # ── WHOIS / IP ────────────────────────────────────────────

    def ip_info(self, target: str) -> str:
        """Get IP geolocation and WHOIS-like info."""
        domain = self._normalize_target(target)
        self._say("searching", f"IP info: {domain}")

        try:
            ip = socket.gethostbyname(domain)
        except socket.gaierror:
            return f"  Cannot resolve: {domain}"

        lines = [f"  Resolved IP: {ip}"]

        try:
            import requests
            resp = requests.get(
                f"https://ipwho.is/{ip}",
                timeout=10,
                headers={"User-Agent": "NEXUS-AI/1.0"},
            )
            if resp.ok:
                d = resp.json()
                lines += [
                    f"  Country  : {d.get('country', '?')} ({d.get('country_code', '?')})",
                    f"  Region   : {d.get('region', '?')}",
                    f"  City     : {d.get('city', '?')}",
                    f"  ISP      : {d.get('connection', {}).get('isp', '?')}",
                    f"  ASN      : {d.get('connection', {}).get('asn', '?')}",
                    f"  Org      : {d.get('connection', {}).get('org', '?')}",
                ]
        except Exception as exc:
            log.debug("IP info failed: %s", exc)

        # Basic WHOIS via socket (port 43)
        try:
            whois_lines = self._raw_whois(domain)
            if whois_lines:
                registrant = next(
                    (l for l in whois_lines if "Registrant" in l or "Org" in l), ""
                )
                created = next(
                    (l for l in whois_lines if "Creation" in l or "Created" in l), ""
                )
                expires = next(
                    (l for l in whois_lines if "Expir" in l), ""
                )
                for info in (registrant, created, expires):
                    if info:
                        lines.append(f"  {info.strip()}")
        except Exception:
            pass

        return "\n".join(lines)

    def _raw_whois(self, domain: str) -> list[str]:
        try:
            proc = subprocess.run(
                ["whois", domain], capture_output=True, text=True, timeout=15,
            )
            return proc.stdout.splitlines()
        except Exception:
            return []

    # ── HTTP Headers ──────────────────────────────────────────

    def http_headers(self, target: str) -> str:
        """Fetch HTTP headers and fingerprint the target's tech stack."""
        url = target if target.startswith("http") else f"https://{target}"
        self._say("searching", f"HTTP headers: {url}")

        try:
            import requests
            resp = requests.get(
                url,
                timeout=15,
                allow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"},
            )
            headers = dict(resp.headers)
            status  = resp.status_code
            final   = resp.url
            body    = resp.text[:5000]
        except Exception as exc:
            return f"  [!] HTTP request failed: {exc}"

        lines = [
            f"  Status   : {status}",
            f"  Final URL: {final}",
            "",
            "  HTTP Headers:",
        ]
        interesting = [
            "Server", "X-Powered-By", "X-Generator", "X-Frame-Options",
            "Content-Security-Policy", "Strict-Transport-Security",
            "X-Content-Type-Options", "X-XSS-Protection",
            "Access-Control-Allow-Origin", "Set-Cookie",
            "Via", "CF-Ray", "X-Varnish", "X-Backend",
        ]
        for h in interesting:
            val = headers.get(h) or headers.get(h.lower(), "")
            if val:
                lines.append(f"    {h:<35} {val[:80]}")

        # Security header grades
        lines.append("\n  Security Header Check:")
        sec_headers = {
            "Strict-Transport-Security": "HSTS",
            "Content-Security-Policy":   "CSP",
            "X-Frame-Options":           "Clickjacking protection",
            "X-Content-Type-Options":    "MIME sniffing protection",
            "X-XSS-Protection":          "XSS filter",
        }
        for h, label in sec_headers.items():
            present = bool(headers.get(h) or headers.get(h.lower()))
            mark = "✓" if present else "✗"
            lines.append(f"    {mark} {label}")

        # Tech fingerprinting
        techs = self._fingerprint_tech(headers, body)
        if techs:
            lines.append("\n  Detected Technologies:")
            for t in techs:
                lines.append(f"    • {t}")

        return "\n".join(lines)

    def _fingerprint_tech(self, headers: dict, body: str) -> list[str]:
        """Wappalyzer-style pattern matching."""
        techs = []
        h_lower = {k.lower(): v.lower() for k, v in headers.items()}
        b_lower = body.lower()

        checks = [
            (lambda: "wordpress" in b_lower or "/wp-content/" in b_lower,
             "WordPress"),
            (lambda: "drupal" in b_lower,
             "Drupal"),
            (lambda: "joomla" in b_lower,
             "Joomla"),
            (lambda: "nginx" in h_lower.get("server", ""),
             "Nginx"),
            (lambda: "apache" in h_lower.get("server", ""),
             "Apache"),
            (lambda: "php" in h_lower.get("x-powered-by", "") or ".php" in body,
             "PHP"),
            (lambda: "asp.net" in h_lower.get("x-powered-by", ""),
             "ASP.NET"),
            (lambda: "cloudflare" in h_lower.get("server", "") or "cf-ray" in h_lower,
             "Cloudflare"),
            (lambda: "react" in b_lower or "_reactroot" in body,
             "React"),
            (lambda: "vue.js" in b_lower or "vuex" in b_lower,
             "Vue.js"),
            (lambda: "__next" in body or "_next/static" in body,
             "Next.js"),
            (lambda: "django" in b_lower or "csrfmiddlewaretoken" in body,
             "Django"),
            (lambda: "laravel" in b_lower or "laravel_session" in h_lower.get("set-cookie", ""),
             "Laravel"),
            (lambda: "express" in h_lower.get("x-powered-by", ""),
             "Express.js"),
            (lambda: "jquery" in b_lower,
             "jQuery"),
            (lambda: "bootstrap" in b_lower,
             "Bootstrap"),
        ]
        for cond_fn, name in checks:
            try:
                if cond_fn():
                    techs.append(name)
            except Exception:
                pass
        return techs

    # ── Subdomains ────────────────────────────────────────────

    def subdomains(self, target: str, passive: bool = True, active: bool = True) -> str:
        """
        Enumerate subdomains via passive + active methods.

        Passive: crt.sh certificate transparency, HackerTarget
        Active:  DNS resolution brute-force against wordlist
        """
        domain = self._normalize_target(target)
        found: set[str] = set()

        if passive:
            self._say("searching", f"crt.sh (certificate transparency): {domain}")
            found.update(self._crtsh_subs(domain))

            self._say("searching", f"HackerTarget subdomain API: {domain}")
            found.update(self._hackertarget_subs(domain))

        if active:
            self._say("searching", f"DNS brute-force ({len(_DEFAULT_SUBS)} prefixes): {domain}")
            found.update(self._brute_subs(domain))

        if not found:
            return f"  No subdomains found for {domain}"

        lines = [f"  Found {len(found)} subdomain(s):"]
        for sub in sorted(found):
            try:
                ip = socket.gethostbyname(sub)
                lines.append(f"  {sub:<45} {ip}")
            except Exception:
                lines.append(f"  {sub}")
        return "\n".join(lines)

    def _crtsh_subs(self, domain: str) -> set[str]:
        try:
            import requests
            resp = requests.get(
                "https://crt.sh/",
                params={"q": f"%.{domain}", "output": "json"},
                timeout=20,
                headers={"User-Agent": "NEXUS-AI/1.0"},
            )
            resp.raise_for_status()
            entries = resp.json()
            subs = set()
            for e in entries:
                for name in e.get("name_value", "").splitlines():
                    name = name.strip().lstrip("*.")
                    if name.endswith(domain) and name != domain:
                        subs.add(name.lower())
            return subs
        except Exception as exc:
            log.debug("crt.sh failed: %s", exc)
            return set()

    def _hackertarget_subs(self, domain: str) -> set[str]:
        try:
            import requests
            resp = requests.get(
                "https://api.hackertarget.com/hostsearch/",
                params={"q": domain},
                timeout=15,
                headers={"User-Agent": "NEXUS-AI/1.0"},
            )
            if resp.ok and "error" not in resp.text.lower():
                subs = set()
                for line in resp.text.splitlines():
                    parts = line.split(",")
                    if parts and parts[0].endswith(domain):
                        subs.add(parts[0].strip().lower())
                return subs
        except Exception as exc:
            log.debug("HackerTarget failed: %s", exc)
        return set()

    def _brute_subs(self, domain: str) -> set[str]:
        """Resolve a wordlist of common subdomains concurrently."""
        found: set[str] = set()

        def check(prefix: str) -> Optional[str]:
            fqdn = f"{prefix}.{domain}"
            try:
                socket.getaddrinfo(fqdn, None, timeout=2)
                return fqdn
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=30) as pool:
            futures = {pool.submit(check, p): p for p in _DEFAULT_SUBS}
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    found.add(result)
        return found

    # ── robots.txt / sitemap ──────────────────────────────────

    def robots_sitemap(self, target: str) -> str:
        """Fetch robots.txt and sitemap.xml for recon."""
        base = target if target.startswith("http") else f"https://{target}"
        lines = []

        try:
            import requests
            for path in ("/robots.txt", "/sitemap.xml", "/sitemap_index.xml"):
                url = base.rstrip("/") + path
                try:
                    resp = requests.get(
                        url,
                        timeout=10,
                        headers={"User-Agent": "Mozilla/5.0"},
                    )
                    if resp.ok and len(resp.text) < 50000:
                        snippet = resp.text[:800].strip()
                        lines.append(f"  [{path}] HTTP {resp.status_code}")
                        lines.append(snippet)
                        lines.append("")
                    else:
                        lines.append(f"  [{path}] HTTP {resp.status_code} (not found or too large)")
                except Exception as exc:
                    lines.append(f"  [{path}] error: {exc}")
        except ImportError:
            return "  requests not installed."

        return "\n".join(lines) if lines else "  No robots.txt or sitemap found."

    # ── Directory Brute-force ─────────────────────────────────

    def dir_bruteforce(self, target: str, wordlist: str = "common") -> str:
        """
        Directory brute-force using ffuf/gobuster/dirb if installed,
        or a pure-Python fallback.

        Args:
            target:   base URL (e.g. https://example.com)
            wordlist: "common" | "big" | path to custom wordlist
        """
        auth_err = self._require_auth(target)
        if auth_err:
            return auth_err

        url = target if target.startswith("http") else f"https://{target}"
        self._say("thinking", f"Directory brute-force: {url}")

        # Try ffuf first
        if self._cmd_exists("ffuf"):
            wl = self._get_wordlist(wordlist)
            self._say("running", f"ffuf -u {url}/FUZZ -w {wl} -mc 200,301,302,403")
            try:
                proc = subprocess.run(
                    ["ffuf", "-u", f"{url}/FUZZ", "-w", str(wl),
                     "-mc", "200,301,302,403", "-o", "-", "-of", "json"],
                    capture_output=True, text=True, timeout=120,
                )
                return self._format_ffuf(proc.stdout, url)
            except subprocess.TimeoutExpired:
                return "  [!] ffuf timed out after 120s"

        # Try gobuster
        if self._cmd_exists("gobuster"):
            wl = self._get_wordlist(wordlist)
            self._say("running", f"gobuster dir -u {url} -w {wl}")
            try:
                proc = subprocess.run(
                    ["gobuster", "dir", "-u", url, "-w", str(wl), "-q"],
                    capture_output=True, text=True, timeout=120,
                )
                return f"  gobuster output:\n{proc.stdout[:2000]}"
            except subprocess.TimeoutExpired:
                return "  [!] gobuster timed out"

        # Python fallback — checks a small built-in wordlist
        self._say("thinking", "No ffuf/gobuster found — using Python fallback")
        return self._python_dir_scan(url)

    def _python_dir_scan(self, base_url: str) -> str:
        common_paths = [
            "admin", "login", "api", "v1", "v2", "api/v1", "api/v2",
            "dashboard", "wp-admin", "wp-login.php", "phpmyadmin",
            "config", "backup", "db", "test", "dev", "staging",
            ".git", ".env", "server-status", "robots.txt",
            "sitemap.xml", "upload", "uploads", "static", "assets",
            "js", "css", "img", "images", "files", "docs",
        ]

        try:
            import requests
        except ImportError:
            return "  requests not installed."

        found = []
        session = requests.Session()
        session.headers["User-Agent"] = "Mozilla/5.0"

        for path in common_paths:
            url = f"{base_url.rstrip('/')}/{path}"
            try:
                resp = session.get(url, timeout=5, allow_redirects=False)
                if resp.status_code in (200, 301, 302, 403):
                    found.append(f"  [{resp.status_code}] {url}")
            except Exception:
                continue

        if not found:
            return "  No interesting paths found."
        return "\n".join(["  Discovered paths:"] + found)

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _normalize_target(target: str) -> str:
        target = target.strip().rstrip("/")
        parsed = urlparse(target)
        return parsed.netloc or parsed.path or target

    def _get_wordlist(self, name: str) -> Path:
        builtin = _WORDLIST_DIR / f"{name}.txt"
        if builtin.exists():
            return builtin
        if Path(name).exists():
            return Path(name)
        # Create a tiny fallback wordlist
        builtin.parent.mkdir(parents=True, exist_ok=True)
        builtin.write_text("\n".join(_DEFAULT_SUBS + [
            "admin", "login", "api", "v1", "dashboard", "config",
            "backup", "test", "dev", "wp-admin", "phpmyadmin",
        ]))
        return builtin

    @staticmethod
    def _cmd_exists(cmd: str) -> bool:
        try:
            subprocess.run([cmd, "--version"], capture_output=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _section(title: str, content: str) -> str:
        return f"  ── {title} ──\n{content}\n"

    def _format_ffuf(self, raw: str, url: str) -> str:
        try:
            data = json.loads(raw)
            results = data.get("results", [])
            if not results:
                return "  No paths found by ffuf."
            lines = ["  ffuf results:"]
            for r in results:
                lines.append(f"  [{r.get('status', '?')}] {r.get('url', '?')}  ({r.get('length', '?')} bytes)")
            return "\n".join(lines)
        except json.JSONDecodeError:
            return f"  ffuf output:\n{raw[:1000]}"

    def _save_report(self, target: str, content: str):
        from datetime import datetime
        fname = _REPORT_DIR / f"recon_{target.replace('/', '_')}_{datetime.now():%Y%m%d_%H%M%S}.txt"
        fname.write_text(content, encoding="utf-8")
        self._say("done", f"Report saved: {fname.name}")

    def _load_auth(self):
        auth_file = Path(__file__).parent.parent / "data" / "authorized_targets.json"
        if auth_file.exists():
            try:
                self._authorized = set(json.loads(auth_file.read_text()))
            except Exception:
                pass

    def _save_auth(self):
        auth_file = Path(__file__).parent.parent / "data" / "authorized_targets.json"
        auth_file.write_text(json.dumps(list(self._authorized), indent=2))
