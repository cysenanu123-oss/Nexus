"""
core/web_safety.py
NEXUS — safety layer for the autonomous web agent.

Same discipline as core/shell_safety.py, applied to the web: **reading is
free, acting is gated.** An agent that fetches URLs on its own is an SSRF risk
(tell it to "research http://169.254.169.254/…" and it might read cloud
metadata or an internal admin panel), so:

  * `is_safe_to_fetch(url)` — allows normal http(s) reads, but refuses
    non-web schemes (file://, javascript:, data:, ftp://) and any host that
    is loopback/private/link-local/reserved (localhost, 127.*, 10.*,
    192.168.*, 172.16–31.*, 169.254.*, ::1, *.local, *.internal).
  * `is_action(method)` — GET/HEAD are reads; POST/PUT/PATCH/DELETE change
    state and must be confirmed.
  * `guard_action(description, confirm)` — the consent gate for any
    state-changing web action (posting, buying, logging in, sending).
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import urlparse

log = logging.getLogger("nexus.web_safety")

ConfirmFn = Callable[[str], bool]

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_BLOCKED_HOST_SUFFIXES = (".local", ".internal", ".localhost")
_BLOCKED_HOST_NAMES = frozenset({"localhost", "ip6-localhost"})


class WebSafetyError(Exception):
    """Raised when a web action is refused by the safety layer."""


@dataclass
class WebVerdict:
    ok: bool
    reason: str = ""


def _host_is_internal(host: str, resolve: bool = False) -> bool:
    """True if host points at loopback/private/link-local/reserved space."""
    if not host:
        return True
    host = host.strip("[]").lower()          # strip IPv6 brackets
    if host in _BLOCKED_HOST_NAMES or host.endswith(_BLOCKED_HOST_SUFFIXES):
        return True

    def _ip_bad(ip: str) -> bool:
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False
        return (addr.is_loopback or addr.is_private or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified)

    # Literal IP host.
    if _ip_bad(host):
        return True

    # Optional DNS resolution (guards DNS-rebinding to internal IPs).
    if resolve:
        try:
            for res in socket.getaddrinfo(host, None):
                if _ip_bad(res[4][0]):
                    return True
        except Exception:
            # Can't resolve → treat as not-obviously-internal (fetch will fail).
            return False
    return False


def is_safe_to_fetch(url: str, resolve: bool = False) -> WebVerdict:
    """Validate a URL for read-only fetching."""
    if not url or not url.strip():
        return WebVerdict(False, "empty url")
    try:
        parsed = urlparse(url.strip())
    except Exception as e:
        return WebVerdict(False, f"unparseable url: {e}")

    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        return WebVerdict(False, f"blocked scheme '{parsed.scheme}' (only http/https allowed)")
    if not parsed.hostname:
        return WebVerdict(False, "no host in url")
    if _host_is_internal(parsed.hostname, resolve=resolve):
        return WebVerdict(False, f"blocked internal/private host '{parsed.hostname}'")
    return WebVerdict(True, "ok")


def filter_fetchable(urls: list[str], resolve: bool = False) -> list[str]:
    """Return only the URLs safe to fetch (logs the ones dropped)."""
    safe = []
    for u in urls:
        v = is_safe_to_fetch(u, resolve=resolve)
        if v.ok:
            safe.append(u)
        else:
            log.info("Dropping unsafe URL %r — %s", u, v.reason)
    return safe


def is_action(method: str) -> bool:
    """True if an HTTP method changes state (needs consent)."""
    return method.upper() not in _READ_METHODS


def guard_action(description: str, confirm: Optional[ConfirmFn]) -> None:
    """Gate a state-changing web action behind consent. Raises on refusal."""
    if confirm is None or not confirm(description):
        raise WebSafetyError(f"web action not confirmed: {description}")


if __name__ == "__main__":
    for u in ["https://en.wikipedia.org/wiki/Rayleigh_scattering",
              "http://169.254.169.254/latest/meta-data/",
              "http://localhost:8080/admin", "file:///etc/passwd",
              "javascript:alert(1)", "http://10.0.0.5/"]:
        v = is_safe_to_fetch(u)
        print(f"  {'OK  ' if v.ok else 'DENY'} {u:55} {v.reason}")
