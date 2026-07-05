"""Tests for core/web_safety.py — read-freely / act-with-consent + SSRF guard."""

import pytest

from core.web_safety import (
    is_safe_to_fetch, filter_fetchable, is_action, guard_action, WebSafetyError,
)


SAFE_URLS = [
    "https://en.wikipedia.org/wiki/Rayleigh_scattering",
    "http://example.com/page",
    "https://news.ycombinator.com",
]

UNSAFE_URLS = [
    "http://169.254.169.254/latest/meta-data/",   # cloud metadata (SSRF)
    "http://localhost:8080/admin",
    "http://127.0.0.1/",
    "http://10.0.0.5/",
    "http://192.168.1.1/",
    "http://172.16.0.1/",
    "http://[::1]/",
    "https://myrouter.local/",
    "http://service.internal/api",
    "file:///etc/passwd",
    "javascript:alert(1)",
    "data:text/html,<script>",
    "ftp://ftp.example.com/x",
    "",
]


@pytest.mark.parametrize("url", SAFE_URLS)
def test_safe_urls_allowed(url):
    assert is_safe_to_fetch(url).ok


@pytest.mark.parametrize("url", UNSAFE_URLS)
def test_unsafe_urls_blocked(url):
    assert not is_safe_to_fetch(url).ok


def test_filter_fetchable_drops_unsafe():
    mixed = ["https://good.com", "http://127.0.0.1", "file:///x", "https://also-good.org"]
    assert filter_fetchable(mixed) == ["https://good.com", "https://also-good.org"]


@pytest.mark.parametrize("method,expect_action", [
    ("GET", False), ("HEAD", False), ("OPTIONS", False),
    ("POST", True), ("PUT", True), ("PATCH", True), ("DELETE", True),
    ("post", True), ("get", False),
])
def test_is_action(method, expect_action):
    assert is_action(method) is expect_action


def test_guard_action_requires_confirmation():
    with pytest.raises(WebSafetyError):
        guard_action("post a comment", confirm=None)
    with pytest.raises(WebSafetyError):
        guard_action("post a comment", confirm=lambda d: False)
    # Approving does not raise.
    guard_action("post a comment", confirm=lambda d: True)
