"""
core/command_router.py
NEXUS — a small, testable command router.

`Brain._route` grew into a ~290-line if/elif chain of substring checks whose
behavior depends entirely on ordering, and where the same phrase ("what is …")
is matched in three different places. That is impossible to unit-test and easy
to break when adding a command.

This module replaces the fragile inline matching with a declarative registry.
Routes are tried in registration order (first match wins — same semantics as an
if/elif chain), but each route is data you can inspect and test in isolation,
with no dependency on Brain or any LLM.

Match kinds:
    exact(*phrases)      → the cleaned text equals one of the phrases
    prefix(*phrases)     → the text starts with one of the phrases
    contains(*phrases)   → one of the phrases appears anywhere in the text
    predicate(fn)        → fn(text) is truthy

Text is compared lowercased and stripped of surrounding whitespace and common
trailing punctuation, matching the old ``t_lower.strip()`` behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


def normalize(text: str) -> str:
    """Lowercase, strip whitespace and surrounding punctuation."""
    return text.lower().strip().strip("!?.,;:")


MatchFn = Callable[[str], bool]
Handler = Callable[[str], object]


@dataclass
class Route:
    kind: str
    test: MatchFn
    handler: Handler
    label: str = ""

    def matches(self, text: str) -> bool:
        return self.test(text)


class CommandRouter:
    """First-match-wins registry of text → handler routes."""

    def __init__(self) -> None:
        self._routes: list[Route] = []

    # ── registration ────────────────────────────────────────────────
    def exact(self, *phrases: str, to: Handler, label: str = "") -> "CommandRouter":
        keys = frozenset(normalize(p) for p in phrases)
        self._routes.append(Route("exact", lambda t: normalize(t) in keys, to, label))
        return self

    def prefix(self, *phrases: str, to: Handler, label: str = "") -> "CommandRouter":
        keys = tuple(p.lower() for p in phrases)
        self._routes.append(
            Route("prefix", lambda t: t.lower().lstrip().startswith(keys), to, label)
        )
        return self

    def contains(self, *phrases: str, to: Handler, label: str = "") -> "CommandRouter":
        keys = tuple(p.lower() for p in phrases)
        self._routes.append(
            Route("contains", lambda t: any(k in t.lower() for k in keys), to, label)
        )
        return self

    def predicate(self, fn: MatchFn, to: Handler, label: str = "") -> "CommandRouter":
        self._routes.append(Route("predicate", fn, to, label))
        return self

    # ── lookup ──────────────────────────────────────────────────────
    def find(self, text: str) -> Optional[Route]:
        """Return the first matching Route, or None."""
        for route in self._routes:
            if route.matches(text):
                return route
        return None

    def dispatch(self, text: str):
        """Run the first matching handler, or return None if nothing matched."""
        route = self.find(text)
        if route is None:
            return None
        return route.handler(text)

    def __len__(self) -> int:
        return len(self._routes)
