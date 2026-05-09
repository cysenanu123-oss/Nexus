"""
core/planner.py
NEXUS Planner — breaks multi-step commands into ordered task sequences.

Examples:
    "open firefox and go to youtube"
    → [Intent(open_application, firefox), Intent(open_url, youtube.com)]

    "search hackthebox then open terminal"
    → [Intent(search_web, hackthebox), Intent(open_application, terminal)]
"""

import re
import logging
from typing import Optional

from core.intent_engine import IntentEngine, Intent

log = logging.getLogger("nexus.planner")

# Phrases that signal a multi-step command
STEP_SPLITTERS = re.compile(
    r"\b(and then|then|after that|and also|also|and|,)\b",
    re.IGNORECASE
)

# Known URL shortcuts
URL_SHORTCUTS: dict[str, str] = {
    "youtube":    "https://youtube.com",
    "github":     "https://github.com",
    "google":     "https://google.com",
    "hackthebox": "https://hackthebox.com",
    "tryhackme":  "https://tryhackme.com",
    "gmail":      "https://mail.google.com",
    "reddit":     "https://reddit.com",
    "stackoverflow": "https://stackoverflow.com",
    "wikipedia":  "https://wikipedia.org",
    "twitter":    "https://twitter.com",
    "linkedin":   "https://linkedin.com",
    "eventconnect": "https://eventconnectgh.com",
}


class Planner:
    """
    Splits compound commands and returns an ordered list of Intents.

    Usage:
        planner = Planner()
        steps = planner.plan("open firefox and go to youtube")
        for step in steps:
            result = dispatcher.dispatch(step)
    """

    def __init__(self):
        self.engine = IntentEngine(backend="rules")
        log.info("Planner ready.")

    def plan(self, text: str) -> list[Intent]:
        """
        Parse a command into 1 or more ordered Intents.
        Single commands return a list with one Intent.
        """
        # Try to split into sub-commands
        segments = self._split(text)

        if len(segments) == 1:
            return [self._parse_segment(segments[0])]

        log.info(f"Multi-step plan: {len(segments)} steps")
        intents = []
        for seg in segments:
            seg = seg.strip()
            if seg:
                intent = self._parse_segment(seg)
                intents.append(intent)
                log.debug(f"  Step: {intent}")

        return intents

    def _split(self, text: str) -> list[str]:
        """Split compound command on conjunctions."""
        parts = STEP_SPLITTERS.split(text)
        # Filter out the splitter words themselves
        splitter_words = {"and then", "then", "after that",
                          "and also", "also", "and", ","}
        return [p.strip() for p in parts
                if p.strip().lower() not in splitter_words and p.strip()]

    def _parse_segment(self, text: str) -> Intent:
        """
        Parse a single segment. Applies URL shortcuts for navigation intents.
        """
        intent = self.engine.parse(text)

        # Expand URL shortcuts
        if intent.intent == "open_url" and intent.target:
            expanded = URL_SHORTCUTS.get(intent.target.lower().strip())
            if expanded:
                intent.target = expanded

        # "go to youtube" → open_url even if parsed as open_application
        if intent.intent == "open_application" and intent.target:
            shortcut = URL_SHORTCUTS.get(intent.target.lower().strip())
            if shortcut:
                intent.intent = "open_url"
                intent.target = shortcut

        return intent

    def format_plan(self, intents: list[Intent]) -> str:
        """Human-readable plan summary."""
        if len(intents) == 1:
            return str(intents[0])
        lines = [f"  {i+1}. {str(intent)}" for i, intent in enumerate(intents)]
        return "Plan:\n" + "\n".join(lines)
