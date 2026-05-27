"""
core/persona.py
NEXUS Persona System — 1,839 expert role prompts from prompts.chat.

Lets NEXUS instantly switch into any expert persona:
  "act as a cyber security specialist"
  "be a linux terminal"
  "switch to python debugger mode"

The active persona is injected into every LLM prompt as a system prefix,
changing how NEXUS responds until the persona is cleared.

Data source: /data/personas.json (exported from prompts.chat, MIT licensed)

Usage:
    pm = PersonaManager(llm=brain.llm)
    pm.activate("cyber security specialist")   # fuzzy match
    prefix = pm.system_prefix()               # inject into prompts
    pm.clear()                                 # back to default NEXUS
    pm.search("linux")                         # find matching personas
    pm.list_categories()                       # grouped by type
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger("nexus.persona")

PERSONAS_PATH = Path("data/personas.json")

_NEXUS_DEFAULT = (
    "You are NEXUS, a personal AI assistant built by Cyril. "
    "You are local-first, privacy-focused, and highly capable across "
    "coding, cybersecurity, research, and automation tasks. "
    "Be concise, direct, and technically accurate."
)

_NEXUS_KEYWORDS = {
    "cyber": ["cyber", "security", "hacker", "penetration", "osint", "malware", "forensic", "cve", "exploit"],
    "code": ["developer", "programmer", "python", "javascript", "sql", "terminal", "debugger", "linux", "git", "bash", "rust"],
    "research": ["researcher", "scientist", "analyst", "academician", "data scientist", "statistician"],
    "assistant": ["coach", "advisor", "consultant", "tutor", "teacher", "mentor"],
}


class PersonaManager:
    """
    Manages NEXUS expert personas sourced from prompts.chat (1,839 roles).
    """

    def __init__(self, llm=None):
        self._llm      = llm
        self._active   : Optional[dict] = None
        self._personas : list[dict]     = []
        self._load()
        log.info("PersonaManager ready — %d personas loaded.", len(self._personas))

    def _load(self):
        if not PERSONAS_PATH.exists():
            log.warning("personas.json not found at %s", PERSONAS_PATH)
            return
        try:
            with open(PERSONAS_PATH) as f:
                self._personas = json.load(f)
        except Exception as e:
            log.warning("Failed to load personas: %s", e)

    # ── Activation ───────────────────────────────────────────────────────

    def activate(self, query: str) -> Optional[dict]:
        """Fuzzy-match query to a persona and activate it."""
        match = self._find_best(query)
        if match:
            self._active = match
            log.info("Persona activated: [%s]", match["act"])
            return match
        return None

    def activate_by_name(self, name: str) -> Optional[dict]:
        """Exact name lookup."""
        for p in self._personas:
            if p["act"].lower() == name.lower():
                self._active = p
                return p
        return self.activate(name)

    def clear(self):
        """Return to default NEXUS persona."""
        self._active = None
        log.info("Persona cleared — back to NEXUS default.")

    @property
    def active(self) -> Optional[dict]:
        return self._active

    @property
    def active_name(self) -> str:
        return self._active["act"] if self._active else "NEXUS"

    # ── Prompt injection ─────────────────────────────────────────────────

    def system_prefix(self) -> str:
        """
        Returns the system prompt for the active persona.
        Inject this at the top of every LLM call when a persona is active.
        """
        if not self._active:
            return _NEXUS_DEFAULT
        return self._active["prompt"]

    def wrap_prompt(self, user_input: str) -> str:
        """
        Wraps user input with the active persona's system context.
        Use when calling LLM directly (not via brain._understand).
        """
        if not self._active:
            return user_input
        return f"{self._active['prompt']}\n\n{user_input}"

    # ── Search & discovery ───────────────────────────────────────────────

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Find personas matching query string."""
        q = query.lower()
        scored = []
        for p in self._personas:
            haystack = (p["act"] + " " + p["prompt"][:200]).lower()
            score = sum(1 for w in q.split() if w in haystack)
            if score > 0:
                scored.append((score, p))
        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored[:limit]]

    def list_categories(self) -> dict[str, list[str]]:
        """Group all personas by NEXUS module category."""
        result = {k: [] for k in _NEXUS_KEYWORDS}
        result["other"] = []
        for p in self._personas:
            placed = False
            name = p["act"].lower()
            prompt = p["prompt"][:100].lower()
            for cat, keys in _NEXUS_KEYWORDS.items():
                if any(k in name or k in prompt for k in keys):
                    result[cat].append(p["act"])
                    placed = True
                    break
            if not placed:
                result["other"].append(p["act"])
        return result

    def cyber_personas(self) -> list[dict]:
        return [p for p in self._personas
                if any(k in p["act"].lower() for k in _NEXUS_KEYWORDS["cyber"])]

    def code_personas(self) -> list[dict]:
        return [p for p in self._personas
                if any(k in p["act"].lower() for k in _NEXUS_KEYWORDS["code"])]

    def count(self) -> int:
        return len(self._personas)

    def summary(self) -> str:
        cats = self.list_categories()
        parts = [f"{k}: {len(v)}" for k, v in cats.items() if v]
        return f"{self.count()} personas — " + ", ".join(parts)

    # ── Internals ─────────────────────────────────────────────────────────

    def _find_best(self, query: str) -> Optional[dict]:
        q = query.lower().strip()
        # Exact match first
        for p in self._personas:
            if p["act"].lower() == q:
                return p
        # Substring match on name
        for p in self._personas:
            if q in p["act"].lower() or p["act"].lower() in q:
                return p
        # Word overlap
        words = set(q.split())
        best_score, best = 0, None
        for p in self._personas:
            name_words = set(p["act"].lower().split())
            score = len(words & name_words)
            if score > best_score:
                best_score, best = score, p
        return best if best_score > 0 else None
