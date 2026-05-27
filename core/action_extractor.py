"""
core/action_extractor.py
NEXUS Action Item & Goal Extractor — inspired by Omi's action_items.py + goals.py routers.

Given any text (conversation transcript, single message, research output),
extracts structured:
    - Action items  (concrete tasks to do)
    - Goals         (longer-term objectives)
    - Reminders     (time-bound items)

All results are persisted to SQLite for later retrieval and completion tracking.

Usage:
    ae = ActionExtractor(llm=brain.llm)
    results = ae.extract_from_text("I need to call John tomorrow and finish the report by Friday")
    # → {"action_items": [...], "goals": [...], "reminders": [...]}

    ae.get_pending()        # all unfinished items across all sources
    ae.mark_done(item_id)
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("nexus.action_extractor")

DB_PATH = Path("data/actions.db")

CATEGORIES = ("task", "goal", "reminder")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS items (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                text        TEXT NOT NULL,
                category    TEXT DEFAULT 'task',
                source      TEXT DEFAULT 'conversation',
                context     TEXT,
                done        INTEGER DEFAULT 0,
                created_at  REAL NOT NULL,
                done_at     REAL
            );
        """)


_EXTRACTION_PROMPT = """Extract all action items, goals, and reminders from the following text.

Rules:
- action items: concrete things to DO (call someone, write something, fix something)
- goals: longer-term objectives (learn X, build Y, improve Z)
- reminders: time-bound items (do X tomorrow, meet Y at 3pm, deadline on Friday)

Return ONLY a valid JSON object with three arrays:
{{
  "action_items": [{{"text": "...", "deadline": "..." or null}}],
  "goals":        [{{"text": "...", "timeframe": "..." or null}}],
  "reminders":    [{{"text": "...", "when": "..." or null}}]
}}

Text:
{text}"""


# ── Rule-based fallback patterns ─────────────────────────────────────────

_TASK_PATTERNS = [
    r"\b(?:need to|have to|must|should|will|gonna|going to)\s+(.+?)(?:\.|,|$)",
    r"\b(?:remind me to|don't forget to|remember to)\s+(.+?)(?:\.|,|$)",
    r"\b(?:todo|to-do|to do):\s*(.+?)(?:\.|$)",
]

_REMINDER_PATTERNS = [
    r"\b(?:tomorrow|today|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b.{5,60}",
    r"\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b.{5,40}",
    r"\bby\s+(?:friday|monday|end of day|eod|eow|next week)\b.{0,40}",
]


def _rule_based_extract(text: str) -> dict:
    items = {"action_items": [], "goals": [], "reminders": []}
    lower = text.lower()

    for pat in _TASK_PATTERNS:
        for m in re.finditer(pat, lower):
            task = m.group(1).strip()
            if 5 < len(task) < 120:
                items["action_items"].append({"text": task.capitalize(), "deadline": None})

    for pat in _REMINDER_PATTERNS:
        for m in re.finditer(pat, lower):
            reminder = m.group(0).strip()
            if 5 < len(reminder) < 120:
                items["reminders"].append({"text": reminder.capitalize(), "when": None})

    return items


class ActionExtractor:
    """Extracts and persists action items, goals, and reminders from text."""

    def __init__(self, llm=None):
        _init_db()
        self._llm = llm
        log.info("ActionExtractor ready.")

    def set_llm(self, llm):
        self._llm = llm

    # ── Core extraction ──────────────────────────────────────────────────

    def extract_from_text(self, text: str, source: str = "conversation") -> dict:
        """Extract and persist action items from any text block."""
        if not text.strip():
            return {"action_items": [], "goals": [], "reminders": []}

        extracted = self._extract(text)
        self._persist(extracted, source=source, context=text[:500])

        total = sum(len(v) for v in extracted.values())
        log.info("Extracted %d items from %s.", total, source)
        return extracted

    def _extract(self, text: str) -> dict:
        if self._llm:
            try:
                prompt = _EXTRACTION_PROMPT.format(text=text[:3000])
                raw = self._llm.ask(prompt, max_tokens=400).strip()
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    return json.loads(raw[start:end])
            except Exception as e:
                log.warning("LLM extraction failed, falling back to rules: %s", e)

        return _rule_based_extract(text)

    def _persist(self, extracted: dict, source: str, context: str):
        rows = []
        for item in extracted.get("action_items", []):
            rows.append((item["text"], "task", source, context))
        for item in extracted.get("goals", []):
            rows.append((item["text"], "goal", source, context))
        for item in extracted.get("reminders", []):
            rows.append((item["text"], "reminder", source, context))

        if not rows:
            return

        with _connect() as conn:
            conn.executemany("""
                INSERT INTO items (text, category, source, context, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, [(r[0], r[1], r[2], r[3], time.time()) for r in rows])

    # ── Retrieval ────────────────────────────────────────────────────────

    def get_pending(self, category: Optional[str] = None) -> list[dict]:
        """Return all unfinished items, optionally filtered by category."""
        with _connect() as conn:
            if category:
                rows = conn.execute(
                    "SELECT * FROM items WHERE done=0 AND category=? ORDER BY created_at DESC",
                    (category,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM items WHERE done=0 ORDER BY created_at DESC"
                ).fetchall()
        return [dict(r) for r in rows]

    def get_all(self, limit: int = 50) -> list[dict]:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT * FROM items ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_done(self, item_id: int):
        with _connect() as conn:
            conn.execute(
                "UPDATE items SET done=1, done_at=? WHERE id=?",
                (time.time(), item_id)
            )
        log.info("Item #%d marked done.", item_id)

    def add_manual(self, text: str, category: str = "task") -> int:
        """Manually add an item (e.g. from 'remember to X' command)."""
        if category not in CATEGORIES:
            category = "task"
        with _connect() as conn:
            cur = conn.execute(
                "INSERT INTO items (text, category, source, created_at) VALUES (?, ?, 'manual', ?)",
                (text, category, time.time())
            )
            return cur.lastrowid

    def summary_text(self) -> str:
        """One-line summary of pending items for status display."""
        pending = self.get_pending()
        if not pending:
            return "No pending action items."
        tasks    = [i for i in pending if i["category"] == "task"]
        goals    = [i for i in pending if i["category"] == "goal"]
        reminders = [i for i in pending if i["category"] == "reminder"]
        parts = []
        if tasks:     parts.append(f"{len(tasks)} task{'s' if len(tasks)>1 else ''}")
        if reminders: parts.append(f"{len(reminders)} reminder{'s' if len(reminders)>1 else ''}")
        if goals:     parts.append(f"{len(goals)} goal{'s' if len(goals)>1 else ''}")
        return "Pending: " + ", ".join(parts) + "."
