"""
core/conversation_session.py
NEXUS Conversation Session Manager — inspired by Omi's conversation lifecycle.

Tracks each conversation as a structured session:
    - Transcript (speaker turns)
    - Auto-generated title + summary
    - Extracted action items and key topics
    - Persistent SQLite storage

Usage:
    session = ConversationSessionManager()
    session.start_session()
    session.add_turn("user", "remind me to call John tomorrow")
    session.add_turn("nexus", "Got it, I'll remind you.")
    result = session.end_session()          # triggers summary + action extraction
    sessions = session.get_sessions(limit=10)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("nexus.conversation_session")

DB_PATH = Path("data/conversations.db")
SILENCE_TIMEOUT = 120  # seconds of inactivity before auto-ending a session


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT,
                summary     TEXT,
                transcript  TEXT NOT NULL DEFAULT '[]',
                topics      TEXT NOT NULL DEFAULT '[]',
                started_at  REAL NOT NULL,
                ended_at    REAL,
                duration    REAL,
                turn_count  INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS action_items (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER REFERENCES sessions(id),
                text        TEXT NOT NULL,
                category    TEXT DEFAULT 'task',
                done        INTEGER DEFAULT 0,
                created_at  REAL NOT NULL
            );
        """)


@dataclass
class Turn:
    speaker: str
    text: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"speaker": self.speaker, "text": self.text, "timestamp": self.timestamp}


class ConversationSession:
    """A single in-progress conversation session."""

    def __init__(self):
        self.id: Optional[int] = None
        self.turns: list[Turn] = []
        self.started_at: float = time.time()
        self.last_activity: float = time.time()

    def add_turn(self, speaker: str, text: str):
        self.turns.append(Turn(speaker=speaker, text=text))
        self.last_activity = time.time()

    def is_stale(self) -> bool:
        return (time.time() - self.last_activity) > SILENCE_TIMEOUT

    def transcript_text(self) -> str:
        return "\n".join(f"{t.speaker.upper()}: {t.text}" for t in self.turns)

    def word_count(self) -> int:
        return sum(len(t.text.split()) for t in self.turns)


class ConversationSessionManager:
    """
    Manages the full lifecycle of NEXUS conversation sessions.

    Inspired by Omi's conversation_capturing pipeline.
    """

    def __init__(self, llm=None):
        _init_db()
        self._active: Optional[ConversationSession] = None
        self._llm = llm
        log.info("ConversationSessionManager ready.")

    def set_llm(self, llm):
        self._llm = llm

    # ── Session lifecycle ────────────────────────────────────────────────

    def start_session(self) -> ConversationSession:
        if self._active and not self._active.is_stale():
            return self._active
        self._active = ConversationSession()
        log.info("New conversation session started.")
        return self._active

    def add_turn(self, speaker: str, text: str):
        if not text.strip():
            return
        if self._active is None or self._active.is_stale():
            self.start_session()
        self._active.add_turn(speaker, text)
        log.debug("Turn added: [%s] %s", speaker, text[:60])

    def has_active_session(self) -> bool:
        return self._active is not None and len(self._active.turns) > 0

    def end_session(self) -> Optional[dict]:
        """Close the current session, generate summary, save to DB."""
        if not self._active or not self._active.turns:
            self._active = None
            return None

        session = self._active
        self._active = None

        if session.word_count() < 5:
            log.debug("Session too short to save (< 5 words).")
            return None

        ended_at = time.time()
        duration = ended_at - session.started_at
        transcript_json = json.dumps([t.to_dict() for t in session.turns])

        title, summary, action_items, topics = self._process(session)

        with _connect() as conn:
            cur = conn.execute("""
                INSERT INTO sessions (title, summary, transcript, topics, started_at, ended_at, duration, turn_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (title, summary, transcript_json, json.dumps(topics),
                  session.started_at, ended_at, duration, len(session.turns)))
            session_id = cur.lastrowid

            for item in action_items:
                conn.execute("""
                    INSERT INTO action_items (session_id, text, category, created_at)
                    VALUES (?, ?, ?, ?)
                """, (session_id, item["text"], item.get("category", "task"), time.time()))

        log.info("Session #%d saved — %d turns, %d action items.", session_id, len(session.turns), len(action_items))

        return {
            "id": session_id,
            "title": title,
            "summary": summary,
            "action_items": action_items,
            "topics": topics,
            "duration": duration,
            "turns": len(session.turns),
        }

    def auto_end_if_stale(self) -> Optional[dict]:
        """Call periodically — ends session if user has been silent too long."""
        if self._active and self._active.is_stale():
            log.info("Auto-ending stale session.")
            return self.end_session()
        return None

    # ── Post-processing (LLM) ────────────────────────────────────────────

    def _process(self, session: ConversationSession) -> tuple[str, str, list[dict], list[str]]:
        transcript = session.transcript_text()
        title = self._generate_title(transcript)
        summary = self._generate_summary(transcript)
        action_items = self._extract_actions(transcript)
        topics = self._extract_topics(transcript)
        return title, summary, action_items, topics

    def _generate_title(self, transcript: str) -> str:
        if not self._llm:
            words = transcript.split()[:8]
            return " ".join(words).capitalize() + "..."
        try:
            prompt = (
                "Generate a short 5-7 word title for this conversation. "
                "Reply with the title only, no punctuation:\n\n" + transcript[:1000]
            )
            return self._llm.ask(prompt, max_tokens=20).strip().strip('"')
        except Exception as e:
            log.warning("Title generation failed: %s", e)
            return f"Conversation {datetime.now().strftime('%b %d %H:%M')}"

    def _generate_summary(self, transcript: str) -> str:
        if not self._llm:
            return transcript[:300] + ("..." if len(transcript) > 300 else "")
        try:
            prompt = (
                "Summarize this conversation in 2-3 sentences. Be concise and factual:\n\n"
                + transcript[:3000]
            )
            return self._llm.ask(prompt, max_tokens=120).strip()
        except Exception as e:
            log.warning("Summary generation failed: %s", e)
            return ""

    def _extract_actions(self, transcript: str) -> list[dict]:
        if not self._llm:
            return []
        try:
            prompt = (
                "Extract all action items, tasks, reminders, and goals from this conversation. "
                "Return a JSON array of objects with 'text' and 'category' (task/reminder/goal). "
                "If none, return []. Only return the JSON:\n\n" + transcript[:3000]
            )
            raw = self._llm.ask(prompt, max_tokens=300).strip()
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except Exception as e:
            log.warning("Action extraction failed: %s", e)
        return []

    def _extract_topics(self, transcript: str) -> list[str]:
        if not self._llm:
            return []
        try:
            prompt = (
                "List 3-5 key topics from this conversation as a JSON array of strings. "
                "Only return the JSON array:\n\n" + transcript[:2000]
            )
            raw = self._llm.ask(prompt, max_tokens=80).strip()
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except Exception as e:
            log.warning("Topic extraction failed: %s", e)
        return []

    # ── Retrieval ────────────────────────────────────────────────────────

    def get_sessions(self, limit: int = 20) -> list[dict]:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT id, title, summary, topics, started_at, duration, turn_count "
                "FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_session_detail(self, session_id: int) -> Optional[dict]:
        with _connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            if not row:
                return None
            actions = conn.execute(
                "SELECT text, category, done FROM action_items WHERE session_id=?",
                (session_id,)
            ).fetchall()
        result = dict(row)
        result["transcript"] = json.loads(result["transcript"])
        result["topics"] = json.loads(result["topics"])
        result["action_items"] = [dict(a) for a in actions]
        return result

    def get_pending_actions(self) -> list[dict]:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT id, session_id, text, category, created_at "
                "FROM action_items WHERE done=0 ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def complete_action(self, action_id: int):
        with _connect() as conn:
            conn.execute("UPDATE action_items SET done=1 WHERE id=?", (action_id,))

    def search_sessions(self, query: str) -> list[dict]:
        q = f"%{query}%"
        with _connect() as conn:
            rows = conn.execute(
                "SELECT id, title, summary, started_at FROM sessions "
                "WHERE title LIKE ? OR summary LIKE ? ORDER BY started_at DESC LIMIT 20",
                (q, q)
            ).fetchall()
        return [dict(r) for r in rows]
