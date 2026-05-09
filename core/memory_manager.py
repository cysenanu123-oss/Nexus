"""
core/memory_manager.py
NEXUS Memory System — two layers:
  - Short-term : in-memory dict, lives for the session
  - Long-term  : SQLite database, persists across sessions

Training data is also written here for future fine-tuning.
"""

import sqlite3
import json
import time
import logging
import os
from typing import Optional, Any

log = logging.getLogger("nexus.memory")

DB_PATH      = "data/nexus_memory.db"
TRAINING_LOG = "data/training_pairs.jsonl"


def _connect() -> sqlite3.Connection:
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS long_term (
                key       TEXT PRIMARY KEY,
                value     TEXT NOT NULL,
                category  TEXT DEFAULT 'general',
                timestamp REAL
            );

            CREATE TABLE IF NOT EXISTS episodes (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_text TEXT NOT NULL,
                nexus_text TEXT NOT NULL,
                intent    TEXT,
                timestamp REAL
            );

            CREATE TABLE IF NOT EXISTS facts (
                subject   TEXT,
                predicate TEXT,
                object    TEXT,
                timestamp REAL,
                PRIMARY KEY (subject, predicate)
            );
        """)


class MemoryManager:
    """
    Two-layer memory system.

    Short-term  → session dict, fast, volatile
    Long-term   → SQLite, persistent across restarts
    Training    → logs (input, output) pairs for future fine-tuning
    """

    def __init__(self):
        _init_db()
        self._short: dict[str, Any] = {}
        log.info("MemoryManager ready.")

    # ── Short-term (session) ─────────────────────────────────────────────

    def remember_now(self, key: str, value: Any):
        """Store in session memory only."""
        self._short[key.lower()] = value
        log.debug(f"Short-term: {key!r} = {value!r}")

    def recall_now(self, key: str) -> Optional[Any]:
        return self._short.get(key.lower())

    def short_term_all(self) -> dict:
        return dict(self._short)

    # ── Long-term (SQLite) ───────────────────────────────────────────────

    def remember(self, key: str, value: Any, category: str = "general"):
        """Persist a key-value fact to SQLite."""
        serialized = json.dumps(value) if not isinstance(value, str) else value
        with _connect() as conn:
            conn.execute("""
                INSERT INTO long_term (key, value, category, timestamp)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    category=excluded.category,
                    timestamp=excluded.timestamp
            """, (key.lower(), serialized, category, time.time()))
        log.debug(f"Long-term stored: {key!r}")

    def recall(self, key: str) -> Optional[str]:
        with _connect() as conn:
            row = conn.execute(
                "SELECT value FROM long_term WHERE key = ?", (key.lower(),)
            ).fetchone()
        return row["value"] if row else None

    def recall_by_category(self, category: str) -> list[dict]:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT key, value FROM long_term WHERE category = ?", (category,)
            ).fetchall()
        return [{"key": r["key"], "value": r["value"]} for r in rows]

    def search_memory(self, query: str) -> list[dict]:
        """Fuzzy keyword search across long-term memory."""
        with _connect() as conn:
            rows = conn.execute(
                "SELECT key, value, category FROM long_term WHERE key LIKE ? OR value LIKE ?",
                (f"%{query}%", f"%{query}%")
            ).fetchall()
        return [{"key": r["key"], "value": r["value"], "category": r["category"]} for r in rows]

    def forget(self, key: str) -> bool:
        with _connect() as conn:
            cur = conn.execute("DELETE FROM long_term WHERE key = ?", (key.lower(),))
        return cur.rowcount > 0

    # ── Fact triples (subject → predicate → object) ──────────────────────

    def store_fact(self, subject: str, predicate: str, obj: str):
        """
        Store structured facts.
        Example: store_fact("cyril", "uses", "kali linux")
        """
        with _connect() as conn:
            conn.execute("""
                INSERT INTO facts (subject, predicate, object, timestamp)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(subject, predicate) DO UPDATE SET
                    object=excluded.object,
                    timestamp=excluded.timestamp
            """, (subject.lower(), predicate.lower(), obj.lower(), time.time()))

    def query_fact(self, subject: str, predicate: str) -> Optional[str]:
        with _connect() as conn:
            row = conn.execute(
                "SELECT object FROM facts WHERE subject=? AND predicate=?",
                (subject.lower(), predicate.lower())
            ).fetchone()
        return row["object"] if row else None

    # ── Episode memory (conversation history) ────────────────────────────

    def log_episode(self, user_text: str, nexus_text: str, intent: str = ""):
        with _connect() as conn:
            conn.execute("""
                INSERT INTO episodes (user_text, nexus_text, intent, timestamp)
                VALUES (?, ?, ?, ?)
            """, (user_text, nexus_text, intent, time.time()))

    def recent_episodes(self, n: int = 10) -> list[dict]:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT user_text, nexus_text, intent FROM episodes ORDER BY timestamp DESC LIMIT ?",
                (n,)
            ).fetchall()
        return [{"user": r["user_text"], "nexus": r["nexus_text"], "intent": r["intent"]}
                for r in reversed(rows)]

    # ── Training data collection ─────────────────────────────────────────

    def log_training_pair(self, user_input: str, nexus_output: str,
                          intent: str = "", quality: float = 1.0):
        """
        Every good (input, output) pair is written to JSONL for future training.
        quality: 1.0 = confirmed good, 0.5 = uncertain, 0.0 = bad (to exclude)
        """
        os.makedirs("data", exist_ok=True)
        pair = {
            "input":     user_input,
            "output":    nexus_output,
            "intent":    intent,
            "quality":   quality,
            "timestamp": time.time(),
        }
        with open(TRAINING_LOG, "a") as f:
            f.write(json.dumps(pair) + "\n")
