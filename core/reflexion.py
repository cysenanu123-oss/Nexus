"""
core/reflexion.py
NEXUS Reflexion Engine — Verbal Reinforcement Learning.
Based on Shinn et al. NeurIPS 2023 (arXiv:2303.11366).

When NEXUS gives a bad response, instead of just logging it, it:
1. Writes a natural-language reflection on WHY it failed.
2. Stores that reflection in SQLite.
3. Next time a similar input arrives, injects the reflection into context.

This gives NEXUS the ability to learn from mistakes without retraining.

Usage:
    ref = ReflexionEngine(llm=brain.llm)

    # After a bad response:
    ref.reflect(user_input="how do I open a port?",
                bad_response="I don't know",
                correction="Use: sudo ufw allow <port>/tcp")

    # Before generating a response:
    hints = ref.get_hints("how do I open a port?", top_k=2)
    # → ["I previously said 'I don't know' but the answer is to use ufw allow..."]
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("nexus.reflexion")

DB_PATH = Path("data/reflexion.db")

_REFLECT_PROMPT = """You are a self-improving AI assistant. You gave a bad response to a user.
Analyze what went wrong and write a concise reflection (2-3 sentences) that captures:
1. What specifically was wrong or missing.
2. What the correct approach should have been.
3. A rule to remember for next time.

User input: {user_input}
Bad response: {bad_response}
Correct answer: {correction}

Write the reflection:"""

_SIMILARITY_THRESHOLD = 0.75


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS reflections (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input   TEXT NOT NULL,
                bad_response TEXT NOT NULL,
                correction   TEXT NOT NULL,
                reflection   TEXT NOT NULL,
                intent_hint  TEXT DEFAULT '',
                times_used   INTEGER DEFAULT 0,
                created_at   REAL NOT NULL
            );
        """)


class ReflexionEngine:
    """
    Verbal reinforcement learning for NEXUS.
    Stores and retrieves failure-driven reflections.
    """

    def __init__(self, llm=None):
        _init_db()
        self._llm = llm
        # Lazy-load vector memory for semantic retrieval
        self._vector = None
        log.info("ReflexionEngine ready — %d reflections stored.", self._count())

    def set_llm(self, llm):
        self._llm = llm

    def _get_vector(self):
        if self._vector is None:
            try:
                from core.vector_memory import VectorMemory
                self._vector = VectorMemory()
            except Exception:
                pass
        return self._vector

    # ── Core API ─────────────────────────────────────────────────────────

    def reflect(self, user_input: str, bad_response: str,
                correction: str = "", intent_hint: str = "") -> str:
        """
        Generate and store a reflection on a failed response.
        Call this when the user marks a response as bad.
        """
        reflection = self._generate_reflection(user_input, bad_response, correction)

        with _connect() as conn:
            cur = conn.execute("""
                INSERT INTO reflections (user_input, bad_response, correction,
                                         reflection, intent_hint, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_input, bad_response, correction, reflection, intent_hint, time.time()))
            rid = cur.lastrowid

        # Also store in vector memory for semantic retrieval
        vm = self._get_vector()
        if vm and vm.ready:
            vm.store(
                f"FAILURE: {user_input}\nREFLECTION: {reflection}",
                metadata={"type": "reflection", "reflection_id": rid},
                doc_id=f"reflection_{rid}"
            )

        log.info("Reflection #%d stored for: %r", rid, user_input[:60])
        return reflection

    def get_hints(self, user_input: str, top_k: int = 2) -> list[str]:
        """
        Retrieve relevant past reflections to inject into the next prompt.
        Returns a list of reflection strings, or empty list if none relevant.
        """
        hints = []

        # Try semantic retrieval first
        vm = self._get_vector()
        if vm and vm.ready:
            results = vm.search(user_input, n=top_k, where={"type": "reflection"})
            for r in results:
                if r["distance"] < (1 - _SIMILARITY_THRESHOLD):
                    rid = r["metadata"].get("reflection_id")
                    if rid:
                        row = self._get_reflection(int(rid))
                        if row:
                            hints.append(row["reflection"])
                            self._increment_usage(int(rid))

        # Fallback: keyword search
        if not hints:
            words = set(user_input.lower().split())
            with _connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM reflections ORDER BY created_at DESC LIMIT 50"
                ).fetchall()
            for row in rows:
                haystack = row["user_input"].lower()
                overlap = sum(1 for w in words if w in haystack)
                if overlap >= 2:
                    hints.append(row["reflection"])
                    self._increment_usage(row["id"])
                    if len(hints) >= top_k:
                        break

        return hints

    def build_context_prefix(self, user_input: str) -> str:
        """
        Returns a prompt prefix with relevant past failures to inject before generating.
        Empty string if no relevant reflections.
        """
        hints = self.get_hints(user_input)
        if not hints:
            return ""
        lines = ["[Past failure notes — apply these lessons:]"]
        for i, h in enumerate(hints, 1):
            lines.append(f"  {i}. {h}")
        return "\n".join(lines) + "\n\n"

    def all_reflections(self, limit: int = 20) -> list[dict]:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT id, user_input, reflection, times_used, created_at "
                "FROM reflections ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Internals ─────────────────────────────────────────────────────────

    def _generate_reflection(self, user_input: str, bad_response: str, correction: str) -> str:
        if self._llm:
            try:
                prompt = _REFLECT_PROMPT.format(
                    user_input=user_input[:500],
                    bad_response=bad_response[:500],
                    correction=correction[:500] or "(user did not specify)"
                )
                return self._llm.ask(prompt, max_tokens=150).strip()
            except Exception as e:
                log.warning("LLM reflection failed: %s", e)

        # Rule-based fallback
        if correction:
            return (f"When asked '{user_input[:80]}', I gave an inadequate response. "
                    f"The correct approach was: {correction[:200]}. "
                    f"Remember this pattern for similar future queries.")
        return (f"My response to '{user_input[:80]}' was marked as incorrect. "
                f"I should reason more carefully about this type of query.")

    def _get_reflection(self, rid: int) -> Optional[dict]:
        with _connect() as conn:
            row = conn.execute("SELECT * FROM reflections WHERE id=?", (rid,)).fetchone()
        return dict(row) if row else None

    def _increment_usage(self, rid: int):
        with _connect() as conn:
            conn.execute("UPDATE reflections SET times_used=times_used+1 WHERE id=?", (rid,))

    def _count(self) -> int:
        try:
            with _connect() as conn:
                return conn.execute("SELECT COUNT(*) FROM reflections").fetchone()[0]
        except Exception:
            return 0
