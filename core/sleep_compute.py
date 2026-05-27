"""
core/sleep_compute.py
NEXUS Sleep-Time Compute — Letta-inspired background memory consolidation.
Based on Letta (MemGPT v2, 2024) sleep-time compute pattern.

Runs a background thread while NEXUS is idle. It:
1. Reads recent conversation sessions.
2. Consolidates facts about the user into a persistent "Human Block".
3. Updates NEXUS's own "Persona Block" based on usage patterns.
4. Prunes stale/contradicted memories.
5. Extracts new entities into the knowledge graph.

The Human Block and Persona Block are always injected into the brain prompt
so NEXUS always knows who it's talking to without the user re-explaining.

Usage:
    sc = SleepCompute(memory=brain.memory, llm=brain.llm, kg=brain.kg)
    sc.start()     # launches background thread
    sc.stop()      # graceful shutdown

    # Read blocks for prompt injection:
    prefix = sc.build_system_prefix()
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("nexus.sleep_compute")

DB_PATH      = Path("data/memory_blocks.db")
IDLE_TRIGGER = 300   # run consolidation after 5 min of no brain activity
MIN_INTERVAL = 600   # run at most once every 10 min


_HUMAN_BLOCK_PROMPT = """You are analyzing conversation history to build a profile of the user.
Extract key facts: name, occupation, projects, preferences, goals, tools they use.
Write 3-5 concise bullet points. Only include facts you are confident about.

Conversation history:
{history}

Existing human block (update/add, don't remove confirmed facts):
{existing}

Write the updated human block (bullet points only):"""

_PERSONA_BLOCK_PROMPT = """You are NEXUS, a personal AI assistant. Based on recent interactions,
update your self-description to reflect your current capabilities and the user's preferences for
how you respond. Be concise (3-4 sentences).

Recent activity summary: {activity}
Existing persona block: {existing}

Write the updated persona block:"""

_PRUNE_PROMPT = """Review these memory entries and identify any that are:
- Contradicted by newer entries
- Outdated (no longer relevant)
- Duplicates

Return a JSON array of IDs to delete. If none, return [].
Entries: {entries}
IDs to delete:"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS blocks (
                label      TEXT PRIMARY KEY,
                value      TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS consolidation_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ran_at     REAL NOT NULL,
                sessions_processed INTEGER DEFAULT 0,
                facts_extracted    INTEGER DEFAULT 0,
                memories_pruned    INTEGER DEFAULT 0
            );
        """)
        # Seed default blocks
        for label in ("human", "persona", "task"):
            exists = conn.execute("SELECT 1 FROM blocks WHERE label=?", (label,)).fetchone()
            if not exists:
                defaults = {
                    "human":   "• User: Cyril\n• System: Linux/WSL2\n• Interests: AI, cybersecurity",
                    "persona": "I am NEXUS, a local-first personal AI assistant built by Cyril. "
                               "I handle voice commands, automation, research, and cybersecurity tasks.",
                    "task":    "",
                }
                conn.execute(
                    "INSERT INTO blocks (label, value, updated_at) VALUES (?, ?, ?)",
                    (label, defaults[label], time.time())
                )


class SleepCompute:
    """
    Background memory consolidation agent.
    Runs after periods of inactivity, consolidates sessions into durable blocks.
    """

    def __init__(self, memory=None, llm=None, kg=None, session_mgr=None):
        _init_db()
        self._memory      = memory
        self._llm         = llm
        self._kg          = kg
        self._session_mgr = session_mgr
        self._running     = False
        self._thread: Optional[threading.Thread] = None
        self._last_activity  = time.time()
        self._last_ran       = 0.0
        log.info("SleepCompute ready.")

    def set_components(self, memory=None, llm=None, kg=None, session_mgr=None):
        if memory:      self._memory      = memory
        if llm:         self._llm         = llm
        if kg:          self._kg          = kg
        if session_mgr: self._session_mgr = session_mgr

    def ping(self):
        """Call this on every brain.think() to reset the idle timer."""
        self._last_activity = time.time()

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="nexus-sleep-compute")
        self._thread.start()
        log.info("SleepCompute background thread started.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        while self._running:
            time.sleep(60)
            idle_secs = time.time() - self._last_activity
            since_last_run = time.time() - self._last_ran
            if idle_secs >= IDLE_TRIGGER and since_last_run >= MIN_INTERVAL:
                try:
                    self.consolidate()
                except Exception as e:
                    log.warning("SleepCompute consolidation error: %s", e)

    # ── Consolidation ────────────────────────────────────────────────────

    def consolidate(self):
        log.info("SleepCompute: starting consolidation pass.")
        self._last_ran = time.time()

        sessions_processed = 0
        facts_extracted    = 0
        memories_pruned    = 0

        # 1. Update Human Block from recent sessions
        if self._session_mgr:
            sessions = self._session_mgr.get_sessions(limit=5)
            if sessions:
                history_parts = []
                for s in sessions:
                    detail = self._session_mgr.get_session_detail(s["id"])
                    if detail and detail.get("transcript"):
                        turns = detail["transcript"][:20]
                        history_parts.append(
                            "\n".join(f"{t['speaker'].upper()}: {t['text']}" for t in turns)
                        )
                        sessions_processed += 1

                if history_parts:
                    history = "\n---\n".join(history_parts[:3])
                    self._update_human_block(history)

        # 2. Update Persona Block from trends
        facts_extracted = self._update_persona_block()

        # 3. Prune stale memories
        memories_pruned = self._prune_stale_memories()

        # 4. Extract entities from recent episodes into knowledge graph
        if self._memory and self._kg:
            episodes = self._memory.recent_episodes(n=20)
            for ep in episodes:
                self._kg.extract_from_text(ep["user"], source="sleep_compute")

        with _connect() as conn:
            conn.execute(
                "INSERT INTO consolidation_log (ran_at, sessions_processed, facts_extracted, memories_pruned) VALUES (?,?,?,?)",
                (time.time(), sessions_processed, facts_extracted, memories_pruned)
            )

        log.info("SleepCompute done — %d sessions, %d facts, %d pruned.",
                 sessions_processed, facts_extracted, memories_pruned)

    def _update_human_block(self, history: str):
        existing = self.read_block("human")
        if self._llm and history.strip():
            try:
                prompt = _HUMAN_BLOCK_PROMPT.format(
                    history=history[:3000], existing=existing
                )
                new_value = self._llm.ask(prompt, max_tokens=200).strip()
                if new_value:
                    self.write_block("human", new_value)
                    log.info("Human block updated.")
                    return
            except Exception as e:
                log.warning("Human block LLM update failed: %s", e)

    def _update_persona_block(self) -> int:
        if not self._llm:
            return 0
        try:
            from core.trends import TrendsTracker
            tr = TrendsTracker()
            activity = tr.daily_summary()
            existing = self.read_block("persona")
            prompt = _PERSONA_BLOCK_PROMPT.format(activity=activity, existing=existing)
            new_value = self._llm.ask(prompt, max_tokens=120).strip()
            if new_value:
                self.write_block("persona", new_value)
                return 1
        except Exception as e:
            log.warning("Persona block update failed: %s", e)
        return 0

    def _prune_stale_memories(self) -> int:
        if not self._memory or not self._llm:
            return 0
        try:
            # Get recent long-term memories
            with sqlite3.connect("data/nexus_memory.db") as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT rowid, key, value FROM long_term ORDER BY timestamp DESC LIMIT 30"
                ).fetchall()

            if len(rows) < 5:
                return 0

            entries_text = json.dumps([
                {"id": r["rowid"], "key": r["key"], "value": r["value"][:100]}
                for r in rows
            ])
            prompt = _PRUNE_PROMPT.format(entries=entries_text[:2000])
            raw = self._llm.ask(prompt, max_tokens=100).strip()

            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start < 0 or end <= start:
                return 0

            ids_to_delete = json.loads(raw[start:end])
            if not isinstance(ids_to_delete, list):
                return 0

            pruned = 0
            with sqlite3.connect("data/nexus_memory.db") as conn:
                for rid in ids_to_delete[:5]:  # safety cap
                    try:
                        conn.execute("DELETE FROM long_term WHERE rowid=?", (int(rid),))
                        pruned += 1
                    except Exception:
                        pass
            return pruned
        except Exception as e:
            log.warning("Memory pruning failed: %s", e)
            return 0

    # ── Block API ────────────────────────────────────────────────────────

    def read_block(self, label: str) -> str:
        with _connect() as conn:
            row = conn.execute("SELECT value FROM blocks WHERE label=?", (label,)).fetchone()
        return row["value"] if row else ""

    def write_block(self, label: str, value: str):
        with _connect() as conn:
            conn.execute("""
                INSERT INTO blocks (label, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(label) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """, (label, value, time.time()))

    def all_blocks(self) -> dict:
        with _connect() as conn:
            rows = conn.execute("SELECT label, value, updated_at FROM blocks").fetchall()
        return {r["label"]: r["value"] for r in rows}

    def build_system_prefix(self) -> str:
        """
        Returns a system context prefix for injection at the top of every brain prompt.
        Contains Human Block + Persona Block for persistent identity awareness.
        """
        human   = self.read_block("human").strip()
        persona = self.read_block("persona").strip()
        task    = self.read_block("task").strip()

        lines = []
        if persona:
            lines.append(f"[PERSONA]\n{persona}")
        if human:
            lines.append(f"[USER CONTEXT]\n{human}")
        if task:
            lines.append(f"[CURRENT TASK]\n{task}")

        return "\n\n".join(lines) + "\n\n" if lines else ""

    def last_consolidation(self) -> Optional[dict]:
        with _connect() as conn:
            row = conn.execute(
                "SELECT * FROM consolidation_log ORDER BY ran_at DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None
