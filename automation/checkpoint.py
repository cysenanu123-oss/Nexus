"""
automation/checkpoint.py
NEXUS Durable Execution — Checkpoint & Replay for long automation tasks.
Based on Phase 15, Lesson 12 (Temporal/workflow orchestration pattern).

Every step result is checkpointed to SQLite. If a multi-step automation
crashes or is interrupted, it resumes from the last completed step
instead of restarting from zero.

Usage:
    cp = CheckpointStore()

    # Before running a plan:
    run_id = cp.start_run(instruction="open firefox and go to github.com")

    # After each step:
    cp.save_step(run_id, step_index=0, step_desc="open firefox",
                 success=True, output="Done")

    # On crash/resume:
    state = cp.load_run(run_id)
    completed = {s["step_index"] for s in state["steps"] if s["success"]}
    # → skip already-done steps

    # On success:
    cp.finish_run(run_id, success=True)

    # List recent runs:
    cp.list_runs(limit=10)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("nexus.automation.checkpoint")

DB_PATH = Path("data/automation_checkpoints.db")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                instruction  TEXT NOT NULL,
                status       TEXT DEFAULT 'running',
                started_at   REAL NOT NULL,
                finished_at  REAL,
                step_count   INTEGER DEFAULT 0,
                success      INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS steps (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id       INTEGER REFERENCES runs(id),
                step_index   INTEGER NOT NULL,
                step_desc    TEXT NOT NULL,
                success      INTEGER NOT NULL,
                output       TEXT DEFAULT '',
                error        TEXT DEFAULT '',
                elapsed_sec  REAL DEFAULT 0.0,
                saved_at     REAL NOT NULL,
                UNIQUE(run_id, step_index)
            );
        """)


class CheckpointStore:
    """Durable checkpoint store for automation runs."""

    def __init__(self):
        _init_db()

    # ── Run lifecycle ────────────────────────────────────────────────────

    def start_run(self, instruction: str, step_count: int = 0) -> int:
        with _connect() as conn:
            cur = conn.execute(
                "INSERT INTO runs (instruction, started_at, step_count) VALUES (?, ?, ?)",
                (instruction, time.time(), step_count)
            )
            run_id = cur.lastrowid
        log.info("Checkpoint run #%d started: %r", run_id, instruction[:60])
        return run_id

    def save_step(self, run_id: int, step_index: int, step_desc: str,
                  success: bool, output: str = "", error: str = "",
                  elapsed_sec: float = 0.0):
        with _connect() as conn:
            conn.execute("""
                INSERT INTO steps (run_id, step_index, step_desc, success, output, error, elapsed_sec, saved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, step_index) DO UPDATE SET
                    success=excluded.success, output=excluded.output,
                    error=excluded.error, elapsed_sec=excluded.elapsed_sec,
                    saved_at=excluded.saved_at
            """, (run_id, step_index, step_desc, int(success),
                  output[:2000], error[:500], elapsed_sec, time.time()))
        log.debug("Step %d checkpointed (run #%d, success=%s)", step_index, run_id, success)

    def finish_run(self, run_id: int, success: bool):
        with _connect() as conn:
            conn.execute(
                "UPDATE runs SET status=?, finished_at=?, success=? WHERE id=?",
                ("done" if success else "failed", time.time(), int(success), run_id)
            )

    # ── Resume ───────────────────────────────────────────────────────────

    def load_run(self, run_id: int) -> Optional[dict]:
        with _connect() as conn:
            run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if not run:
                return None
            steps = conn.execute(
                "SELECT * FROM steps WHERE run_id=? ORDER BY step_index ASC", (run_id,)
            ).fetchall()
        result = dict(run)
        result["steps"] = [dict(s) for s in steps]
        return result

    def completed_steps(self, run_id: int) -> set[int]:
        """Return set of step indices that completed successfully."""
        with _connect() as conn:
            rows = conn.execute(
                "SELECT step_index FROM steps WHERE run_id=? AND success=1", (run_id,)
            ).fetchall()
        return {r["step_index"] for r in rows}

    def find_incomplete_run(self, instruction: str) -> Optional[int]:
        """
        Check if there's an unfinished run for this exact instruction.
        Returns run_id if found, None otherwise.
        """
        with _connect() as conn:
            row = conn.execute(
                "SELECT id FROM runs WHERE instruction=? AND status='running' "
                "ORDER BY started_at DESC LIMIT 1",
                (instruction,)
            ).fetchone()
        return row["id"] if row else None

    # ── History ──────────────────────────────────────────────────────────

    def list_runs(self, limit: int = 10) -> list[dict]:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT id, instruction, status, started_at, success, step_count "
                "FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def run_summary(self, run_id: int) -> str:
        run = self.load_run(run_id)
        if not run:
            return f"Run #{run_id} not found."
        steps = run["steps"]
        passed = sum(1 for s in steps if s["success"])
        failed = sum(1 for s in steps if not s["success"])
        status = "✓ done" if run["success"] else ("✗ failed" if run["status"] == "failed" else "⋯ running")
        return (f"Run #{run_id}: {status} | "
                f"{passed} passed, {failed} failed | "
                f"'{run['instruction'][:50]}'")
