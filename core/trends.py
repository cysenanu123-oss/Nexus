"""
core/trends.py
NEXUS Trends & Focus Sessions — inspired by Omi's focus_sessions.py + trends.py.

Tracks:
    - Command usage patterns (most used commands, intents)
    - Focus sessions (bursts of activity = productive periods)
    - Active hours (when you use NEXUS most)
    - Daily/weekly summaries

Usage:
    trends = TrendsTracker()
    trends.log_command("open firefox", intent="launch_app")
    trends.log_command("research quantum computing", intent="research")

    trends.get_focus_sessions()     # detected focus bursts
    trends.peak_hours()             # hours of day you're most active
    trends.top_commands(n=10)       # most used commands/intents
    trends.daily_summary()          # today's activity summary
    trends.weekly_report()          # last 7 days overview
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger("nexus.trends")

DB_PATH = Path("data/trends.db")

FOCUS_WINDOW_SECONDS = 1800   # 30 min — if 5+ commands in this window = focus session
FOCUS_MIN_COMMANDS   = 5


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS commands (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                text        TEXT NOT NULL,
                intent      TEXT DEFAULT '',
                category    TEXT DEFAULT '',
                timestamp   REAL NOT NULL,
                hour        INTEGER,
                weekday     INTEGER
            );

            CREATE TABLE IF NOT EXISTS focus_sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at  REAL NOT NULL,
                ended_at    REAL NOT NULL,
                duration    REAL NOT NULL,
                command_count INTEGER NOT NULL,
                top_intent  TEXT DEFAULT '',
                summary     TEXT DEFAULT ''
            );
        """)


class TrendsTracker:
    """Tracks NEXUS usage patterns and detects focus sessions."""

    def __init__(self):
        _init_db()
        log.info("TrendsTracker ready — %d commands logged.", self._total_commands())

    # ── Logging ──────────────────────────────────────────────────────────

    def log_command(self, text: str, intent: str = "", category: str = ""):
        now = time.time()
        dt = datetime.fromtimestamp(now)
        with _connect() as conn:
            conn.execute(
                "INSERT INTO commands (text, intent, category, timestamp, hour, weekday) VALUES (?,?,?,?,?,?)",
                (text[:200], intent, category, now, dt.hour, dt.weekday())
            )
        self._check_focus_session()

    # ── Focus session detection ──────────────────────────────────────────

    def _check_focus_session(self):
        """Detect and save a focus session if a burst of activity just ended or is happening."""
        cutoff = time.time() - FOCUS_WINDOW_SECONDS
        with _connect() as conn:
            rows = conn.execute(
                "SELECT timestamp, intent FROM commands WHERE timestamp > ? ORDER BY timestamp ASC",
                (cutoff,)
            ).fetchall()

        if len(rows) < FOCUS_MIN_COMMANDS:
            return

        started = rows[0]["timestamp"]
        ended   = rows[-1]["timestamp"]
        duration = ended - started

        if duration < 60:
            return

        intents = [r["intent"] for r in rows if r["intent"]]
        top_intent = Counter(intents).most_common(1)[0][0] if intents else ""

        with _connect() as conn:
            existing = conn.execute(
                "SELECT id FROM focus_sessions WHERE started_at=? AND ended_at=?",
                (started, ended)
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO focus_sessions (started_at, ended_at, duration, command_count, top_intent) VALUES (?,?,?,?,?)",
                    (started, ended, duration, len(rows), top_intent)
                )

    # ── Analytics ────────────────────────────────────────────────────────

    def top_commands(self, n: int = 10, days: int = 7) -> list[dict]:
        cutoff = time.time() - (days * 86400)
        with _connect() as conn:
            rows = conn.execute(
                "SELECT intent, COUNT(*) as count FROM commands WHERE timestamp > ? AND intent != '' "
                "GROUP BY intent ORDER BY count DESC LIMIT ?",
                (cutoff, n)
            ).fetchall()
        return [{"intent": r["intent"], "count": r["count"]} for r in rows]

    def peak_hours(self, days: int = 30) -> list[dict]:
        cutoff = time.time() - (days * 86400)
        with _connect() as conn:
            rows = conn.execute(
                "SELECT hour, COUNT(*) as count FROM commands WHERE timestamp > ? "
                "GROUP BY hour ORDER BY count DESC",
                (cutoff,)
            ).fetchall()
        return [{"hour": r["hour"], "count": r["count"]} for r in rows]

    def peak_days(self, days: int = 30) -> list[dict]:
        cutoff = time.time() - (days * 86400)
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        with _connect() as conn:
            rows = conn.execute(
                "SELECT weekday, COUNT(*) as count FROM commands WHERE timestamp > ? "
                "GROUP BY weekday ORDER BY count DESC",
                (cutoff,)
            ).fetchall()
        return [{"day": day_names[r["weekday"]], "count": r["count"]} for r in rows]

    def get_focus_sessions(self, limit: int = 10) -> list[dict]:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT * FROM focus_sessions ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["started_str"] = datetime.fromtimestamp(d["started_at"]).strftime("%Y-%m-%d %H:%M")
            d["duration_min"] = round(d["duration"] / 60, 1)
            result.append(d)
        return result

    def today_stats(self) -> dict:
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        with _connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM commands WHERE timestamp > ?", (today_start,)
            ).fetchone()[0]
            top = conn.execute(
                "SELECT intent, COUNT(*) as c FROM commands WHERE timestamp > ? AND intent != '' "
                "GROUP BY intent ORDER BY c DESC LIMIT 3",
                (today_start,)
            ).fetchall()
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "commands": count,
            "top_intents": [r["intent"] for r in top],
        }

    def weekly_report(self) -> str:
        """Human-readable weekly summary."""
        week_start = (datetime.now() - timedelta(days=7)).timestamp()
        with _connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM commands WHERE timestamp > ?", (week_start,)
            ).fetchone()[0]
            focus_count = conn.execute(
                "SELECT COUNT(*) FROM focus_sessions WHERE started_at > ?", (week_start,)
            ).fetchone()[0]

        top = self.top_commands(n=3, days=7)
        peak = self.peak_hours(days=7)
        peak_hour = peak[0]["hour"] if peak else None

        lines = [
            "── NEXUS WEEKLY REPORT ─────────────────────",
            f"  Commands this week : {total}",
            f"  Focus sessions     : {focus_count}",
        ]
        if top:
            lines.append(f"  Top activities     : {', '.join(t['intent'] for t in top)}")
        if peak_hour is not None:
            am_pm = "am" if peak_hour < 12 else "pm"
            h = peak_hour if peak_hour <= 12 else peak_hour - 12
            lines.append(f"  Most active hour   : {h}{am_pm}")

        return "\n".join(lines)

    def daily_summary(self) -> str:
        stats = self.today_stats()
        if stats["commands"] == 0:
            return "No activity recorded today."
        intents = ", ".join(stats["top_intents"]) if stats["top_intents"] else "various"
        return f"Today: {stats['commands']} commands. Top activities: {intents}."

    def _total_commands(self) -> int:
        try:
            with _connect() as conn:
                return conn.execute("SELECT COUNT(*) FROM commands").fetchone()[0]
        except Exception:
            return 0
