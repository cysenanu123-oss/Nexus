"""
core/knowledge.py
NEXUS Self-Learning Knowledge Base — stores what NEXUS learns online.

When NEXUS goes online to solve a problem it hasn't seen before, it stores
what it finds here so it never has to look it up again.

Storage: SQLite at data/knowledge.db
"""

import sqlite3
import logging
import datetime
import json
from pathlib import Path
from typing import Optional

log = logging.getLogger("nexus.knowledge")

DB_PATH = Path(__file__).parent.parent / "data" / "knowledge.db"


class KnowledgeBase:
    """
    Persistent knowledge store for self-learned information.

    NEXUS writes here when it learns something new from the web.
    Before going online, it searches here first to avoid redundant lookups.

    Schema
    ------
    topic    : normalized search key
    content  : full learned content
    source   : where it came from (url, 'web_search', 'code_analysis', etc.)
    tags     : JSON list of topic tags
    quality  : 0.0–1.0 confidence score
    created  : ISO timestamp
    accessed : how many times this entry was recalled
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        log.info("KnowledgeBase ready — %d entries", self.count())

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic     TEXT NOT NULL,
                    content   TEXT NOT NULL,
                    source    TEXT    DEFAULT '',
                    tags      TEXT    DEFAULT '[]',
                    quality   REAL    DEFAULT 1.0,
                    created   TEXT    DEFAULT '',
                    accessed  INTEGER DEFAULT 0
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge(topic)"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    # ── Write ─────────────────────────────────────────────────

    def store(
        self,
        topic: str,
        content: str,
        source: str = "",
        tags: list[str] = None,
        quality: float = 1.0,
    ) -> int:
        """
        Store a learned piece of knowledge.

        Returns the new row ID.
        Deduplicates — if same topic+source exists, updates content instead.
        """
        topic_key = topic.lower().strip()
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id FROM knowledge WHERE topic = ? AND source = ? LIMIT 1",
                (topic_key, source),
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE knowledge SET content=?, quality=?, accessed=accessed+1 WHERE id=?",
                    (content, quality, existing[0]),
                )
                log.debug("Knowledge updated: %r", topic_key)
                return existing[0]

            cursor = conn.execute(
                "INSERT INTO knowledge (topic, content, source, tags, quality, created) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    topic_key,
                    content,
                    source,
                    json.dumps(tags or []),
                    quality,
                    datetime.datetime.now().isoformat(),
                ),
            )
            log.info("Knowledge stored: %r (source=%r)", topic_key, source)
            return cursor.lastrowid

    # ── Read ──────────────────────────────────────────────────

    def search(self, query: str, limit: int = 3) -> list[dict]:
        """
        Search the knowledge base for entries relevant to query.

        Scores by word overlap; returns top N sorted by score then recency.
        """
        words = [w for w in query.lower().split() if len(w) > 2]
        if not words:
            return []

        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, topic, content, source, created FROM knowledge "
                "ORDER BY accessed DESC, quality DESC"
            ).fetchall()

        results = []
        for kid, topic, content, source, created in rows:
            score = sum(
                (2 if w in topic else 0) + (1 if w in content.lower() else 0)
                for w in words
            )
            if score > 0:
                results.append(
                    {
                        "id": kid,
                        "topic": topic,
                        "content": content,
                        "source": source,
                        "created": created,
                        "score": score,
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)
        top = results[:limit]

        if top:
            with self._conn() as conn:
                for r in top:
                    conn.execute(
                        "UPDATE knowledge SET accessed = accessed + 1 WHERE id = ?",
                        (r["id"],),
                    )

        return top

    def has_topic(self, topic: str, min_quality: float = 0.5) -> bool:
        """Return True if we already have knowledge about this topic."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM knowledge WHERE topic LIKE ? AND quality >= ? LIMIT 1",
                (f"%{topic.lower()}%", min_quality),
            ).fetchone()
        return row is not None

    def get_recent(self, limit: int = 5) -> list[dict]:
        """Return the most recently learned entries."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT topic, content, source, created FROM knowledge "
                "ORDER BY created DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"topic": t, "content": c, "source": s, "created": d}
            for t, c, s, d in rows
        ]

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]

    def stats(self) -> dict:
        with self._conn() as conn:
            total   = conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
            sources = conn.execute(
                "SELECT source, COUNT(*) FROM knowledge GROUP BY source"
            ).fetchall()
        return {"total": total, "by_source": dict(sources)}


# ── Singleton ─────────────────────────────────────────────────

_kb: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb
