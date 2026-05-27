"""
core/knowledge_graph.py
NEXUS Knowledge Graph — entity and relationship tracking over time.
Inspired by Omi's knowledge_graph.py router.

Builds a persistent graph of entities (people, places, projects, concepts)
and the relationships between them, extracted from conversations.

Schema:
    entities   → id, name, type (person/place/project/concept/tool), attrs JSON
    relations  → id, from_id, relation, to_id, confidence, source, created_at

Usage:
    kg = KnowledgeGraph(llm=brain.llm)
    kg.add_entity("John", "person", {"role": "colleague"})
    kg.add_relation("Cyril", "works with", "John")

    # Auto-extract from conversation
    kg.extract_from_text("I met John from the cybersecurity team at the conference")

    # Query
    kg.query_entity("John")           # → all relations involving John
    kg.related_to("Cyril")            # → everything connected to Cyril
    kg.visualize()                    # → text graph
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("nexus.knowledge_graph")

DB_PATH = Path("data/knowledge_graph.db")

ENTITY_TYPES = {"person", "place", "project", "concept", "tool", "event", "organization", "other"}


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL COLLATE NOCASE,
                type        TEXT DEFAULT 'other',
                attrs       TEXT DEFAULT '{}',
                first_seen  REAL NOT NULL,
                last_seen   REAL NOT NULL,
                mention_count INTEGER DEFAULT 1,
                UNIQUE(name)
            );

            CREATE TABLE IF NOT EXISTS relations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                from_name   TEXT NOT NULL COLLATE NOCASE,
                relation    TEXT NOT NULL,
                to_name     TEXT NOT NULL COLLATE NOCASE,
                confidence  REAL DEFAULT 1.0,
                source      TEXT DEFAULT 'conversation',
                created_at  REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_rel_from ON relations(from_name);
            CREATE INDEX IF NOT EXISTS idx_rel_to   ON relations(to_name);
        """)


_EXTRACT_PROMPT = """Extract entities and relationships from the following text.

Return ONLY a valid JSON object:
{{
  "entities": [
    {{"name": "...", "type": "person|place|project|concept|tool|event|organization|other"}}
  ],
  "relations": [
    {{"from": "...", "relation": "...", "to": "..."}}
  ]
}}

Be concise. Only extract meaningful named entities, not generic words.

Text:
{text}"""


class KnowledgeGraph:
    """Persistent entity-relationship graph for NEXUS long-term intelligence."""

    def __init__(self, llm=None):
        _init_db()
        self._llm = llm
        log.info("KnowledgeGraph ready — %d entities, %d relations.",
                 self._count("entities"), self._count("relations"))

    def set_llm(self, llm):
        self._llm = llm

    # ── Entity management ────────────────────────────────────────────────

    def add_entity(self, name: str, entity_type: str = "other", attrs: Optional[dict] = None) -> int:
        name = name.strip()
        if not name:
            return -1
        if entity_type not in ENTITY_TYPES:
            entity_type = "other"
        now = time.time()
        with _connect() as conn:
            existing = conn.execute("SELECT id FROM entities WHERE name=?", (name,)).fetchone()
            if existing:
                conn.execute(
                    "UPDATE entities SET last_seen=?, mention_count=mention_count+1 WHERE name=?",
                    (now, name)
                )
                if attrs:
                    row = conn.execute("SELECT attrs FROM entities WHERE name=?", (name,)).fetchone()
                    merged = {**json.loads(row["attrs"]), **attrs}
                    conn.execute("UPDATE entities SET attrs=? WHERE name=?", (json.dumps(merged), name))
                return existing["id"]
            cur = conn.execute(
                "INSERT INTO entities (name, type, attrs, first_seen, last_seen) VALUES (?,?,?,?,?)",
                (name, entity_type, json.dumps(attrs or {}), now, now)
            )
            return cur.lastrowid

    def add_relation(self, from_name: str, relation: str, to_name: str,
                     confidence: float = 1.0, source: str = "conversation"):
        from_name = from_name.strip()
        to_name = to_name.strip()
        relation = relation.strip().lower()
        if not from_name or not to_name or not relation:
            return

        self.add_entity(from_name)
        self.add_entity(to_name)

        with _connect() as conn:
            existing = conn.execute(
                "SELECT id FROM relations WHERE from_name=? AND relation=? AND to_name=?",
                (from_name, relation, to_name)
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO relations (from_name, relation, to_name, confidence, source, created_at) VALUES (?,?,?,?,?,?)",
                    (from_name, relation, to_name, confidence, source, time.time())
                )

    # ── Extraction ───────────────────────────────────────────────────────

    def extract_from_text(self, text: str, source: str = "conversation") -> dict:
        """Auto-extract entities and relations from text using LLM or rules."""
        if not text.strip():
            return {"entities": [], "relations": []}

        extracted = self._extract(text)

        entities_added = 0
        for e in extracted.get("entities", []):
            if e.get("name"):
                self.add_entity(e["name"], e.get("type", "other"))
                entities_added += 1

        relations_added = 0
        for r in extracted.get("relations", []):
            if r.get("from") and r.get("relation") and r.get("to"):
                self.add_relation(r["from"], r["relation"], r["to"], source=source)
                relations_added += 1

        log.info("Extracted %d entities, %d relations from text.", entities_added, relations_added)
        return {"entities": extracted.get("entities", []), "relations": extracted.get("relations", [])}

    def _extract(self, text: str) -> dict:
        if self._llm:
            try:
                prompt = _EXTRACT_PROMPT.format(text=text[:2000])
                raw = self._llm.ask(prompt, max_tokens=500).strip()
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    return json.loads(raw[start:end])
            except Exception as e:
                log.warning("LLM extraction failed, falling back to rules: %s", e)

        return _rule_based_extract(text)

    # ── Querying ─────────────────────────────────────────────────────────

    def query_entity(self, name: str) -> dict:
        """Get all info and relations for a named entity."""
        with _connect() as conn:
            entity = conn.execute(
                "SELECT * FROM entities WHERE name=?", (name,)
            ).fetchone()
            outgoing = conn.execute(
                "SELECT relation, to_name FROM relations WHERE from_name=?", (name,)
            ).fetchall()
            incoming = conn.execute(
                "SELECT from_name, relation FROM relations WHERE to_name=?", (name,)
            ).fetchall()

        if not entity:
            return {}

        return {
            "name": entity["name"],
            "type": entity["type"],
            "attrs": json.loads(entity["attrs"]),
            "mentions": entity["mention_count"],
            "outgoing": [{"relation": r["relation"], "to": r["to_name"]} for r in outgoing],
            "incoming": [{"from": r["from_name"], "relation": r["relation"]} for r in incoming],
        }

    def related_to(self, name: str) -> list[str]:
        """All entity names connected to this entity."""
        with _connect() as conn:
            rows = conn.execute(
                "SELECT to_name FROM relations WHERE from_name=? "
                "UNION SELECT from_name FROM relations WHERE to_name=?",
                (name, name)
            ).fetchall()
        return [r[0] for r in rows]

    def search_entities(self, query: str) -> list[dict]:
        q = f"%{query}%"
        with _connect() as conn:
            rows = conn.execute(
                "SELECT name, type, mention_count FROM entities WHERE name LIKE ? ORDER BY mention_count DESC LIMIT 20",
                (q,)
            ).fetchall()
        return [dict(r) for r in rows]

    def most_mentioned(self, limit: int = 10) -> list[dict]:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT name, type, mention_count FROM entities ORDER BY mention_count DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def all_entities(self, entity_type: Optional[str] = None) -> list[dict]:
        with _connect() as conn:
            if entity_type:
                rows = conn.execute(
                    "SELECT name, type, mention_count FROM entities WHERE type=? ORDER BY mention_count DESC",
                    (entity_type,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT name, type, mention_count FROM entities ORDER BY mention_count DESC"
                ).fetchall()
        return [dict(r) for r in rows]

    # ── Visualization ────────────────────────────────────────────────────

    def visualize(self, limit: int = 30) -> str:
        """Return a text representation of the graph."""
        with _connect() as conn:
            relations = conn.execute(
                "SELECT from_name, relation, to_name FROM relations ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()

        if not relations:
            return "Knowledge graph is empty."

        lines = ["── NEXUS KNOWLEDGE GRAPH ──────────────────"]
        for r in relations:
            lines.append(f"  [{r['from_name']}] --{r['relation']}--> [{r['to_name']}]")
        lines.append(f"  ({self._count('entities')} entities, {self._count('relations')} relations total)")
        return "\n".join(lines)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _count(self, table: str) -> int:
        try:
            with _connect() as conn:
                return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except Exception:
            return 0


# ── Rule-based fallback ──────────────────────────────────────────────────

_PERSON_PATTERNS = [
    r"\b(?:my\s+)?(?:friend|colleague|boss|manager|professor|doctor|dr\.?)\s+([A-Z][a-z]+)\b",
    r"\b([A-Z][a-z]+)\s+(?:told|said|asked|helped|called|texted|emailed)\b",
]

_PROJECT_PATTERNS = [
    r"\b(?:project|app|tool|system|module)\s+(?:called\s+)?[\"']?([A-Z][a-zA-Z0-9_\-]+)[\"']?\b",
]


def _rule_based_extract(text: str) -> dict:
    entities = []
    relations = []

    for pat in _PERSON_PATTERNS:
        for m in re.finditer(pat, text):
            name = m.group(1)
            if len(name) > 1:
                entities.append({"name": name, "type": "person"})

    for pat in _PROJECT_PATTERNS:
        for m in re.finditer(pat, text, re.I):
            name = m.group(1)
            entities.append({"name": name, "type": "project"})

    seen = set()
    unique_entities = []
    for e in entities:
        if e["name"] not in seen:
            seen.add(e["name"])
            unique_entities.append(e)

    return {"entities": unique_entities, "relations": relations}
