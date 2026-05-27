"""
core/skill_registry.py
NEXUS Skill Registry — SQLite-backed catalog of every capability NEXUS can invoke.

A skill is a named, callable unit of capability:
  builtin   — hardcoded in NEXUS (cyber, calendar, memory, etc.)
  acquired  — learned from a GitHub repo or URL via SkillAcquirer
  created   — LLM-generated on demand by TaskPlanner
  learned   — extracted from a webpage / documentation

The registry is the source of truth for "what can I do?"  The planner searches
it before going online so skills accumulate over time.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("nexus.skill_registry")

_DB_PATH   = Path(__file__).parent.parent / "data" / "skills" / "registry.db"
_SKILL_DIR = Path(__file__).parent.parent / "data" / "skills"


# ─────────────────────────────────────────────────────────────
#  DATA CLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class Skill:
    name:          str
    description:   str
    category:      str           # cyber | communication | system | research | code | utility | calendar | memory
    source:        str           # "builtin" | "github:<url>" | "created" | "learned:<url>"
    tags:          list[str]     = field(default_factory=list)
    usage_example: str           = ""
    code_path:     str           = ""   # path to skill's Python file (relative to repo root)
    invoke_fn:     str           = ""   # function name inside that file
    invoke_module: str           = ""   # importable module path
    parameters:    dict          = field(default_factory=dict)  # param_name → description
    created_at:    str           = field(default_factory=lambda: datetime.now().isoformat())
    last_used:     str           = ""
    use_count:     int           = 0
    id:            int           = 0

    @property
    def is_invokable(self) -> bool:
        return bool(self.code_path or self.invoke_module or self.invoke_fn)

    def summary(self) -> str:
        return f"[{self.category}] {self.name}: {self.description}"

    def __str__(self) -> str:
        return self.summary()


# ─────────────────────────────────────────────────────────────
#  REGISTRY
# ─────────────────────────────────────────────────────────────

class SkillRegistry:
    """Persistent skill catalog backed by SQLite with Voyager-style vector retrieval."""

    def __init__(self, db_path: Path = _DB_PATH):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # Must be set before _seed_builtins() so _index_skill() can check it
        self._vector = None
        self._vector_indexed = False
        self._init_db()
        self._seed_builtins()
        self._init_vector_index()

    # ── Vector index (Voyager-style semantic retrieval) ───────

    def _init_vector_index(self):
        try:
            from core.vector_memory import VectorMemory
            self._vector = VectorMemory(persist_path=str(Path(__file__).parent.parent / "data" / "skill_vectors"))
            if not self._vector.ready:
                self._vector = None
                return
            # Index all skills into the vector store on first run
            if self._vector._memory_col.count() < self.count():
                self._rebuild_vector_index()
            self._vector_indexed = True
        except Exception as e:
            log.debug("Skill vector index unavailable: %s", e)
            self._vector = None

    def _rebuild_vector_index(self):
        if not self._vector or not self._vector.ready:
            return
        skills = self.all()
        for skill in skills:
            text = f"{skill.name}: {skill.description}. Tags: {' '.join(skill.tags)}. Example: {skill.usage_example}"
            self._vector.store(text, metadata={"skill_name": skill.name}, doc_id=f"skill_{skill.name}")
        log.info("Skill vector index built: %d skills.", len(skills))

    def _index_skill(self, skill: "Skill"):
        if not self._vector or not self._vector.ready:
            return
        text = f"{skill.name}: {skill.description}. Tags: {' '.join(skill.tags)}. Example: {skill.usage_example}"
        self._vector.store(text, metadata={"skill_name": skill.name}, doc_id=f"skill_{skill.name}")

    # ── DB plumbing ───────────────────────────────────────────

    def _conn(self):
        return sqlite3.connect(self._db_path)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    name          TEXT UNIQUE NOT NULL,
                    description   TEXT,
                    category      TEXT,
                    source        TEXT,
                    tags          TEXT,
                    usage_example TEXT,
                    code_path     TEXT,
                    invoke_fn     TEXT,
                    invoke_module TEXT,
                    parameters    TEXT,
                    created_at    TEXT,
                    last_used     TEXT,
                    use_count     INTEGER DEFAULT 0
                )
            """)

    # ── CRUD ──────────────────────────────────────────────────

    def register(self, skill: Skill) -> Skill:
        """Insert or update a skill (upsert by name)."""
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO skills (name, description, category, source, tags,
                    usage_example, code_path, invoke_fn, invoke_module,
                    parameters, created_at, last_used, use_count)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(name) DO UPDATE SET
                    description   = excluded.description,
                    category      = excluded.category,
                    source        = excluded.source,
                    tags          = excluded.tags,
                    usage_example = excluded.usage_example,
                    code_path     = excluded.code_path,
                    invoke_fn     = excluded.invoke_fn,
                    invoke_module = excluded.invoke_module,
                    parameters    = excluded.parameters
            """, (
                skill.name, skill.description, skill.category, skill.source,
                json.dumps(skill.tags), skill.usage_example,
                skill.code_path, skill.invoke_fn, skill.invoke_module,
                json.dumps(skill.parameters),
                skill.created_at, skill.last_used, skill.use_count,
            ))
        log.info("Registered skill: %s [%s]", skill.name, skill.source)
        self._index_skill(skill)
        return skill

    def get(self, name: str) -> Optional[Skill]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM skills WHERE name=?", (name,)).fetchone()
        return self._row(row) if row else None

    def delete(self, name: str) -> bool:
        with self._conn() as conn:
            conn.execute("DELETE FROM skills WHERE name=?", (name,))
        return True

    def update_usage(self, name: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE skills SET use_count=use_count+1, last_used=? WHERE name=?",
                (datetime.now().isoformat(), name),
            )

    # ── Search ────────────────────────────────────────────────

    def search(self, query: str, limit: int = 8) -> list[Skill]:
        """Semantic skill retrieval (Voyager pattern) with keyword fallback."""
        # Try vector search first
        if self._vector and self._vector.ready:
            try:
                results = self._vector.search(query, n=limit)
                if results:
                    skills = []
                    for r in results:
                        name = r["metadata"].get("skill_name")
                        if name:
                            skill = self.get(name)
                            if skill:
                                skills.append(skill)
                    if skills:
                        return skills
            except Exception as e:
                log.debug("Vector skill search failed, falling back: %s", e)

        # Fallback: keyword overlap search
        words = set(query.lower().split())
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM skills").fetchall()

        scored: list[tuple[int, Skill]] = []
        for row in rows:
            skill = self._row(row)
            haystack = f"{skill.name} {skill.description} {' '.join(skill.tags)}".lower()
            score = sum(1 for w in words if w in haystack)
            if score > 0:
                scored.append((score, skill))

        scored.sort(key=lambda x: (-x[0], -x[1].use_count))
        return [s for _, s in scored[:limit]]

    def list_by_category(self, category: str) -> list[Skill]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM skills WHERE category=? ORDER BY use_count DESC",
                (category,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def all(self) -> list[Skill]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM skills ORDER BY category, name"
            ).fetchall()
        return [self._row(r) for r in rows]

    def categories(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT category FROM skills ORDER BY category"
            ).fetchall()
        return [r[0] for r in rows]

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]

    # ── Row → Skill ───────────────────────────────────────────

    def _row(self, row) -> Skill:
        cols = [
            "id", "name", "description", "category", "source", "tags",
            "usage_example", "code_path", "invoke_fn", "invoke_module",
            "parameters", "created_at", "last_used", "use_count",
        ]
        d = dict(zip(cols, row))
        d["tags"]       = json.loads(d.get("tags") or "[]")
        d["parameters"] = json.loads(d.get("parameters") or "{}")
        return Skill(**d)

    # ── Built-in seed ─────────────────────────────────────────

    def _seed_builtins(self):
        """Register the built-in NEXUS skills (idempotent — skip if already exist)."""
        builtins = [
            # Cyber
            Skill("port_scan",       "Scan open ports on a target host",
                  "cyber",    "builtin", ["scan","nmap","ports","network"],
                  "port scan 192.168.1.1"),
            Skill("subdomain_enum",  "Enumerate subdomains of a domain (passive + active DNS)",
                  "cyber",    "builtin", ["recon","subdomain","dns","osint"],
                  "find subdomains example.com"),
            Skill("vuln_scan",       "Run nuclei/nikto/nmap vuln scripts against a target",
                  "cyber",    "builtin", ["vulnerability","nuclei","nikto","pentest"],
                  "vuln scan 192.168.1.1"),
            Skill("full_recon",      "Full recon pipeline: DNS → IP/WHOIS → headers → subdomains",
                  "cyber",    "builtin", ["recon","osint","pentest","reconnaissance"],
                  "recon on example.com"),
            Skill("cve_lookup",      "Look up CVE details from NVD",
                  "cyber",    "builtin", ["cve","nvd","vulnerability","exploit"],
                  "cve lookup CVE-2024-1234"),
            Skill("exploit_search",  "Search Exploit-DB for exploits matching a keyword",
                  "cyber",    "builtin", ["exploit","searchsploit","exploitdb"],
                  "find exploit vsftpd"),
            Skill("cyber_news",      "Get latest cybersecurity news from RSS feeds",
                  "cyber",    "builtin", ["news","threat","intel","rss"],
                  "latest cyber news"),
            Skill("sandbox_test",    "Clone target services in Docker and run vuln tests",
                  "cyber",    "builtin", ["sandbox","docker","isolation","pentest"],
                  "sandbox 192.168.1.1"),
            Skill("monitor_target",  "Watch a target for port/service changes over time",
                  "cyber",    "builtin", ["monitor","watch","diff","changes"],
                  "monitor target 192.168.1.1"),
            # Calendar / scheduling
            Skill("schedule_meeting","Create a .ics calendar event and set a desktop reminder",
                  "calendar", "builtin", ["calendar","meeting","schedule","ics","reminder"],
                  "remind me I have a meeting tomorrow at 3pm on Zoom"),
            Skill("set_reminder",    "Set a timed desktop reminder via the at command",
                  "calendar", "builtin", ["reminder","alarm","alert","notify"],
                  "remind me to take medicine at 8pm"),
            Skill("show_calendar",   "List stored calendar events for a given date",
                  "calendar", "builtin", ["calendar","events","show","list"],
                  "show my calendar"),
            # Memory
            Skill("store_memory",    "Persist a key-value fact in NEXUS long-term memory",
                  "memory",   "builtin", ["remember","store","note","fact"],
                  "remember that my API key is ..."),
            Skill("recall_memory",   "Search NEXUS memory for stored facts",
                  "memory",   "builtin", ["recall","search","memory","lookup"],
                  "what do I know about project X"),
            # Research
            Skill("web_search",      "Search the web and return result snippets",
                  "research", "builtin", ["search","google","web","internet"],
                  "search the web for latest AI news"),
            Skill("web_fetch",       "Fetch and read the content of a URL",
                  "research", "builtin", ["fetch","read","url","scrape","website"],
                  "fetch https://example.com"),
            Skill("learn_from_url",  "Read a URL, summarise it, store findings in knowledge base",
                  "research", "builtin", ["learn","url","read","knowledge","summarize"],
                  "learn from https://docs.example.com"),
            # Code
            Skill("code_analyze",    "Analyze a Python file with AST, show structure",
                  "code",     "builtin", ["analyze","ast","python","structure","read"],
                  "analyze core/brain.py"),
            Skill("code_plan",       "Narrated step-by-step plan for a coding task",
                  "code",     "builtin", ["plan","implement","feature","bug","code"],
                  "plan how to add a login system"),
            Skill("acquire_skill",   "Clone a GitHub repo and extract skills from it",
                  "code",     "builtin", ["github","clone","learn","acquire","repo"],
                  "acquire skill from https://github.com/user/repo"),
            Skill("create_skill",    "Generate a new Python skill function using LLM",
                  "code",     "builtin", ["create","generate","skill","new","write"],
                  "create a skill to send emails via Gmail"),
            # System
            Skill("run_shell",       "Execute a shell command and return output",
                  "system",   "builtin", ["shell","bash","command","run","execute"],
                  "run ls -la"),
            Skill("send_notification","Send a desktop notification via notify-send",
                  "system",   "builtin", ["notify","alert","desktop","popup"],
                  "send me a notification when done"),
            Skill("take_screenshot", "Capture the current screen",
                  "system",   "builtin", ["screenshot","screen","capture"],
                  "take a screenshot"),
        ]

        for s in builtins:
            if not self.get(s.name):
                self.register(s)


# ── Singleton ──────────────────────────────────────────────────

_registry: Optional[SkillRegistry] = None


def get_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry
