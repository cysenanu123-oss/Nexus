"""
research/memory.py
NEXUS Research Memory — persistent vector storage for everything NEXUS learns.

Stores research summaries and facts in ChromaDB (local, no cloud).
Retrieves relevant past research using semantic similarity search.

Pipeline:
    research/searcher.py
           ↓
    research/fetcher.py
           ↓
    research/summarizer.py
           ↓
    research/memory.py     ← YOU ARE HERE

Usage:
    from research.memory import ResearchMemory
    mem = ResearchMemory()

    # Store research
    mem.store(topic="buffer overflow", text="...", url="https://...", source="wikipedia")

    # Retrieve relevant past research
    results = mem.recall("how do buffer overflows work")
    for r in results:
        print(r["text"][:200])

    # Check if we've already researched something
    if mem.has_researched("SQL injection"):
        print("Already know about this — pulling from memory")
"""

from __future__ import annotations

import logging
import time
import json
import os
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

log = logging.getLogger("nexus.research.memory")

# ── paths ─────────────────────────────────────────────────────
VECTOR_STORE_DIR = Path("data/research/vectors")
METADATA_DB      = Path("data/research/metadata.json")
COLLECTION_NAME  = "nexus_research"

# ── lazy imports ──────────────────────────────────────────────
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _CHROMA_OK = True
except ImportError:
    _CHROMA_OK = False

try:
    import numpy as np
    _NP_OK = True
except ImportError:
    _NP_OK = False


# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """One stored research entry."""
    id:        str
    topic:     str
    text:      str
    url:       str
    source:    str
    timestamp: float = field(default_factory=time.time)
    metadata:  dict  = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id":        self.id,
            "topic":     self.topic,
            "text":      self.text,
            "url":       self.url,
            "source":    self.source,
            "timestamp": self.timestamp,
            "metadata":  self.metadata,
        }

    def age_hours(self) -> float:
        return (time.time() - self.timestamp) / 3600


@dataclass
class RecallResult:
    """One result returned from a memory search."""
    entry:      MemoryEntry
    relevance:  float        # 0.0 – 1.0 (higher = more similar)
    distance:   float        # raw distance from ChromaDB

    def __str__(self) -> str:
        return (
            f"RecallResult(topic={self.entry.topic!r}, "
            f"relevance={self.relevance:.2f}, "
            f"age={self.entry.age_hours():.1f}h)"
        )


# ─────────────────────────────────────────────────────────────
#  RESEARCH MEMORY
# ─────────────────────────────────────────────────────────────

class ResearchMemory:
    """
    Persistent vector memory for NEXUS research.

    Stores research summaries as embeddings so NEXUS can recall
    semantically related information from past research sessions.

    Two storage layers:
        1. ChromaDB  — vector embeddings for semantic search
        2. JSON file — fast metadata index, topic deduplication

    Falls back to JSON-only storage if ChromaDB isn't installed.

    Usage:
        mem = ResearchMemory()
        mem.store(topic="XSS attacks", text="...", url="https://owasp.org/...")
        results = mem.recall("cross-site scripting web security")
        mem.stats()
    """

    def __init__(
        self,
        vector_dir: Path = VECTOR_STORE_DIR,
        metadata_path: Path = METADATA_DB,
    ):
        self.vector_dir    = Path(vector_dir)
        self.metadata_path = Path(metadata_path)
        self._collection   = None
        self._metadata: dict[str, dict] = {}

        # Ensure directories
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # Load metadata index
        self._load_metadata()

        # Init ChromaDB
        if _CHROMA_OK:
            self._init_chroma()
        else:
            log.warning(
                "ChromaDB not installed — using JSON-only fallback. "
                "Run: pip install chromadb  for semantic search."
            )

        log.info(
            "ResearchMemory ready — %d entries, chroma=%s",
            len(self._metadata), _CHROMA_OK,
        )

    # ── ChromaDB init ─────────────────────────────────────────

    def _init_chroma(self) -> None:
        """Initialize the ChromaDB collection."""
        try:
            client = chromadb.PersistentClient(
                path=str(self.vector_dir),
            )
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            log.info(
                "ChromaDB collection '%s' ready — %d documents",
                COLLECTION_NAME, self._collection.count(),
            )
        except Exception as e:
            log.error("ChromaDB init failed: %s — falling back to JSON.", e)
            self._collection = None

    # ── Public API ────────────────────────────────────────────

    # ── Quality gate ─────────────────────────────────────────────

    _FEED_PATTERNS = [
        "follow this topic", "share\ngraphic", "hours ago\n",
        "days ago\n", "by maxwell", "by bloomberg", "more\npope",
        "\nmore\n", "follow this topic\nshare",
        "hours ago\nby ", "hours ago\nwired", "hours ago\ncnbc",
        "hours ago\nfortune", "hours ago\nreuters",
    ]
    # Stub entries that sneak through as single-sentence "summaries"
    _STUB_PATTERNS = [
        "technology usually creates jobs for young",
        "workers historically filled new tech",
        "artificial intelligence\nfollow this topic",
        "follow this topic\nshare",
    ]

    def _is_junk(self, text: str) -> bool:
        """Return True if text looks like a raw news feed / nav dump / stub."""
        if not text or len(text.strip()) < 40:
            return True
        t = text.lower()
        # Exact stub patterns
        if any(p in t for p in self._STUB_PATTERNS):
            return True
        feed_hits = sum(1 for p in self._FEED_PATTERNS if p in t)
        if feed_hits >= 2:
            return True
        # High ratio of short lines = nav/feed structure
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if lines:
            short = sum(1 for l in lines if len(l) < 30)
            if short / len(lines) > 0.55 and len(lines) > 20:
                return True
        return False

    def wipe_junk(self) -> int:
        """Delete all stored entries that look like raw feed/nav dumps. Returns count deleted."""
        bad_ids = [eid for eid, e in self._metadata.items()
                   if self._is_junk(e.get("text", ""))]
        if not bad_ids:
            return 0
        if self._collection is not None:
            try:
                self._collection.delete(ids=bad_ids)
            except Exception as e:
                log.warning("ChromaDB delete failed: %s", e)
        for eid in bad_ids:
            self._metadata.pop(eid, None)
        self._save_metadata()
        log.info("Wiped %d junk entries from research memory.", len(bad_ids))
        return len(bad_ids)

    def store(
        self,
        topic:    str,
        text:     str,
        url:      str    = "",
        source:   str    = "",
        metadata: dict   = None,
    ) -> str:
        """
        Store a research result in memory.

        Parameters
        ----------
        topic    : research topic / query
        text     : summary or full text to store
        url      : source URL
        source   : where this came from (wikipedia, duckduckgo, etc.)
        metadata : optional extra fields

        Returns
        -------
        str — entry ID
        """
        if self._is_junk(text):
            log.warning("Rejected junk content from %r — not storing in memory.", url)
            return ""

        entry_id = self._make_id(topic, url)

        entry = MemoryEntry(
            id        = entry_id,
            topic     = topic,
            text      = text,
            url       = url,
            source    = source,
            timestamp = time.time(),
            metadata  = metadata or {},
        )

        # Store in ChromaDB (vector)
        if self._collection is not None:
            try:
                self._collection.upsert(
                    ids        = [entry_id],
                    documents  = [text],
                    metadatas  = [{
                        "topic":  topic,
                        "url":    url,
                        "source": source,
                        "ts":     str(entry.timestamp),
                    }],
                )
            except Exception as e:
                log.error("ChromaDB upsert failed: %s", e)

        # Store in metadata index
        self._metadata[entry_id] = entry.to_dict()
        self._save_metadata()

        log.info("Stored research: topic=%r, id=%s", topic, entry_id)
        return entry_id

    def recall(
        self,
        query:       str,
        max_results: int   = 5,
        min_relevance: float = 0.3,
    ) -> list[RecallResult]:
        """
        Retrieve research relevant to a query using semantic search.

        Parameters
        ----------
        query        : natural language query
        max_results  : max entries to return
        min_relevance: minimum similarity score (0.0 – 1.0)

        Returns
        -------
        List of RecallResult sorted by relevance (highest first)
        """
        if not self._metadata:
            return []

        # ChromaDB semantic search
        if self._collection is not None and self._collection.count() > 0:
            return self._chroma_recall(query, max_results, min_relevance)

        # JSON fallback: simple keyword search
        return self._keyword_recall(query, max_results)

    def has_researched(self, topic: str, max_age_hours: float = 24.0) -> bool:
        """
        Check if we've already researched this topic recently.

        Parameters
        ----------
        topic         : topic to check
        max_age_hours : ignore entries older than this

        Returns
        -------
        bool
        """
        topic_lower = topic.lower()
        cutoff      = time.time() - (max_age_hours * 3600)

        for entry in self._metadata.values():
            if topic_lower in entry.get("topic", "").lower():
                if entry.get("timestamp", 0) >= cutoff:
                    return True
        return False

    def get_recent(self, n: int = 10) -> list[MemoryEntry]:
        """Return the N most recently stored entries."""
        entries = sorted(
            self._metadata.values(),
            key=lambda e: e.get("timestamp", 0),
            reverse=True,
        )[:n]
        return [self._dict_to_entry(e) for e in entries]

    def forget(self, entry_id: str) -> bool:
        """Remove a specific entry from memory."""
        if entry_id not in self._metadata:
            return False

        del self._metadata[entry_id]
        self._save_metadata()

        if self._collection is not None:
            try:
                self._collection.delete(ids=[entry_id])
            except Exception as e:
                log.error("ChromaDB delete failed: %s", e)

        log.info("Forgot entry: %s", entry_id)
        return True

    def forget_topic(self, topic: str) -> int:
        """Remove all entries matching a topic. Returns number removed."""
        to_remove = [
            id_ for id_, e in self._metadata.items()
            if topic.lower() in e.get("topic", "").lower()
        ]
        for id_ in to_remove:
            self.forget(id_)
        return len(to_remove)

    def clear(self) -> None:
        """Wipe all research memory. Irreversible."""
        self._metadata = {}
        self._save_metadata()
        if self._collection is not None:
            try:
                ids = self._collection.get()["ids"]
                if ids:
                    self._collection.delete(ids=ids)
            except Exception as e:
                log.error("ChromaDB clear failed: %s", e)
        log.warning("Research memory cleared.")

    def stats(self) -> dict:
        """Return memory statistics."""
        chroma_count = 0
        if self._collection is not None:
            try:
                chroma_count = self._collection.count()
            except Exception:
                pass

        topics = list({e.get("topic", "?") for e in self._metadata.values()})

        return {
            "total_entries":  len(self._metadata),
            "chroma_entries": chroma_count,
            "unique_topics":  len(topics),
            "recent_topics":  topics[:5],
            "vector_dir":     str(self.vector_dir),
        }

    def print_stats(self) -> None:
        """Print memory stats to terminal."""
        s = self.stats()
        print(f"\n  Research Memory Stats")
        print(f"  {'─' * 40}")
        print(f"  Entries        : {s['total_entries']}")
        print(f"  ChromaDB docs  : {s['chroma_entries']}")
        print(f"  Unique topics  : {s['unique_topics']}")
        if s["recent_topics"]:
            print(f"  Recent topics  :")
            for t in s["recent_topics"]:
                print(f"    • {t}")
        print()

    # ── Internal search backends ──────────────────────────────

    def _chroma_recall(
        self, query: str, max_results: int, min_relevance: float
    ) -> list[RecallResult]:
        """ChromaDB semantic similarity search."""
        try:
            results = self._collection.query(
                query_texts    = [query],
                n_results      = min(max_results, self._collection.count()),
                include        = ["documents", "metadatas", "distances"],
            )
        except Exception as e:
            log.error("ChromaDB query failed: %s", e)
            return self._keyword_recall(query, max_results)

        if not results or not results.get("ids"):
            return []

        recall_results = []
        ids       = results["ids"][0]
        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        for id_, doc, meta, dist in zip(ids, docs, metas, distances):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to relevance 0–1
            relevance = max(0.0, 1.0 - (dist / 2.0))
            if relevance < min_relevance:
                continue

            entry_dict = self._metadata.get(id_, {
                "id": id_, "topic": meta.get("topic", ""),
                "text": doc, "url": meta.get("url", ""),
                "source": meta.get("source", ""), "timestamp": 0.0,
            })
            entry = self._dict_to_entry(entry_dict)
            recall_results.append(RecallResult(
                entry=entry, relevance=relevance, distance=dist
            ))

        return sorted(recall_results, key=lambda r: -r.relevance)

    def _keyword_recall(self, query: str, max_results: int) -> list[RecallResult]:
        """Simple keyword overlap search (fallback)."""
        query_words = set(query.lower().split())
        scored = []

        for entry_dict in self._metadata.values():
            text  = (entry_dict.get("text", "") + " " + entry_dict.get("topic", "")).lower()
            overlap = sum(1 for w in query_words if w in text)
            if overlap > 0:
                scored.append((overlap, entry_dict))

        scored.sort(key=lambda x: -x[0])
        max_overlap = max((s for s, _ in scored), default=1)

        results = []
        for score, entry_dict in scored[:max_results]:
            entry = self._dict_to_entry(entry_dict)
            results.append(RecallResult(
                entry     = entry,
                relevance = score / max_overlap,
                distance  = 1.0 - (score / max_overlap),
            ))

        return results

    # ── Storage helpers ───────────────────────────────────────

    def _make_id(self, topic: str, url: str) -> str:
        """Generate a stable ID from topic + URL."""
        content = f"{topic.lower().strip()}|{url}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _load_metadata(self) -> None:
        if self.metadata_path.exists():
            try:
                self._metadata = json.loads(self.metadata_path.read_text())
                log.debug("Loaded %d metadata entries.", len(self._metadata))
            except Exception as e:
                log.error("Failed to load metadata: %s", e)
                self._metadata = {}

    def _save_metadata(self) -> None:
        try:
            self.metadata_path.write_text(
                json.dumps(self._metadata, indent=2, ensure_ascii=False)
            )
        except Exception as e:
            log.error("Failed to save metadata: %s", e)

    def _dict_to_entry(self, d: dict) -> MemoryEntry:
        return MemoryEntry(
            id        = d.get("id", ""),
            topic     = d.get("topic", ""),
            text      = d.get("text", ""),
            url       = d.get("url", ""),
            source    = d.get("source", ""),
            timestamp = d.get("timestamp", 0.0),
            metadata  = d.get("metadata", {}),
        )


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("\n  NEXUS Research Memory Test\n")
    mem = ResearchMemory()

    if "--clear" in sys.argv:
        mem.clear()
        print("  Memory cleared.\n")
        sys.exit(0)

    if "--stats" in sys.argv:
        mem.print_stats()
        sys.exit(0)

    if "--recall" in sys.argv:
        idx   = sys.argv.index("--recall")
        query = " ".join(sys.argv[idx + 1:])
        print(f"  Recalling: {query!r}\n")
        results = mem.recall(query)
        if results:
            for r in results:
                print(f"  [{r.relevance:.2f}] {r.entry.topic}")
                print(f"         {r.entry.text[:150]}...")
                print()
        else:
            print("  Nothing found.\n")
        sys.exit(0)

    # Default: store a test entry and recall it
    print("  Storing test entry...")
    mem.store(
        topic  = "buffer overflow exploits",
        text   = "A buffer overflow occurs when data written to a buffer exceeds its size, overwriting adjacent memory. Common in C/C++ programs. Can be exploited to overwrite return addresses and redirect code execution.",
        url    = "https://example.com/buffer-overflow",
        source = "test",
    )
    mem.print_stats()

    print("  Recalling 'memory corruption'...\n")
    results = mem.recall("memory corruption exploit")
    for r in results:
        print(f"  Relevance: {r.relevance:.2f}")
        print(f"  Topic    : {r.entry.topic}")
        print(f"  Text     : {r.entry.text[:200]}\n")