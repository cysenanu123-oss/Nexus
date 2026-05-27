"""
core/vector_memory.py
NEXUS Vector Memory — semantic search layer backed by ChromaDB.

Upgrades plain SQLite keyword search to meaning-based recall.
Inspired by Omi's vector_db.py + memories.py.

Usage:
    vm = VectorMemory()
    vm.store("my exam is on Friday", metadata={"category": "reminder"})
    results = vm.search("when is my exam", n=3)
    for r in results:
        print(r["text"], r["distance"])
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("nexus.vector_memory")

CHROMA_PATH = "data/vector_memory"
COLLECTION_MEMORY   = "nexus_memory"
COLLECTION_EPISODES = "nexus_episodes"
COLLECTION_SESSIONS = "nexus_sessions"


class VectorMemory:
    """
    Semantic memory store using ChromaDB with sentence-transformers embeddings.

    Three collections:
        memory   — explicit facts the user tells NEXUS
        episodes — individual conversation turns
        sessions — full conversation summaries
    """

    def __init__(self, persist_path: str = CHROMA_PATH):
        self._ready = False
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            Path(persist_path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_path)

            ef = embedding_functions.DefaultEmbeddingFunction()

            self._memory_col   = self._client.get_or_create_collection(COLLECTION_MEMORY,   embedding_function=ef)
            self._episodes_col = self._client.get_or_create_collection(COLLECTION_EPISODES, embedding_function=ef)
            self._sessions_col = self._client.get_or_create_collection(COLLECTION_SESSIONS, embedding_function=ef)

            self._ready = True
            log.info("VectorMemory ready — %d memories, %d episodes, %d sessions.",
                     self._memory_col.count(), self._episodes_col.count(), self._sessions_col.count())
        except ImportError:
            log.warning("chromadb not installed — VectorMemory disabled. Run: pip install chromadb")
        except Exception as e:
            log.warning("VectorMemory init failed: %s", e)

    @property
    def ready(self) -> bool:
        return self._ready

    # ── Memory store ─────────────────────────────────────────────────────

    def store(self, text: str, metadata: Optional[dict] = None, doc_id: Optional[str] = None):
        """Store a fact/memory with optional metadata."""
        if not self._ready or not text.strip():
            return
        doc_id = doc_id or _hash(text)
        meta = {"timestamp": time.time(), **(metadata or {})}
        try:
            self._memory_col.upsert(ids=[doc_id], documents=[text], metadatas=[meta])
            log.debug("Stored memory: %s", text[:60])
        except Exception as e:
            log.warning("Vector store failed: %s", e)

    def search(self, query: str, n: int = 5, where: Optional[dict] = None) -> list[dict]:
        """Semantic search across stored memories."""
        if not self._ready or not query.strip():
            return []
        try:
            kwargs = {"query_texts": [query], "n_results": min(n, self._memory_col.count() or 1)}
            if where:
                kwargs["where"] = where
            results = self._memory_col.query(**kwargs)
            return _format_results(results)
        except Exception as e:
            log.warning("Vector search failed: %s", e)
            return []

    def delete(self, doc_id: str):
        if not self._ready:
            return
        try:
            self._memory_col.delete(ids=[doc_id])
        except Exception as e:
            log.warning("Vector delete failed: %s", e)

    # ── Episode store (conversation turns) ──────────────────────────────

    def store_episode(self, user_text: str, nexus_text: str, intent: str = ""):
        if not self._ready:
            return
        combined = f"USER: {user_text}\nNEXUS: {nexus_text}"
        meta = {"user": user_text, "nexus": nexus_text, "intent": intent, "timestamp": time.time()}
        try:
            self._episodes_col.upsert(
                ids=[_hash(combined + str(time.time()))],
                documents=[combined],
                metadatas=[meta]
            )
        except Exception as e:
            log.warning("Episode store failed: %s", e)

    def search_episodes(self, query: str, n: int = 5) -> list[dict]:
        if not self._ready or not query.strip():
            return []
        try:
            count = self._episodes_col.count()
            if count == 0:
                return []
            results = self._episodes_col.query(query_texts=[query], n_results=min(n, count))
            return _format_results(results)
        except Exception as e:
            log.warning("Episode search failed: %s", e)
            return []

    # ── Session summaries ────────────────────────────────────────────────

    def store_session_summary(self, session_id: int, title: str, summary: str, topics: list[str]):
        if not self._ready or not summary.strip():
            return
        text = f"{title}\n{summary}\nTopics: {', '.join(topics)}"
        meta = {"session_id": session_id, "title": title, "timestamp": time.time()}
        try:
            self._sessions_col.upsert(
                ids=[f"session_{session_id}"],
                documents=[text],
                metadatas=[meta]
            )
        except Exception as e:
            log.warning("Session summary store failed: %s", e)

    def search_sessions(self, query: str, n: int = 5) -> list[dict]:
        if not self._ready or not query.strip():
            return []
        try:
            count = self._sessions_col.count()
            if count == 0:
                return []
            results = self._sessions_col.query(query_texts=[query], n_results=min(n, count))
            return _format_results(results)
        except Exception as e:
            log.warning("Session search failed: %s", e)
            return []

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        if not self._ready:
            return {"ready": False}
        return {
            "ready": True,
            "memories": self._memory_col.count(),
            "episodes": self._episodes_col.count(),
            "sessions": self._sessions_col.count(),
        }


# ── Helpers ──────────────────────────────────────────────────────────────

def _hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def _format_results(results: dict) -> list[dict]:
    out = []
    docs      = results.get("documents", [[]])[0]
    metas     = results.get("metadatas",  [[]])[0]
    distances = results.get("distances",  [[]])[0]
    for doc, meta, dist in zip(docs, metas, distances):
        out.append({"text": doc, "metadata": meta, "distance": dist})
    return out
