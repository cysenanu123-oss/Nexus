"""
core/web_agent.py
NEXUS — autonomous web-research agent.

The "go search the web on its own, then come back with an answer" loop. It
orchestrates the existing research/ modules into an agentic cycle:

    search → filter (web_safety) → fetch → summarize → assess
        ├─ enough?  → synthesize a cited answer
        └─ not yet? → refine the query and loop (up to max_iterations)

Every URL is passed through core.web_safety before fetching, so the agent
can't be steered into internal/metadata endpoints (SSRF). It only *reads* the
web; any state-changing action must go through web_safety.guard_action.

All collaborators (searcher, fetcher, summarizer, llm) are injected, so the
loop is unit-testable without touching the network or a model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from core.web_safety import filter_fetchable

log = logging.getLogger("nexus.web_agent")

ProgressFn = Callable[[str], None]


@dataclass
class Source:
    url: str
    title: str = ""


@dataclass
class ResearchResult:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    iterations: int = 0
    success: bool = False


class WebAgent:
    def __init__(
        self,
        searcher,
        fetcher,
        summarizer,
        llm=None,
        max_iterations: int = 2,
        pages_per_iter: int = 3,
        min_sources: int = 2,
        resolve_dns: bool = False,
    ):
        self.searcher = searcher
        self.fetcher = fetcher
        self.summarizer = summarizer
        self.llm = llm
        self.max_iterations = max_iterations
        self.pages_per_iter = pages_per_iter
        self.min_sources = min_sources
        self.resolve_dns = resolve_dns

    # ── public ───────────────────────────────────────────────────────
    def research(self, question: str, on_progress: Optional[ProgressFn] = None) -> ResearchResult:
        def say(msg: str):
            log.info(msg)
            if on_progress:
                on_progress(msg)

        summaries: list = []
        sources: list[Source] = []
        queries: list[str] = []
        seen_urls: set[str] = set()

        query = question.strip()
        iterations = 0
        for i in range(self.max_iterations):
            iterations = i + 1
            queries.append(query)
            say(f"Searching (round {iterations}): {query!r}")

            resp = self.searcher.search(query, max_results=self.pages_per_iter * 2)
            if not getattr(resp, "success", False):
                say("No search results.")
                new_q = self._refine_query(question, query, summaries)
                if not new_q or new_q == query:
                    break
                query = new_q
                continue

            urls = [u for u in filter_fetchable(resp.urls, resolve=self.resolve_dns)
                    if u not in seen_urls][: self.pages_per_iter]
            say(f"Reading {len(urls)} page(s)…")

            for url in urls:
                seen_urls.add(url)
                page = self.fetcher.fetch(url)
                if not getattr(page, "success", False):
                    continue
                summary = self.summarizer.summarize(page, topic=question)
                if getattr(summary, "success", False):
                    summaries.append(summary)
                    sources.append(Source(page.url, getattr(page, "title", "")))

            if len(summaries) >= self.min_sources:
                say("Gathered enough material.")
                break

            new_q = self._refine_query(question, query, summaries)
            if not new_q or new_q == query:
                break
            query = new_q

        answer = self._synthesize(question, summaries)
        return ResearchResult(
            question=question,
            answer=answer,
            sources=self._dedupe(sources),
            queries=queries,
            iterations=iterations,
            success=bool(summaries),
        )

    # ── internals ────────────────────────────────────────────────────
    @staticmethod
    def _dedupe(sources: list[Source]) -> list[Source]:
        seen: set[str] = set()
        out: list[Source] = []
        for s in sources:
            if s.url not in seen:
                seen.add(s.url)
                out.append(s)
        return out

    def _llm_ready(self) -> bool:
        return bool(self.llm) and getattr(self.llm, "is_ready", True)

    def _refine_query(self, question: str, last_query: str, summaries: list) -> str:
        """Ask the LLM for a better query to fill the gap. Empty string = stop."""
        if not self._llm_ready():
            return ""
        found = "; ".join(getattr(s, "title", "") for s in summaries[-4:]) or "nothing useful yet"
        prompt = (
            f"You are refining a web search. Original question: {question!r}\n"
            f"Last query: {last_query!r}\n"
            f"So far found: {found}\n"
            "Give ONE improved search query that would fill the remaining gap. "
            "Reply with only the query, no quotes, no explanation."
        )
        try:
            out = self.llm.chat(prompt=prompt, task="fast")
        except Exception as e:
            log.warning("Query refine failed: %s", e)
            return ""
        line = (out or "").strip().splitlines()[0].strip().strip('"') if out else ""
        # Guard against the model echoing the same query or returning junk.
        if len(line) < 3 or line.lower() == last_query.lower():
            return ""
        return line[:200]

    def _synthesize(self, question: str, summaries: list) -> str:
        if not summaries:
            return ("I searched but couldn't find usable information for that. "
                    "Try rephrasing, or check your internet connection.")

        notes = []
        for idx, s in enumerate(summaries, 1):
            title = getattr(s, "title", "") or getattr(s, "url", f"source {idx}")
            text = getattr(s, "text", "")
            notes.append(f"[{idx}] {title}: {text}")
        notes_block = "\n".join(notes)

        if self._llm_ready():
            prompt = (
                f"Answer the question using ONLY the notes below. Be concise and "
                f"factual, and cite sources inline like [1], [2].\n\n"
                f"Question: {question}\n\nNotes:\n{notes_block}\n\nAnswer:"
            )
            try:
                out = self.llm.chat(prompt=prompt, task="chat")
                if out and out.strip():
                    return out.strip()
            except Exception as e:
                log.warning("Synthesis failed: %s", e)

        # Fallback: stitch the summaries together.
        joined = "\n\n".join(f"• {getattr(s, 'text', '')}" for s in summaries[:4])
        return f"Here's what I found:\n\n{joined}"


def build_default(llm=None, max_iterations: int = 2) -> Optional[WebAgent]:
    """Construct a WebAgent from the research/ modules. None if unavailable."""
    try:
        from research.searcher import WebSearcher
        from research.fetcher import PageFetcher
        from research.summarizer import Summarizer
    except Exception as e:
        log.warning("Web agent unavailable (research modules missing): %s", e)
        return None
    try:
        summarizer = Summarizer(llm=llm) if llm is not None else Summarizer()
    except TypeError:
        summarizer = Summarizer()
    return WebAgent(WebSearcher(), PageFetcher(), summarizer, llm=llm,
                    max_iterations=max_iterations)
