"""
research/researcher.py
NEXUS Research Agent — the main orchestrator.

Ties together searcher + fetcher + summarizer + memory into one
clean interface. The Brain calls this when NEXUS needs to learn.

Pipeline:
    Brain.think("research buffer overflows")
           ↓
    research/researcher.py   ← YOU ARE HERE
           ↓
    searcher → fetcher → summarizer → memory → report

Usage:
    from research.researcher import Researcher
    r = Researcher()

    # Research a topic
    report = r.research("how does ASLR work in Linux")
    print(report.synthesis)

    # Quick answer (checks memory first, then web)
    answer = r.answer("what is heap spraying")
    print(answer)

    # Check what NEXUS has already learned
    r.memory.print_stats()
"""

from __future__ import annotations

import logging
import time
from typing import Optional

log = logging.getLogger("nexus.research.researcher")


def _format_output(topic: str, content: str, sources: list, source: str = "web") -> str:
    """Render a research result in NEXUS's styled format."""
    from datetime import datetime
    width = 62
    bar   = "═" * width
    thin  = "─" * width
    ts    = datetime.now().strftime("%Y-%m-%d  %H:%M")
    src_label = "memory" if source == "memory" else "web research"

    lines = [
        f"\n  {bar}",
        f"  NEXUS RESEARCH  ·  {ts}",
        f"  Topic : {topic}",
        f"  {thin}",
        "",
    ]

    # Indent each paragraph of the content
    for para in content.strip().split("\n"):
        stripped = para.strip()
        if not stripped:
            lines.append("")
        elif stripped.startswith(("•", "-", "*", "–")):
            lines.append(f"    {stripped}")
        else:
            # Wrap long lines at ~70 chars
            words, cur = [], 0
            row: list[str] = []
            for word in stripped.split():
                if cur + len(word) + 1 > 68 and row:
                    lines.append("  " + " ".join(row))
                    row, cur = [], 0
                row.append(word)
                cur += len(word) + 1
            if row:
                lines.append("  " + " ".join(row))

    # Sources footer
    unique_sources = list(dict.fromkeys(s for s in sources if s and "nexus://" not in s))
    if unique_sources:
        lines += ["", f"  {thin}", f"  Sources ({src_label}):"]
        for i, url in enumerate(unique_sources[:3], 1):
            lines.append(f"    [{i}] {url[:72]}")

    lines.append(f"  {bar}\n")
    return "\n".join(lines)


class Researcher:
    """
    The NEXUS Research Agent.

    On each .research() call:
      1. Checks memory for existing knowledge
      2. Searches the web for new sources
      3. Fetches top pages
      4. Summarizes content with Ollama
      5. Stores everything in ChromaDB memory
      6. Returns a ResearchReport

    Usage:
        researcher = Researcher()
        report = researcher.research("SQL injection OWASP top 10")
        print(report.synthesis)
    """

    def __init__(
        self,
        max_sources: int   = 3,
        search_delay: float = 0.8,
        use_memory:  bool  = True,
        memory_max_age_hours: float = 48.0,
    ):
        self.max_sources   = max_sources
        self.search_delay  = search_delay
        self.use_memory    = use_memory
        self.memory_max_age = memory_max_age_hours

        from research.searcher   import WebSearcher
        from research.fetcher    import PageFetcher
        from research.summarizer import Summarizer, ResearchReport
        from research.memory     import ResearchMemory

        self._ResearchReport = ResearchReport

        self.searcher   = WebSearcher()
        self.fetcher    = PageFetcher()
        self.summarizer = Summarizer()
        self.memory     = ResearchMemory()

        log.info(
            "Researcher ready — max_sources=%d, memory=%s",
            max_sources, use_memory,
        )

    # ─────────────────────────────────────────────
    # Main API
    # ─────────────────────────────────────────────

    def research(
        self,
        query: str,
        topic: str = "",
        force_refresh: bool = False,
    ) -> "ResearchReport":
        """
        Research a topic end-to-end.

        Parameters
        ----------
        query         : the research question / search query
        topic         : optional short label (defaults to query)
        force_refresh : ignore cached memory and re-fetch from web

        Returns
        -------
        ResearchReport with synthesis + individual summaries
        """
        from research.summarizer import ResearchReport

        topic = topic or query
        t0    = time.time()

        log.info("Researching: %r", query)
        print(f"\n  [NEXUS] Researching: {query!r}")

        # ── 1. Check memory first ──────────────────────────────
        if self.use_memory and not force_refresh:
            cached = self.memory.recall(query, max_results=3, min_relevance=0.5)
            if cached:
                log.info("Found %d cached results in memory.", len(cached))
                print(f"  [NEXUS] Found {len(cached)} result(s) from memory.\n")
                return self._report_from_memory(query, topic, cached, t0)

        # ── 2. Search web ──────────────────────────────────────
        print(f"  [NEXUS] Searching the web...")
        search_response = self.searcher.search(query, max_results=self.max_sources + 2)

        if not search_response.success:
            log.warning("Search failed: %s", search_response.error)
            return ResearchReport(
                topic=topic, query=query,
                error=f"Search failed: {search_response.error}",
                elapsed=time.time() - t0,
            )

        log.info("Found %d results via %s.", len(search_response.results), search_response.source)
        print(f"  [NEXUS] Found {len(search_response.results)} results. Fetching pages...")

        # ── 3. Fetch pages ─────────────────────────────────────
        pages = self.fetcher.fetch_many(
            urls      = search_response.urls,
            delay     = self.search_delay,
            max_pages = self.max_sources,
        )

        if not pages:
            return ResearchReport(
                topic=topic, query=query,
                error="All page fetches failed.",
                elapsed=time.time() - t0,
            )

        print(f"  [NEXUS] Fetched {len(pages)} page(s). Summarizing with LLM...")

        # ── 4. Summarize pages ─────────────────────────────────
        summaries = self.summarizer.summarize_many(pages, topic=topic)

        if not summaries:
            return ResearchReport(
                topic=topic, query=query,
                error="Summarization produced no output.",
                elapsed=time.time() - t0,
            )

        # ── 5. Synthesize ──────────────────────────────────────
        print(f"  [NEXUS] Synthesizing {len(summaries)} source(s)...")
        synthesis = self.summarizer.synthesize(summaries, topic=topic)

        # ── 6. Store in memory ─────────────────────────────────
        if self.use_memory:
            self._store_research(query, topic, summaries, synthesis)

        report = ResearchReport(
            topic     = topic,
            query     = query,
            summaries = summaries,
            synthesis = synthesis,
            elapsed   = time.time() - t0,
        )

        print(f"  [NEXUS] Research complete in {report.elapsed:.1f}s — {len(summaries)} source(s).\n")
        log.info("Research complete: %s", report)

        return report

    def answer(self, question: str, force_refresh: bool = False) -> str:
        """
        Quick Q&A — researches the question and returns a concise answer.

        Checks memory first. Falls back to web research if nothing found.

        Parameters
        ----------
        question      : natural language question
        force_refresh : skip memory and re-research

        Returns
        -------
        str — answer text
        """
        # Check memory for existing knowledge
        if not force_refresh and self.use_memory:
            cached = self.memory.recall(question, max_results=3, min_relevance=0.55)
            if cached:
                log.info("Answering from memory (relevance=%.2f)", cached[0].relevance)
                # Re-synthesize from memory rather than dumping raw text
                from research.summarizer import Summary
                mem_summaries = [
                    Summary(url=r.entry.url, title=r.entry.topic,
                            topic=question, text=r.entry.text, model="memory-recall")
                    for r in cached
                ]
                synthesis = self.summarizer.synthesize(mem_summaries, topic=question)
                if synthesis:
                    return _format_output(question, synthesis, [r.entry.url for r in cached], source="memory")

        # Full research
        report = self.research(question, force_refresh=force_refresh)

        if not report.success:
            return f"I couldn't research that: {report.error}"

        content = report.synthesis or (report.summaries[0].text if report.summaries else "")
        if not content:
            return "Research completed but produced no content."

        sources = [s.url for s in report.summaries]
        return _format_output(question, content, sources, source="web")

    def recall(self, query: str, max_results: int = 5) -> str:
        """
        Recall past research from memory without hitting the web.

        Returns
        -------
        str — formatted recall results
        """
        results = self.memory.recall(query, max_results=max_results)

        if not results:
            return f"I haven't researched anything about '{query}' yet."

        lines = [f"From my research notes ({len(results)} match(es)):\n"]
        for i, r in enumerate(results, 1):
            age_h = r.entry.age_hours()
            age_s = f"{int(age_h)}h ago" if age_h < 48 else f"{int(age_h/24)}d ago"
            lines.append(
                f"[{i}] {r.entry.topic}  ({age_s}, relevance={r.relevance:.2f})\n"
                f"    {r.entry.text[:200]}...\n"
                f"    Source: {r.entry.url}\n"
            )

        return "\n".join(lines)

    def learn_from_url(self, url: str, topic: str = "") -> str:
        """
        Directly fetch and learn from a specific URL.
        Useful when the user pastes a link they want NEXUS to read.

        Parameters
        ----------
        url   : the URL to fetch
        topic : optional topic label

        Returns
        -------
        str — summary of what was learned
        """
        print(f"\n  [NEXUS] Reading: {url}")
        page = self.fetcher.fetch(url)

        if not page.success:
            return f"Couldn't fetch that page: {page.error}"

        topic = topic or page.title or url
        print(f"  [NEXUS] Summarizing...")
        summary = self.summarizer.summarize(page, topic=topic)

        if not summary.success:
            return f"Fetched the page but couldn't summarize it: {summary.error}"

        if self.use_memory:
            self.memory.store(
                topic  = topic,
                text   = summary.text,
                url    = url,
                source = "direct",
            )
            print(f"  [NEXUS] Stored in memory.\n")

        return summary.text

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    def _report_from_memory(self, query, topic, cached, t0) -> "ResearchReport":
        """Build a ResearchReport from cached memory results."""
        from research.summarizer import Summary, ResearchReport

        summaries = [
            Summary(
                url       = r.entry.url,
                title     = r.entry.topic,
                topic     = topic,
                text      = r.entry.text,
                key_facts = [],
                model     = "memory-recall",
            )
            for r in cached
        ]

        synthesis = self.summarizer.synthesize(summaries, topic=topic)

        return ResearchReport(
            topic     = topic,
            query     = query,
            summaries = summaries,
            synthesis = synthesis,
            elapsed   = time.time() - t0,
        )

    def _store_research(self, query, topic, summaries, synthesis) -> None:
        """Store all summaries and the synthesis in memory."""
        # Store synthesis as the primary entry
        if synthesis:
            self.memory.store(
                topic  = topic,
                text   = synthesis,
                url    = "nexus://synthesis",
                source = "synthesis",
            )

        # Store each individual summary
        for s in summaries:
            self.memory.store(
                topic  = topic,
                text   = s.text,
                url    = s.url,
                source = s.model,
            )

        log.info("Stored %d summaries + synthesis in memory.", len(summaries))


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logging.basicConfig(level=logging.INFO)

    researcher = Researcher(max_sources=3)

    if "--answer" in sys.argv:
        idx      = sys.argv.index("--answer")
        question = " ".join(sys.argv[idx + 1:])
        print(f"\n  Q: {question}\n")
        answer = researcher.answer(question)
        print(f"  A: {answer}\n")
        sys.exit(0)

    if "--recall" in sys.argv:
        idx   = sys.argv.index("--recall")
        query = " ".join(sys.argv[idx + 1:])
        print(researcher.recall(query))
        sys.exit(0)

    if "--url" in sys.argv:
        idx   = sys.argv.index("--url")
        url   = sys.argv[idx + 1]
        topic = " ".join(sys.argv[idx + 2:]) if len(sys.argv) > idx + 2 else ""
        result = researcher.learn_from_url(url, topic=topic)
        print(f"\n  Learned:\n  {result[:600]}\n")
        sys.exit(0)

    if "--stats" in sys.argv:
        researcher.memory.print_stats()
        sys.exit(0)

    # Default: research a query
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "ASLR address space layout randomization Linux"
    report = researcher.research(query)

    if report.success:
        print(f"\n  ─── Research Report ───")
        print(f"  Topic   : {report.topic}")
        print(f"  Sources : {report.source_count}")
        print(f"  Elapsed : {report.elapsed:.1f}s\n")
        print(f"  Synthesis:\n")
        print(f"  {report.synthesis[:800]}\n")
    else:
        print(f"\n  Research failed: {report.error}\n")