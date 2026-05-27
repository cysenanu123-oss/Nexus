"""
research/summarizer.py
NEXUS Research Summarizer — compresses web pages into useful knowledge
using the local Ollama LLM (same LLM stack as core/llm.py).

Pipeline:
    research/searcher.py
           ↓
    research/fetcher.py
           ↓
    research/summarizer.py   ← YOU ARE HERE
           ↓
    research/memory.py

Usage:
    from research.summarizer import Summarizer
    from research.fetcher import PageFetcher

    fetcher    = PageFetcher()
    summarizer = Summarizer()

    page    = fetcher.fetch("https://owasp.org/www-community/attacks/SQL_Injection")
    summary = summarizer.summarize(page, topic="SQL injection")
    print(summary.text)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("nexus.research.summarizer")

# Max characters to send to the LLM — keeps token usage sane
MAX_CONTENT_FOR_LLM = 4_000


# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class Summary:
    """Output from the summarizer for one page."""
    url:         str
    title:       str
    topic:       str
    text:        str          # full summary text
    key_facts:   list[str] = field(default_factory=list)
    elapsed:     float     = 0.0
    model:       str       = ""
    error:       str       = ""

    @property
    def success(self) -> bool:
        return bool(self.text) and not self.error

    def __str__(self) -> str:
        if self.error:
            return f"Summary(FAILED: {self.error})"
        preview = self.text[:120].replace("\n", " ")
        return f"Summary(topic={self.topic!r}, model={self.model!r}): {preview}..."


@dataclass
class ResearchReport:
    """Aggregated summary across multiple pages on one topic."""
    topic:     str
    query:     str
    summaries: list[Summary] = field(default_factory=list)
    synthesis: str           = ""   # combined synthesis from all summaries
    elapsed:   float         = 0.0
    error:     str           = ""

    @property
    def success(self) -> bool:
        return bool(self.summaries) and not self.error

    @property
    def source_count(self) -> int:
        return len(self.summaries)

    def full_text(self) -> str:
        """Concatenate all summaries for storage."""
        parts = [f"Topic: {self.topic}\n", f"Synthesis:\n{self.synthesis}\n"]
        for i, s in enumerate(self.summaries, 1):
            parts.append(f"\n--- Source {i}: {s.title} ---\n{s.text}")
        return "\n".join(parts)

    def __str__(self) -> str:
        return (
            f"ResearchReport(topic={self.topic!r}, "
            f"sources={self.source_count}, "
            f"elapsed={self.elapsed:.1f}s)"
        )


# ─────────────────────────────────────────────────────────────
#  SUMMARIZER
# ─────────────────────────────────────────────────────────────

class Summarizer:
    """
    Uses the NEXUS LLM (Ollama) to summarize fetched pages.

    Falls back to extractive summarization (no LLM) if Ollama is offline,
    so research still works without a running model.

    Usage:
        summarizer = Summarizer()
        summary = summarizer.summarize(page, topic="SQL injection")
        print(summary.text)
        print(summary.key_facts)
    """

    def __init__(self):
        try:
            from core.llm import get_llm
            self._llm = get_llm()
            if self._llm.is_ready:
                log.info("Summarizer using LLM: %s", self._llm._pick_model("research"))
            else:
                log.warning("LLM offline — summarizer will use extractive fallback.")
        except Exception as e:
            self._llm = None
            log.warning("LLM unavailable for summarizer: %s — using extractive fallback.", e)

    # ── Public API ────────────────────────────────────────────

    def summarize(self, page, topic: str = "") -> Summary:
        """
        Summarize a FetchedPage.

        Parameters
        ----------
        page  : FetchedPage from research.fetcher
        topic : the research topic (helps the LLM focus)

        Returns
        -------
        Summary
        """
        if not page.success:
            return Summary(
                url=page.url, title=page.title, topic=topic,
                text="", error=f"Page fetch failed: {page.error}"
            )

        log.info("Summarizing: %s (topic=%r)", page.url, topic)
        t0 = time.time()

        # Truncate content to LLM token budget
        content = page.text[:MAX_CONTENT_FOR_LLM]

        if self._llm and self._llm.is_ready:
            summary_text, key_facts, model = self._llm_summarize(content, topic, page.title)
        else:
            summary_text, key_facts, model = self._extractive_summarize(content, topic)

        elapsed = time.time() - t0

        return Summary(
            url       = page.url,
            title     = page.title,
            topic     = topic,
            text      = summary_text,
            key_facts = key_facts,
            elapsed   = elapsed,
            model     = model,
        )

    def summarize_many(self, pages: list, topic: str = "") -> list[Summary]:
        """Summarize a list of FetchedPages."""
        summaries = []
        for page in pages:
            s = self.summarize(page, topic)
            if s.success:
                summaries.append(s)
        return summaries

    def synthesize(self, summaries: list[Summary], topic: str) -> str:
        """
        Combine multiple page summaries into one coherent synthesis.
        Used after summarizing all sources for a research query.
        """
        if not summaries:
            return ""

        combined = "\n\n".join(
            f"Source {i+1} [{s.title}]:\n{s.text}"
            for i, s in enumerate(summaries)
        )
        combined = combined[:MAX_CONTENT_FOR_LLM * 2]

        if self._llm and self._llm.is_ready:
            prompt = (
                f"You are NEXUS, a sharp and knowledgeable AI assistant. "
                f"Using the research sources below, write a well-structured answer about: \"{topic}\"\n\n"
                f"Rules:\n"
                f"- Write in your own words. Do NOT copy article titles or headlines.\n"
                f"- Start with a 2-3 sentence overview of the topic.\n"
                f"- Then list 4-6 key findings or developments as bullet points (• ).\n"
                f"- End with a 1-sentence takeaway.\n"
                f"- Be concise and technically accurate. Avoid padding.\n\n"
                f"Research sources:\n{combined}"
            )
            try:
                return self._llm.chat(prompt, task="research")
            except Exception as e:
                log.error("LLM synthesis failed: %s", e)

        # Fallback: first summary only (avoid dumping all raw text)
        return summaries[0].text if summaries else ""

    # ── LLM-based summarization ───────────────────────────────

    def _llm_summarize(
        self, content: str, topic: str, title: str
    ) -> tuple[str, list[str], str]:
        """Use Ollama to summarize content."""
        topic_ctx = f"Research topic: {topic}\n" if topic else ""
        prompt = (
            f"{topic_ctx}"
            f"Page title: {title}\n\n"
            f"Summarize the following web page content. "
            f"Focus on facts, technical details, and actionable information. "
            f"Be concise but thorough. "
            f"End with 3-5 key facts as bullet points prefixed with '• '.\n\n"
            f"Content:\n{content}"
        )

        try:
            model  = self._llm._pick_model("research")
            result = self._llm.chat(prompt, task="research")
            facts  = self._extract_bullet_facts(result)
            return result, facts, model
        except Exception as e:
            log.error("LLM summarization failed: %s", e)
            # Fallback to extractive
            text, facts, _ = self._extractive_summarize(content, topic)
            return text, facts, "extractive-fallback"

    # ── Extractive fallback (no LLM needed) ──────────────────

    def _extractive_summarize(
        self, content: str, topic: str
    ) -> tuple[str, list[str], str]:
        """
        Simple extractive summarization — no LLM required.
        Picks the most relevant sentences using keyword overlap.
        """
        sentences = self._split_sentences(content)
        if not sentences:
            return "", [], "extractive"

        topic_words = set(topic.lower().split()) if topic else set()

        # Score sentences by length and topic keyword overlap
        scored = []
        for s in sentences:
            if len(s) < 30:
                continue
            s_lower = s.lower()
            score   = sum(1 for w in topic_words if w in s_lower)
            # Prefer sentences that look like they contain facts
            if any(c.isdigit() for c in s):
                score += 0.5
            scored.append((score, s))

        scored.sort(key=lambda x: -x[0])

        # Take top 8 sentences in their original order
        top    = {s for _, s in scored[:8]}
        result = [s for s in sentences if s in top][:8]

        text  = " ".join(result)
        facts = [f"• {s}" for s in result[:4] if len(s) > 40]

        return text, facts, "extractive"

    def _split_sentences(self, text: str) -> list[str]:
        """Basic sentence splitter."""
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def _extract_bullet_facts(self, text: str) -> list[str]:
        """Pull bullet-point facts from LLM output."""
        facts = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith(("•", "-", "*", "–")) and len(line) > 10:
                facts.append(line)
        return facts[:8]


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    url   = sys.argv[1] if len(sys.argv) > 1 else "https://en.wikipedia.org/wiki/Buffer_overflow"
    topic = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "buffer overflow"

    print(f"\n  Fetching and summarizing: {url}")
    print(f"  Topic: {topic}\n")

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from research.fetcher import PageFetcher
    fetcher = PageFetcher()
    page    = fetcher.fetch(url)

    if not page.success:
        print(f"  Fetch error: {page.error}")
        sys.exit(1)

    summarizer = Summarizer()
    summary    = summarizer.summarize(page, topic=topic)

    print(f"  Model   : {summary.model}")
    print(f"  Elapsed : {summary.elapsed:.2f}s")
    print(f"\n  Summary:\n")
    print(f"  {summary.text[:800]}\n")

    if summary.key_facts:
        print(f"  Key facts:")
        for f in summary.key_facts:
            print(f"    {f}")
    print()