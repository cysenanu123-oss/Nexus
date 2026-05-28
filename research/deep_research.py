"""
research/deep_research.py
NEXUS Deep Research Pipeline — iterative multi-round research.

Pipeline:
    Round 1  → search + fetch + summarize initial topic
    Gap pass → identify follow-up queries (LLM or keyword fallback)
    Round 2+ → research each follow-up query
    Rewrite  → synthesize all rounds into one structured report
    Save     → write reports/TOPIC_TIMESTAMP.md + .html
    Publish  → optionally POST to paste.rs and return URL
    Open     → launch browser to view the report

Works WITHOUT Ollama — extractive fallback handles summarization,
keyword heuristics handle gap analysis. With Ollama, quality is better.

Usage:
    python research/deep_research.py "quantum computing cryptography"
    python research/deep_research.py "XSS attack techniques" --rounds 3
    python research/deep_research.py "buffer overflow" --publish
    python research/deep_research.py "ASLR Linux" --no-browser
"""

from __future__ import annotations

import json
import logging
import re
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("nexus.deep_research")

REPORTS_DIR  = Path(__file__).parent.parent / "reports"
MAX_FOLLOWUPS = 3
PASTE_API     = "https://paste.rs/"


# ─────────────────────────────────────────────────────────────
#  Deep Researcher
# ─────────────────────────────────────────────────────────────

class DeepResearcher:
    """
    Iterative multi-round research pipeline.

    Each .run() call:
      1. Searches + fetches + summarizes the initial topic
      2. Extracts follow-up queries from the result
      3. Researches each follow-up
      4. Synthesizes all rounds into one clean report
      5. Saves .md + .html to reports/
      6. Optionally publishes to paste.rs
      7. Opens the HTML report in the browser
    """

    def __init__(
        self,
        rounds:       int  = 2,
        open_browser: bool = True,
        publish:      bool = False,
    ):
        self.rounds       = max(1, rounds)
        self.open_browser = open_browser
        self.publish      = publish

        from research.researcher import Researcher
        self.researcher = Researcher(max_sources=3)

        self._llm = None
        try:
            from core.llm import get_llm
            llm = get_llm()
            if llm.is_ready:
                self._llm = llm
                log.info("Deep researcher: LLM online (%s)", llm._pick_model("research"))
            else:
                log.info("Deep researcher: LLM offline — extractive fallback active")
        except Exception:
            log.info("Deep researcher: LLM unavailable — extractive fallback active")

    # ── Main entry ────────────────────────────────────────────

    def run(self, topic: str) -> str:
        """
        Execute the full pipeline.

        Parameters
        ----------
        topic : the research topic / question

        Returns
        -------
        str — path to the saved report file (empty string on total failure)
        """
        self._banner(topic)
        t0 = time.time()

        all_summaries: list = []
        all_sources:   list = []

        # ── Round 1: initial research ──────────────────────────
        print(f"\n  [1/{self.rounds}] Researching: {topic!r}")
        report1 = self.researcher.research(topic, force_refresh=True)

        if not report1.success:
            print(f"\n  [!] Initial research failed: {report1.error}")
            return ""

        all_summaries.extend(report1.summaries)
        all_sources.extend(s.url for s in report1.summaries)

        # ── Gap analysis → follow-up queries ──────────────────
        followups: list[str] = []
        if self.rounds > 1:
            seed_text = report1.synthesis or (report1.summaries[0].text if report1.summaries else "")
            print(f"\n  [NEXUS] Analysing gaps in round-1 results...")
            followups = self._extract_followups(seed_text, topic)
            for i, q in enumerate(followups, 1):
                print(f"    → follow-up {i}: {q!r}")

        # ── Rounds 2+: follow-up searches ─────────────────────
        for round_num in range(2, self.rounds + 1):
            if not followups:
                break
            query = followups.pop(0)
            print(f"\n  [{round_num}/{self.rounds}] Follow-up: {query!r}")
            rep = self.researcher.research(query, force_refresh=True)
            if rep.success:
                all_summaries.extend(rep.summaries)
                all_sources.extend(s.url for s in rep.summaries)
            else:
                print(f"  [!] Follow-up failed: {rep.error}")

        # ── Rewrite: synthesize all rounds ─────────────────────
        print(f"\n  [NEXUS] Rewriting {len(all_summaries)} source(s) into final report...")
        final_content = self._synthesize_all(topic, all_summaries)

        # ── Save ───────────────────────────────────────────────
        report_path = self._save(topic, final_content, all_sources)
        html_path   = report_path.with_suffix(".html")
        elapsed     = time.time() - t0

        print(f"\n  [NEXUS] Saved: {report_path}")

        # ── Publish ────────────────────────────────────────────
        pub_url = ""
        if self.publish:
            print(f"  [NEXUS] Publishing to paste.rs...")
            pub_url = self._publish(topic, final_content) or ""
            if pub_url:
                print(f"  [NEXUS] Published: {pub_url}")
            else:
                print(f"  [NEXUS] Publish failed (offline?)")

        # ── Open browser ───────────────────────────────────────
        if self.open_browser:
            target = f"file://{html_path}" if html_path.exists() else f"file://{report_path}"
            print(f"  [NEXUS] Opening in browser...")
            webbrowser.open(target)
            if pub_url:
                webbrowser.open(pub_url)

        print(f"\n  {'═'*56}")
        print(f"  Done in {elapsed:.1f}s — {len(set(all_sources))} unique source(s)")
        print(f"  Report : {report_path.name}")
        if pub_url:
            print(f"  URL    : {pub_url}")
        print(f"  {'═'*56}\n")

        return str(report_path)

    # ── Follow-up query extraction ────────────────────────────

    def _extract_followups(self, text: str, topic: str) -> list[str]:
        if self._llm:
            result = self._llm_followups(text, topic)
            if result:
                return result
        return self._keyword_followups(text, topic)

    def _llm_followups(self, text: str, topic: str) -> list[str]:
        prompt = (
            f"You are a research assistant. Based on this summary about '{topic}', "
            f"generate {MAX_FOLLOWUPS} specific follow-up search queries to fill knowledge gaps "
            f"and deepen understanding.\n\n"
            f"Rules:\n"
            f"- Each query must explore a different sub-aspect\n"
            f"- Keep each under 10 words\n"
            f"- Return ONLY a JSON array of strings, no markdown\n\n"
            f"Summary:\n{text[:2000]}\n\n"
            f"Follow-up queries:"
        )
        try:
            raw   = self._llm.chat(prompt, task="research")
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            if start >= 0 and end > start:
                queries = json.loads(raw[start:end])
                if isinstance(queries, list) and queries:
                    return [str(q) for q in queries[:MAX_FOLLOWUPS]]
        except Exception as e:
            log.warning("LLM followup extraction failed: %s", e)
        return []

    def _keyword_followups(self, text: str, topic: str) -> list[str]:
        """No-LLM fallback: extract capitalised noun phrases from the summary."""
        topic_words = set(topic.lower().split())
        seen: set[str] = set()
        candidates: list[str] = []

        for phrase in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text):
            low = phrase.lower()
            if low not in topic_words and low not in seen and len(phrase) > 4:
                seen.add(low)
                candidates.append(phrase)

        queries: list[str] = []
        for c in candidates[:MAX_FOLLOWUPS]:
            queries.append(f"{c} {topic}")

        templates = [
            f"{topic} examples and use cases",
            f"{topic} best practices guide",
            f"{topic} latest developments 2024",
        ]
        while len(queries) < MAX_FOLLOWUPS:
            queries.append(templates[len(queries) % len(templates)])

        return queries[:MAX_FOLLOWUPS]

    # ── Final synthesis ───────────────────────────────────────

    def _synthesize_all(self, topic: str, summaries: list) -> str:
        if not summaries:
            return "No research content was collected."

        if self._llm:
            from research.summarizer import Summarizer
            return Summarizer().synthesize(summaries, topic=topic) or _extractive_concat(summaries, topic)

        return _extractive_concat(summaries, topic)

    # ── Save ──────────────────────────────────────────────────

    def _save(self, topic: str, content: str, sources: list) -> Path:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        safe  = re.sub(r'[^\w\s-]', '_', topic)[:50].strip().replace(" ", "_")
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        base  = REPORTS_DIR / f"{safe}_{ts}"

        unique_sources = list(dict.fromkeys(s for s in sources if s))
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        md = f"# {topic}\n\n"
        md += f"*Generated by NEXUS — {ts_str}*\n\n"
        md += "---\n\n"
        md += content.strip() + "\n\n"
        if unique_sources:
            md += "---\n\n## Sources\n\n"
            for i, url in enumerate(unique_sources[:10], 1):
                md += f"{i}. {url}\n"

        md_path = base.with_suffix(".md")
        md_path.write_text(md, encoding="utf-8")

        html_path = base.with_suffix(".html")
        html_path.write_text(_to_html(topic, md, ts_str), encoding="utf-8")

        return md_path

    # ── Publish ───────────────────────────────────────────────

    def _publish(self, topic: str, content: str) -> str:
        """POST report to paste.rs — returns public URL or empty string."""
        import urllib.request
        try:
            data = content[:50_000].encode("utf-8")
            req  = urllib.request.Request(PASTE_API, data=data, method="POST")
            req.add_header("Content-Type", "text/plain; charset=utf-8")
            with urllib.request.urlopen(req, timeout=12) as resp:
                return resp.read().decode().strip()
        except Exception as e:
            log.warning("paste.rs publish failed: %s", e)
            return ""

    # ── Banner ────────────────────────────────────────────────

    def _banner(self, topic: str) -> None:
        bar = "═" * 58
        print(f"\n  {bar}")
        print(f"  NEXUS DEEP RESEARCH PIPELINE")
        print(f"  Topic  : {topic}")
        print(f"  Rounds : {self.rounds}")
        print(f"  LLM    : {'online' if self._llm else 'offline — extractive fallback'}")
        print(f"  {bar}")


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _extractive_concat(summaries: list, topic: str) -> str:
    """Merge summaries when LLM is offline."""
    parts = [f"Research summary for: {topic}\n"]
    for i, s in enumerate(summaries, 1):
        title = getattr(s, "title", f"Source {i}")
        text  = getattr(s, "text",  "")
        parts.append(f"\n### {title}\n\n{text}")
    return "\n".join(parts)


def _to_html(topic: str, md: str, ts: str) -> str:
    """Convert the markdown report to a styled HTML page."""
    body = md
    body = re.sub(r'^# (.+)$',   r'<h1>\1</h1>',   body, flags=re.MULTILINE)
    body = re.sub(r'^## (.+)$',  r'<h2>\1</h2>',   body, flags=re.MULTILINE)
    body = re.sub(r'^### (.+)$', r'<h3>\1</h3>',   body, flags=re.MULTILINE)
    body = re.sub(r'^[•\-\*] (.+)$', r'<li>\1</li>', body, flags=re.MULTILINE)
    body = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', body)
    body = re.sub(r'\*(.+?)\*',     r'<em>\1</em>',         body)
    body = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2" target="_blank">\1</a>', body)
    body = re.sub(r'---+', r'<hr>', body)
    body = re.sub(r'\n\n+', r'</p><p>', body)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{topic} — NEXUS</title>
  <style>
    *  {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      max-width: 860px; margin: 40px auto; padding: 0 24px;
      background: #0d1117; color: #c9d1d9; line-height: 1.75;
    }}
    .topbar {{
      background: #161b22; border: 1px solid #30363d; border-radius: 8px;
      padding: 14px 20px; margin-bottom: 2em; display: flex;
      justify-content: space-between; align-items: center;
    }}
    .topbar .brand {{ color: #58a6ff; font-weight: 700; font-size: 1.05em; }}
    .topbar .ts    {{ color: #8b949e; font-size: 0.85em; }}
    h1  {{ color: #58a6ff; font-size: 1.8em; border-bottom: 1px solid #30363d;
           padding-bottom: 10px; margin: 1.2em 0 0.6em; }}
    h2  {{ color: #79c0ff; font-size: 1.3em; margin: 1.8em 0 0.5em; }}
    h3  {{ color: #d2a8ff; font-size: 1.1em; margin: 1.4em 0 0.4em; }}
    p   {{ margin: 0.8em 0; }}
    li  {{ margin: 6px 0 6px 1.4em; list-style: disc; }}
    a   {{ color: #58a6ff; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    hr  {{ border: none; border-top: 1px solid #30363d; margin: 2em 0; }}
    em  {{ color: #8b949e; }}
    strong {{ color: #e6edf3; }}
  </style>
</head>
<body>
  <div class="topbar">
    <span class="brand">NEXUS</span>
    <span class="ts">Research Report &nbsp;·&nbsp; {ts}</span>
  </div>
  <p>{body}</p>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    # Parse flags
    rounds       = 2
    open_browser = True
    publish      = False
    topic_parts: list[str] = []

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--rounds" and i + 1 < len(args):
            rounds = int(args[i + 1]); i += 2
        elif a == "--no-browser":
            open_browser = False; i += 1
        elif a == "--publish":
            publish = True; i += 1
        else:
            topic_parts.append(a); i += 1

    topic = " ".join(topic_parts).strip()
    if not topic:
        print("  Usage: python research/deep_research.py \"your topic\" [--rounds N] [--publish] [--no-browser]")
        sys.exit(1)

    dr = DeepResearcher(rounds=rounds, open_browser=open_browser, publish=publish)
    dr.run(topic)
