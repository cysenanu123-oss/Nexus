"""
core/code_engine.py
NEXUS Code Engine — Claude Code-style coding assistant.

Every action is narrated in real time:
  ↳ [READING]   core/brain.py — 645 lines
  ◆ [THINKING]  Found _route() at line 185 — this is where we add the hook
  ✎ [EDITING]   core/brain.py:185 — adding CodeEngine dispatch
  ▶ [RUNNING]   python3 -c "from core.brain import Brain; print('OK')"
  ★ [LEARNED]   pyautogui click syntax — saved to knowledge base

Usage (from Brain or terminal):
    from core.code_engine import get_engine
    engine = get_engine()
    result = engine.work("add task tracking to brain.py")
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.stream_output import get_output
from core.knowledge import get_knowledge_base

log = logging.getLogger("nexus.code_engine")

PROJECT_ROOT = Path(__file__).parent.parent


# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────


@dataclass
class FileInfo:
    """Structural summary of an analysed Python file."""

    path: str
    lines: int = 0
    classes: list[dict] = field(default_factory=list)    # {"name", "line"}
    functions: list[dict] = field(default_factory=list)  # {"name", "line"}
    imports: list[str] = field(default_factory=list)

    def describe(self) -> str:
        parts = [f"{self.lines} lines"]
        if self.classes:
            parts.append(
                "classes: " + ", ".join(f"{c['name']}:{c['line']}" for c in self.classes)
            )
        if self.functions:
            parts.append(
                "functions: " + ", ".join(f"{f['name']}:{f['line']}" for f in self.functions[:8])
            )
        return " | ".join(parts)


@dataclass
class PlanStep:
    """One step in a CodePlan."""

    n: int
    action: str         # read | edit | create | run | search | think
    target: str         # file path, command, or description
    line: int = 0
    description: str = ""
    detail: str = ""    # specific code diff or content hint

    def __str__(self) -> str:
        ref = f":{self.line}" if self.line else ""
        return f"Step {self.n} [{self.action.upper()}] {self.target}{ref} — {self.description}"


@dataclass
class CodePlan:
    """A structured plan for one coding task."""

    goal: str
    analysis: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    files_to_read: list[str] = field(default_factory=list)
    files_to_modify: list[str] = field(default_factory=list)
    new_files: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def success(self) -> bool:
        return not self.error and bool(self.steps)

    def summary(self) -> str:
        lines = [f"Goal: {self.goal}", f"Analysis: {self.analysis}"]
        for step in self.steps:
            lines.append(f"  {step}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  CODE ENGINE
# ─────────────────────────────────────────────────────────────


class CodeEngine:
    """
    Claude Code-style coding assistant for NEXUS.

    Narrates every action in real time — file reads, analysis, edits,
    runs, searches — so you always know exactly what's happening and why.

    Pipeline for engine.work(instruction):
        1. Check knowledge base (maybe we already know the answer)
        2. Identify relevant files from instruction keywords
        3. Analyse each file (AST: classes, functions, line numbers)
        4. Plan changes with LLM or rule fallback
        5. Show plan to user
        6. On LLM failure → go online, learn, store, return
    """

    def __init__(self, llm=None, researcher=None):
        self._llm        = llm
        self._researcher = researcher
        self._out        = get_output()
        self._kb         = get_knowledge_base()
        log.info("CodeEngine ready.")

    # ── Public API ────────────────────────────────────────────

    def read_file(self, path: str, start: int = 1, end: Optional[int] = None) -> str:
        """
        Read a file, narrate the action, return numbered content.

        Example output:
           1 │ class Brain:
           2 │     def think(self, text: str) -> str:
        """
        full = self._resolve(path)
        if not full.exists():
            self._out.fail(f"File not found: {path}")
            return ""

        source = full.read_text(encoding="utf-8", errors="replace")
        lines  = source.splitlines()
        end    = end or len(lines)

        range_label = (
            f"lines {start}–{end}" if (start > 1 or end < len(lines)) else f"{len(lines)} lines"
        )
        self._out.reading(path, range_label)

        selected = lines[start - 1 : end]
        return "\n".join(f"{start + i:4d} │ {line}" for i, line in enumerate(selected))

    def analyze_file(self, path: str) -> FileInfo:
        """
        Parse a Python file's AST and return its structural summary.
        Narrates the read action.
        """
        full = self._resolve(path)
        if not full.exists():
            return FileInfo(path=path)

        source = full.read_text(encoding="utf-8", errors="replace")
        info   = FileInfo(path=path, lines=len(source.splitlines()))

        self._out.reading(path, f"{info.lines} lines — analysing structure")

        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    info.classes.append({"name": node.name, "line": node.lineno})
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    info.functions.append({"name": node.name, "line": node.lineno})
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    info.imports.append(ast.unparse(node))
        except SyntaxError as e:
            log.debug("AST parse failed for %s: %s", path, e)

        return info

    def plan(
        self,
        instruction: str,
        context_files: Optional[list[str]] = None,
    ) -> CodePlan:
        """
        Analyse the request and build a narrated step-by-step plan.

        1. Search knowledge base
        2. Find relevant files
        3. Analyse each file
        4. Ask LLM (or fallback to rules)
        """
        self._out.thinking(f"Parsing request: {instruction!r}")

        # ── Knowledge base check ──────────────────────────────
        kb_hits = self._kb.search(instruction, limit=2)
        if kb_hits:
            self._out.thinking(
                f"Found {len(kb_hits)} matching knowledge entries — using as context"
            )

        # ── Find relevant files ───────────────────────────────
        self._out.thinking("Identifying relevant files...")
        relevant = self._find_relevant_files(instruction, context_files)
        if relevant:
            self._out.thinking(f"Relevant files: {', '.join(relevant)}")

        # ── Analyse each file ─────────────────────────────────
        analyses: dict[str, FileInfo] = {}
        for fpath in relevant[:6]:
            analyses[fpath] = self.analyze_file(fpath)

        # ── Build plan ────────────────────────────────────────
        if self._llm and self._llm.is_ready:
            code_plan = self._llm_plan(instruction, analyses, kb_hits)
        else:
            code_plan = self._rule_plan(instruction, relevant)

        return code_plan

    def work(self, instruction: str) -> str:
        """
        Main entry point — plan changes and return a formatted plan summary.

        Does NOT apply edits automatically — shows the plan and returns the
        analysis. The user (or brain) decides whether to execute.
        """
        self._out.blank()
        self._out.planning(f"Working on: {instruction}")
        self._out.blank()

        code_plan = self.plan(instruction)

        if not code_plan.success:
            self._out.thinking("Planning hit a wall — going online for help...")
            online = self._go_online(instruction)
            if online:
                return online
            return f"Could not plan: {code_plan.error or 'no steps generated'}"

        self._show_plan(code_plan)
        self._out.blank()

        return self._format_response(code_plan)

    def run_code(self, path: str, args: Optional[list[str]] = None) -> str:
        """Run a Python file and return its output (stdout + stderr)."""
        full = self._resolve(path)
        cmd  = ["python3", str(full)] + (args or [])
        self._out.running(" ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(PROJECT_ROOT),
            )
            out = result.stdout.strip()
            err = result.stderr.strip()
            if err:
                out += f"\n[stderr] {err}"
            return out or "(no output)"
        except subprocess.TimeoutExpired:
            return "[timed out after 30s]"
        except Exception as exc:
            return f"[error] {exc}"

    def run_snippet(self, code: str) -> str:
        """Run a Python snippet and return its output."""
        cmd = ["python3", "-c", code]
        self._out.running(f"python3 -c {code[:60]!r}...")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
                cwd=str(PROJECT_ROOT),
            )
            return (result.stdout + result.stderr).strip() or "(no output)"
        except Exception as exc:
            return f"[error] {exc}"

    def search_and_learn(self, query: str) -> str:
        """
        Go online, find information, store it in the knowledge base.

        Called automatically when planning or LLM fails.
        Also callable directly: engine.search_and_learn("pyautogui syntax")
        """
        self._out.searching(query)

        if not self._researcher:
            self._out.warn("Researcher module not available — cannot go online.")
            return ""

        try:
            result = self._researcher.answer(query)
            if result and len(result) > 50:
                self._kb.store(
                    topic   = query,
                    content = result,
                    source  = "web_search",
                    tags    = ["auto-learned"],
                    quality = 0.8,
                )
                self._out.learned(
                    query[:60],
                    f"stored to knowledge base ({self._kb.count()} total entries)",
                )
            return result
        except Exception as exc:
            log.warning("search_and_learn failed: %s", exc)
            return ""

    def recall(self, query: str, limit: int = 3) -> str:
        """Look up what NEXUS knows about a topic from its knowledge base."""
        hits = self._kb.search(query, limit=limit)
        if not hits:
            return f"Nothing stored about {query!r} yet."
        parts = []
        for h in hits:
            snippet = h["content"][:400].replace("\n", " ")
            parts.append(f"[{h['topic']}]\n{snippet}")
        return "\n\n".join(parts)

    # ── Internal: file resolution ─────────────────────────────

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    # ── Internal: file discovery ──────────────────────────────

    def _find_relevant_files(
        self,
        instruction: str,
        hints: Optional[list[str]] = None,
    ) -> list[str]:
        """Return the most relevant project files for this instruction."""
        relevant: list[str] = list(hints or [])

        # Explicit file mentions in the instruction
        for match in re.findall(r"[\w/\-]+\.py", instruction):
            relevant.append(match)

        # Keyword → file heuristics
        kw_map: list[tuple[tuple[str, ...], list[str]]] = [
            (("brain", "think", "route", "dispatch"),     ["core/brain.py"]),
            (("planner", "plan", "automation"),            ["automation/planner.py"]),
            (("memory", "remember", "recall", "episode"), ["core/memory_manager.py"]),
            (("llm", "model", "ollama", "chat"),           ["core/llm.py"]),
            (("voice", "tts", "speak", "wake"),            ["voice/engine.py"]),
            (("intent", "parse", "intent_engine"),         ["core/intent_engine.py"]),
            (("conversation", "respond", "convo"),         ["core/conversation.py"]),
            (("code", "coding", "code_engine"),            ["core/code_engine.py",
                                                            "core/coding_assistant.py"]),
            (("knowledge", "learn", "knowledge base"),     ["core/knowledge.py"]),
            (("stream", "output", "narrat"),               ["core/stream_output.py"]),
            (("cyber", "hack", "scan", "nmap"),            ["cyber/cyber.py"]),
            (("router", "route", "category"),              ["core/router.py"]),
            (("skill", "plugin"),                          ["core/skill_manager.py",
                                                            "core/plugins.py"]),
            (("main", "banner", "repl", "prompt"),         ["main.py"]),
        ]
        instr_lower = instruction.lower()
        for keywords, files in kw_map:
            if any(k in instr_lower for k in keywords):
                relevant.extend(files)

        # Deduplicate, keep only existing files
        seen: set[str] = set()
        result: list[str] = []
        for f in relevant:
            if f not in seen:
                seen.add(f)
                if self._resolve(f).exists():
                    result.append(f)
        return result

    # ── Internal: LLM plan ────────────────────────────────────

    def _llm_plan(
        self,
        instruction: str,
        analyses: dict[str, FileInfo],
        kb_hits: list[dict],
    ) -> CodePlan:
        self._out.thinking("Asking LLM to build a detailed plan...")

        file_summaries = [
            f"  {path} ({info.describe()})" for path, info in analyses.items()
        ]
        kb_context = (
            "\n\nKnown context from knowledge base:\n"
            + "\n".join(f"  - {h['topic']}: {h['content'][:200]}" for h in kb_hits)
            if kb_hits
            else ""
        )

        system = textwrap.dedent("""
            You are NEXUS's internal code planner.
            Given a coding task and file analysis, produce a structured JSON plan.

            Return ONLY valid JSON, nothing else:
            {
              "goal": "one-line goal",
              "analysis": "2-3 sentence analysis — what needs to change and why, with specific file:line references",
              "steps": [
                {
                  "n": 1,
                  "action": "read|edit|create|run|search|think",
                  "target": "file path or command",
                  "line": 0,
                  "description": "exactly what this step does",
                  "detail": "specific code change, function name, or content"
                }
              ],
              "files_to_read": [],
              "files_to_modify": [],
              "new_files": []
            }

            Actions:
              read   — read/analyse a file
              edit   — modify existing file (specify line)
              create — create a new file
              run    — execute a command or snippet
              search — search online (when uncertain about an API or approach)
              think  — reasoning checkpoint (no file action)
        """).strip()

        user_msg = (
            f"Task: {instruction}\n\n"
            f"Files analysed:\n" + "\n".join(file_summaries) + kb_context
        )

        try:
            raw = self._llm.chat(user_msg, system=system, task="code")
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(raw)
        except Exception as exc:
            log.warning("LLM plan JSON failed: %s", exc)
            return self._rule_plan(instruction, list(analyses.keys()))

        steps = [
            PlanStep(
                n           = s.get("n", i + 1),
                action      = s.get("action", "think"),
                target      = s.get("target", ""),
                line        = int(s.get("line", 0)),
                description = s.get("description", ""),
                detail      = s.get("detail", ""),
            )
            for i, s in enumerate(data.get("steps", []))
        ]

        return CodePlan(
            goal             = data.get("goal", instruction),
            analysis         = data.get("analysis", ""),
            steps            = steps,
            files_to_read    = data.get("files_to_read", []),
            files_to_modify  = data.get("files_to_modify", []),
            new_files        = data.get("new_files", []),
        )

    # ── Internal: rule-based fallback plan ───────────────────

    def _rule_plan(self, instruction: str, files: list[str]) -> CodePlan:
        self._out.thinking("Building rule-based plan (LLM unavailable)...")

        if not files:
            return CodePlan(
                goal     = instruction,
                analysis = "",
                error    = "No relevant files identified.",
            )

        steps = [
            PlanStep(
                n           = i,
                action      = "read",
                target      = fpath,
                description = f"Read and analyse {fpath}",
            )
            for i, fpath in enumerate(files, 1)
        ]

        return CodePlan(
            goal          = instruction,
            analysis      = f"Will read {len(files)} file(s) and identify what needs changing.",
            steps         = steps,
            files_to_read = files,
        )

    # ── Internal: plan display ────────────────────────────────

    def _show_plan(self, plan: CodePlan):
        """Print the plan in formatted, coloured output."""
        import sys
        _R   = "\033[0m"
        _DIM = "\033[2m"
        _B   = "\033[1m"
        _CYN = "\033[96m"
        _GRN = "\033[92m"
        _YLW = "\033[93m"
        _MAG = "\033[95m"
        _BLU = "\033[94m"
        _WHT = "\033[97m"

        ACTION_COLOR = {
            "read":   _CYN,
            "edit":   _MAG,
            "create": _GRN,
            "run":    _BLU,
            "search": _YLW,
            "think":  _DIM,
        }

        print(f"  {_B}Goal{_R}      {plan.goal}", flush=True)
        if plan.analysis:
            wrapped = textwrap.fill(plan.analysis, width=70, subsequent_indent="            ")
            print(f"  {_B}Analysis{_R}  {wrapped}", flush=True)

        if plan.files_to_modify:
            print(f"  {_DIM}Modifies: {', '.join(plan.files_to_modify)}{_R}", flush=True)
        if plan.new_files:
            print(f"  {_DIM}Creates:  {', '.join(plan.new_files)}{_R}", flush=True)

        print(flush=True)

        for step in plan.steps:
            color = ACTION_COLOR.get(step.action, _WHT)
            ref   = f":{step.line}" if step.line else ""
            label = f"{color}{step.action.upper():<8}{_R}"
            target = f"{step.target}{ref}"
            print(f"  {_DIM}{step.n:2d}.{_R}  {label}  {target}", flush=True)
            if step.description:
                print(f"            {_DIM}{step.description}{_R}", flush=True)
            if step.detail:
                print(f"            {_DIM}→ {step.detail}{_R}", flush=True)

    # ── Internal: response formatting ────────────────────────

    def _format_response(self, plan: CodePlan) -> str:
        lines = [f"Here's my plan for: {plan.goal}"]
        if plan.analysis:
            lines.append(f"\n{plan.analysis}")
        lines.append(f"\n{len(plan.steps)} step(s) planned.")
        if plan.files_to_modify:
            lines.append(f"Files to modify: {', '.join(plan.files_to_modify)}")
        if plan.new_files:
            lines.append(f"New files: {', '.join(plan.new_files)}")
        return "\n".join(lines)

    # ── Internal: online fallback ─────────────────────────────

    def _go_online(self, query: str) -> str:
        return self.search_and_learn(query)


# ── Singleton ─────────────────────────────────────────────────

_engine: Optional[CodeEngine] = None


def get_engine(llm=None, researcher=None) -> CodeEngine:
    global _engine
    if _engine is None:
        _engine = CodeEngine(llm=llm, researcher=researcher)
    elif llm and _engine._llm is None:
        _engine._llm = llm
    elif researcher and _engine._researcher is None:
        _engine._researcher = researcher
    return _engine
