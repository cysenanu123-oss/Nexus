"""
core/orchestrator.py
NEXUS Supervisor / Orchestrator Pattern — Phase 16, Lesson 05.
Inspired by Anthropic's Research system (+90.2% vs single-agent on research evals).

For complex tasks, instead of a single LLM call, spawns parallel worker agents,
each with a fresh context and narrow instruction, then synthesizes results.

Architecture:
    Lead (Brain) → decomposes task
        ↓ ↓ ↓
    Worker1 | Worker2 | Worker3  ← parallel, fresh context each
        ↓ ↓ ↓
    Lead synthesizes → final answer

Usage:
    orch = Orchestrator(llm=brain.llm, researcher=brain.researcher)
    result = orch.run("Explain quantum computing and its impact on cryptography")

    # Returns the synthesized answer
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("nexus.orchestrator")

MAX_WORKERS      = 4
WORKER_TIMEOUT   = 60    # seconds per worker
MAX_SUBTASKS     = 5

_DECOMPOSE_PROMPT = """Break this complex task into {max_sub} independent sub-questions or sub-tasks.
Each sub-task should be narrow and self-contained — a worker agent will handle it alone.
Return a JSON array of strings. Only return the JSON.

Task: {task}

Sub-tasks:"""

_WORKER_PROMPT = """You are a specialized worker agent. Answer this specific sub-task thoroughly and concisely.
Focus only on this sub-task. Do not discuss other aspects.

Sub-task: {subtask}
Context: {context}

Answer:"""

_SYNTHESIZE_PROMPT = """You are a lead agent. Synthesize these worker results into a coherent, comprehensive answer.
Eliminate redundancy. Ensure logical flow. The final answer should directly address the original task.

Original task: {task}

Worker results:
{results}

Synthesized answer:"""

_COMPLEXITY_KEYWORDS = (
    "explain and", "compare", "analyze", "research", "comprehensive",
    "in depth", "detailed", "thorough", "all aspects", "everything about",
    "what are all", "pros and cons", "history and"
)


@dataclass
class WorkerResult:
    subtask:  str
    answer:   str
    success:  bool
    error:    str = ""


@dataclass
class OrchestrationResult:
    task:          str
    subtasks:      list[str]
    worker_results: list[WorkerResult]
    final_answer:  str
    used_parallel: bool = False

    def __str__(self) -> str:
        return self.final_answer


class Orchestrator:
    """
    Supervisor-worker multi-agent orchestrator.
    Decides when to parallelize vs handle directly.
    """

    def __init__(self, llm=None, researcher=None, memory=None):
        self._llm        = llm
        self._researcher = researcher
        self._memory     = memory
        log.info("Orchestrator ready.")

    def set_components(self, llm=None, researcher=None, memory=None):
        if llm:        self._llm        = llm
        if researcher: self._researcher = researcher
        if memory:     self._memory     = memory

    # ── Public API ────────────────────────────────────────────────────────

    def should_orchestrate(self, task: str) -> bool:
        """Decide if this task warrants multi-agent decomposition."""
        t = task.lower()
        if len(t) < 40:
            return False
        return any(kw in t for kw in _COMPLEXITY_KEYWORDS)

    def run(self, task: str, context: str = "") -> str:
        """
        Main entry. Decomposes task, runs workers in parallel, synthesizes.
        Falls back to direct LLM call if no LLM available.
        """
        if not self._llm:
            return self._direct_answer(task)

        log.info("Orchestrating: %r", task[:80])

        # 1. Decompose
        subtasks = self._decompose(task)
        if not subtasks or len(subtasks) < 2:
            return self._direct_answer(task)

        log.info("Decomposed into %d subtasks.", len(subtasks))

        # 2. Run workers in parallel
        worker_results = self._run_workers(subtasks, context)

        # 3. Synthesize
        final = self._synthesize(task, worker_results)

        result = OrchestrationResult(
            task=task,
            subtasks=subtasks,
            worker_results=worker_results,
            final_answer=final,
            used_parallel=True,
        )

        return result.final_answer

    # ── Internals ─────────────────────────────────────────────────────────

    def _decompose(self, task: str) -> list[str]:
        try:
            import json
            prompt = _DECOMPOSE_PROMPT.format(task=task[:800], max_sub=min(MAX_SUBTASKS, 4))
            raw = self._llm.ask(prompt, max_tokens=300).strip()
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                subtasks = json.loads(raw[start:end])
                if isinstance(subtasks, list) and subtasks:
                    return [str(s) for s in subtasks[:MAX_SUBTASKS]]
        except Exception as e:
            log.warning("Decomposition failed: %s", e)
        return []

    def _run_workers(self, subtasks: list[str], context: str) -> list[WorkerResult]:
        results = []
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(subtasks))) as pool:
            futures = {
                pool.submit(self._worker, subtask, context): subtask
                for subtask in subtasks
            }
            for future in as_completed(futures, timeout=WORKER_TIMEOUT * len(subtasks)):
                subtask = futures[future]
                try:
                    answer = future.result(timeout=WORKER_TIMEOUT)
                    results.append(WorkerResult(subtask=subtask, answer=answer, success=True))
                    log.debug("Worker done: %r", subtask[:50])
                except TimeoutError:
                    results.append(WorkerResult(subtask=subtask, answer="", success=False, error="timeout"))
                except Exception as e:
                    results.append(WorkerResult(subtask=subtask, answer="", success=False, error=str(e)))
        return results

    def _worker(self, subtask: str, context: str) -> str:
        """Single worker — runs in its own thread with a fresh LLM call."""
        # Try researcher first for research-type subtasks
        research_keywords = ("what is", "who is", "explain", "history", "how does", "why does")
        if self._researcher and any(kw in subtask.lower() for kw in research_keywords):
            try:
                result = self._researcher.answer(subtask)
                if result and len(result) > 50:
                    return result[:1500]
            except Exception:
                pass

        # Fallback to LLM
        try:
            prompt = _WORKER_PROMPT.format(
                subtask=subtask[:600],
                context=context[:300] if context else "No additional context."
            )
            return self._llm.ask(prompt, max_tokens=500).strip()
        except Exception as e:
            log.warning("Worker LLM call failed: %s", e)
            return f"[Worker failed: {e}]"

    def _synthesize(self, task: str, results: list[WorkerResult]) -> str:
        successful = [r for r in results if r.success and r.answer]
        if not successful:
            return self._direct_answer(task)

        results_text = "\n\n".join(
            f"Sub-task: {r.subtask}\nAnswer: {r.answer[:800]}"
            for r in successful
        )
        try:
            prompt = _SYNTHESIZE_PROMPT.format(
                task=task[:600], results=results_text[:4000]
            )
            return self._llm.ask(prompt, max_tokens=800).strip()
        except Exception as e:
            log.warning("Synthesis failed: %s", e)
            return "\n\n".join(r.answer for r in successful)

    def _direct_answer(self, task: str) -> str:
        if not self._llm:
            return "LLM unavailable."
        try:
            return self._llm.ask(task, max_tokens=600).strip()
        except Exception as e:
            return f"Error: {e}"
