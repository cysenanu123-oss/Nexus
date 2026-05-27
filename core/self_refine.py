"""
core/self_refine.py
NEXUS Self-Refine Loop — Madaan et al. NeurIPS 2023 + CRITIC (Gou et al. 2023).

Three-role loop: generate → critique → refine → critique → refine → stop.
Average gain: +20 absolute across 7 tasks (original paper).

CRITIC variant: routes verification through external tools (code runner, search)
instead of pure self-evaluation.

Applied in NEXUS for:
  - Code generation (runs the code, feeds errors back)
  - Research answers (fact-checks key claims)
  - Any response where quality > speed

Usage:
    sr = SelfRefine(llm=brain.llm)
    final = sr.refine_code("write a function that reverses a string")
    final = sr.refine_answer("what is the capital of Ghana?")
    final = sr.refine(task, initial_output, verifier_fn=my_checker)
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import textwrap
from typing import Callable, Optional

log = logging.getLogger("nexus.self_refine")

MAX_ITERATIONS  = 3
STOP_SIGNAL     = "NO_ISSUES"


_FEEDBACK_PROMPT = """You are a critical reviewer. Evaluate this response and give specific, actionable feedback.
If it is correct and complete, respond with exactly: {stop}
Otherwise, describe 1-2 specific problems and how to fix them. Be concise.

Task: {task}
Response: {response}

Feedback:"""

_REFINE_PROMPT = """Improve this response based on the feedback. Make the minimal necessary changes.

Task: {task}
Current response: {response}
Feedback: {feedback}
History of previous attempts: {history}

Improved response:"""

_CODE_FEEDBACK_PROMPT = """You are a code reviewer. Evaluate this Python code.
If it is correct, efficient, and complete, respond with exactly: {stop}
Otherwise, list specific bugs or improvements needed. Be brief.

Task: {task}
Code:
```python
{response}
```
Execution result: {exec_result}

Feedback:"""


class SelfRefine:
    """
    Iterative output improvement via generate → critique → refine loop.
    Supports external verifiers (CRITIC pattern) for grounded feedback.
    """

    def __init__(self, llm=None):
        self._llm = llm

    def set_llm(self, llm):
        self._llm = llm

    # ── Main entry points ────────────────────────────────────────────────

    def refine(self, task: str, initial_output: str,
               verifier_fn: Optional[Callable[[str], str]] = None,
               max_iter: int = MAX_ITERATIONS) -> str:
        """
        Generic self-refine loop.
        verifier_fn(output) → feedback string or STOP_SIGNAL
        """
        if not self._llm:
            return initial_output

        output  = initial_output
        history = []

        for i in range(max_iter):
            feedback = self._get_feedback(task, output, verifier_fn)
            log.debug("Iter %d feedback: %s", i + 1, feedback[:80])

            if STOP_SIGNAL in feedback.upper() or not feedback.strip():
                log.info("Self-refine converged after %d iteration(s).", i + 1)
                break

            history.append(output)
            output = self._get_refinement(task, output, feedback, history)

        return output

    def refine_code(self, task: str, initial_code: str = "",
                    max_iter: int = MAX_ITERATIONS) -> str:
        """
        Code-specific refine loop with execution feedback (CRITIC pattern).
        Runs the code in a sandbox and feeds errors back to the model.
        """
        if not self._llm:
            return initial_code

        if not initial_code:
            initial_code = self._generate_initial(task)

        code    = initial_code
        history = []

        for i in range(max_iter):
            exec_result = self._run_code_safe(code)
            feedback    = self._get_code_feedback(task, code, exec_result)
            log.debug("Code iter %d: exec=%r, feedback=%s", i + 1, exec_result[:60], feedback[:60])

            if STOP_SIGNAL in feedback.upper() or (not exec_result.strip() and not feedback.strip()):
                log.info("Code self-refine converged after %d iteration(s).", i + 1)
                break

            history.append(code)
            code = self._refine_code_step(task, code, feedback, exec_result, history)

        return code

    def refine_answer(self, task: str, initial_answer: str = "",
                      max_iter: int = 2) -> str:
        """
        Answer refinement — simpler, lower budget (max 2 iters by default).
        """
        if not self._llm:
            return initial_answer

        if not initial_answer:
            initial_answer = self._generate_initial(task)

        return self.refine(task, initial_answer, max_iter=max_iter)

    # ── Internals ─────────────────────────────────────────────────────────

    def _generate_initial(self, task: str) -> str:
        try:
            return self._llm.ask(task, max_tokens=600).strip()
        except Exception as e:
            log.warning("Initial generation failed: %s", e)
            return ""

    def _get_feedback(self, task: str, response: str,
                      verifier_fn: Optional[Callable]) -> str:
        if verifier_fn:
            try:
                return verifier_fn(response)
            except Exception as e:
                log.warning("External verifier failed: %s", e)

        try:
            prompt = _FEEDBACK_PROMPT.format(
                stop=STOP_SIGNAL, task=task[:500], response=response[:1000]
            )
            return self._llm.ask(prompt, max_tokens=120).strip()
        except Exception as e:
            log.warning("Feedback generation failed: %s", e)
            return STOP_SIGNAL

    def _get_refinement(self, task: str, response: str,
                        feedback: str, history: list[str]) -> str:
        history_text = "\n---\n".join(f"Attempt {i+1}: {h[:200]}" for i, h in enumerate(history[-2:]))
        try:
            prompt = _REFINE_PROMPT.format(
                task=task[:500],
                response=response[:1000],
                feedback=feedback[:300],
                history=history_text or "None"
            )
            return self._llm.ask(prompt, max_tokens=800).strip()
        except Exception as e:
            log.warning("Refinement failed: %s", e)
            return response

    def _get_code_feedback(self, task: str, code: str, exec_result: str) -> str:
        try:
            prompt = _CODE_FEEDBACK_PROMPT.format(
                stop=STOP_SIGNAL, task=task[:400],
                response=code[:1200], exec_result=exec_result[:300]
            )
            return self._llm.ask(prompt, max_tokens=120).strip()
        except Exception as e:
            log.warning("Code feedback failed: %s", e)
            return STOP_SIGNAL

    def _refine_code_step(self, task: str, code: str, feedback: str,
                          exec_result: str, history: list[str]) -> str:
        history_text = f"Previous attempts: {len(history)}"
        try:
            prompt = (
                f"Fix this Python code based on the feedback and execution result.\n\n"
                f"Task: {task[:400]}\n\n"
                f"Current code:\n```python\n{code[:1200]}\n```\n\n"
                f"Execution result: {exec_result[:300]}\n"
                f"Feedback: {feedback[:300]}\n"
                f"{history_text}\n\n"
                f"Write only the corrected Python code, no explanation:"
            )
            raw = self._llm.ask(prompt, max_tokens=800).strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            return raw
        except Exception as e:
            log.warning("Code refinement failed: %s", e)
            return code

    def _run_code_safe(self, code: str, timeout: int = 5) -> str:
        """Run Python code in a subprocess sandbox. Returns stdout+stderr."""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                fname = f.name

            result = subprocess.run(
                ["python3", fname],
                capture_output=True, text=True, timeout=timeout
            )
            output = result.stdout.strip()
            if result.stderr.strip():
                output = (output + "\n" + result.stderr.strip()).strip()
            return output[:500]
        except subprocess.TimeoutExpired:
            return "TimeoutError: code ran too long"
        except Exception as e:
            return f"Error: {e}"
