"""
automation/reporter.py
NEXUS Automation Reporter — real-time progress display for running plans.

Shows the user what NEXUS is doing as it does it:
  - Live step-by-step status (running / done / failed)
  - Rich console output with icons and timing
  - Optional silent mode (just logs, no terminal output)
  - Summary after execution

Usage:
    from automation.reporter import Reporter
    from automation.executor import Executor
    from automation.planner  import TaskPlanner

    reporter = Reporter()
    planner  = TaskPlanner()
    executor = Executor()

    plan   = planner.plan("open firefox and go to github.com")
    result = executor.run(plan, on_progress=reporter.on_step)
    reporter.summarize(result)
"""

from __future__ import annotations

import time
import logging
from typing import Optional

log = logging.getLogger("nexus.automation.reporter")


# ─────────────────────────────────────────────────────────────
#  REPORTER
# ─────────────────────────────────────────────────────────────

class Reporter:
    """
    Real-time progress reporter for automation plans.

    Modes:
      - verbose (default) : prints every step as it runs
      - silent            : only logs, no terminal output
      - minimal           : prints a single summary line at the end
    """

    # ANSI color codes for terminal
    _GREEN  = "\033[92m"
    _RED    = "\033[91m"
    _YELLOW = "\033[93m"
    _CYAN   = "\033[96m"
    _GREY   = "\033[90m"
    _BOLD   = "\033[1m"
    _RESET  = "\033[0m"

    def __init__(
        self,
        verbose: bool = True,
        use_color: bool = True,
        prefix: str = "  ",
    ):
        """
        Parameters
        ----------
        verbose   : print each step as it completes
        use_color : use ANSI colors in terminal output
        prefix    : line prefix (indentation)
        """
        self.verbose   = verbose
        self.use_color = use_color
        self.prefix    = prefix
        self._t_start  = time.time()
        self._step_log: list[dict] = []

        # Callback buffer for voice/UI output
        self._output_lines: list[str] = []

    # ── Step callback (used as on_progress) ───────────────────

    def on_step(self, step, step_result) -> None:
        """
        Called by the Executor after each step completes.

        Parameters
        ----------
        step        : Step object from planner
        step_result : StepResult from executor
        """
        self._step_log.append({
            "step":   step,
            "result": step_result,
        })

        if not self.verbose:
            log.info("%s", step_result)
            return

        # Format the line
        if step_result.skipped:
            icon  = self._color("⊘", self._YELLOW)
            state = self._color("SKIPPED", self._YELLOW)
        elif step_result.success:
            icon  = self._color("✓", self._GREEN)
            state = self._color("OK", self._GREEN)
        else:
            icon  = self._color("✗", self._RED)
            state = self._color("FAIL", self._RED)

        timing  = self._color(f"({step_result.elapsed_sec:.2f}s)", self._GREY)
        desc    = step.description or f"[{step.type}] {step.action}"
        line    = f"{self.prefix}{icon} {desc} {timing}"

        print(line)
        self._output_lines.append(line)

        # Show short output/error
        if step_result.output and not step_result.success:
            err_line = f"{self.prefix}   ↳ {self._color(step_result.output[:120], self._RED)}"
            print(err_line)
        elif step_result.output and len(step_result.output.strip()) > 0:
            # Only show output if it's brief and interesting
            out = step_result.output.strip()
            if len(out) <= 100 and out != "(no output)" and out != "[dry run]":
                out_line = f"{self.prefix}   ↳ {self._color(out, self._GREY)}"
                print(out_line)

    def on_start(self, plan) -> None:
        """Call before execution starts."""
        self._t_start      = time.time()
        self._step_log     = []
        self._output_lines = []

        if not self.verbose:
            return

        header = (
            f"\n{self.prefix}{self._color('▶ NEXUS AUTOMATION', self._BOLD + self._CYAN)}\n"
            f"{self.prefix}  Task : {plan.instruction!r}\n"
            f"{self.prefix}  Steps: {plan.step_count}  |  "
            f"Risk: {self._risk_color(plan.risk_level)}"
        )
        print(header)
        self._output_lines.append(header)

    def on_end(self, result) -> None:
        """Call after execution ends."""
        self.summarize(result)

    # ── Summary ───────────────────────────────────────────────

    def summarize(self, result) -> str:
        """
        Print and return a summary of the execution result.

        Returns a clean string suitable for voice output.
        """
        elapsed = result.total_elapsed
        passed  = result.steps_passed
        total   = result.steps_run

        if result.success:
            status_icon = self._color("✓", self._GREEN)
            status_word = self._color("DONE", self._GREEN)
        else:
            status_icon = self._color("✗", self._RED)
            status_word = self._color("FAILED", self._RED)

        lines = [
            f"\n{self.prefix}{'─' * 40}",
            (f"{self.prefix}{status_icon} {status_word}  "
             f"{self._color(f'{passed}/{total} steps', self._GREY)}  "
             f"{self._color(f'{elapsed:.1f}s', self._GREY)}"),
        ]

        if result.aborted_at:
            lines.append(
                f"{self.prefix}   Aborted at step {result.aborted_at}"
            )

        # Failed steps detail
        for sr in result.step_results:
            if not sr.success and not sr.skipped and sr.error:
                lines.append(
                    f"{self.prefix}   {self._color('✗', self._RED)} "
                    f"Step {sr.step_index}: {sr.error[:80]}"
                )

        summary_block = "\n".join(lines)

        if self.verbose:
            print(summary_block)

        # Voice-friendly summary
        if result.success:
            voice = f"Done. Completed {passed} step{'s' if passed != 1 else ''} in {elapsed:.1f} seconds."
        else:
            failed = result.steps_failed
            voice  = (
                f"Task failed at step {result.aborted_at}. "
                f"{passed} of {total} steps completed."
            )

        return voice

    # ── Output buffer (for voice pipeline) ───────────────────

    def get_output(self) -> str:
        """Return all printed output as a single string."""
        return "\n".join(self._output_lines)

    def clear(self) -> None:
        """Clear the output buffer."""
        self._output_lines = []
        self._step_log     = []

    # ── ANSI helpers ──────────────────────────────────────────

    def _color(self, text: str, code: str) -> str:
        if not self.use_color:
            return text
        return f"{code}{text}{self._RESET}"

    def _risk_color(self, risk: str) -> str:
        color_map = {
            "low":    self._GREEN,
            "medium": self._YELLOW,
            "high":   self._RED,
        }
        return self._color(risk.upper(), color_map.get(risk, self._GREY))


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    # Allow running as: python automation/reporter.py ...
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    logging.basicConfig(level=logging.INFO)

    from automation.planner  import TaskPlanner
    from automation.executor import Executor

    reporter = Reporter(verbose=True)
    planner  = TaskPlanner(use_llm=False)
    executor = Executor(confirm_high_risk=False)

    tests = [
        "wait 1 seconds",
        "run command echo 'NEXUS automation test'",
        "run command ls /tmp",
    ]

    if len(sys.argv) > 1:
        tests = [" ".join(sys.argv[1:])]

    for instruction in tests:
        plan   = planner.plan(instruction)
        reporter.on_start(plan)
        result = executor.run(plan, on_progress=reporter.on_step)
        voice  = reporter.on_end(result)
        print(f"\n  Voice: {voice!r}\n")
