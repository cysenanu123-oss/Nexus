"""
automation/automation.py
NEXUS Automation — the unified brain interface for computer automation.

This is the single entry point that brain.py calls.
It glues together:
    TaskPlanner  → breaks instruction into steps
    Executor     → runs each step (shell, GUI, web, wait, check)
    Reporter     → shows live progress and returns voice summary

Usage from brain.py:
    auto   = Automation()
    result = auto.run("open firefox and go to github.com")
    print(result.voice_summary)  # "Done. Completed 3 steps in 2.1 seconds."

Usage standalone:
    from automation.automation import Automation
    auto = Automation()
    result = auto.run("create file /tmp/nexus_test.txt")
    print(result.summary())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

log = logging.getLogger("nexus.automation")


# ─────────────────────────────────────────────────────────────
#  AUTOMATION RESULT
# ─────────────────────────────────────────────────────────────

@dataclass
class AutomationResult:
    """
    Final result returned to brain.py after automation completes.

    Attributes
    ----------
    instruction   : original user instruction
    success       : True if all required steps passed
    voice_summary : concise string for voice/TTS output
    detail        : full step-by-step log (for display)
    steps_run     : how many steps executed
    steps_passed  : how many steps succeeded
    elapsed       : total time in seconds
    plan_error    : set if planning itself failed
    dry_run       : True if this was a dry run
    """
    instruction:   str
    success:       bool  = False
    voice_summary: str   = ""
    detail:        str   = ""
    steps_run:     int   = 0
    steps_passed:  int   = 0
    elapsed:       float = 0.0
    plan_error:    str   = ""
    dry_run:       bool  = False

    def summary(self) -> str:
        return self.detail or self.voice_summary

    def __str__(self) -> str:
        status = "OK" if self.success else "FAIL"
        return (
            f"AutomationResult({status}, "
            f"{self.steps_passed}/{self.steps_run} steps, "
            f"{self.elapsed:.2f}s, "
            f"instruction={self.instruction!r})"
        )


# ─────────────────────────────────────────────────────────────
#  AUTOMATION
# ─────────────────────────────────────────────────────────────

class Automation:
    """
    Unified automation interface — the layer between brain.py and the
    planner / executor / reporter.

    Example:
        auto   = Automation()
        result = auto.run("open firefox and go to youtube.com")
        print(result.voice_summary)

    For dry runs (plan only, no execution):
        result = auto.run("delete /home/user/docs", dry_run=True)
        print(result.detail)
    """

    def __init__(
        self,
        verbose:          bool = True,
        confirm_high_risk: bool = True,
        use_llm:          bool = True,
    ):
        """
        Parameters
        ----------
        verbose           : show live step progress in terminal
        confirm_high_risk : prompt before running high-risk plans
        use_llm           : allow LLM planning fallback for complex instructions
        """
        self.verbose           = verbose
        self.confirm_high_risk = confirm_high_risk
        self.use_llm           = use_llm

        # Lazy-load sub-modules
        self._planner  = None
        self._executor = None
        self._reporter = None

        log.info(
            "Automation ready — verbose=%s, confirm_risky=%s, llm=%s",
            verbose, confirm_high_risk, use_llm,
        )

    # ── Public API ────────────────────────────────────────────

    def run(
        self,
        instruction: str,
        dry_run:     bool             = False,
        on_progress: Optional[Callable] = None,
    ) -> AutomationResult:
        """
        Plan and execute a natural language instruction.

        Parameters
        ----------
        instruction : what to do (e.g. "open firefox and go to github.com")
        dry_run     : plan but don't actually execute anything
        on_progress : optional callback(step, step_result) for custom UIs

        Returns
        -------
        AutomationResult with voice_summary and detail
        """
        instruction = instruction.strip()
        if not instruction:
            return AutomationResult(
                instruction="",
                success=False,
                voice_summary="No instruction given.",
            )

        t0       = time.time()
        planner  = self._get_planner()
        executor = self._get_executor()
        reporter = self._get_reporter()

        # ── Planning ──────────────────────────────────────────
        log.info("Planning: %r", instruction)
        plan = planner.plan(instruction)

        if not plan.success:
            log.warning("Planning failed: %s", plan.error)
            return AutomationResult(
                instruction=instruction,
                success=False,
                voice_summary=f"I couldn't figure out how to do that: {plan.error}",
                plan_error=plan.error,
                elapsed=time.time() - t0,
            )

        # ── Announce plan ─────────────────────────────────────
        reporter.on_start(plan)

        # Combine reporter callback with any external on_progress
        def _progress_cb(step, step_result):
            reporter.on_step(step, step_result)
            if on_progress:
                try:
                    on_progress(step, step_result)
                except Exception as e:
                    log.debug("External on_progress callback error: %s", e)

        # ── Execution ─────────────────────────────────────────
        exec_result = executor.run(plan, dry_run=dry_run, on_progress=_progress_cb)

        # ── Report ────────────────────────────────────────────
        voice = reporter.summarize(exec_result)

        return AutomationResult(
            instruction  = instruction,
            success      = exec_result.success,
            voice_summary= voice,
            detail       = exec_result.summary(),
            steps_run    = exec_result.steps_run,
            steps_passed = exec_result.steps_passed,
            elapsed      = exec_result.total_elapsed,
            dry_run      = dry_run,
        )

    def plan_only(self, instruction: str) -> str:
        """
        Plan without executing — returns a human-readable plan description.
        Useful for "what would you do if I said..." queries.
        """
        planner = self._get_planner()
        plan    = planner.plan(instruction)

        if not plan.success:
            return f"I couldn't plan that: {plan.error}"

        return plan.summary()

    def dry_run(self, instruction: str) -> AutomationResult:
        """Plan and simulate without executing."""
        return self.run(instruction, dry_run=True)

    # ── Lazy loaders ──────────────────────────────────────────

    def _get_planner(self):
        if self._planner is None:
            from automation.planner import TaskPlanner
            self._planner = TaskPlanner(use_llm=self.use_llm)
        return self._planner

    def _get_executor(self):
        if self._executor is None:
            from automation.executor import Executor
            self._executor = Executor(confirm_high_risk=self.confirm_high_risk)
        return self._executor

    def _get_reporter(self):
        if self._reporter is None:
            from automation.reporter import Reporter
            self._reporter = Reporter(verbose=self.verbose)
        return self._reporter


# ─────────────────────────────────────────────────────────────
#  SINGLETON for brain.py
# ─────────────────────────────────────────────────────────────

_automation_instance: Optional[Automation] = None


def get_automation(verbose: bool = True) -> Automation:
    """Return the shared Automation instance."""
    global _automation_instance
    if _automation_instance is None:
        _automation_instance = Automation(verbose=verbose)
    return _automation_instance


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    # Allow running as: python automation/automation.py ...
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    logging.basicConfig(level=logging.INFO)

    auto = Automation(verbose=True, confirm_high_risk=False)

    tests = [
        "run command echo 'hello from NEXUS automation'",
        "run command ls /tmp",
        "wait 1 seconds",
        "create file /tmp/nexus_auto_test.txt",
        "run command cat /tmp/nexus_auto_test.txt",
        "open firefox and wait 2 seconds",
    ]

    if len(sys.argv) > 1:
        tests = [" ".join(sys.argv[1:])]

    for instruction in tests:
        print(f"\n{'═'*60}")
        print(f"  Instruction: {instruction!r}")
        print(f"{'─'*60}")
        result = auto.run(instruction)
        print(f"\n  Voice: {result.voice_summary!r}")
        print(f"  Result: {result}")
