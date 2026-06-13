"""
automation/executor.py
NEXUS Task Executor — runs an ExecutionPlan step by step.

Receives an ExecutionPlan from the planner and executes each Step
using the appropriate agent (shell, gui, web, wait, check).

Features:
  - Dependency resolution (steps wait for their depends_on)
  - Retry logic with configurable attempts
  - Timeout enforcement per step
  - Partial execution (continues on optional step failures)
  - Live progress callbacks for the Reporter
  - Dry-run mode (plans without executing)

Pipeline:
    ExecutionPlan
           ↓
    executor.run(plan) → ExecutionResult
           ↓
    per Step → ShellAgent | GUIAgent | WebAgent | ...
           ↓
    StepResult (success, output, elapsed)

Usage:
    from automation.planner  import TaskPlanner
    from automation.executor import Executor

    planner  = TaskPlanner()
    executor = Executor()

    plan   = planner.plan("open firefox and go to github.com")
    result = executor.run(plan)
    print(result.summary())
"""

from __future__ import annotations

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, Any

from automation.checkpoint import CheckpointStore

log = logging.getLogger("nexus.automation.executor")


# ─────────────────────────────────────────────────────────────
#  RESULT DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Result of executing one Step."""
    step_index:  int
    step_desc:   str
    success:     bool
    output:      str   = ""
    error:       str   = ""
    elapsed_sec: float = 0.0
    skipped:     bool  = False
    retried:     int   = 0

    def __str__(self) -> str:
        icon = "✓" if self.success else ("⊘" if self.skipped else "✗")
        timing = f" ({self.elapsed_sec:.2f}s)"
        retry_note = f" [retried {self.retried}x]" if self.retried else ""
        return f"  {icon} Step {self.step_index}: {self.step_desc}{timing}{retry_note}"


@dataclass
class ExecutionResult:
    """Aggregated result of running an entire ExecutionPlan."""
    instruction:   str
    step_results:  list[StepResult] = field(default_factory=list)
    success:       bool             = False
    aborted_at:    Optional[int]    = None   # step index that caused abort
    total_elapsed: float            = 0.0
    dry_run:       bool             = False

    @property
    def steps_run(self) -> int:
        return sum(1 for r in self.step_results if not r.skipped)

    @property
    def steps_passed(self) -> int:
        return sum(1 for r in self.step_results if r.success)

    @property
    def steps_failed(self) -> int:
        return sum(1 for r in self.step_results if not r.success and not r.skipped)

    def summary(self) -> str:
        lines = [
            f"{'[DRY RUN] ' if self.dry_run else ''}Execution: {self.instruction!r}",
            f"Result  : {'SUCCESS' if self.success else 'FAILED'}",
            f"Steps   : {self.steps_passed}/{self.steps_run} passed",
            f"Elapsed : {self.total_elapsed:.2f}s",
        ]
        if self.aborted_at:
            lines.append(f"Aborted : at step {self.aborted_at}")
        lines.append("")
        for r in self.step_results:
            lines.append(str(r))
            if r.error:
                lines.append(f"    ↳ Error: {r.error}")
            if r.output and len(r.output) < 200:
                lines.append(f"    ↳ Output: {r.output.strip()}")
        return "\n".join(lines)

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (f"ExecutionResult({status}, "
                f"{self.steps_passed}/{self.steps_run} steps, "
                f"{self.total_elapsed:.2f}s)")


# ─────────────────────────────────────────────────────────────
#  EXECUTOR
# ─────────────────────────────────────────────────────────────

class Executor:
    """
    Runs an ExecutionPlan step by step using the right agent.

    Usage:
        executor = Executor()
        result   = executor.run(plan)
        print(result.summary())

        # With progress callback
        def on_progress(step, step_result):
            print(f"  {step_result}")

        result = executor.run(plan, on_progress=on_progress)

        # Dry run (plan only, no execution)
        result = executor.run(plan, dry_run=True)
    """

    def __init__(self, confirm_high_risk: bool = True):
        """
        Parameters
        ----------
        confirm_high_risk : prompt user before running high-risk plans
        """
        self.confirm_high_risk = confirm_high_risk

        # Lazy-load agents — only import when actually needed
        self._shell_agent = None
        self._gui_agent   = None
        self._web_agent   = None

        self._checkpoints = CheckpointStore()

        log.info("Executor ready.")

    # ── Public API ────────────────────────────────────────────

    def run(
        self,
        plan,
        dry_run:     bool     = False,
        on_progress: Optional[Callable] = None,
    ) -> ExecutionResult:
        """
        Execute an ExecutionPlan.

        Parameters
        ----------
        plan        : ExecutionPlan from TaskPlanner
        dry_run     : if True, log steps but don't execute
        on_progress : callback(step, step_result) fired after each step

        Returns
        -------
        ExecutionResult
        """
        if not plan.success:
            log.error("Cannot execute a failed plan: %s", plan.error)
            return ExecutionResult(
                instruction=plan.instruction,
                success=False,
            )

        # Safety gate for high-risk plans
        if (not dry_run and
                plan.risk_level == "high" and
                plan.requires_confirmation and
                self.confirm_high_risk):
            if not self._confirm(plan):
                log.info("User declined high-risk plan.")
                return ExecutionResult(
                    instruction=plan.instruction,
                    success=False,
                    aborted_at=0,
                )

        log.info("Executing plan: %r (%d steps, risk=%s%s)",
                 plan.instruction, plan.step_count, plan.risk_level,
                 " [DRY RUN]" if dry_run else "")

        t0           = time.time()
        step_results: list[StepResult] = []
        completed:    dict[int, bool]  = {}   # step_index → success

        # ── Durable execution: checkpoint + resume ────────────
        run_id = None
        already_done: set[int] = set()
        if not dry_run:
            # Check for an unfinished run with the same instruction
            existing_id = self._checkpoints.find_incomplete_run(plan.instruction)
            if existing_id:
                already_done = self._checkpoints.completed_steps(existing_id)
                run_id = existing_id
                log.info("Resuming run #%d — %d steps already done.", run_id, len(already_done))
            else:
                run_id = self._checkpoints.start_run(plan.instruction, plan.step_count)

        for step in plan.steps:
            # Check if dependencies are satisfied
            for dep in step.depends_on:
                if not completed.get(dep, False):
                    log.warning("Step %d skipped — dependency %d failed/not run.",
                                step.index, dep)
                    sr = StepResult(
                        step_index=step.index,
                        step_desc=step.description,
                        success=False,
                        skipped=True,
                        error=f"Dependency step {dep} failed.",
                    )
                    step_results.append(sr)
                    completed[step.index] = False
                    if on_progress:
                        on_progress(step, sr)
                    if not step.optional:
                        # Non-optional skipped step aborts the plan
                        return ExecutionResult(
                            instruction   = plan.instruction,
                            step_results  = step_results,
                            success       = False,
                            aborted_at    = step.index,
                            total_elapsed = time.time() - t0,
                            dry_run       = dry_run,
                        )
                    break

            # Resume: skip steps that already completed in a prior run
            if step.index in already_done:
                log.info("Skipping already-done step %d (resume).", step.index)
                sr = StepResult(
                    step_index=step.index, step_desc=step.description,
                    success=True, output="[resumed from checkpoint]"
                )
                step_results.append(sr)
                completed[step.index] = True
                continue

            # Execute the step
            sr = self._run_step(step, dry_run=dry_run)
            step_results.append(sr)
            completed[step.index] = sr.success

            # Checkpoint this step result
            if run_id and not dry_run:
                self._checkpoints.save_step(
                    run_id, step.index, step.description,
                    sr.success, sr.output, sr.error, sr.elapsed_sec
                )

            if on_progress:
                try:
                    on_progress(step, sr)
                except Exception as e:
                    log.debug("on_progress callback error: %s", e)

            # Abort on non-optional failure
            if not sr.success and not step.optional and not dry_run:
                log.warning("Step %d failed — aborting plan.", step.index)
                if run_id:
                    self._checkpoints.finish_run(run_id, success=False)
                return ExecutionResult(
                    instruction   = plan.instruction,
                    step_results  = step_results,
                    success       = False,
                    aborted_at    = step.index,
                    total_elapsed = time.time() - t0,
                    dry_run       = dry_run,
                )

        overall_success = all(
            r.success or r.skipped
            for r in step_results
            if not any(plan.steps[i].optional
                       for i, s in enumerate(plan.steps)
                       if s.index == r.step_index)
        )

        if run_id and not dry_run:
            self._checkpoints.finish_run(run_id, success=overall_success)

        return ExecutionResult(
            instruction   = plan.instruction,
            step_results  = step_results,
            success       = overall_success,
            total_elapsed = time.time() - t0,
            dry_run       = dry_run,
        )

    # ── Step runner ───────────────────────────────────────────

    def _run_step(self, step, dry_run: bool = False) -> StepResult:
        """Execute a single step with advanced retry logic."""
        if dry_run:
            log.info("[DRY RUN] %s", step)
            return StepResult(
                step_index=step.index,
                step_desc=step.description,
                success=True,
                output="[dry run]",
            )

        log.info("Running: %s", step)

        # Use new retry system for robust execution
        from core.retry_system import RetryExecutor, RetryConfig

        # Configure retry based on step settings
        max_attempts = max(step.retry + 1, 5)  # Ensure at least 5 attempts
        config = RetryConfig(
            max_attempts=max_attempts,
            base_timeout=getattr(step, 'timeout', 30.0)
        )

        executor = RetryExecutor(config)

        # Create a wrapper function for the step execution
        def execute_step():
            return self._dispatch_step(step)

        # Execute with retry system
        retry_result = executor.execute_with_retry(execute_step)

        return StepResult(
            step_index  = step.index,
            step_desc   = step.description,
            success     = retry_result.success,
            output      = retry_result.final_stdout if retry_result.success else "",
            error       = retry_result.failure_reason if not retry_result.success else "",
            elapsed_sec = retry_result.total_elapsed,
            retried     = max(0, retry_result.attempt_count - 1),
        )

    # ── Step dispatcher ───────────────────────────────────────

    def _dispatch_step(self, step) -> tuple[bool, str]:
        """
        Route a step to the correct agent.

        Returns
        -------
        (success: bool, output: str)
        """
        stype = step.type.lower()

        if stype == "shell":
            return self._get_shell_agent().run(step)

        elif stype == "gui":
            return self._get_gui_agent().run(step)

        elif stype == "web":
            return self._get_shell_agent().run(step)   # web steps via shell (browser)

        elif stype == "wait":
            return self._run_wait(step)

        elif stype == "check":
            return self._get_shell_agent().check(step)

        elif stype == "api":
            return self._run_api(step)

        elif stype == "llm":
            return self._run_llm(step)

        else:
            return False, f"Unknown step type: {stype!r}"

    # ── Built-in step handlers ────────────────────────────────

    def _run_wait(self, step) -> tuple[bool, str]:
        """Handle wait steps (sleep, wait_for_text, wait_for_window)."""
        action = step.action.lower()

        if action == "sleep":
            seconds = float(step.params.get("seconds", 1.0))
            log.info("Waiting %.1f seconds...", seconds)
            time.sleep(seconds)
            return True, f"Waited {seconds}s"

        elif action == "wait_for_text":
            timeout = float(step.params.get("timeout", 30.0))
            text    = step.target
            log.info("Waiting for text %r on screen (timeout=%.1fs)...", text, timeout)
            try:
                import sys, os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from vision.vision import Vision
                v = Vision()
                found = v.find_text(text, timeout=timeout)
                return found, f"Text {'found' if found else 'not found'}: {text!r}"
            except ImportError:
                log.warning("Vision module not available for wait_for_text.")
                time.sleep(min(timeout, 5.0))
                return True, "Vision unavailable — waited instead"

        elif action == "wait_for_window":
            seconds = float(step.params.get("timeout", 10.0))
            time.sleep(seconds)
            return True, f"Waited {seconds}s for window"

        return False, f"Unknown wait action: {action!r}"

    def _run_api(self, step) -> tuple[bool, str]:
        """Handle API steps (notifications, webhooks, etc.)."""
        action = step.action.lower()

        if action == "notify":
            msg = step.params.get("message", step.target)
            try:
                import subprocess
                subprocess.run(
                    ["notify-send", "NEXUS", msg],
                    capture_output=True, timeout=5
                )
                return True, f"Notification sent: {msg}"
            except Exception as e:
                return False, str(e)

        return False, f"API action not implemented: {action!r}"

    def _run_llm(self, step) -> tuple[bool, str]:
        """Run an LLM sub-task step."""
        try:
            from core.llm import get_llm
            llm = get_llm()
            if not llm.is_ready:
                return False, "LLM offline"
            output = llm.chat(step.target or step.params.get("prompt", ""))
            return True, output
        except Exception as e:
            return False, str(e)

    # ── Agent lazy loaders ────────────────────────────────────

    def _get_shell_agent(self):
        if self._shell_agent is None:
            from automation.shell_agent import ShellAgent
            self._shell_agent = ShellAgent()
        return self._shell_agent

    def _get_gui_agent(self):
        if self._gui_agent is None:
            from automation.gui_agent import GUIAgent
            self._gui_agent = GUIAgent()
        return self._gui_agent

    # ── Safety ────────────────────────────────────────────────

    def _confirm(self, plan) -> bool:
        """Ask user to confirm a high-risk plan."""
        print(f"\n  ⚠️  HIGH RISK PLAN DETECTED")
        print(f"  Instruction: {plan.instruction!r}")
        print(f"  Steps: {plan.step_count}")
        for step in plan.steps[:5]:
            print(f"    {step}")
        if plan.step_count > 5:
            print(f"    ... and {plan.step_count - 5} more")
        print()
        try:
            ans = input("  Execute this plan? [y/N] ").strip().lower()
            return ans == "y"
        except (EOFError, KeyboardInterrupt):
            return False


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    # Allow running as: python automation/executor.py ...
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    logging.basicConfig(level=logging.INFO)

    from automation.planner import TaskPlanner

    planner  = TaskPlanner(use_llm=False)
    executor = Executor(confirm_high_risk=False)

    tests = [
        ("wait 2 seconds",               False),
        ("run command echo hello nexus", False),
        ("run command ls -la /tmp",      False),
    ]

    if len(sys.argv) > 1:
        tests = [(" ".join(sys.argv[1:]), False)]

    for instruction, dry in tests:
        print(f"\n  Instruction: {instruction!r}  dry_run={dry}")
        plan   = planner.plan(instruction)
        result = executor.run(plan, dry_run=dry)
        print(result.summary())