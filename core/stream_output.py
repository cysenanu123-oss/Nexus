"""
core/stream_output.py
NEXUS narrated real-time output — shows every action as it happens.

Like Claude Code's tool narration: NEXUS tells you what file it's reading,
what it's thinking, what it's changing, line by line, before and after.
"""

import sys
import datetime
from typing import Optional

# Standalone ANSI codes — no circular imports with main.py
_R   = "\033[0m"
_DIM = "\033[2m"
_B   = "\033[1m"
_CYN = "\033[96m"
_GRN = "\033[92m"
_YLW = "\033[93m"
_MAG = "\033[95m"
_BLU = "\033[94m"
_WHT = "\033[97m"
_RED = "\033[91m"


def _ts() -> str:
    return f"{_DIM}{datetime.datetime.now().strftime('%H:%M:%S')}{_R}"


def _strip_ansi(s: str) -> str:
    import re
    return re.sub(r"\033\[[0-9;]*m", "", s)


def _supports_color() -> bool:
    import os, platform
    if platform.system() == "Windows":
        return "WT_SESSION" in os.environ
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_COLOR = _supports_color()


def _c(code: str, text: str) -> str:
    return f"{code}{text}{_R}" if _COLOR else text


class StreamOutput:
    """
    Real-time narrated output for NEXUS — prints exactly what it's doing.

    thinking  ◆  internal reasoning steps
    reading   ↳  file reads with path:line reference
    editing   ✎  file edits with what changed
    running   ▶  shell/python execution
    searching ⌖  web searches
    learned   ★  new knowledge stored
    planning  ⊞  plan steps
    step      →  numbered plan step
    done      ✓  completion
    fail      ✗  failure
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def _print(self, icon: str, color: str, label: str, msg: str, detail: str = ""):
        if not self.enabled:
            return
        detail_part = f"  {_c(_DIM, detail)}" if detail else ""
        print(f"{_ts()} {_c(color, f'{icon} {label}')}  {msg}{detail_part}", flush=True)

    def thinking(self, msg: str):
        self._print("◆", _YLW, "[THINKING]", msg)

    def reading(self, path: str, detail: str = ""):
        self._print("↳", _CYN, "[READING] ", _c(_WHT, path), detail)

    def editing(self, path: str, detail: str = ""):
        self._print("✎", _MAG, "[EDITING] ", _c(_WHT, path), detail)

    def running(self, cmd: str):
        short = cmd if len(cmd) <= 80 else cmd[:77] + "..."
        self._print("▶", _BLU, "[RUNNING] ", _c(_DIM, short))

    def searching(self, query: str):
        self._print("⌖", _CYN, "[SEARCH]  ", query)

    def learned(self, topic: str, detail: str = ""):
        self._print("★", _GRN, "[LEARNED] ", topic, detail)

    def planning(self, msg: str):
        self._print("⊞", _WHT, "[PLAN]    ", msg)

    def step(self, n: int, total: int, msg: str, done: bool = False):
        if not self.enabled:
            return
        icon  = _c(_GRN, "✓") if done else _c(_DIM, "→")
        label = _c(_DIM, f"Step {n}/{total}:")
        print(f"  {icon}  {label} {msg}", flush=True)

    def done(self, msg: str):
        self._print("✓", _GRN, "[DONE]    ", msg)

    def fail(self, msg: str):
        self._print("✗", _RED, "[FAIL]    ", msg)

    def warn(self, msg: str):
        self._print("⚠", _YLW, "[WARN]    ", msg)

    def blank(self):
        if self.enabled:
            print(flush=True)

    def divider(self, char: str = "─", width: int = 60):
        if self.enabled:
            print(_c(_DIM, char * width), flush=True)

    def plan_header(self, goal: str, n_steps: int):
        """Print a plan summary header."""
        if not self.enabled:
            return
        self.blank()
        self.divider()
        print(f"  {_c(_B + _WHT, 'PLAN')}  {goal}", flush=True)
        print(f"  {_c(_DIM, f'{n_steps} step(s) — confirm to proceed')}", flush=True)
        self.divider()
        self.blank()

    def file_ref(self, path: str, line: int = 0, description: str = "") -> str:
        """Return a formatted file:line reference string."""
        ref = f"{path}:{line}" if line else path
        return f"{_c(_CYN, ref)}{(' — ' + description) if description else ''}"


# ── Singleton ─────────────────────────────────────────────────

_output: Optional[StreamOutput] = None


def get_output() -> StreamOutput:
    global _output
    if _output is None:
        _output = StreamOutput()
    return _output


def set_output_enabled(enabled: bool):
    get_output().enabled = enabled
