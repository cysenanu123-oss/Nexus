"""
core/context.py
Holds the current session state — what's happening RIGHT NOW.
Shared across brain, memory, planner, and conversation.
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    role:      str    # "user" or "nexus"
    text:      str
    timestamp: float  = field(default_factory=time.time)
    intent:    str    = ""
    action:    str    = ""


class Context:
    """
    Live session context — resets every time NEXUS starts.
    Tracks conversation history, active task, current user focus,
    and the last action output for follow-up questions.
    """

    def __init__(self, max_history: int = 20):
        self.max_history   = max_history
        self.history:  list[Turn] = []
        self.user_name:    str  = "Cyril"
        self.active_task:  Optional[str] = None
        self.last_intent:  Optional[str] = None
        self.last_target:  Optional[str] = None
        self.session_start = time.time()

        # ── Action output memory (for follow-up questions) ────────────────
        self.last_output:        Optional[str] = None   # raw text from last action
        self.last_output_domain: Optional[str] = None   # "cyber", "system", "research", "code"
        self.last_output_time:   float         = 0.0    # when it was produced

    # ── Output tracking ──────────────────────────────────────────────────

    def set_last_output(self, output: str, domain: str):
        """Store the output of the last action so follow-up questions can reference it."""
        self.last_output        = output
        self.last_output_domain = domain
        self.last_output_time   = time.time()

    def get_recent_output(self, max_age_seconds: int = 300) -> Optional[str]:
        """Get the last output if it's recent enough (default: within 5 minutes)."""
        if self.last_output and (time.time() - self.last_output_time) < max_age_seconds:
            return self.last_output
        return None

    # ── Conversation history ─────────────────────────────────────────────

    def add_turn(self, role: str, text: str, intent: str = "", action: str = ""):
        turn = Turn(role=role, text=text, intent=intent, action=action)
        self.history.append(turn)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def last_user_text(self) -> Optional[str]:
        for turn in reversed(self.history):
            if turn.role == "user":
                return turn.text
        return None

    def last_nexus_text(self) -> Optional[str]:
        for turn in reversed(self.history):
            if turn.role == "nexus":
                return turn.text
        return None

    def recent_turns(self, n: int = 6) -> list[Turn]:
        return self.history[-n:]

    def summary(self) -> str:
        lines = [f"{t.role.upper()}: {t.text}" for t in self.recent_turns()]
        return "\n".join(lines)

    def clear(self):
        self.history.clear()
        self.active_task        = None
        self.last_intent        = None
        self.last_target        = None
        self.last_output        = None
        self.last_output_domain = None
        self.last_output_time   = 0.0
