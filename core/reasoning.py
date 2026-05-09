"""
core/reasoning.py
NEXUS Reasoning Layer — decides HOW to handle each input.

This sits between the brain and the engines.
It answers: "What KIND of response does this input need?"

Decision types:
    ACTION       → dispatcher should execute something
    CONVERSATION → conversation engine should reply
    MEMORY_STORE → user is telling us something to remember
    MEMORY_QUERY → user is asking us to recall something
    PLAN         → multi-step execution needed
    CLARIFY      → we need more info before acting
"""

import logging
import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from core.intent_engine import Intent

log = logging.getLogger("nexus.reasoning")


class DecisionType(Enum):
    ACTION       = "action"
    CONVERSATION = "conversation"
    MEMORY_STORE = "memory_store"
    MEMORY_QUERY = "memory_query"
    PLAN         = "plan"
    CLARIFY      = "clarify"


@dataclass
class Decision:
    type:       DecisionType
    confidence: float
    reason:     str
    needs_clarification: Optional[str] = None


# Intents that map directly to system actions
ACTION_INTENTS = {
    "open_application", "close_application", "search_web", "open_url",
    "open_file", "open_folder", "list_files", "run_command",
    "system_control", "volume_control", "brightness_control",
    "network_scan", "system_info", "get_time",
}

# Intents that are conversational
CONVO_INTENTS = {
    "greet", "farewell", "ask_question", "unknown",
}

# Phrases that signal multi-step intent
MULTI_STEP_SIGNALS = re.compile(
    r"\b(and then|then|after that|and also|also)\b", re.I
)


class ReasoningEngine:
    """
    Decides what category of response an input needs.
    This keeps brain.py clean — all decision logic lives here.
    """

    def decide(self, text: str, intent: Intent) -> Decision:
        # ── Multi-step plan? ─────────────────────────────────────────────
        if MULTI_STEP_SIGNALS.search(text) and intent.intent in ACTION_INTENTS:
            return Decision(
                type=DecisionType.PLAN,
                confidence=0.85,
                reason="Compound command detected."
            )

        # ── Memory store ─────────────────────────────────────────────────
        if intent.intent == "remember":
            return Decision(
                type=DecisionType.MEMORY_STORE,
                confidence=0.95,
                reason="User wants to store information."
            )

        # ── Memory query ─────────────────────────────────────────────────
        if intent.intent == "recall":
            return Decision(
                type=DecisionType.MEMORY_QUERY,
                confidence=0.95,
                reason="User wants to retrieve stored information."
            )

        # ── System action ─────────────────────────────────────────────────
        if intent.intent in ACTION_INTENTS and intent.confidence >= 0.7:
            return Decision(
                type=DecisionType.ACTION,
                confidence=intent.confidence,
                reason=f"Intent '{intent.intent}' maps to a system action."
            )

        # ── Needs clarification ───────────────────────────────────────────
        if intent.intent in ACTION_INTENTS and intent.confidence < 0.7:
            return Decision(
                type=DecisionType.CLARIFY,
                confidence=0.6,
                reason="Low confidence on action intent.",
                needs_clarification=f"Did you want me to {intent.intent.replace('_', ' ')}?"
            )

        # ── Conversation ──────────────────────────────────────────────────
        return Decision(
            type=DecisionType.CONVERSATION,
            confidence=0.8,
            reason="No action intent matched — routing to conversation engine."
        )
