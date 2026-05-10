"""
core/brain.py
NEXUS Central Brain — the main coordinator.

Everything flows through here:
    text in → think() → response out

Internally it uses:
    IntentEngine     → parse meaning
    ReasoningEngine  → decide response type
    Planner          → break multi-step commands
    Dispatcher       → execute system actions
    ConversationEngine → handle natural language
    MemoryManager    → remember and recall
    Context          → track session state
"""

import logging
import time

from core.intent_engine  import IntentEngine
from core.dispatcher     import Dispatcher
from core.memory_manager import MemoryManager
from core.context        import Context
from core.reasoning      import ReasoningEngine, DecisionType
from core.planner        import Planner
from core.conversation   import ConversationEngine

log = logging.getLogger("nexus.brain")


class Brain:
    """
    The central coordinator of NEXUS.

    Usage:
        brain = Brain()
        response = brain.think("open firefox")
        response = brain.think("what time is it")
        response = brain.think("remember my exam is on friday")
    """

    def __init__(self, user_name: str = "Cyril", use_llm: bool = True):
        log.info("Brain initializing...")

        self.context    = Context()
        self.context.user_name = user_name

        self.memory     = MemoryManager()
        self.intent_eng = IntentEngine(backend="rules")
        self.reasoning  = ReasoningEngine()
        self.planner    = Planner()
        self.dispatcher = Dispatcher()
        
        try:
            from vision.vision import Vision
            self.vision = Vision()
            log.info("Vision system ready.")
        except Exception as e:
            self.vision = None
            log.warning(f"Vision unavailable: {e}")

        self.convo      = ConversationEngine(
            context=self.context,
            memory=self.memory,
            use_llm=use_llm,
        )

        log.info("Brain ready.")

    # ─────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────

    def think(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "I didn't hear anything."

        # Log user turn
        self.context.add_turn("user", text)

        # Parse intent
        intent = self.intent_eng.parse(text)
        log.info(f"Intent: {intent}")

        # Decide how to handle it
        decision = self.reasoning.decide(text, intent)
        log.info(f"Decision: {decision.type.value} ({decision.reason})")

        # Route to correct handler
        response = self._route(text, intent, decision)

        # Log NEXUS turn + training pair
        self.context.add_turn("nexus", response,
                               intent=intent.intent, action=decision.type.value)
        self.context.last_intent = intent.intent
        self.context.last_target = intent.target

        self.memory.log_episode(text, response, intent.intent)
        self.memory.log_training_pair(text, response, intent.intent)

        return response

    # ─────────────────────────────────────────────
    # Router
    # ─────────────────────────────────────────────

    def _route(self, text, intent, decision) -> str:

        # ── Multi-step plan ──────────────────────────────────────────────
        if decision.type == DecisionType.PLAN:
            return self._execute_plan(text)

        # ── System action ────────────────────────────────────────────────
        if decision.type == DecisionType.ACTION:
            result = self.dispatcher.dispatch(intent)
            # Store successful actions as facts
            if result.success and intent.target:
                self.memory.store_fact(
                    "nexus", f"last_{intent.intent}", intent.target
                )
            return result.message

        # ── Memory store ─────────────────────────────────────────────────
        if decision.type == DecisionType.MEMORY_STORE:
            key   = intent.target or "note"
            value = intent.query  or text
            self.memory.remember(key, value, category="user_note")
            self.memory.remember_now(key, value)
            return f"Got it — remembered: \"{value}\""

        # ── Memory query ─────────────────────────────────────────────────
        if decision.type == DecisionType.MEMORY_QUERY:
            query   = intent.query or intent.target or ""
            results = self.memory.search_memory(query)
            if results:
                lines = [f"  • {r['key']}: {r['value']}" for r in results[:5]]
                return "I remember:\n" + "\n".join(lines)
            recent = self.memory.recent_episodes(3)
            if recent:
                lines = [f"  • {e['user']}" for e in recent]
                return "Recent conversation:\n" + "\n".join(lines)
            return "I don't have anything stored about that."

        # ── Needs clarification ───────────────────────────────────────────
        if decision.type == DecisionType.CLARIFY:
            return decision.needs_clarification or "Could you rephrase that?"

        # ── Screen queries ────────────────────────────────────────────
        if self.vision and any(w in text.lower() for w in [
            "what's on screen", "what is on screen", "read the screen",
            "what do you see", "is there an error", "screen"
        ]):
            return self.vision.answer_about_screen(text)

        # ── Conversation (default) ────────────────────────────────────────
        return self.convo.respond(text)

    # ─────────────────────────────────────────────
    # Plan executor
    # ─────────────────────────────────────────────

    def _execute_plan(self, text: str) -> str:
        steps   = self.planner.plan(text)
        results = []

        for i, step_intent in enumerate(steps, 1):
            result = self.dispatcher.dispatch(step_intent)
            status = "✓" if result.success else "✗"
            results.append(f"  {status} Step {i}: {result.message}")
            if not result.success:
                log.warning(f"Plan step {i} failed: {result.message}")

        return "\n".join(results)

    # ─────────────────────────────────────────────
    # Training data tools
    # ─────────────────────────────────────────────

    def mark_good(self):
        """Call after a response to mark the last pair as high quality."""
        episodes = self.memory.recent_episodes(1)
        if episodes:
            e = episodes[0]
            self.memory.log_training_pair(
                e["user"], e["nexus"], e["intent"], quality=1.0
            )
            return "Training pair marked as good."
        return "Nothing to mark."

    def mark_bad(self, correction: str = ""):
        """Call when a response was wrong. Optionally provide correct answer."""
        episodes = self.memory.recent_episodes(1)
        if episodes:
            e = episodes[0]
            self.memory.log_training_pair(
                e["user"], correction or e["nexus"], e["intent"], quality=0.0
            )
            return "Got it — noted as a bad response."
        return "Nothing to mark."
