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
from core.llm import LLM
from core.coding_assistant import CodingAssistant

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
        
        # LLM
        try:
            self.llm = LLM()
            log.info(f"LLM ready — models: {self.llm.available_models()}")
        except Exception as e:
            self.llm = None
            log.warning(f"LLM unavailable: {e}")

        # Coding assistant
        try:
            self.coder = CodingAssistant()
        except Exception as e:
            self.coder = None
            log.warning(f"Coding assistant unavailable: {e}")

        # Research module
        try:
            from research.researcher import Researcher
            self.researcher = Researcher(max_sources=3)
            log.info("Researcher ready.")
        except Exception as e:
            self.researcher = None
            log.warning(f"Researcher unavailable: {e}")

        # Vision
        try:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from vision.vision import Vision
            self.vision = Vision()
            log.info("Vision ready.")
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

        # ── Research intent ───────────────────────────────────────
        research_triggers = [
            "research", "look up", "find out about", "what is",
            "explain", "tell me about", "learn about",
            "search for", "who is", "how does", "why does",
            "read this url", "fetch this", "summarize this url",
        ]
        if self.researcher and intent.intent in (
            "research_topic", "search_web", "ask_question"
        ) and any(t in text.lower() for t in research_triggers):
            # Check if it's a URL read request
            if any(t in text.lower() for t in ["read url", "fetch url", "summarize url", "learn from"]):
                import re
                url_match = re.search(r"https?://\S+", text)
                if url_match:
                    return self.researcher.learn_from_url(url_match.group(0))

            return self.researcher.answer(text)

        # ── recall from research memory ────────────────────────────
        if self.researcher and intent.intent in ("recall", "memory_query") and (
            "research" in text.lower() or "what did you learn" in text.lower()
        ):
            return self.researcher.recall(text)

        # ── Screen queries ────────────────────────────────────────────
        if self.vision and any(w in text.lower() for w in [
            "what's on screen", "what is on screen", "read the screen",
            "what do you see", "what can you see", "is there an error", 
            "screen", "scrren", "see on"
        ]):
            return self.vision.answer_about_screen(text)

        # ── Coding assistant ──────────────────────────────────────────
        coding_triggers = [
            "explain", "fix", "debug", "what does this", "how do i",
            "write a function", "write code", "help me code",
            "what's wrong with", "refactor", "improve this code",
            "explain this code", "document this",
        ]
        if self.coder and any(t in text.lower() for t in coding_triggers):
            # Check if there's code on screen to work with
            if any(t in text.lower() for t in ["on screen", "this code", "explain this", "fix this"]):
                return self.coder.explain_screen() if "explain" in text.lower() else self.coder.fix_screen()
            return self.coder.ask(text)

        # ── Direct LLM conversation ───────────────────────────────────
        if self.llm and self.llm.is_ready:
            # Upgrade conversation engine to use real LLM
            # (ConversationEngine already calls Ollama if available,
            #  but this is the explicit fallback for complex questions)
            pass  # ConversationEngine handles this via _llm_response

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
