"""
core/conversation.py
NEXUS Conversation Engine — handles natural language replies,
personality, small talk, and fallback responses.

This is also where Ollama LLM gets wired in for real conversation.
"""

import logging
import random
import datetime
from typing import Optional

from core.context        import Context
from core.memory_manager import MemoryManager

log = logging.getLogger("nexus.conversation")


# ─────────────────────────────────────────────
# Static personality responses
# ─────────────────────────────────────────────

GREETINGS = [
    "Hey {name}. What do you need?",
    "NEXUS online. Go ahead.",
    "Ready.",
    "What's up, {name}?",
]

FAREWELLS = [
    "Later, {name}.",
    "NEXUS signing off.",
    "Shutting down. Stay sharp.",
]

CONFUSION = [
    "I don't have a response for that yet.",
    "Not sure how to answer that. Rephrase?",
    "I understood the words. Not the meaning.",
    "That one's beyond me right now.",
]

AFFIRMATIONS = ["Got it.", "Done.", "Sure.", "On it.", "Roger that."]

IDENTITY = [
    "I'm NEXUS — your personal AI assistant. Built by Cyril.",
    "NEXUS. Personal AI. Still learning.",
    "An AI assistant running locally on your machine.",
]

CAPABILITIES = """I can:
  • open and close applications
  • search the web
  • run shell commands
  • tell you the time and system info
  • remember things you tell me
  • control volume
  • answer basic questions
  
I'm still learning to do more."""


# ─────────────────────────────────────────────
# Removed unused _ollama_chat function
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# Conversation Engine
# ─────────────────────────────────────────────

class ConversationEngine:
    """
    Handles all responses that are NOT system actions.
    Falls back to Ollama if installed, otherwise uses
    rule-based personality responses.
    """

    def __init__(self, context: Context, memory: MemoryManager,
                 use_llm: bool = True, llm_model: str = "mistral"):
        self.context   = context
        self.memory    = memory
        self.use_llm   = use_llm
        self.llm_model = llm_model
        log.info(f"ConversationEngine ready (llm={use_llm}, model={llm_model!r})")

    # ── Rule-based matchers ──────────────────────────────────────────────

    def _rule_response(self, text: str) -> Optional[str]:
        name = self.context.user_name
        t = text.lower().strip()

        # Greetings
        if any(w in t for w in ["hello", "hi", "hey", "good morning",
                                  "good afternoon", "good evening", "sup"]):
            return random.choice(GREETINGS).format(name=name)

        # Farewells
        if any(w in t for w in ["bye", "goodbye", "see you", "later",
                                  "goodnight", "night"]):
            return random.choice(FAREWELLS).format(name=name)

        # Identity
        if any(p in t for p in ["who are you", "what are you",
                                  "introduce yourself", "your name"]):
            return random.choice(IDENTITY)

        # Capabilities
        if any(p in t for p in ["what can you do", "help", "capabilities",
                                  "what do you know", "commands"]):
            return CAPABILITIES

        # How are you
        if any(p in t for p in ["how are you", "you okay", "you good",
                                  "how do you feel"]):
            return "Running well. All systems nominal."

        # Creator / origin
        if any(p in t for p in ["who made you", "who built you",
                                  "who created you", "who is your creator"]):
            return "Cyril built me. Still a work in progress."

        # Time / date (conversational)
        if "what day" in t or "what's today" in t:
            return datetime.datetime.now().strftime("Today is %A, %B %d %Y.")

        # Affirmation follow-up
        if t in ["ok", "okay", "sure", "alright", "got it", "thanks",
                 "thank you", "cool", "nice"]:
            return random.choice(AFFIRMATIONS)

        # Memory recall conversational
        if any(p in t for p in ["do you remember", "what did i say",
                                  "what do you know about me"]):
            results = self.memory.search_memory(t.replace("do you remember", "").strip())
            if results:
                lines = [f"  • {r['key']}: {r['value']}" for r in results[:4]]
                return "I have:\n" + "\n".join(lines)
            return "I don't have anything stored about that."

        return None  # no rule matched

    # ── LLM-based response ───────────────────────────────────────────────

    def _llm_response(self, text: str) -> Optional[str]:
        if not self.use_llm:
            return None

        try:
            from core.llm import get_llm
            llm = get_llm()
            if not llm.is_ready:
                return None

            # Build minimal history for context
            history = []
            for turn in self.context.recent_turns(6):
                role = "user" if turn.role == "user" else "assistant"
                history.append({"role": role, "content": turn.text})

            system = (
                "You are NEXUS, a smart, concise, no-nonsense AI assistant "
                f"built for {self.context.user_name}. "
                "You are running locally on their Linux machine. "
                "You help with cybersecurity, coding, and general tasks. "
                "Keep replies short and direct. Never be sycophantic."
            )
            
            return llm.chat(text, system=system, history=history)
        except Exception as e:
            log.warning(f"LLM integration failed: {e}")
            return None

    # ── Public API ───────────────────────────────────────────────────────

    def respond(self, text: str) -> str:
        # 1. Try rule-based first (fast, no dependencies)
        reply = self._rule_response(text)
        if reply:
            log.debug("Rule-based reply used.")
            return reply

        # 2. Try LLM (requires Ollama running)
        reply = self._llm_response(text)
        if reply:
            log.debug("LLM reply used.")
            return reply

        # 3. Fallback
        return random.choice(CONFUSION)
