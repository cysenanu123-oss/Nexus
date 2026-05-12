"""
core/router.py
NEXUS Intelligent Router — the traffic controller of the brain.

Every question passes through here BEFORE going to any module.
The router decides:
    - What category is this question?
    - Which modules are needed?
    - Does it need the internet?
    - Does it need vision (screen reading)?
    - How confident are we?

Pipeline:
    User input
         ↓
    router.classify(text)        ← YOU ARE HERE
         ↓
    Route object (category + modules + flags)
         ↓
    brain._route() dispatches to correct module(s)

Design goals:
    - Fast: rule-based first, LLM fallback only when uncertain
    - Lightweight: uses qwen2.5:1.5b (smallest model) for classification
    - Transparent: every decision is logged with reasoning
    - Extensible: add new categories by adding rules + a handler

Usage:
    from core.router import Router
    router = Router()
    route = router.classify("how do I exploit a buffer overflow")
    print(route.category)        # "CYBER"
    print(route.modules)         # ["cyber", "research"]
    print(route.needs_internet)  # True
    print(route.confidence)      # 0.92
"""

import re
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger("nexus.router")


# ─────────────────────────────────────────────────────────────
#  ROUTE CATEGORIES
#  Each category maps to one or more NEXUS modules
# ─────────────────────────────────────────────────────────────

class Category(Enum):
    CODING       = "coding"        # code help, debugging, writing functions
    CYBER        = "cyber"         # hacking, security, pentesting, networking
    RESEARCH     = "research"      # needs internet, current info, learning
    VISION       = "vision"        # screen reading, OCR, screenshot questions
    MEMORY       = "memory"        # remember/recall personal notes
    SYSTEM       = "system"        # open apps, control system, shell commands
    CONVERSATION = "conversation"  # general chat, no tools needed
    SKILL        = "skill"         # load/run/learn a skill
    MULTI        = "multi"         # needs multiple modules combined


# ─────────────────────────────────────────────────────────────
#  MODULE REGISTRY
#  Maps each category to the modules it uses
# ─────────────────────────────────────────────────────────────

CATEGORY_MODULES: dict[Category, list[str]] = {
    Category.CODING:       ["coding_assistant", "vision"],
    Category.CYBER:        ["cyber", "research", "coding_assistant"],
    Category.RESEARCH:     ["research", "llm"],
    Category.VISION:       ["vision"],
    Category.MEMORY:       ["memory"],
    Category.SYSTEM:       ["dispatcher"],
    Category.CONVERSATION: ["llm"],
    Category.SKILL:        ["skill_manager"],
    Category.MULTI:        ["research", "coding_assistant", "cyber", "llm"],
}


# ─────────────────────────────────────────────────────────────
#  ROUTE RESULT
# ─────────────────────────────────────────────────────────────

@dataclass
class Route:
    """
    Result of classifying a user input.

    Attributes
    ----------
    category       : primary category of the question
    modules        : list of NEXUS modules to activate
    confidence     : 0.0–1.0, how sure the router is
    needs_internet : True if the question requires web search
    needs_vision   : True if the question requires screen reading
    needs_llm      : True if a language model is required
    reasoning      : short explanation of why this route was chosen
    sub_tasks      : if MULTI, list of sub-routes for each part
    raw            : original user input
    elapsed        : time taken to classify (seconds)
    """
    category:       Category
    modules:        list[str]
    confidence:     float         = 1.0
    needs_internet: bool          = False
    needs_vision:   bool          = False
    needs_llm:      bool          = True
    reasoning:      str           = ""
    sub_tasks:      list["Route"] = field(default_factory=list)
    raw:            str           = ""
    elapsed:        float         = 0.0

    def __str__(self) -> str:
        return (
            f"Route({self.category.value}, "
            f"modules={self.modules}, "
            f"conf={self.confidence:.2f}, "
            f"web={self.needs_internet}, "
            f"vision={self.needs_vision})"
        )

    def primary_module(self) -> str:
        return self.modules[0] if self.modules else "llm"


# ─────────────────────────────────────────────────────────────
#  RULE DEFINITIONS
#  Fast pattern matching — runs before LLM classification
#  Format: (category, keyword_list, internet_needed, vision_needed)
# ─────────────────────────────────────────────────────────────

RULES: list[tuple[Category, list[str], bool, bool]] = [

    # ── Vision / Screen ───────────────────────────────────────
    (Category.VISION, [
        "on screen", "on my screen", "what's on screen", "what is on screen",
        "read screen", "what do you see", "what can you see",
        "take screenshot", "screenshot", "screen shows", "is there an error",
        "what's open", "what app", "look at screen", "on the screen",
    ], False, True),

    # ── Memory ────────────────────────────────────────────────
    (Category.MEMORY, [
        "remember", "recall", "remind me", "what did i", "do you remember",
        "save this", "note that", "store this", "what did you save",
        "what do you know about me", "my notes",
    ], False, False),

    # ── System / Dispatcher ───────────────────────────────────
    (Category.SYSTEM, [
        "open ", "close ", "launch ", "run ", "start ", "execute ",
        "shutdown", "reboot", "volume", "brightness",
        "what time", "system info", "system status",
    ], False, False),

    # ── Cybersecurity ─────────────────────────────────────────
    (Category.CYBER, [
        "hack", "exploit", "vulnerability", "pentest", "penetration",
        "nmap", "scan port", "scan network", "open ports",
        "sql injection", "xss", "buffer overflow", "reverse shell",
        "payload", "privilege escalation", "metasploit", "burp",
        "wireshark", "packet", "firewall", "bypass", "ctf",
        "capture the flag", "network scan", "arp scan",
        "reconnaissance", "recon", "enumeration", "brute force",
        "password crack", "hash", "decrypt", "malware", "ransomware",
        "backdoor", "rootkit", "keylogger", "phishing", "social engineering",
    ], True, False),

    # ── Coding ────────────────────────────────────────────────
    (Category.CODING, [
        "write code", "write a function", "write a script",
        "write a program", "code to", "debug", "fix this code",
        "fix this bug", "explain this code", "what does this code",
        "refactor", "optimize this", "improve this code",
        "error in my code", "traceback", "syntax error",
        "how do i code", "how do i write", "implement",
        "algorithm", "function that", "class that", "loop",
        "python code", "bash script", "javascript",
        "git ", "git commit", "git push",
    ], False, False),

    # ── Research (needs internet) ─────────────────────────────
    (Category.RESEARCH, [
        "latest ", "recent ", "current ", "news about", "what happened",
        "today", "this year", "2025", "2026",
        "research ", "look up", "find out about", "what is ",
        "who is ", "how does ", "explain ", "tell me about",
        "learn about", "read this url", "fetch this",
        "summarize this", "what are the", "how do",
    ], True, False),

    # ── Skill ────────────────────────────────────────────────
    (Category.SKILL, [
        "load skill", "run skill", "use skill", "learn this skill",
        "save this as a skill", "add this skill", "teach you",
        "here is a script", "here is a function",
    ], False, False),
]


# ─────────────────────────────────────────────────────────────
#  ROUTER CLASS
# ─────────────────────────────────────────────────────────────

class Router:
    """
    Classifies user input and returns a Route object.

    Classification happens in two stages:
        1. Rule-based (fast, keyword matching, no LLM needed)
        2. LLM-based (slower, used when rules are uncertain)

    Usage:
        router = Router()
        route  = router.classify("how do I scan for open ports")
        print(route)
        # Route(CYBER, modules=['cyber', 'research'], conf=0.95, web=True)
    """

    def __init__(self, use_llm: bool = True, llm_threshold: float = 0.5):
        """
        Parameters
        ----------
        use_llm       : use LLM as fallback when rules are uncertain
        llm_threshold : if rule confidence < this, fall back to LLM
        """
        self.use_llm       = use_llm
        self.llm_threshold = llm_threshold
        self._llm          = None   # lazy-loaded
        log.info("Router initialized — llm_fallback=%s", use_llm)

    # ── Public API ────────────────────────────────────────────

    def classify(self, text: str) -> Route:
        """
        Classify user input into a Route.

        Parameters
        ----------
        text : raw user input string

        Returns
        -------
        Route object with category, modules, flags, confidence
        """
        t0      = time.time()
        text    = text.strip()
        lower   = text.lower()

        # ── Stage 1: Rule-based classification ────────────────
        route = self._rule_classify(text, lower)

        # ── Stage 2: LLM fallback if confidence is low ────────
        if route.confidence < self.llm_threshold and self.use_llm:
            log.debug(
                "Rule confidence %.2f < threshold %.2f — trying LLM classifier.",
                route.confidence, self.llm_threshold,
            )
            llm_route = self._llm_classify(text)
            if llm_route and llm_route.confidence > route.confidence:
                route = llm_route

        # ── Finalize ──────────────────────────────────────────
        route.raw     = text
        route.elapsed = time.time() - t0

        log.info(
            "Route: %s | conf=%.2f | web=%s | vision=%s | modules=%s | %.3fs",
            route.category.value, route.confidence,
            route.needs_internet, route.needs_vision,
            route.modules, route.elapsed,
        )

        return route

    def classify_batch(self, texts: list[str]) -> list[Route]:
        """Classify multiple inputs at once."""
        return [self.classify(t) for t in texts]

    def explain(self, text: str) -> str:
        """Return a human-readable explanation of the routing decision."""
        route = self.classify(text)
        return (
            f"Category  : {route.category.value}\n"
            f"Modules   : {', '.join(route.modules)}\n"
            f"Internet  : {'Yes' if route.needs_internet else 'No'}\n"
            f"Vision    : {'Yes' if route.needs_vision else 'No'}\n"
            f"Confidence: {route.confidence:.0%}\n"
            f"Reasoning : {route.reasoning}\n"
            f"Time      : {route.elapsed*1000:.1f}ms"
        )

    # ── Rule-based classification ─────────────────────────────

    def _rule_classify(self, text: str, lower: str) -> Route:
        """Match against keyword rules. Returns best match."""
        scores: dict[Category, tuple[float, bool, bool, str]] = {}

        for category, keywords, needs_web, needs_vis in RULES:
            matches = [kw for kw in keywords if kw in lower]
            if not matches:
                continue

            # Score = proportion of matched keywords, capped at 0.95
            score = min(0.95, 0.5 + (len(matches) * 0.15))
            reason = f"Matched keywords: {', '.join(matches[:3])}"
            scores[category] = (score, needs_web, needs_vis, reason)

        if not scores:
            # No rules matched — default to CONVERSATION with low confidence
            return Route(
                category       = Category.CONVERSATION,
                modules        = CATEGORY_MODULES[Category.CONVERSATION],
                confidence     = 0.3,
                needs_internet = False,
                needs_vision   = False,
                needs_llm      = True,
                reasoning      = "No rules matched — defaulting to conversation.",
            )

        # Pick the highest-scoring category
        best_category = max(scores, key=lambda c: scores[c][0])
        score, needs_web, needs_vis, reason = scores[best_category]

        # Check for multi-category (multiple high scores)
        high_scores = [c for c, (s, _, _, _) in scores.items() if s >= 0.5]
        if len(high_scores) > 1 and best_category not in (Category.SYSTEM, Category.MEMORY):
            # Combine modules from all high-scoring categories
            combined_modules = []
            for c in high_scores:
                for m in CATEGORY_MODULES[c]:
                    if m not in combined_modules:
                        combined_modules.append(m)
            return Route(
                category       = Category.MULTI,
                modules        = combined_modules,
                confidence     = score,
                needs_internet = any(scores[c][1] for c in high_scores),
                needs_vision   = any(scores[c][2] for c in high_scores),
                needs_llm      = True,
                reasoning      = f"Multiple categories detected: {[c.value for c in high_scores]}",
            )

        return Route(
            category       = best_category,
            modules        = CATEGORY_MODULES[best_category],
            confidence     = score,
            needs_internet = needs_web,
            needs_vision   = needs_vis,
            needs_llm      = True,
            reasoning      = reason,
        )

    # ── LLM-based classification ──────────────────────────────

    def _llm_classify(self, text: str) -> Optional[Route]:
        """
        Use the lightweight local LLM to classify ambiguous inputs.
        Uses qwen2.5:1.5b — fastest model, minimal resources.
        """
        try:
            llm = self._get_llm()
            if not llm or not llm.is_ready:
                return None

            system = """You are a classifier for an AI assistant called NEXUS.
Classify the user input into exactly ONE category.

Categories:
- CODING: code help, debugging, writing functions/scripts
- CYBER: hacking, security, pentesting, network scanning
- RESEARCH: needs internet, current info, factual lookup
- VISION: screen reading, screenshots, what's on screen
- MEMORY: remember/recall personal notes
- SYSTEM: open apps, control system, run commands
- CONVERSATION: general chat, no tools needed
- SKILL: load/save/run a skill or script

Reply ONLY with this JSON:
{"category": "CATEGORY", "confidence": 0.85, "needs_internet": true, "needs_vision": false, "reason": "one sentence"}

No explanation. No markdown. JSON only."""

            raw = llm.chat(text, system=system, task="fast")

            # Parse JSON response
            import json
            raw = raw.strip().strip("```json").strip("```").strip()
            data = json.loads(raw)

            category_str = data.get("category", "CONVERSATION").upper()
            try:
                category = Category[category_str]
            except KeyError:
                category = Category.CONVERSATION

            return Route(
                category       = category,
                modules        = CATEGORY_MODULES[category],
                confidence     = float(data.get("confidence", 0.7)),
                needs_internet = bool(data.get("needs_internet", False)),
                needs_vision   = bool(data.get("needs_vision", False)),
                needs_llm      = True,
                reasoning      = data.get("reason", "LLM classification"),
            )

        except Exception as e:
            log.warning("LLM classification failed: %s", e)
            return None

    # ── Lazy loader ───────────────────────────────────────────

    def _get_llm(self):
        if self._llm is None:
            try:
                from core.llm import get_llm
                self._llm = get_llm()
            except Exception as e:
                log.warning("Could not load LLM for router: %s", e)
        return self._llm


# ─────────────────────────────────────────────────────────────
#  SINGLETON
# ─────────────────────────────────────────────────────────────

_router_instance: Optional[Router] = None


def get_router() -> Router:
    """Return the shared Router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = Router()
    return _router_instance


# ─────────────────────────────────────────────────────────────
#  CLI TEST — python core/router.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    router = Router(use_llm=True)

    test_inputs = [
        "how do I exploit a buffer overflow",
        "write a python function to reverse a string",
        "what's on my screen right now",
        "remember that my exam is on friday",
        "open firefox",
        "what is the latest news about AI",
        "can you help me with pentesting",
        "scan my network for open ports",
        "explain SQL injection to me",
        "hey what's up",
        "write a bash script to monitor CPU usage",
        "research zero day vulnerabilities in 2026",
    ]

    if len(sys.argv) > 1:
        test_inputs = [" ".join(sys.argv[1:])]

    print("\n─── NEXUS Router Test ───\n")

    for text in test_inputs:
        route = router.classify(text)
        print(f"  Input    : {text!r}")
        print(f"  Category : {route.category.value}")
        print(f"  Modules  : {route.modules}")
        print(f"  Internet : {route.needs_internet}")
        print(f"  Vision   : {route.needs_vision}")
        print(f"  Conf     : {route.confidence:.0%}")
        print(f"  Reason   : {route.reasoning}")
        print(f"  Time     : {route.elapsed*1000:.1f}ms")
        print()
