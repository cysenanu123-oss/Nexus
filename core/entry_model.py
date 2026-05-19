"""
core/entry_model.py
NEXUS Entry Model — lightweight fast-path classifier at the top of the pipeline.

This is the "entrance point" before anything else touches the input.
It runs in milliseconds using fuzzy matching + rules (no LLM needed).

Responsibilities:
  1. High-level category classification  (ACTION / PLAN / CHAT / CYBER / CODE / MEMORY / CALENDAR)
  2. Confidence scoring — if confidence is high, skip heavy routing downstream
  3. Flag complex multi-step requests for the AutonomousPlanner
  4. Attach extracted entities from the Normalizer to the classification

EntryModel is NOT a replacement for IntentEngine — it sits above it
and guides routing so the brain doesn't have to guess.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz, process as rfuzz_process

log = logging.getLogger("nexus.entry_model")


# ─────────────────────────────────────────────────────────────
#  CATEGORIES
# ─────────────────────────────────────────────────────────────

class Cat:
    ACTION   = "action"       # single system action (open app, run command)
    PLAN     = "plan"         # multi-step autonomous plan
    CHAT     = "chat"         # general conversation / question
    CYBER    = "cyber"        # cybersecurity operation
    CODE     = "code"         # coding / planning code changes
    MEMORY   = "memory"       # store or recall a memory
    CALENDAR = "calendar"     # calendar / reminder / scheduling
    RESEARCH = "research"     # web research / information lookup
    UNKNOWN  = "unknown"


# ─────────────────────────────────────────────────────────────
#  RESULT DATACLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class EntryResult:
    """
    Result of the entry model classification.

    category:       high-level category (Cat.*)
    confidence:     0.0 – 1.0
    needs_planning: True if this should go to AutonomousPlanner
    intent_hint:    pre-parsed intent name hint (optional, speeds up IntentEngine)
    matched_phrase: the phrase that triggered this classification
    entities:       entities from NormalizedInput (date, time, app, etc.)
    corrected_text: spell-corrected text to use downstream
    """
    category:       str
    confidence:     float
    needs_planning: bool = False
    intent_hint:    Optional[str] = None
    matched_phrase: str = ""
    entities:       dict = field(default_factory=dict)
    corrected_text: str = ""

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.80

    def __str__(self) -> str:
        return (
            f"EntryResult(cat={self.category!r}, conf={self.confidence:.2f}, "
            f"plan={self.needs_planning}, hint={self.intent_hint!r})"
        )


# ─────────────────────────────────────────────────────────────
#  PHRASE BANKS  (keyword → category, confidence, intent_hint, needs_plan)
# ─────────────────────────────────────────────────────────────
#
# Each entry: (phrase, category, confidence, intent_hint, needs_planning)
#
# Longer, more specific phrases should come first in the list — they
# override shorter partial matches.

_PHRASE_BANKS: list[tuple[str, str, float, Optional[str], bool]] = [

    # ── Calendar / Scheduling (ALWAYS plan — multi-step) ─────────────
    ("remind me i have a meeting",    Cat.CALENDAR, 0.95, "schedule_meeting", True),
    ("remind me about",               Cat.CALENDAR, 0.90, "set_reminder",     True),
    ("remind me to",                  Cat.CALENDAR, 0.90, "set_reminder",     True),
    ("set a reminder",                Cat.CALENDAR, 0.90, "set_reminder",     True),
    ("set a meeting",                 Cat.CALENDAR, 0.90, "schedule_meeting", True),
    ("schedule a meeting",            Cat.CALENDAR, 0.90, "schedule_meeting", True),
    ("schedule meeting",              Cat.CALENDAR, 0.88, "schedule_meeting", True),
    ("add to calendar",               Cat.CALENDAR, 0.88, "add_calendar",     True),
    ("add to my calendar",            Cat.CALENDAR, 0.90, "add_calendar",     True),
    ("create a calendar event",       Cat.CALENDAR, 0.92, "add_calendar",     True),
    ("show my calendar",              Cat.CALENDAR, 0.90, "show_calendar",    False),
    ("what's on my calendar",         Cat.CALENDAR, 0.92, "show_calendar",    False),
    ("show today's calendar",         Cat.CALENDAR, 0.90, "show_calendar",    False),
    ("i have a meeting",              Cat.CALENDAR, 0.85, "schedule_meeting", True),
    ("meeting tomorrow",              Cat.CALENDAR, 0.85, "schedule_meeting", True),
    ("meeting on zoom",               Cat.CALENDAR, 0.87, "schedule_meeting", True),
    ("zoom call",                     Cat.CALENDAR, 0.85, "schedule_meeting", True),
    ("google meet",                   Cat.CALENDAR, 0.83, "schedule_meeting", True),
    ("set an alarm",                  Cat.CALENDAR, 0.88, "set_reminder",     True),
    ("wake me up",                    Cat.CALENDAR, 0.86, "set_reminder",     True),
    ("alert me",                      Cat.CALENDAR, 0.80, "set_reminder",     True),
    ("don't let me forget",           Cat.CALENDAR, 0.85, "set_reminder",     True),
    ("before the meeting",            Cat.CALENDAR, 0.83, "set_reminder",     True),
    ("30 minutes before",             Cat.CALENDAR, 0.83, "set_reminder",     True),

    # ── Memory ────────────────────────────────────────────────────────
    ("remember that",                 Cat.MEMORY, 0.92, "remember",       False),
    ("remember my",                   Cat.MEMORY, 0.88, "remember",       False),
    ("note that",                     Cat.MEMORY, 0.88, "remember",       False),
    ("make a note",                   Cat.MEMORY, 0.88, "remember",       False),
    ("note to self",                  Cat.MEMORY, 0.90, "remember",       False),
    ("store that",                    Cat.MEMORY, 0.85, "remember",       False),
    ("what did i say about",          Cat.MEMORY, 0.88, "recall",         False),
    ("do you remember",               Cat.MEMORY, 0.85, "recall",         False),
    ("what do you know about",        Cat.MEMORY, 0.85, "recall",         False),
    ("recall",                        Cat.MEMORY, 0.82, "recall",         False),
    ("remind me what",                Cat.MEMORY, 0.83, "recall",         False),

    # ── Cyber ─────────────────────────────────────────────────────────
    ("scan my network",               Cat.CYBER, 0.97, "cyber_full_network_scan",  False),
    ("scan ports",                    Cat.CYBER, 0.93, "cyber_port_scan",          False),
    ("port scan",                     Cat.CYBER, 0.93, "cyber_port_scan",          False),
    ("vuln scan",                     Cat.CYBER, 0.95, "vuln_scan",                True),
    ("vulnerability scan",            Cat.CYBER, 0.95, "vuln_scan",                True),
    ("recon on",                      Cat.CYBER, 0.95, "full_recon",               True),
    ("full recon",                    Cat.CYBER, 0.95, "full_recon",               True),
    ("find subdomains",               Cat.CYBER, 0.93, "subdomains",               False),
    ("exploit search",                Cat.CYBER, 0.93, "exploit_search",           False),
    ("find exploit",                  Cat.CYBER, 0.93, "exploit_search",           False),
    ("cve lookup",                    Cat.CYBER, 0.93, "cve_search",               False),
    ("search cve",                    Cat.CYBER, 0.93, "cve_search",               False),
    ("cyber news",                    Cat.CYBER, 0.93, "cyber_news",               False),
    ("hacking news",                  Cat.CYBER, 0.93, "cyber_news",               False),
    ("latest news",                   Cat.CYBER, 0.85, "cyber_news",               False),
    ("authorize",                     Cat.CYBER, 0.88, "authorize_target",         False),
    ("penetration test",              Cat.CYBER, 0.93, "pentest",                  True),
    ("pentest",                       Cat.CYBER, 0.93, "pentest",                  True),
    ("sandbox",                       Cat.CYBER, 0.90, "create_sandbox",           True),
    ("monitor target",                Cat.CYBER, 0.92, "monitor_target",           True),
    ("check for suspicious",          Cat.CYBER, 0.92, "cyber_analyze_logs",       False),
    ("failed logins",                 Cat.CYBER, 0.90, "cyber_check_logins",       False),
    ("download exploit",              Cat.CYBER, 0.93, "exploit_download",         False),

    # ── Code / Planning ───────────────────────────────────────────────
    ("plan how",                      Cat.CODE,  0.90, "code_plan",    True),
    ("plan to",                       Cat.CODE,  0.85, "code_plan",    True),
    ("add a feature",                 Cat.CODE,  0.90, "code_plan",    True),
    ("add feature",                   Cat.CODE,  0.88, "code_plan",    True),
    ("implement",                     Cat.CODE,  0.85, "code_plan",    True),
    ("write a module",                Cat.CODE,  0.88, "code_plan",    True),
    ("write a class",                 Cat.CODE,  0.88, "code_plan",    True),
    ("analyse the code",              Cat.CODE,  0.88, "code_analyze", False),
    ("analyze the code",              Cat.CODE,  0.88, "code_analyze", False),
    ("read through",                  Cat.CODE,  0.85, "code_analyze", False),
    ("what files",                    Cat.CODE,  0.80, "code_analyze", False),
    ("fix the bug",                   Cat.CODE,  0.88, "code_fix",     True),
    ("debug",                         Cat.CODE,  0.82, "code_fix",     False),
    ("refactor",                      Cat.CODE,  0.85, "code_plan",    True),
    ("search and learn",              Cat.CODE,  0.85, "search_learn", False),

    # ── Research ──────────────────────────────────────────────────────
    ("research",                      Cat.RESEARCH, 0.85, "research_topic", False),
    ("look up",                       Cat.RESEARCH, 0.82, "research_topic", False),
    ("find out about",                Cat.RESEARCH, 0.85, "research_topic", False),
    ("tell me about",                 Cat.RESEARCH, 0.80, "research_topic", False),
    ("what is",                       Cat.RESEARCH, 0.70, "ask_question",   False),
    ("who is",                        Cat.RESEARCH, 0.72, "ask_question",   False),
    ("how does",                      Cat.RESEARCH, 0.70, "ask_question",   False),
    ("explain",                       Cat.RESEARCH, 0.72, "ask_question",   False),

    # ── Action ────────────────────────────────────────────────────────
    ("open",                          Cat.ACTION, 0.82, "open_application", False),
    ("launch",                        Cat.ACTION, 0.85, "open_application", False),
    ("close",                         Cat.ACTION, 0.82, "close_application",False),
    ("run",                           Cat.ACTION, 0.80, "run_command",      False),
    ("execute",                       Cat.ACTION, 0.80, "run_command",      False),
    ("take a screenshot",             Cat.ACTION, 0.93, "screenshot",       False),
    ("volume up",                     Cat.ACTION, 0.93, "volume_control",   False),
    ("volume down",                   Cat.ACTION, 0.93, "volume_control",   False),
    ("shutdown",                      Cat.ACTION, 0.93, "system_control",   False),
    ("restart",                       Cat.ACTION, 0.93, "system_control",   False),

    # ── Chat / General ────────────────────────────────────────────────
    ("hello",                         Cat.CHAT, 0.90, "greet",   False),
    ("hi",                            Cat.CHAT, 0.88, "greet",   False),
    ("hey",                           Cat.CHAT, 0.85, "greet",   False),
    ("good morning",                  Cat.CHAT, 0.90, "greet",   False),
    ("good evening",                  Cat.CHAT, 0.90, "greet",   False),
    ("thank you",                     Cat.CHAT, 0.90, "thanks",  False),
    ("thanks",                        Cat.CHAT, 0.88, "thanks",  False),
    ("what time",                     Cat.CHAT, 0.88, "get_time",False),
    ("what date",                     Cat.CHAT, 0.88, "get_time",False),
    ("how are you",                   Cat.CHAT, 0.90, "greet",   False),
]


# Build a fast lookup from phrase → tuple
_BANK_MAP: dict[str, tuple] = {p: rest for p, *rest in _PHRASE_BANKS}
_ALL_PHRASES: list[str] = [p for p, *_ in _PHRASE_BANKS]

# ── Complexity signals → needs_planning=True ──────────────────
_PLANNING_SIGNALS = re.compile(
    r"\b(?:and then|then|after that|first|next|also|as well|make sure|"
    r"don.?t forget|schedule|set up|set a|create a|open and|launch and|"
    r"go to and|and open|and send|remind me|plan|step by step)\b",
    re.I,
)


# ─────────────────────────────────────────────────────────────
#  ENTRY MODEL
# ─────────────────────────────────────────────────────────────

class EntryModel:
    """
    Lightweight entry-point classifier. Runs in <5ms.

    Uses fuzzy string matching against a phrase bank to classify
    user input before any heavy processing happens.
    """

    def __init__(self, fuzzy_threshold: int = 72):
        self.fuzzy_threshold = fuzzy_threshold
        log.info("EntryModel ready (fuzzy_threshold=%d)", fuzzy_threshold)

    def classify(self, normalized_input) -> EntryResult:
        """
        Classify a NormalizedInput (from Normalizer) into an EntryResult.

        Tries:
          1. Exact phrase match (case-insensitive)
          2. Starts-with match
          3. Fuzzy partial match (rapidfuzz)
          4. Complexity heuristic for planning signals
          5. Fallback: CHAT
        """
        from core.normalizer import NormalizedInput
        if isinstance(normalized_input, str):
            # Convenience: accept raw string too
            from core.normalizer import normalize
            ni = normalize(normalized_input)
        else:
            ni = normalized_input

        text    = ni.corrected.lower().strip()
        entities = ni.entities

        # ── 1. Exact phrase scan ──────────────────────────────────
        for phrase in _ALL_PHRASES:
            if phrase in text:
                cat, conf, hint, plan = _BANK_MAP[phrase]
                # Boost planning if multi-step signals present
                if not plan and _PLANNING_SIGNALS.search(text):
                    plan = True
                return EntryResult(
                    category       = cat,
                    confidence     = conf,
                    needs_planning = plan,
                    intent_hint    = hint,
                    matched_phrase = phrase,
                    entities       = entities,
                    corrected_text = ni.corrected,
                )

        # ── 2. Fuzzy match ────────────────────────────────────────
        best = rfuzz_process.extractOne(
            text, _ALL_PHRASES,
            scorer=fuzz.partial_ratio,
            score_cutoff=self.fuzzy_threshold,
        )
        if best:
            matched_phrase, score, _ = best
            cat, conf, hint, plan = _BANK_MAP[matched_phrase]
            fuzzy_conf = conf * (score / 100)
            if not plan and _PLANNING_SIGNALS.search(text):
                plan = True
            return EntryResult(
                category       = cat,
                confidence     = fuzzy_conf,
                needs_planning = plan,
                intent_hint    = hint,
                matched_phrase = matched_phrase,
                entities       = entities,
                corrected_text = ni.corrected,
            )

        # ── 3. Planning signal without specific category ───────────
        if _PLANNING_SIGNALS.search(text):
            return EntryResult(
                category       = Cat.PLAN,
                confidence     = 0.60,
                needs_planning = True,
                entities       = entities,
                corrected_text = ni.corrected,
            )

        # ── 4. Fallback ───────────────────────────────────────────
        return EntryResult(
            category       = Cat.CHAT,
            confidence     = 0.40,
            needs_planning = False,
            entities       = entities,
            corrected_text = ni.corrected,
        )


# ── Singleton ─────────────────────────────────────────────────

_model: Optional[EntryModel] = None


def get_entry_model() -> EntryModel:
    global _model
    if _model is None:
        _model = EntryModel()
    return _model
