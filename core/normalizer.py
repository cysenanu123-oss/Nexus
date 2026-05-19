"""
core/normalizer.py
NEXUS Input Normalizer — spell-correction + phrase dictionary + entity cleanup.

Runs before everything else. Fixes typos, expands abbreviations, maps
user-friendly phrases to canonical forms, and extracts key entities
(dates, times, app names, URLs) from raw input.

Pipeline:
    raw text
        → strip / lowercase
        → phrase substitution (dict lookup)
        → spell correction (word-level)
        → entity extraction (date, time, app, url)
        → NormalizedInput
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional

log = logging.getLogger("nexus.normalizer")

# ─────────────────────────────────────────────────────────────
#  PHRASE DICTIONARY
#  Maps common typos, abbreviations, and shorthand to
#  canonical phrases the intent engine + brain understand.
# ─────────────────────────────────────────────────────────────

PHRASE_DICT: dict[str, str] = {
    # Typos / misspellings
    "nexs":          "nexus",
    "nexsu":         "nexus",
    "remeber":       "remember",
    "remmber":       "remember",
    "opn":           "open",
    "shwow":         "show",
    "tomorow":       "tomorrow",
    "tommorow":      "tomorrow",
    "tommorrow":     "tomorrow",
    "tomorow":       "tomorrow",
    "calander":      "calendar",
    "calender":      "calendar",
    "metting":       "meeting",
    "meating":       "meeting",
    "schedul":       "schedule",
    "scedule":       "schedule",
    "reserch":       "research",
    "reasearch":     "research",
    "hackin":        "hacking",
    "vulnerabilty":  "vulnerability",
    "vunerability":  "vulnerability",
    "expoit":        "exploit",
    "exploite":      "exploit",
    "reckon":        "recon",        # common auto-correct mis-fix
    "recone":        "recon",
    "infomation":    "information",
    "inforamtion":   "information",
    "dowload":       "download",
    "donwload":      "download",
    "instlal":       "install",
    "instll":        "install",
    "pasword":       "password",
    "passord":       "password",
    "netwrok":       "network",
    "netwrk":        "network",
    "porst":         "ports",
    "scna":          "scan",
    "sacn":          "scan",
    "alret":         "alert",
    "notifcation":   "notification",
    "notifiation":   "notification",
    "remaind":       "remind",
    "remaindr":      "reminder",
    "remidner":      "reminder",
    "mkae":          "make",
    "mak":           "make",
    "sart":          "start",
    "statr":         "start",
    "oen":           "open",
    "clsoe":         "close",

    # Abbreviations / shorthand
    "mtg":           "meeting",
    "cal":           "calendar",
    "tmrw":          "tomorrow",
    "tmr":           "tomorrow",
    "2moro":         "tomorrow",
    "2day":          "today",
    "td":            "today",
    "asap":          "as soon as possible",
    "mins":          "minutes",
    "min":           "minute",
    "hrs":           "hours",
    "hr":            "hour",
    "sec":           "second",
    "secs":          "seconds",
    "pls":           "please",
    "plz":           "please",
    "thx":           "thanks",
    "ty":            "thank you",
    "btw":           "by the way",
    "fyi":           "for your information",
    "eta":           "estimated time of arrival",
    "imo":           "in my opinion",
    "atm":           "at the moment",
    "w/":            "with",
    "b/w":           "between",
    "vs":            "versus",
    "b4":            "before",
    "cya":           "see you",
    "nxt":           "next",
    "wk":            "week",
    "wks":           "weeks",
    "msg":           "message",
    "msgs":          "messages",
    "notif":         "notification",
    "pw":            "password",
    "pwd":           "password",
    "usr":           "user",
    "db":            "database",
    "ip":            "ip address",
    "infosec":       "information security",
    "appsec":        "application security",
    "pentest":       "penetration test",
    "bugbounty":     "bug bounty",
    "vuln":          "vulnerability",
    "vulns":         "vulnerabilities",
    "cve":           "CVE",
    "poc":           "proof of concept",
    "rce":           "remote code execution",
    "sqli":          "SQL injection",
    "xss":           "cross-site scripting",
    "lfi":           "local file inclusion",
    "rfi":           "remote file inclusion",
    "ssrf":          "server-side request forgery",
    "csrf":          "cross-site request forgery",
    "priv esc":      "privilege escalation",
    "privesc":       "privilege escalation",
    "rev shell":     "reverse shell",
    "revshell":      "reverse shell",

    # Phrase → canonical intent phrases
    "set a reminder":       "remind me",
    "set a meeting":        "schedule a meeting",
    "book a meeting":       "schedule a meeting",
    "put on my calendar":   "add to calendar",
    "put it on my calendar":"add to calendar",
    "add it to my calendar":"add to calendar",
    "check my schedule":    "show my calendar",
    "what do i have today": "show today's calendar",
    "what's on today":      "show today's calendar",
    "open my calendar":     "show my calendar",
    "schedule call":        "schedule a meeting",
    "set alarm":            "set a reminder",
    "set an alarm":         "set a reminder",
    "wake me up":           "set a reminder",
    "alert me":             "set a reminder",
    "let me know when":     "remind me when",
    "don't forget":         "remind me",
    "note to self":         "remember",
    "jot down":             "remember",
    "take note":            "remember",
    "make a note":          "remember",
    "put a note":           "remember",
    "what time is it now":  "what time is it",
    "current time":         "what time is it",
    "today's date":         "what is today's date",
    "hack into":            "penetration test",
    "break into":           "penetration test",
    "crack":                "penetration test",
    "go online":            "search online",
    "go on the internet":   "search online",
    "look it up":           "search online",
    "check the web":        "search online",
    "show me the news":     "latest cyber news",
    "hacking news":         "latest cyber news",
    "security news":        "latest cyber news",
}

# Regex patterns for multi-word phrase replacement (order matters: longer first)
_PHRASE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(k) + r"\b", re.I), v)
    for k, v in sorted(PHRASE_DICT.items(), key=lambda x: -len(x[0]))
]

# ─────────────────────────────────────────────────────────────
#  RELATIVE DATE PATTERNS
# ─────────────────────────────────────────────────────────────

_DAY_NAMES = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _next_weekday(weekday: int) -> date:
    today = date.today()
    days_ahead = weekday - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return today + timedelta(days=days_ahead)


def _resolve_relative_date(text: str) -> Optional[str]:
    """
    Convert relative date phrases to absolute dates.
    Returns ISO date string e.g. "2026-05-18" or None.
    """
    t = text.lower()
    today = date.today()

    if "today" in t:
        return today.isoformat()
    if "tomorrow" in t:
        return (today + timedelta(days=1)).isoformat()
    if "day after tomorrow" in t:
        return (today + timedelta(days=2)).isoformat()
    if "next week" in t:
        return (today + timedelta(weeks=1)).isoformat()
    if "in two days" in t or "in 2 days" in t:
        return (today + timedelta(days=2)).isoformat()
    if "in three days" in t or "in 3 days" in t:
        return (today + timedelta(days=3)).isoformat()
    if "in a week" in t or "in one week" in t:
        return (today + timedelta(weeks=1)).isoformat()

    for day_name, day_num in _DAY_NAMES.items():
        if f"next {day_name}" in t or f"on {day_name}" in t or f"this {day_name}" in t or day_name in t:
            return _next_weekday(day_num).isoformat()

    return None


def _resolve_time(text: str) -> Optional[str]:
    """
    Extract time from text.
    Returns "HH:MM" string or None.
    """
    # "at 3pm", "at 15:00", "at 3:30 pm", "at noon", "at midnight"
    noon_midnight = re.search(r"\b(noon|midnight)\b", text, re.I)
    if noon_midnight:
        word = noon_midnight.group(1).lower()
        return "12:00" if word == "noon" else "00:00"

    m = re.search(
        r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
        text, re.I
    )
    if m:
        hour   = int(m.group(1))
        minute = int(m.group(2) or 0)
        ampm   = (m.group(3) or "").lower()
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"

    m2 = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if m2:
        return f"{int(m2.group(1)):02d}:{int(m2.group(2)):02d}"

    return None


# ─────────────────────────────────────────────────────────────
#  RESULT DATACLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class NormalizedInput:
    """
    Result of running raw text through the normalizer.

    original    : the raw input from the user
    corrected   : spell-corrected + phrase-substituted text
    was_changed : True if anything was modified
    corrections : list of (original_word, corrected_word) pairs
    date        : ISO date string if a date was found ("2026-05-18")
    time        : "HH:MM" if a time was found
    entities    : {"app": "zoom", "url": "...", ...}
    """
    original:    str
    corrected:   str
    was_changed: bool
    corrections: list[tuple[str, str]] = field(default_factory=list)
    date:        Optional[str] = None
    time:        Optional[str] = None
    entities:    dict = field(default_factory=dict)

    def __str__(self) -> str:
        if self.was_changed:
            return f"[normalized] {self.corrected!r} (was: {self.original!r})"
        return f"[clean] {self.corrected!r}"


# ─────────────────────────────────────────────────────────────
#  NORMALIZER
# ─────────────────────────────────────────────────────────────

class Normalizer:
    """
    Input normalizer — the first thing raw text touches in the pipeline.

    1. Phrase substitution  (phrase dictionary — multi-word patterns)
    2. Spell correction     (pyspellchecker — word level)
    3. Entity extraction    (date, time, app, URL)
    """

    # Words the spell-checker must NOT correct
    _WHITELIST = {
        "nexus", "nmap", "ncat", "metasploit", "msfconsole", "burpsuite",
        "python", "linux", "ubuntu", "kali", "wifi", "bluetooth", "shodan",
        "nuclei", "nikto", "gobuster", "ffuf", "dirb", "hashcat", "hydra",
        "aircrack", "wireshark", "tcpdump", "sqlmap", "mimikatz", "netcat",
        "github", "gitlab", "api", "url", "http", "https", "ftp", "ssh",
        "cvss", "cve", "ghsa", "edb", "poc", "rce", "xss", "sqli",
        "lfi", "rfi", "ssrf", "csrf", "jwt", "oauth", "ldap", "smb",
        "powershell", "bash", "zsh", "vim", "nano", "tmux", "venv",
        "cyber", "cyril", "senanu",  # user names + project words
        "zoom", "meet", "teams", "slack", "discord", "whatsapp", "telegram",
        "gmail", "outlook", "calendar", "notion", "trello", "jira",
    }

    def __init__(self):
        self._spell = None
        self._init_spell()

    def _init_spell(self):
        try:
            from spellchecker import SpellChecker
            self._spell = SpellChecker()
            # Add project-specific words
            self._spell.word_frequency.load_words(list(self._WHITELIST))
            log.info("Spell checker ready.")
        except ImportError:
            log.warning("pyspellchecker not installed — spell correction disabled. pip install pyspellchecker")

    def normalize(self, text: str) -> NormalizedInput:
        """
        Run the full normalization pipeline on raw input text.
        Returns a NormalizedInput with corrected text + extracted entities.
        """
        original = text
        working  = text.strip()

        corrections: list[tuple[str, str]] = []

        # ── Step 1: Phrase substitution ───────────────────────────────
        for pattern, replacement in _PHRASE_PATTERNS:
            new = pattern.sub(replacement, working)
            if new != working:
                # Track what changed (approximate)
                corrections.append((pattern.pattern[3:-3], replacement))
                working = new

        # ── Step 2: Spell correction ──────────────────────────────────
        if self._spell:
            words    = working.split()
            fixed    = []
            for word in words:
                # Skip: very short, punctuation-only, numbers, URLs, already whitelisted
                bare = re.sub(r"[^a-zA-Z]", "", word).lower()
                if not bare or len(bare) < 3 or bare in self._WHITELIST:
                    fixed.append(word)
                    continue
                # Only correct if the word is clearly misspelled
                if bare not in self._spell:
                    candidate = self._spell.correction(bare)
                    if candidate and candidate != bare:
                        # Preserve capitalization pattern
                        corrected_word = self._match_case(word, candidate)
                        corrections.append((word, corrected_word))
                        fixed.append(corrected_word)
                        continue
                fixed.append(word)
            working = " ".join(fixed)

        # ── Step 3: Entity extraction ─────────────────────────────────
        extracted_date = _resolve_relative_date(working)
        extracted_time = _resolve_time(working)
        entities       = self._extract_entities(working)

        if extracted_date:
            entities["date"] = extracted_date
        if extracted_time:
            entities["time"] = extracted_time

        was_changed = working.strip() != original.strip()

        return NormalizedInput(
            original    = original,
            corrected   = working,
            was_changed = was_changed,
            corrections = corrections,
            date        = extracted_date,
            time        = extracted_time,
            entities    = entities,
        )

    def _extract_entities(self, text: str) -> dict:
        """Extract named entities: URLs, apps, durations, reminder times."""
        entities = {}

        # URLs
        url = re.search(r"https?://\S+", text)
        if url:
            entities["url"] = url.group(0)

        # Known apps / services
        app_patterns = {
            "zoom":       r"\bzoom\b",
            "google meet":r"\bgoogle meet\b|\bgmeet\b",
            "teams":      r"\bteams\b|\bms teams\b|\bmicrosoft teams\b",
            "slack":      r"\bslack\b",
            "discord":    r"\bdiscord\b",
            "whatsapp":   r"\bwhatsapp\b",
            "telegram":   r"\btelegram\b",
            "gmail":      r"\bgmail\b",
            "outlook":    r"\boutlook\b",
            "calendar":   r"\bcalendar\b|\bgoogle calendar\b",
            "notion":     r"\bnotion\b",
            "github":     r"\bgithub\b",
            "firefox":    r"\bfirefox\b",
            "chrome":     r"\bchrome\b|\bgoogle chrome\b",
            "vscode":     r"\bvs ?code\b|\bvscode\b",
            "terminal":   r"\bterminal\b",
        }
        for app, pat in app_patterns.items():
            if re.search(pat, text, re.I):
                entities["app"] = app
                break  # first match wins

        # Duration (e.g. "30 minutes before", "1 hour before")
        dur = re.search(
            r"(\d+)\s*(minute|min|hour|hr|second|sec)s?\s*(?:before|prior|early)", text, re.I
        )
        if dur:
            entities["reminder_offset"] = f"{dur.group(1)} {dur.group(2)}"

        # Person name (rudimentary: "with John", "invite Sarah")
        person = re.search(r"\b(?:with|invite|and|call|email)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", text)
        if person:
            entities["person"] = person.group(1)

        return entities

    @staticmethod
    def _match_case(original: str, replacement: str) -> str:
        """Preserve capitalization style of original word."""
        if original.isupper():
            return replacement.upper()
        if original.istitle():
            return replacement.capitalize()
        return replacement


# ── Singleton ─────────────────────────────────────────────────

_normalizer: Optional[Normalizer] = None


def get_normalizer() -> Normalizer:
    global _normalizer
    if _normalizer is None:
        _normalizer = Normalizer()
    return _normalizer


def normalize(text: str) -> NormalizedInput:
    """Convenience function — normalize text through the global normalizer."""
    return get_normalizer().normalize(text)
