"""
core/intent_engine.py
NEXUS Intent Engine — converts raw text into structured intent objects.

Supports two backends:
  - 'rules'  : fast, zero-dependency pattern matching (default)
  - 'llm'    : Ollama-based NLU for ambiguous / complex commands (optional)
"""

import re
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

log = logging.getLogger("nexus.intent")

# ─────────────────────────────────────────────
# Intent result dataclass
# ─────────────────────────────────────────────

@dataclass
class Intent:
    intent:     str            # e.g. "open_application"
    target:     Optional[str]  # e.g. "firefox"
    action:     Optional[str]  # e.g. "open", "close", "search"
    query:      Optional[str]  # free-form payload for conversation/search
    confidence: float          # 0.0 – 1.0
    raw:        str            # original input

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self):
        return (
            f"Intent({self.intent!r}, target={self.target!r}, "
            f"action={self.action!r}, confidence={self.confidence:.2f})"
        )


# ─────────────────────────────────────────────
# Rule-based pattern definitions
# ─────────────────────────────────────────────

# Each entry: (intent_name, compiled_regex, action_hint)
# Groups named 'target' and 'query' are extracted automatically.
RULES: list[tuple[str, re.Pattern, str]] = [

    # ── Applications ──────────────────────────────────────────────────────
    ("open_application",
     re.compile(r"^(?:open|launch|start|run|execute)\s+(?P<target>\w[\w\s-]*)$", re.I),
     "open"),

    ("close_application",
     re.compile(r"^(?:close|quit|kill|exit)\s+(?P<target>\w[\w\s-]*)$", re.I),
     "close"),

    # ── Files & folders ───────────────────────────────────────────────────
    ("open_file",
     re.compile(r"^(?:open|show|read)\s+(?:file\s+)?(?P<target>[\w./~\-]+\.\w+)", re.I),
     "open"),

    ("open_folder",
     re.compile(r"^(?:open|go to|navigate to)\s+(?:folder|directory|dir)\s+(?P<target>[\w./~\-]+)", re.I),
     "open"),

    ("list_files",
     re.compile(r"^(?:list|show|ls)\s+(?:files|contents?)(?:\s+(?:in|of|at)\s+(?P<target>[\w./~\-]+))?", re.I),
     "list"),

    # ── Web / Browser ─────────────────────────────────────────────────────
    ("search_web",
     re.compile(r"^(?:search|google|look up|find)\s+(?:for\s+)?(?P<query>.+)$", re.I),
     "search"),

    ("open_url",
     re.compile(r"^(?:open|go to|visit|navigate to)\s+(?P<target>https?://\S+|www\.\S+|\w+\.\w{2,})", re.I),
     "open"),

    # ── System controls ───────────────────────────────────────────────────
    ("system_control",
     re.compile(r"^(?P<action>shutdown|reboot|restart|lock|sleep|hibernate|log ?out)\s*(?:the\s+)?(?:system|computer|machine|laptop|pc)?$", re.I),
     None),

    ("volume_control",
     re.compile(r"(?:set\s+)?volume\s+(?P<action>up|down|mute|unmute|max|min|to)\s*(?P<target>\d+)?", re.I),
     None),

    ("brightness_control",
     re.compile(r"(?:set\s+)?brightness\s+(?P<action>up|down|to)\s*(?P<target>\d+)?", re.I),
     None),

    # ── Shell / Terminal ──────────────────────────────────────────────────
    ("run_command",
     re.compile(r"^(?:run|execute|do)\s+(?:command\s+)?(?P<query>.+)$", re.I),
     "shell"),

    # ── Memory & notes ────────────────────────────────────────────────────
    ("remember",
     re.compile(r"^(?:remember|note|save|store)\s+(?:that\s+)?(?P<query>.+)$", re.I),
     "store"),

    ("recall",
     re.compile(r"^(?:what did i|recall|remind me|do you remember)\s+(?:about\s+)?(?P<query>.+)?$", re.I),
     "retrieve"),

    # ── Cybersecurity — scanning ──────────────────────────────────────────
    ("cyber_full_network_scan",
     re.compile(r"^(?:scan my network|scan network|network scan|scan local network)", re.I),
     "scan"),

    ("cyber_discover_devices",
     re.compile(r"(?:what devices|who is on|devices on|hosts on|find devices|discover hosts|who.?s on|what.?s on my network)", re.I),
     "discover"),

    # ── Cybersecurity — toolkit (must be before port scan to avoid false match) —
    ("cyber_recommend_tool",
     re.compile(r"(?:best tool for|recommend tool|which tool)\s+(?P<query>.+)?", re.I),
     "recommend"),

    ("cyber_port_scan",
     re.compile(r"(?:open ports on|ports on|scan ports|check ports|port scan|what ports|scan host|scan ip)\s*(?:on\s+|at\s+|for\s+)?(?P<target>[\d.:/\w-]+)?", re.I),
     "scan"),

    ("cyber_quick_scan",
     re.compile(r"(?:quick scan|fast scan)\s+(?P<target>[\d.:/\w-]+)", re.I),
     "scan"),

    ("cyber_full_scan",
     re.compile(r"(?:full scan|deep scan|detailed scan|service scan)\s+(?P<target>[\d.:/\w-]+)", re.I),
     "scan"),

    ("cyber_stealth_scan",
     re.compile(r"(?:stealth scan|silent scan|syn scan)\s+(?P<target>[\d.:/\w-]+)", re.I),
     "scan"),

    # ── Cybersecurity — network intel ─────────────────────────────────────
    ("cyber_show_interfaces",
     re.compile(r"(?:show interfaces|network interfaces|my ip|local ip|ip address|what is my ip|interface info)", re.I),
     "query"),

    ("cyber_external_ip",
     re.compile(r"(?:external ip|public ip|my public ip|internet ip)", re.I),
     "query"),

    ("cyber_show_arp",
     re.compile(r"(?:arp table|arp cache|local arp)", re.I),
     "query"),

    ("cyber_show_connections",
     re.compile(r"(?:active connections|open connections|network connections|established connections)", re.I),
     "query"),

    ("cyber_show_routes",
     re.compile(r"(?:routing table|ip routes|show routes)", re.I),
     "query"),

    ("cyber_show_subnet",
     re.compile(r"(?:my subnet|local subnet|network range|cidr)", re.I),
     "query"),

    # ── Cybersecurity — log / threat analysis ─────────────────────────────
    ("cyber_analyze_logs",
     re.compile(r"(?:suspicious activity|check logs|analyze logs|security scan|intrusion|threats|attacks|log analysis|security report)", re.I),
     "analyze"),

    ("cyber_check_logins",
     re.compile(r"(?:failed login|brute force|ssh attacks|login attempts)", re.I),
     "analyze"),

    ("cyber_check_processes",
     re.compile(r"(?:suspicious process|malicious process|check processes)", re.I),
     "analyze"),

    ("cyber_active_sessions",
     re.compile(r"(?:active sessions|who is logged|current users)", re.I),
     "query"),

    ("cyber_check_listening",
     re.compile(r"(?:listening ports|open ports local|services running)", re.I),
     "query"),

    # ── Cybersecurity — toolkit ───────────────────────────────────────────
    ("cyber_toolkit_status",
     re.compile(r"(?:toolkit status|tool status|what tools|available tools|installed tools|show tools)", re.I),
     "query"),

    ("cyber_install_tool",
     re.compile(r"^(?:install|download)\s+(?P<target>[\w\-]+)", re.I),
     "install"),

    ("cyber_help",
     re.compile(r"^cyber help$|^what can nexus do.*cyber|^cyber commands", re.I),
     "help"),

    # ── Network scan (legacy / generic) ───────────────────────────────────
    ("network_scan",
     re.compile(r"^(?:scan|ping|probe)\s+(?:network|host|ip|target)?\s*(?P<target>[\d./\w-]+)?", re.I),
     "scan"),

    # ── Status / info ─────────────────────────────────────────────────────
    ("system_info",
     re.compile(r"^(?:system\s+)?(?:status|info|information|stats|resources|cpu|memory|ram|disk)", re.I),
     "query"),

    ("get_time",
     re.compile(r"^(?:what(?:'s|\s+is)\s+the\s+)?(?:time|date|day|clock)", re.I),
     "query"),

    # ── Conversation / general ────────────────────────────────────────────
    ("greet",
     re.compile(r"^(?:hello|hi|hey|good\s+(?:morning|afternoon|evening)|what(?:'s|\s+is)\s+up)", re.I),
     None),

    ("farewell",
     re.compile(r"^(?:bye|goodbye|exit|quit|stop|see\s+you|later)", re.I),
     None),

    # ── Research ──────────────────────────────────────────────────────────
    ("research_topic",
     re.compile(r"^(?:research|study|learn about|look up|find out about)\s+(?P<query>.+)$", re.I),
     "research"),

    ("read_url",
     re.compile(r"^(?:read|fetch|summarize|learn from)\s+(?:url\s+|this\s+)?(?P<target>https?://\S+)", re.I),
     "read"),

    ("ask_question",
     re.compile(r"^(?:what|who|when|where|why|how|can you|could you|tell me)\b(?P<query>.+)?$", re.I),
     "answer"),
]


# ─────────────────────────────────────────────
# App name aliases — normalize before matching
# "vs code" → "code", "chrome" → "google-chrome"
# ─────────────────────────────────────────────

APP_NAME_ALIASES: dict[str, str] = {
    "vs code":     "code",
    "vscode":      "code",
    "chrome":      "google-chrome",
    "browser":     "firefox",
    "terminal":    "x-terminal-emulator",
    "burp":        "burpsuite",
    "burp suite":  "burpsuite",
    "metasploit":  "msfconsole",
    "files":       "nautilus",
    "file manager":"nautilus",
}


def _normalize_target(text: str) -> str:
    """Swap user-friendly app names for their actual binary names."""
    lowered = text.lower().strip()
    return APP_NAME_ALIASES.get(lowered, lowered)
# ─────────────────────────────────────────────
# Rule-based engine
# ─────────────────────────────────────────────

def _rules_parse(text: str) -> Intent:
    """Try every rule in order; return first match."""
    # Strip whitespace and common trailing punctuation added by STT
    cleaned = text.strip().rstrip(".?!,;")

    for intent_name, pattern, action_hint in RULES:
        m = pattern.search(cleaned)
        if not m:
            continue

        groups = m.groupdict()
        target = groups.get("target", None)
        query  = groups.get("query",  None)
        action = groups.get("action", action_hint)

        # Clean up captured strings
        if target:
            target = _normalize_target(target.strip())
        if query:
            query = query.strip()
        if action:
            action = action.strip().lower()

        log.debug(f"Rule match: {intent_name!r} on {cleaned!r}")
        return Intent(
            intent=intent_name,
            target=target,
            action=action,
            query=query,
            confidence=0.90,
            raw=text,
        )

    # Nothing matched
    log.debug(f"No rule matched: {cleaned!r}")
    return Intent(
        intent="unknown",
        target=None,
        action=None,
        query=cleaned,
        confidence=0.0,
        raw=text,
    )


# ─────────────────────────────────────────────
# Optional LLM backend (Ollama)
# ─────────────────────────────────────────────

def _llm_parse(text: str, model: str = "mistral") -> Optional[Intent]:
    """
    Ask a local Ollama model to parse the intent.
    Returns None if Ollama is unavailable.
    Requires: pip install ollama
    """
    try:
        import ollama  # type: ignore
    except ImportError:
        log.warning("ollama package not installed — falling back to rules")
        return None

    prompt = f"""You are an intent classifier for an AI assistant called NEXUS.
Parse the user command into JSON with these keys:
  intent      : string  (snake_case label for what the user wants)
  target      : string | null  (app name, filename, URL, IP, etc.)
  action      : string | null  (verb: open, close, search, scan…)
  query       : string | null  (free-form text payload)
  confidence  : float 0-1

Respond ONLY with the JSON object, no explanation.

User command: "{text}"
"""
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_json = response["message"]["content"].strip()
        data = json.loads(raw_json)
        return Intent(
            intent=data.get("intent", "unknown"),
            target=data.get("target"),
            action=data.get("action"),
            query=data.get("query"),
            confidence=float(data.get("confidence", 0.75)),
            raw=text,
        )
    except Exception as e:
        log.warning(f"LLM parse failed: {e}")
        return None


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

class IntentEngine:
    """
    Usage:
        engine = IntentEngine(backend="rules")   # or "llm"
        intent = engine.parse("open firefox")
        print(intent)
        # Intent('open_application', target='firefox', action='open', confidence=0.90)
    """

    def __init__(self, backend: str = "rules", llm_model: str = "mistral"):
        self.backend   = backend
        self.llm_model = llm_model
        log.info(f"IntentEngine initialized — backend={backend!r}")

    def parse(self, text: str) -> Intent:
        if not text or not text.strip():
            return Intent("empty", None, None, None, 0.0, text)

        if self.backend == "llm":
            result = _llm_parse(text, self.llm_model)
            if result:
                return result
            # Fall through to rules if LLM fails

        return _rules_parse(text)

    def parse_batch(self, texts: list[str]) -> list[Intent]:
        return [self.parse(t) for t in texts]


# ─────────────────────────────────────────────
# CLI test mode
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    engine = IntentEngine(backend="rules")

    test_phrases = [
        "open firefox",
        "launch VS Code",
        "search for python tutorials",
        "shutdown the computer",
        "what time is it",
        "remember that my project deadline is Friday",
        "scan network 192.168.1.0/24",
        "run command ls -la",
        "hello nexus",
        "how does gradient descent work",
        "gibberish phrase that matches nothing",
    ]

    # If args provided, test those instead
    if len(sys.argv) > 1:
        test_phrases = [" ".join(sys.argv[1:])]

    print("\n─── NEXUS Intent Engine Test ───\n")
    for phrase in test_phrases:
        result = engine.parse(phrase)
        print(f"  Input : {phrase!r}")
        print(f"  Result: {result}")
        print()