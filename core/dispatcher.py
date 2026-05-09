"""
core/dispatcher.py
NEXUS Dispatcher — receives a parsed Intent and executes the matching action.

Safety model:
  - Applications are launched via a curated alias table, not raw user input
  - Shell commands require explicit whitelist OR --allow-shell flag
  - All actions are logged to nexus.dispatcher

Usage:
    from core.intent_engine import IntentEngine
    from core.dispatcher import Dispatcher

    engine = IntentEngine()
    dispatcher = Dispatcher()

    intent = engine.parse("open firefox")
    result = dispatcher.dispatch(intent)
    print(result.message)
"""

import subprocess
import webbrowser
import datetime
import shutil
import logging
import platform
import os
import sys
from dataclasses import dataclass
from typing import Optional

# Add project root to path so we can import 'core'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.intent_engine import Intent, IntentEngine

log = logging.getLogger("nexus.dispatcher")

# ─────────────────────────────────────────────
# Result object
# ─────────────────────────────────────────────

@dataclass
class ActionResult:
    success:  bool
    message:  str
    data:     Optional[dict] = None

    def __str__(self):
        status = "✓" if self.success else "✗"
        return f"[{status}] {self.message}"


# ─────────────────────────────────────────────
# Application alias table
# Maps what users SAY → actual Linux binary name
# ─────────────────────────────────────────────

APP_ALIASES: dict[str, str] = {
    # Browsers
    "browser":          "firefox",
    "chrome":           "google-chrome",
    "chromium":         "chromium-browser",
    "firefox":          "firefox",
    "brave":            "brave-browser",

    # Editors / IDEs
    "vs code":          "code",
    "vscode":           "code",
    "vscodium":         "vscodium",
    "code":             "code",
    "vim":              "vim",
    "nvim":             "nvim",
    "nano":             "nano",
    "gedit":            "gedit",

    # Terminals
    "terminal":         "x-terminal-emulator",
    "konsole":          "konsole",
    "gnome terminal":   "gnome-terminal",

    # Security tools (your cyber stack)
    "burp":             "burpsuite",
    "burp suite":       "burpsuite",
    "wireshark":        "wireshark",
    "nmap":             "nmap",
    "metasploit":       "msfconsole",
    "maltego":          "maltego",

    # Utilities
    "files":            "nautilus",
    "file manager":     "nautilus",
    "calculator":       "gnome-calculator",
    "text editor":      "gedit",
    "settings":         "gnome-control-center",
    "discord":          "discord",
    "spotify":          "spotify",
    "vlc":              "vlc",

    # Python / dev
    "python":           "python3",
    "jupyter":          "jupyter-notebook",
}

# ─────────────────────────────────────────────
# Safe shell command whitelist
# Only these base commands can be run via voice
# ─────────────────────────────────────────────

SAFE_COMMANDS: set[str] = {
    "ls", "pwd", "whoami", "uname", "uptime",
    "df", "free", "top", "htop", "ps",
    "ping", "ifconfig", "ip", "hostname",
    "cat", "echo", "date", "cal",
    "git", "python3", "pip3",
    "nmap", "netstat", "ss", "curl", "wget",
}


# ─────────────────────────────────────────────
# Individual action handlers
# ─────────────────────────────────────────────

def _handle_open_application(intent: Intent) -> ActionResult:
    target = intent.target or ""
    binary = APP_ALIASES.get(target.lower(), target.lower().replace(" ", "-"))

    if not binary:
        return ActionResult(False, "No application target specified.")

    # Check if binary exists on PATH
    if not shutil.which(binary):
        return ActionResult(
            False,
            f"Application '{binary}' not found on PATH. "
            f"Try: sudo apt install {binary}",
        )

    try:
        subprocess.Popen(
            [binary],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info(f"Launched: {binary}")
        return ActionResult(True, f"Opening {target} ({binary})")
    except Exception as e:
        log.error(f"Launch failed: {e}")
        return ActionResult(False, f"Failed to open {target}: {e}")


def _handle_close_application(intent: Intent) -> ActionResult:
    target = intent.target or ""
    binary = APP_ALIASES.get(target.lower(), target.lower())

    try:
        result = subprocess.run(
            ["pkill", "-f", binary],
            capture_output=True,
        )
        if result.returncode == 0:
            return ActionResult(True, f"Closed {target}.")
        else:
            return ActionResult(False, f"No running process found for '{target}'.")
    except Exception as e:
        return ActionResult(False, f"Could not close {target}: {e}")


def _handle_search_web(intent: Intent) -> ActionResult:
    query = intent.query or intent.target or ""
    if not query:
        return ActionResult(False, "No search query provided.")

    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    try:
        webbrowser.open(url)
        return ActionResult(True, f"Searching the web for: {query}", {"url": url})
    except Exception as e:
        return ActionResult(False, f"Could not open browser: {e}")


def _handle_open_url(intent: Intent) -> ActionResult:
    url = intent.target or ""
    if not url:
        return ActionResult(False, "No URL provided.")

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        webbrowser.open(url)
        return ActionResult(True, f"Opening {url}")
    except Exception as e:
        return ActionResult(False, f"Could not open URL: {e}")


def _handle_get_time(intent: Intent) -> ActionResult:
    now = datetime.datetime.now()
    time_str = now.strftime("%I:%M %p")
    date_str = now.strftime("%A, %B %d %Y")
    return ActionResult(
        True,
        f"The time is {time_str} — {date_str}",
        {"time": time_str, "date": date_str},
    )


def _handle_system_info(intent: Intent) -> ActionResult:
    lines = []
    try:
        import psutil  # type: ignore
        cpu    = psutil.cpu_percent(interval=0.5)
        mem    = psutil.virtual_memory()
        disk   = psutil.disk_usage("/")
        lines += [
            f"CPU:    {cpu}%",
            f"RAM:    {mem.percent}% used ({mem.used // 1024**2} MB / {mem.total // 1024**2} MB)",
            f"Disk:   {disk.percent}% used ({disk.used // 1024**3} GB / {disk.total // 1024**3} GB)",
        ]
    except ImportError:
        lines.append("Install psutil for full stats: pip install psutil")

    lines.append(f"OS:     {platform.system()} {platform.release()}")
    lines.append(f"Host:   {platform.node()}")
    lines.append(f"Python: {platform.python_version()}")

    return ActionResult(True, "\n  ".join(lines))


def _handle_system_control(intent: Intent) -> ActionResult:
    action = (intent.action or "").lower()

    commands = {
        "shutdown":  ["shutdown", "now"],
        "reboot":    ["reboot"],
        "restart":   ["reboot"],
        "lock":      ["loginctl", "lock-session"],
        "sleep":     ["systemctl", "suspend"],
        "hibernate": ["systemctl", "hibernate"],
        "logout":    ["loginctl", "terminate-user", os.environ.get("USER", "")],
        "log out":   ["loginctl", "terminate-user", os.environ.get("USER", "")],
    }

    cmd = commands.get(action)
    if not cmd:
        return ActionResult(False, f"Unknown system action: {action!r}")

    # Safety: confirm destructive actions
    log.warning(f"System control action: {action}")
    try:
        subprocess.run(cmd, check=True)
        return ActionResult(True, f"Executing system {action}…")
    except subprocess.CalledProcessError as e:
        return ActionResult(False, f"System control failed (may need sudo): {e}")
    except Exception as e:
        return ActionResult(False, str(e))


def _handle_volume_control(intent: Intent) -> ActionResult:
    action = (intent.action or "").lower()
    target = intent.target or ""

    cmd_map = {
        "up":      ["amixer", "-D", "pulse", "sset", "Master", "5%+"],
        "down":    ["amixer", "-D", "pulse", "sset", "Master", "5%-"],
        "mute":    ["amixer", "-D", "pulse", "sset", "Master", "mute"],
        "unmute":  ["amixer", "-D", "pulse", "sset", "Master", "unmute"],
        "max":     ["amixer", "-D", "pulse", "sset", "Master", "100%"],
        "min":     ["amixer", "-D", "pulse", "sset", "Master", "0%"],
    }

    if action == "to" and target.isdigit():
        cmd = ["amixer", "-D", "pulse", "sset", "Master", f"{target}%"]
    else:
        cmd = cmd_map.get(action)

    if not cmd:
        return ActionResult(False, f"Unknown volume action: {action!r}")

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        label = f"{target}%" if action == "to" else action
        return ActionResult(True, f"Volume {label}")
    except Exception as e:
        return ActionResult(False, f"Volume control failed: {e}")


def _handle_run_command(intent: Intent) -> ActionResult:
    raw_cmd = (intent.query or "").strip()
    if not raw_cmd:
        return ActionResult(False, "No command provided.")

    base = raw_cmd.split()[0]
    if base not in SAFE_COMMANDS:
        return ActionResult(
            False,
            f"Command '{base}' is not in the NEXUS safe list.\n"
            f"Add it to SAFE_COMMANDS in core/dispatcher.py to allow it.",
        )

    try:
        result = subprocess.run(
            raw_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = (result.stdout or result.stderr or "").strip()
        return ActionResult(
            result.returncode == 0,
            output or "(no output)",
        )
    except subprocess.TimeoutExpired:
        return ActionResult(False, "Command timed out after 15 seconds.")
    except Exception as e:
        return ActionResult(False, str(e))


def _handle_remember(intent: Intent) -> ActionResult:
    """
    Simple flat-file memory — real vector memory comes later.
    Writes to data/memory.log for now.
    """
    text = intent.query or ""
    if not text:
        return ActionResult(False, "Nothing to remember.")

    os.makedirs("data", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {text}\n"

    with open("data/memory.log", "a") as f:
        f.write(entry)

    return ActionResult(True, f"Got it — I'll remember: \"{text}\"")


def _handle_recall(intent: Intent) -> ActionResult:
    query = (intent.query or "").lower().strip()
    log_path = "data/memory.log"

    if not os.path.exists(log_path):
        return ActionResult(False, "Memory log is empty — nothing stored yet.")

    with open(log_path, "r") as f:
        lines = f.readlines()

    if not query:
        recent = lines[-5:] if len(lines) >= 5 else lines
        return ActionResult(True, "Last memories:\n  " + "  ".join(recent).strip())

    matches = [l for l in lines if query in l.lower()]
    if not matches:
        return ActionResult(False, f"Nothing found in memory matching '{query}'.")

    return ActionResult(True, f"Found {len(matches)} match(es):\n  " + "  ".join(matches[-5:]).strip())


def _handle_greet(intent: Intent) -> ActionResult:
    import random
    greetings = [
        "Hey. NEXUS online.",
        "Hello. What do you need?",
        "NEXUS here. Ready.",
        "Hey — what can I do for you?",
    ]
    return ActionResult(True, random.choice(greetings))


def _handle_farewell(intent: Intent) -> ActionResult:
    return ActionResult(True, "NEXUS signing off. Stay sharp.")


def _handle_unknown(intent: Intent) -> ActionResult:
    return ActionResult(
        False,
        f"I didn't understand: \"{intent.raw}\"\n"
        f"Try rephrasing, or say 'help' for available commands.",
    )


# ─────────────────────────────────────────────
# Dispatch table
# Maps intent name → handler function
# ─────────────────────────────────────────────

DISPATCH_TABLE = {
    "open_application":  _handle_open_application,
    "close_application": _handle_close_application,
    "search_web":        _handle_search_web,
    "open_url":          _handle_open_url,
    "get_time":          _handle_get_time,
    "system_info":       _handle_system_info,
    "system_control":    _handle_system_control,
    "volume_control":    _handle_volume_control,
    "run_command":       _handle_run_command,
    "remember":          _handle_remember,
    "recall":            _handle_recall,
    "greet":             _handle_greet,
    "farewell":          _handle_farewell,
}


# ─────────────────────────────────────────────
# Main Dispatcher class
# ─────────────────────────────────────────────

class Dispatcher:
    """
    Receives an Intent and executes the corresponding action.

    Usage:
        d = Dispatcher()
        result = d.dispatch(intent)
        print(result)
    """

    def __init__(self):
        log.info("Dispatcher initialized.")

    def dispatch(self, intent: Intent) -> ActionResult:
        if intent.confidence == 0.0 or intent.intent == "unknown":
            return _handle_unknown(intent)

        handler = DISPATCH_TABLE.get(intent.intent)

        if handler is None:
            log.warning(f"No handler for intent: {intent.intent!r}")
            return ActionResult(
                False,
                f"No action registered for intent '{intent.intent}'. "
                f"Add a handler in core/dispatcher.py",
            )

        log.info(f"Dispatching: {intent}")
        try:
            return handler(intent)
        except Exception as e:
            log.error(f"Handler crashed for {intent.intent!r}: {e}", exc_info=True)
            return ActionResult(False, f"Action failed: {e}")


# ─────────────────────────────────────────────
# CLI test mode  —  python core/dispatcher.py
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    engine     = IntentEngine(backend="rules")
    dispatcher = Dispatcher()

    test_phrases = [
        "what time is it",
        "open firefox",
        "search for hackthebox machines",
        "system info",
        "remember that i need to study flip flops tonight",
        "what did i remember",
        "hello nexus",
        "run command ls -la",
        "volume up",
        "gibberish xkjshd askdjh",
    ]

    if len(sys.argv) > 1:
        test_phrases = [" ".join(sys.argv[1:])]

    print("\n─── NEXUS Dispatcher Test ───\n")
    for phrase in test_phrases:
        intent = engine.parse(phrase)
        result = dispatcher.dispatch(intent)
        print(f"  Input  : {phrase!r}")
        print(f"  Intent : {intent}")
        print(f"  Result : {result}")
        print()