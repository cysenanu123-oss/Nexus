"""
NEXUS — Personal General AI Assistant & Autonomous Intelligence System
main.py — Core launcher, ASCII banner, colorized terminal output, command parser

Phase 1 — Foundation Setup
"""

import os
import sys
import time
import shutil
import platform
import datetime
import subprocess
from typing import Optional
from core.logger import setup_logging


# ─────────────────────────────────────────────────────────────
#  ANSI COLOR SYSTEM
# ─────────────────────────────────────────────────────────────

class Color:
    """ANSI escape code color palette for NEXUS terminal output."""

    RESET       = "\033[0m"
    BOLD        = "\033[1m"
    DIM         = "\033[2m"
    ITALIC      = "\033[3m"
    UNDERLINE   = "\033[4m"

    # Foreground
    BLACK       = "\033[30m"
    RED         = "\033[31m"
    GREEN       = "\033[32m"
    YELLOW      = "\033[33m"
    BLUE        = "\033[34m"
    MAGENTA     = "\033[35m"
    CYAN        = "\033[36m"
    WHITE       = "\033[37m"

    # Bright foreground
    BRIGHT_RED      = "\033[91m"
    BRIGHT_GREEN    = "\033[92m"
    BRIGHT_YELLOW   = "\033[93m"
    BRIGHT_BLUE     = "\033[94m"
    BRIGHT_MAGENTA  = "\033[95m"
    BRIGHT_CYAN     = "\033[96m"
    BRIGHT_WHITE    = "\033[97m"

    # Background
    BG_BLACK    = "\033[40m"
    BG_GREEN    = "\033[42m"
    BG_CYAN     = "\033[46m"

    @staticmethod
    def supported() -> bool:
        """Return True if the terminal likely supports ANSI colors."""
        if platform.system() == "Windows":
            return os.environ.get("ANSICON") is not None or "WT_SESSION" in os.environ
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


# Fallback: strip colors when the terminal doesn't support them
if not Color.supported():
    for attr in [a for a in dir(Color) if not a.startswith("_") and a != "supported"]:
        if isinstance(getattr(Color, attr), str):
            setattr(Color, attr, "")


# ─────────────────────────────────────────────────────────────
#  PRINTER — styled terminal output helpers
# ─────────────────────────────────────────────────────────────

class Printer:
    """Formatted terminal output for NEXUS."""

    PREFIX = {
        "info":    f"{Color.BRIGHT_CYAN}[INFO]{Color.RESET}",
        "ok":      f"{Color.BRIGHT_GREEN}[ OK ]{Color.RESET}",
        "warn":    f"{Color.BRIGHT_YELLOW}[WARN]{Color.RESET}",
        "error":   f"{Color.BRIGHT_RED}[FAIL]{Color.RESET}",
        "cmd":     f"{Color.BRIGHT_MAGENTA}[ >> ]{Color.RESET}",
        "system":  f"{Color.BRIGHT_BLUE}[ SYS]{Color.RESET}",
        "nexus":   f"{Color.BRIGHT_GREEN}[NEXUS]{Color.RESET}",
    }

    @staticmethod
    def _ts() -> str:
        return f"{Color.DIM}{datetime.datetime.now().strftime('%H:%M:%S')}{Color.RESET}"

    @classmethod
    def info(cls, msg: str):
        print(f"{cls._ts()} {cls.PREFIX['info']} {msg}")

    @classmethod
    def ok(cls, msg: str):
        print(f"{cls._ts()} {cls.PREFIX['ok']} {Color.BRIGHT_GREEN}{msg}{Color.RESET}")

    @classmethod
    def warn(cls, msg: str):
        print(f"{cls._ts()} {cls.PREFIX['warn']} {Color.BRIGHT_YELLOW}{msg}{Color.RESET}")

    @classmethod
    def error(cls, msg: str):
        print(f"{cls._ts()} {cls.PREFIX['error']} {Color.BRIGHT_RED}{msg}{Color.RESET}")

    @classmethod
    def cmd(cls, msg: str):
        print(f"{cls._ts()} {cls.PREFIX['cmd']} {Color.BRIGHT_MAGENTA}{msg}{Color.RESET}")

    @classmethod
    def system(cls, msg: str):
        print(f"{cls._ts()} {cls.PREFIX['system']} {Color.BRIGHT_BLUE}{msg}{Color.RESET}")

    @classmethod
    def nexus(cls, msg: str):
        print(f"{cls._ts()} {cls.PREFIX['nexus']} {Color.BRIGHT_WHITE}{msg}{Color.RESET}")

    @staticmethod
    def divider(char: str = "─", width: Optional[int] = None):
        w = width or shutil.get_terminal_size().columns
        print(f"{Color.DIM}{char * w}{Color.RESET}")

    @staticmethod
    def blank():
        print()


# ─────────────────────────────────────────────────────────────
#  ASCII BANNER
# ─────────────────────────────────────────────────────────────

BANNER = r"""
███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗
████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝
██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗
██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║
██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║
╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
"""

TAGLINE    = "Personal General AI Assistant  //  Autonomous Intelligence System"
VERSION    = "v0.1.0-alpha"
PHASE      = "Phase 1 — Foundation"


def print_banner():
    """Print the animated NEXUS startup banner."""
    os.system("clear" if platform.system() != "Windows" else "cls")

    term_width = shutil.get_terminal_size().columns

    # Stream banner lines with a slight delay for effect
    for line in BANNER.splitlines():
        centered = line.center(term_width)
        print(f"{Color.BRIGHT_GREEN}{Color.BOLD}{centered}{Color.RESET}")
        time.sleep(0.04)

    # Tagline
    print(f"{Color.DIM}{TAGLINE.center(term_width)}{Color.RESET}")
    print()

    # Version / phase bar
    bar_items = [
        f"{Color.BRIGHT_GREEN}{VERSION}{Color.RESET}",
        f"{Color.DIM}|{Color.RESET}",
        f"{Color.BRIGHT_CYAN}{PHASE}{Color.RESET}",
        f"{Color.DIM}|{Color.RESET}",
        f"{Color.BRIGHT_YELLOW}Python {platform.python_version()}{Color.RESET}",
        f"{Color.DIM}|{Color.RESET}",
        f"{Color.BRIGHT_MAGENTA}{platform.system()} {platform.release()}{Color.RESET}",
    ]
    bar = "  ".join(bar_items)
    # Simple center by padding — strip ANSI for length calc
    import re
    plain_bar = re.sub(r"\033\[[0-9;]*m", "", bar)
    pad = max(0, (term_width - len(plain_bar)) // 2)
    print(" " * pad + bar)
    print()

    Printer.divider("═")
    print()


# ─────────────────────────────────────────────────────────────
#  BOOT SEQUENCE
# ─────────────────────────────────────────────────────────────

def run_boot_sequence():
    """Display a staged boot sequence checking core subsystems."""

    checks = [
        ("Python runtime",        lambda: f"{platform.python_version()}"),
        ("Operating system",      lambda: f"{platform.system()} {platform.release()}"),
        ("Terminal width",        lambda: f"{shutil.get_terminal_size().columns} cols"),
        ("Working directory",     lambda: os.getcwd()),
        ("Core modules",          lambda: "os, sys, subprocess — OK"),
        ("Config directory",      lambda: _ensure_dir("config")),
        ("Logs directory",        lambda: _ensure_dir("logs")),
        ("Data directory",        lambda: _ensure_dir("data")),
        ("Plugin directory",      lambda: _ensure_dir("plugins")),
        ("NEXUS core",            lambda: "LOADED"),
    ]

    Printer.system("Running boot sequence...")
    Printer.blank()

    for label, check_fn in checks:
        time.sleep(0.08)
        try:
            result = check_fn()
            print(
                f"  {Color.DIM}{'·' * 3}{Color.RESET}  "
                f"{Color.WHITE}{label:<30}{Color.RESET}"
                f"{Color.BRIGHT_GREEN}✓  {Color.DIM}{result}{Color.RESET}"
            )
        except Exception as exc:
            print(
                f"  {Color.DIM}{'·' * 3}{Color.RESET}  "
                f"{Color.WHITE}{label:<30}{Color.RESET}"
                f"{Color.BRIGHT_RED}✗  {exc}{Color.RESET}"
            )

    Printer.blank()
    Printer.ok("All systems nominal. NEXUS is online.")
    Printer.blank()


def _ensure_dir(name: str) -> str:
    """Create a subdirectory relative to the launcher if missing."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    os.makedirs(path, exist_ok=True)
    return path


# ─────────────────────────────────────────────────────────────
#  BUILT-IN COMMANDS
# ─────────────────────────────────────────────────────────────

HELP_TEXT = f"""
{Color.BRIGHT_GREEN}{Color.BOLD}NEXUS — Available Commands{Color.RESET}

  {Color.BRIGHT_CYAN}help{Color.RESET}                   Show this help message
  {Color.BRIGHT_CYAN}status{Color.RESET}                 Display current system status
  {Color.BRIGHT_CYAN}version{Color.RESET}                Show NEXUS version info
  {Color.BRIGHT_CYAN}sysinfo{Color.RESET}                Display host system information
  {Color.BRIGHT_CYAN}clear{Color.RESET}                  Clear the terminal screen
  {Color.BRIGHT_CYAN}modules{Color.RESET}                List all NEXUS modules and their state
  {Color.BRIGHT_CYAN}run <module>{Color.RESET}           Run a specific NEXUS module
  {Color.BRIGHT_CYAN}voice{Color.RESET}                  Manually trigger voice listening mode
  {Color.BRIGHT_CYAN}shell <cmd>{Color.RESET}            Execute a shell command (use carefully)
  {Color.BRIGHT_CYAN}echo <text>{Color.RESET}            Echo text back to the terminal
  {Color.BRIGHT_CYAN}history{Color.RESET}                Show command history for this session
  {Color.BRIGHT_CYAN}exit / quit{Color.RESET}            Shutdown NEXUS

{Color.DIM}Phase 2+ commands (voice, vision, memory) will be added as modules are built.{Color.RESET}
"""

# Module registry — expands as phases are completed
MODULE_REGISTRY = {
    "core":        {"status": "ACTIVE",      "desc": "Core launcher and CLI"},
    "voice":       {"status": "NOT STARTED", "desc": "Voice recognition & wake-word"},
    "speaker":     {"status": "NOT STARTED", "desc": "Speaker identification"},
    "vision":      {"status": "NOT STARTED", "desc": "Screen capture & OCR"},
    "orchestrator":{"status": "NOT STARTED", "desc": "Multi-model routing"},
    "llm":         {"status": "NOT STARTED", "desc": "Local LLM inference"},
    "memory":      {"status": "NOT STARTED", "desc": "Short & long-term memory"},
    "research":    {"status": "NOT STARTED", "desc": "Self-learning & research"},
    "cyber":       {"status": "ACTIVE",      "desc": "Cybersecurity tools (scanner, network intel, log analyzer, toolkit)"},
    "coding":      {"status": "NOT STARTED", "desc": "Coding assistant"},
    "security":    {"status": "NOT STARTED", "desc": "Permissions & auth"},
}

STATUS_COLORS = {
    "ACTIVE":       Color.BRIGHT_GREEN,
    "IN PROGRESS":  Color.BRIGHT_YELLOW,
    "NOT STARTED":  Color.DIM,
}


def cmd_help():
    print(HELP_TEXT)


def cmd_status():
    Printer.blank()
    Printer.system("NEXUS System Status")
    Printer.divider()
    Printer.info(f"Version  : {VERSION}")
    Printer.info(f"Phase    : {PHASE}")
    Printer.info(f"Uptime   : {_uptime()}")
    Printer.info(f"Python   : {platform.python_version()}")
    Printer.info(f"Platform : {platform.system()} {platform.release()} ({platform.machine()})")
    Printer.info(f"CWD      : {os.getcwd()}")
    Printer.blank()


def cmd_version():
    print(
        f"\n  {Color.BRIGHT_GREEN}NEXUS{Color.RESET} "
        f"{Color.BRIGHT_WHITE}{VERSION}{Color.RESET}  "
        f"{Color.DIM}({PHASE}){Color.RESET}\n"
    )


def cmd_sysinfo():
    Printer.blank()
    Printer.system("Host System Information")
    Printer.divider()
    info = {
        "OS":           platform.system(),
        "OS Release":   platform.release(),
        "OS Version":   platform.version(),
        "Machine":      platform.machine(),
        "Processor":    platform.processor() or "unknown",
        "Python":       platform.python_version(),
        "Python Impl":  platform.python_implementation(),
        "Node":         platform.node(),
    }
    for k, v in info.items():
        print(f"  {Color.BRIGHT_CYAN}{k:<16}{Color.RESET} {v}")
    Printer.blank()


def cmd_modules():
    Printer.blank()
    Printer.system("NEXUS Module Registry")
    Printer.divider()
    for name, meta in MODULE_REGISTRY.items():
        color = STATUS_COLORS.get(meta["status"], Color.DIM)
        print(
            f"  {Color.BRIGHT_WHITE}{name:<16}{Color.RESET}"
            f"{color}{meta['status']:<14}{Color.RESET}"
            f"{Color.DIM}{meta['desc']}{Color.RESET}"
        )
    Printer.blank()


def cmd_run(args: list[str]):
    if not args:
        Printer.error("Usage: run <module>")
        return
    module = args[0].lower()
    if module not in MODULE_REGISTRY:
        Printer.error(f"Unknown module '{module}'. Use 'modules' to see available modules.")
        return
    meta = MODULE_REGISTRY[module]
    if meta["status"] == "NOT STARTED":
        Printer.warn(f"Module '{module}' has not been implemented yet. Status: NOT STARTED")
    elif meta["status"] == "ACTIVE":
        Printer.ok(f"Module '{module}' is already running as part of the core.")
    else:
        Printer.info(f"Module '{module}' status: {meta['status']}")


def cmd_shell(args: list[str]):
    if not args:
        Printer.error("Usage: shell <command>")
        return
    command = " ".join(args)
    Printer.warn(f"Executing shell command: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"{Color.BRIGHT_RED}{result.stderr}{Color.RESET}")
        if result.returncode != 0:
            Printer.error(f"Command exited with code {result.returncode}")
        else:
            Printer.ok("Done.")
    except subprocess.TimeoutExpired:
        Printer.error("Command timed out after 30 seconds.")
    except Exception as exc:
        Printer.error(f"Shell error: {exc}")


def cmd_history(history: list[str]):
    if not history:
        Printer.info("No commands in session history yet.")
        return
    Printer.blank()
    for i, entry in enumerate(history, 1):
        print(f"  {Color.DIM}{i:>3}.{Color.RESET}  {entry}")
    Printer.blank()


# ─────────────────────────────────────────────────────────────
#  COMMAND PARSER
# ─────────────────────────────────────────────────────────────

def parse_and_dispatch(raw: str, history: list[str], voice_engine=None, brain=None, speaker=None, is_voice: bool = False) -> bool:
    """
    Parse a raw input string, dispatch to the correct handler.
    Returns False when the user wants to exit, True otherwise.
    """
    raw = raw.strip()
    if not raw:
        return True

    # Record in session history
    history.append(raw)

    parts  = raw.split()
    cmd    = parts[0].lower()
    args   = parts[1:]

    dispatch = {
        "help":    lambda: cmd_help(),
        "?":       lambda: cmd_help(),
        "status":  lambda: cmd_status(),
        "version": lambda: cmd_version(),
        "sysinfo": lambda: cmd_sysinfo(),
        "modules": lambda: cmd_modules(),
        "clear":   lambda: os.system("clear" if platform.system() != "Windows" else "cls"),
        "echo":    lambda: print(" ".join(args)),
        "run":     lambda: cmd_run(args),
        "voice":   lambda: voice_engine.trigger() if voice_engine else Printer.error("Voice Engine not initialized"),
        "shell":   lambda: cmd_shell(args),
        "history": lambda: cmd_history(history),
        "exit":    None,
        "quit":    None,
    }

    if cmd in ("exit", "quit"):
        Printer.blank()
        Printer.nexus("Shutting down. Stay sharp.")
        Printer.blank()
        return False

    if cmd in dispatch:
        try:
            dispatch[cmd]()
        except Exception as exc:
            Printer.error(f"Command '{cmd}' raised an error: {exc}")
    else:
        # All natural language routed through Brain (memory + reasoning + conversation)
        if brain:
            response = brain.think(raw)
            Printer.nexus(response)
            if speaker:
                speaker.say(response, block=is_voice)
        else:
            Printer.warn(
                f"Unknown command '{Color.BRIGHT_WHITE}{cmd}{Color.RESET}"
                f"{Color.BRIGHT_YELLOW}'. Type 'help' for available commands."
            )

    return True


# ─────────────────────────────────────────────────────────────
#  SESSION UTILITIES
# ─────────────────────────────────────────────────────────────

_START_TIME = time.time()


def _uptime() -> str:
    elapsed = int(time.time() - _START_TIME)
    h, rem  = divmod(elapsed, 3600)
    m, s    = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def prompt_string() -> str:
    now = datetime.datetime.now().strftime("%H:%M")
    return (
        f"{Color.DIM}{now}{Color.RESET} "
        f"{Color.BRIGHT_GREEN}NEXUS{Color.RESET}"
        f"{Color.DIM}>{Color.RESET} "
    )


# ─────────────────────────────────────────────────────────────
#  MAIN REPL LOOP
# ─────────────────────────────────────────────────────────────

from voice.engine import VoiceEngine
from core.brain import Brain
from voice.tts import Speaker

def main():
    setup_logging()
    print_banner()
    run_boot_sequence()

    Printer.info("Type 'help' for available commands. Type 'exit' to quit.")
    Printer.blank()

    session_history: list[str] = []

    brain    = Brain(user_name="Senanu")
    speaker  = Speaker()

    voice_engine = VoiceEngine(
        command_callback=lambda text:
            parse_and_dispatch(text, session_history, voice_engine, brain=brain, speaker=speaker, is_voice=True)
    )
    voice_engine.start()

    while True:
        try:
            raw = input(prompt_string())
        except (KeyboardInterrupt, EOFError):
            Printer.blank()
            Printer.nexus("Interrupt received. Shutting down.")
            Printer.blank()
            break

        alive = parse_and_dispatch(raw, session_history, voice_engine, brain=brain, speaker=speaker, is_voice=False)
        if not alive:
            break

    speaker.shutdown()
    voice_engine.stop()


if __name__ == "__main__":
    main()