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
        "info":     f"{Color.BRIGHT_CYAN}[INFO]{Color.RESET}",
        "ok":       f"{Color.BRIGHT_GREEN}[ OK ]{Color.RESET}",
        "warn":     f"{Color.BRIGHT_YELLOW}[WARN]{Color.RESET}",
        "error":    f"{Color.BRIGHT_RED}[FAIL]{Color.RESET}",
        "cmd":      f"{Color.BRIGHT_MAGENTA}[ >> ]{Color.RESET}",
        "system":   f"{Color.BRIGHT_BLUE}[ SYS]{Color.RESET}",
        "nexus":    f"{Color.BRIGHT_GREEN}[NEXUS]{Color.RESET}",
        "thinking": f"{Color.BRIGHT_YELLOW}[ ◆  ]{Color.RESET}",
        "reading":  f"{Color.BRIGHT_CYAN}[ ↳  ]{Color.RESET}",
        "editing":  f"{Color.BRIGHT_MAGENTA}[ ✎  ]{Color.RESET}",
        "learned":  f"{Color.BRIGHT_GREEN}[ ★  ]{Color.RESET}",
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

    @classmethod
    def thinking(cls, msg: str):
        print(f"{cls._ts()} {cls.PREFIX['thinking']} {Color.BRIGHT_YELLOW}{msg}{Color.RESET}")

    @classmethod
    def reading(cls, path: str, detail: str = ""):
        detail_part = f"  {Color.DIM}{detail}{Color.RESET}" if detail else ""
        print(f"{cls._ts()} {cls.PREFIX['reading']} {Color.BRIGHT_CYAN}{path}{Color.RESET}{detail_part}")

    @classmethod
    def editing(cls, path: str, detail: str = ""):
        detail_part = f"  {Color.DIM}{detail}{Color.RESET}" if detail else ""
        print(f"{cls._ts()} {cls.PREFIX['editing']} {Color.BRIGHT_MAGENTA}{path}{Color.RESET}{detail_part}")

    @classmethod
    def learned(cls, topic: str, detail: str = ""):
        detail_part = f"  {Color.DIM}{detail}{Color.RESET}" if detail else ""
        print(f"{cls._ts()} {cls.PREFIX['learned']} {Color.BRIGHT_GREEN}{topic}{Color.RESET}{detail_part}")

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
  {Color.BRIGHT_CYAN}knowledge{Color.RESET}              Show knowledge base stats and recent entries
  {Color.BRIGHT_CYAN}learn <topic>{Color.RESET}          Go online, learn about topic, store to KB
  {Color.BRIGHT_CYAN}plan <task>{Color.RESET}            Show a detailed code change plan for a task
  {Color.BRIGHT_CYAN}read <file>{Color.RESET}            Read and analyse a project file

{Color.BRIGHT_RED}{Color.BOLD}CYBER COMMANDS (just say them naturally){Color.RESET}
  {Color.BRIGHT_RED}latest cyber news{Color.RESET}       Headlines from THN, Krebs, SANS, CISA, Exploit-DB
  {Color.BRIGHT_RED}cve lookup CVE-2024-1234{Color.RESET}  NVD CVE details + CVSS score
  {Color.BRIGHT_RED}search cve apache{Color.RESET}       Search NVD for matching CVEs
  {Color.BRIGHT_RED}find exploit vsftpd{Color.RESET}     Exploit-DB / searchsploit search
  {Color.BRIGHT_RED}download exploit 47887{Color.RESET}  Download exploit to data/exploits/
  {Color.BRIGHT_RED}authorize example.com{Color.RESET}   Add domain to authorized recon scope
  {Color.BRIGHT_RED}recon on example.com{Color.RESET}    Full recon: DNS, IP, headers, subdomains
  {Color.BRIGHT_RED}find subdomains example.com{Color.RESET}  Passive + active subdomain enum
  {Color.BRIGHT_RED}vuln scan 192.168.1.1{Color.RESET}   Nuclei + Nikto + nmap vuln scripts
  {Color.BRIGHT_RED}sandbox 192.168.1.1{Color.RESET}     Clone services → isolated vuln test
  {Color.BRIGHT_RED}monitor target 192.168.1.1{Color.RESET}  Watch for port/service changes
  {Color.BRIGHT_RED}cyber help{Color.RESET}              Full cyber command reference

{Color.BRIGHT_MAGENTA}{Color.BOLD}SKILL & TASK COMMANDS{Color.RESET}
  {Color.BRIGHT_MAGENTA}skills{Color.RESET}                 List all registered skills (builtin + acquired + created)
  {Color.BRIGHT_MAGENTA}skills search <q>{Color.RESET}      Find skills matching a query
  {Color.BRIGHT_MAGENTA}skills acquire <url>{Color.RESET}   Clone a GitHub repo → extract + register skills
  {Color.BRIGHT_MAGENTA}skills create <desc>{Color.RESET}   Ask LLM to write and register a new skill
  {Color.BRIGHT_MAGENTA}task <description>{Color.RESET}     Run the task planner on any arbitrary task

  {Color.BRIGHT_MAGENTA}models{Color.RESET}                 Show device capability + which models fit / are installed
  {Color.BRIGHT_MAGENTA}models recommend <task>{Color.RESET}  Best chat/code/reasoning/vision model for this device
  {Color.BRIGHT_MAGENTA}models get <name>{Color.RESET}      Download a model (asks for confirmation first)
  {Color.BRIGHT_MAGENTA}ask <question>{Color.RESET}         Answer via tiered router (reflex → local → cloud)
  {Color.BRIGHT_MAGENTA}router{Color.RESET}                 Show router backends + cloud policy
  {Color.BRIGHT_MAGENTA}research <question>{Color.RESET}    Autonomous web research → cited answer
  {Color.BRIGHT_MAGENTA}look [image]{Color.RESET}           Describe the camera view / an image (vision model)
  {Color.BRIGHT_MAGENTA}place where [image]{Color.RESET}    Recognize the current place
  {Color.BRIGHT_MAGENTA}place enroll <name> <imgs>{Color.RESET}  Teach NEXUS a place from photos
  {Color.BRIGHT_MAGENTA}awareness [start|stop]{Color.RESET} Live world-state + proactive alerts (fusion loop)

  {Color.DIM}Natural language equivalents (just say them):{Color.RESET}
  {Color.BRIGHT_MAGENTA}acquire skill from https://github.com/...{Color.RESET}
  {Color.BRIGHT_MAGENTA}create a skill to send emails via Gmail{Color.RESET}
  {Color.BRIGHT_MAGENTA}list my skills{Color.RESET}
  {Color.BRIGHT_MAGENTA}search skills for email{Color.RESET}

{Color.BRIGHT_BLUE}{Color.BOLD}PREDICTION & MARKET ANALYSIS{Color.RESET}
  {Color.BRIGHT_BLUE}predict <question>{Color.RESET}          Make AI predictions on any event
  {Color.BRIGHT_BLUE}predict crypto bitcoin{Color.RESET}      Predict cryptocurrency price movements
  {Color.BRIGHT_BLUE}market analysis crypto{Color.RESET}      Get comprehensive market analysis
  {Color.BRIGHT_BLUE}market summary{Color.RESET}              Current market overview and trends
  {Color.BRIGHT_BLUE}crypto summary{Color.RESET}              Cryptocurrency market data and sentiment
  {Color.BRIGHT_BLUE}prediction performance{Color.RESET}      View prediction accuracy statistics
  {Color.BRIGHT_BLUE}prediction markets{Color.RESET}          Browse active prediction markets

  {Color.DIM}Natural language examples (just say them):{Color.RESET}
  {Color.BRIGHT_BLUE}Will Bitcoin reach $100k by end of 2025?{Color.RESET}
  {Color.BRIGHT_BLUE}predict ethereum price movement{Color.RESET}
  {Color.BRIGHT_BLUE}analyze the crypto market{Color.RESET}
  {Color.BRIGHT_BLUE}forecast Tesla stock performance{Color.RESET}

{Color.DIM}Say anything else and NEXUS routes it through brain → LLM → web research.{Color.RESET}
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
    "cyber":       {"status": "ACTIVE",      "desc": "Scanning, recon, vuln scan, sandbox, threat intel, CVE/exploit search, news"},
    "coding":      {"status": "ACTIVE",      "desc": "Code engine — narrated planning, file analysis, self-learning"},
    "skills":      {"status": "ACTIVE",      "desc": "Skill registry, acquirer, task planner — learn/create/invoke skills"},
    "prediction":  {"status": "ACTIVE",      "desc": "AI prediction engine — market forecasting, event prediction, trading analysis"},
    "security":    {"status": "NOT STARTED", "desc": "Permissions & auth"},
}

STATUS_COLORS = {
    "ACTIVE":       Color.BRIGHT_GREEN,
    "IN PROGRESS":  Color.BRIGHT_YELLOW,
    "NOT STARTED":  Color.DIM,
}


def cmd_help():
    print(HELP_TEXT)


def cmd_status(brain=None):
    Printer.blank()
    Printer.system("NEXUS System Status")
    Printer.divider()
    Printer.info(f"Version  : {VERSION}")
    Printer.info(f"Phase    : {PHASE}")
    Printer.info(f"Uptime   : {_uptime()}")
    Printer.info(f"Python   : {platform.python_version()}")
    Printer.info(f"Platform : {platform.system()} {platform.release()} ({platform.machine()})")
    Printer.info(f"CWD      : {os.getcwd()}")
    # Live subsystem health (reflects what actually initialized, not a static list)
    if brain is not None and hasattr(brain, "status_report"):
        Printer.blank()
        print(brain.status_report())
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

    # Use retry system for robust shell execution
    try:
        from core.retry_system import execute_shell_with_retry

        result = execute_shell_with_retry(
            command=command,
            max_attempts=5,
            timeout=30.0
        )

        if result.success:
            if result.final_stdout:
                print(result.final_stdout)
            if result.final_stderr:
                print(f"{Color.BRIGHT_RED}{result.final_stderr}{Color.RESET}")

            successful_attempt = result.successful_attempt
            if successful_attempt and successful_attempt.attempt_number > 1:
                Printer.ok(f"Done after {result.attempt_count} attempt(s) using {successful_attempt.strategy.value} strategy.")
            else:
                Printer.ok("Done.")
        else:
            Printer.error(f"Command failed after {result.attempt_count} attempts: {result.failure_reason}")

            # Show brief retry summary
            if result.attempts:
                Printer.warn(f"Tried strategies: {', '.join(a.strategy.value for a in result.attempts)}")

    except ImportError:
        # Fallback to original implementation if retry system is not available
        Printer.warn("Retry system not available, using simple execution")
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
    except Exception as exc:
        Printer.error(f"Retry system error: {exc}")


def cmd_knowledge(brain=None):
    """Show knowledge base stats and recent entries."""
    try:
        from core.knowledge import get_knowledge_base
        kb = get_knowledge_base()
        stats = kb.stats()
        Printer.blank()
        Printer.system(f"Knowledge Base — {stats['total']} entries")
        Printer.divider()
        for source, count in stats.get("by_source", {}).items():
            print(f"  {Color.BRIGHT_CYAN}{source:<20}{Color.RESET} {count} entries")
        Printer.blank()
        recent = kb.get_recent(5)
        if recent:
            Printer.info("Recent entries:")
            for entry in recent:
                topic = entry["topic"][:50]
                date  = entry["created"][:10]
                print(f"  {Color.DIM}{date}{Color.RESET}  {Color.BRIGHT_WHITE}{topic}{Color.RESET}")
        Printer.blank()
    except Exception as exc:
        Printer.error(f"Knowledge base error: {exc}")


def cmd_learn(args: list[str], brain=None):
    """Go online, learn about a topic, store to knowledge base."""
    if not args:
        Printer.error("Usage: learn <topic>")
        return
    topic = " ".join(args)
    if brain and brain._code_engine:
        Printer.info(f"Going online to learn about: {topic}")
        result = brain._code_engine.search_and_learn(topic)
        if result:
            Printer.nexus(result[:500] + ("..." if len(result) > 500 else ""))
        else:
            Printer.warn("Could not retrieve information online.")
    else:
        Printer.warn("Code engine or researcher not available.")


def cmd_plan(args: list[str], brain=None):
    """Show a detailed narrated plan for a coding task."""
    if not args:
        Printer.error("Usage: plan <task description>")
        return
    task = " ".join(args)
    if brain and brain._code_engine:
        result = brain._code_engine.work(task)
        Printer.blank()
        Printer.nexus(result)
    else:
        Printer.warn("Code engine not available.")


def cmd_read_file(args: list[str], brain=None):
    """Read and analyse a project file."""
    if not args:
        Printer.error("Usage: read <file>")
        return
    path = args[0]
    if brain and brain._code_engine:
        content = brain._code_engine.read_file(path)
        if content:
            Printer.blank()
            print(content)
            Printer.blank()
            info = brain._code_engine.analyze_file(path)
            Printer.info(info.describe())
    else:
        # Fallback: just cat the file
        cmd_shell(["cat", path])


def cmd_skills(args: list[str], brain=None):
    """List, search, acquire, or create skills."""
    if not args:
        # Default: list all
        if brain and brain._skill_registry:
            print(brain._cmd_list_skills())
        else:
            Printer.warn("Skill registry not available.")
        return

    sub = args[0].lower()

    if sub in ("list", "all"):
        if brain:
            print(brain._cmd_list_skills())

    elif sub == "search" and len(args) > 1:
        query = " ".join(args[1:])
        if brain:
            print(brain._cmd_search_skills(query))

    elif sub in ("acquire", "from", "learn") and len(args) > 1:
        url = args[1]
        if brain:
            result = brain._cmd_acquire_skill(f"acquire skill from {url}")
            Printer.nexus(result)
        else:
            Printer.warn("Brain not available.")

    elif sub == "create" and len(args) > 1:
        desc = " ".join(args[1:])
        if brain:
            result = brain._cmd_create_skill(f"create a skill to {desc}")
            Printer.nexus(result)
        else:
            Printer.warn("Brain not available.")

    else:
        Printer.info("Skills sub-commands:")
        print("  skills list             — list all skills")
        print("  skills search <query>   — find matching skills")
        print("  skills acquire <url>    — clone a GitHub repo and extract skills")
        print("  skills create <desc>    — create a new skill with LLM")


def cmd_task(args: list[str], brain=None):
    """Run the task planner on an arbitrary task."""
    if not args:
        Printer.error("Usage: task <description>")
        return
    task = " ".join(args)
    if brain and brain._task_planner:
        result = brain._task_planner.plan_and_execute(task)
        Printer.blank()
        Printer.nexus(result)
    else:
        Printer.warn("Task planner not available.")


def cmd_history(history: list[str]):
    if not history:
        Printer.info("No commands in session history yet.")
        return
    Printer.blank()
    for i, entry in enumerate(history, 1):
        print(f"  {Color.DIM}{i:>3}.{Color.RESET}  {entry}")
    Printer.blank()


def cmd_models(args: list[str]):
    """Inspect device capability and manage local models (consent-based)."""
    try:
        from core.model_manager import get_model_manager
    except Exception as e:
        Printer.error(f"Model manager unavailable: {e}")
        return

    mgr = get_model_manager()
    sub = args[0].lower() if args else "list"

    if sub in ("list", "status"):
        Printer.blank()
        print(mgr.report())
        Printer.blank()
        Printer.info("Try: models get <name> · models recommend <chat|code|reasoning|vision>")
        return

    if sub == "available":
        Printer.blank()
        for s in mgr.available_for_device():
            print(f"  {s.name:<18} {s.size_gb:>4}GB  [{'/'.join(s.tags)}] — {s.description}")
        Printer.blank()
        return

    if sub == "recommend":
        task = args[1] if len(args) > 1 else "reasoning"
        rec = mgr.recommend(task)
        if not rec:
            Printer.warn(f"No '{task}' model fits this device. Try a smaller task or upgrade hardware.")
            return
        status = "installed" if mgr.is_installed(rec.name) else "available to download"
        Printer.nexus(f"Best '{task}' model for this device: {rec.name} "
                      f"(~{rec.size_gb} GB) — {status}.")
        return

    if sub == "get":
        if len(args) < 2:
            Printer.error("Usage: models get <name>")
            return
        name = args[1]

        def _confirm(msg: str) -> bool:
            try:
                return input(f"  {msg} [y/N] ").strip().lower() in ("y", "yes")
            except (EOFError, KeyboardInterrupt):
                return False

        Printer.info(f"Preparing to install {name}…")
        res = mgr.ensure(name, confirm=_confirm,
                         on_progress=lambda line: print(f"    {line}", end="\r"))
        print()
        (Printer.nexus if res.ok else Printer.warn)(res.message)
        return

    Printer.error("Usage: models [list|available|recommend <task>|get <name>]")


def cmd_router(brain=None):
    """Show the tiered brain router's backends and cloud policy."""
    if not brain or not getattr(brain, "router", None):
        Printer.warn("Brain router not available (is the brain loaded?).")
        return
    Printer.blank()
    print(brain.router.describe())
    Printer.blank()


def cmd_ask(args: list[str], brain=None):
    """Answer a question through the tiered router (reflex → local → cloud)."""
    if not args:
        Printer.error("Usage: ask <question>")
        return
    if not brain or not getattr(brain, "router", None):
        Printer.warn("Brain router not available.")
        return
    question = " ".join(args)

    def _confirm(msg: str) -> bool:
        try:
            return input(f"  {msg} [y/N] ").strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    res = brain.ask_tiered(question, confirm=_confirm)
    Printer.blank()
    Printer.nexus(res.text)
    if res.tier_used is not None:
        note = f"answered by {res.backend_name}"
        if res.escalated:
            note += f" (escalated through {' → '.join(res.tiers_tried)})"
        Printer.info(note)


def cmd_research(args: list[str], brain=None):
    """Autonomous web research: search → read → refine → cited answer."""
    if not args:
        Printer.error("Usage: research <question>")
        return
    if not brain or not getattr(brain, "web_agent", None):
        Printer.warn("Web-research agent not available "
                     "(needs the research modules + internet).")
        return
    question = " ".join(args)
    Printer.info("Researching the web…")
    res = brain.deep_research(
        question, on_progress=lambda msg: print(f"    {Color.DIM}{msg}{Color.RESET}"))
    Printer.blank()
    Printer.nexus(res.answer)
    if res.sources:
        Printer.blank()
        print(f"  {Color.DIM}Sources ({len(res.sources)}) · "
              f"{res.iterations} search round(s):{Color.RESET}")
        for i, s in enumerate(res.sources, 1):
            print(f"    [{i}] {s.title or s.url}\n        {Color.DIM}{s.url}{Color.RESET}")


def cmd_look(args: list[str], brain=None):
    """Describe the camera view (or an image file) via the vision model."""
    if not brain or not getattr(brain, "scene_describer", None):
        Printer.warn("Scene description not available.")
        return
    image = args[0] if args else None

    def _confirm(msg: str) -> bool:
        try:
            return input(f"  {msg} [y/N] ").strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    Printer.info("Looking…")
    res = brain.describe_scene(image, confirm=_confirm)
    Printer.blank()
    (Printer.nexus if res.ok else Printer.warn)(res.text)
    if res.ok and res.model:
        Printer.info(f"(via {res.model})")


def cmd_place(args: list[str], brain=None):
    """Place recognition: enroll / list / forget / identify the current place."""
    if not brain or not getattr(brain, "place_recognizer", None):
        Printer.warn("Place recognition not available.")
        return
    sub = args[0].lower() if args else "where"
    pr = brain.place_recognizer

    if sub in ("list", "places"):
        names = pr.store.names()
        Printer.nexus("Enrolled places: " + (", ".join(names) if names else "none yet"))
        return

    if sub == "enroll":
        if len(args) < 3:
            Printer.error("Usage: place enroll <name> <image1> [image2 ...]")
            return
        name, images = args[1], args[2:]
        try:
            n = brain.enroll_place(name, images)
            Printer.nexus(f"Enrolled '{name}' from {n} image(s).")
        except Exception as e:
            Printer.error(f"Enrollment failed: {e}")
        return

    if sub == "forget":
        if len(args) < 2:
            Printer.error("Usage: place forget <name>")
            return
        Printer.nexus("Removed." if pr.store.remove(args[1]) else "No such place.")
        return

    if sub == "where":
        image = args[1] if len(args) > 1 else None
        res = brain.where_am_i(image)
        if res is None:
            Printer.warn("No camera and no image given. Try: place where <image>")
        elif res.known:
            Printer.nexus(f"You're in the {res.place} (confidence {res.score:.2f}).")
        else:
            Printer.nexus("I don't recognize this place. Enroll it: "
                          "place enroll <name> <image>")
        return

    Printer.error("Usage: place [where <img>|enroll <name> <imgs>|list|forget <name>]")


def cmd_awareness(args: list[str], brain=None):
    """Show or control the always-on fusion loop (live world-state + proactivity)."""
    if not brain or not getattr(brain, "fusion", None):
        Printer.warn("Fusion loop not available.")
        return
    fusion = brain.fusion
    sub = args[0].lower() if args else "status"

    if sub == "start":
        fusion.start()
        Printer.nexus(f"Awareness on — fusing {len(fusion.sensors)} sensor(s). "
                      "I'll speak up when something changes.")
        return
    if sub == "stop":
        fusion.stop()
        Printer.nexus("Awareness off.")
        return
    if sub == "tick":
        # One manual fusion step — useful without a live camera/thread.
        msgs = fusion.tick()
        for m in msgs:
            Printer.nexus(m)
        if not msgs:
            Printer.info("(tick: nothing new)")

    # status (default) — show the current world state.
    Printer.blank()
    print(fusion.state.summary())
    Printer.info(f"loop: {'running' if fusion.running else 'stopped'} · "
                 f"{len(fusion.sensors)} sensor(s)")
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
        "help":      lambda: cmd_help(),
        "?":         lambda: cmd_help(),
        "status":    lambda: cmd_status(brain=brain),
        "version":   lambda: cmd_version(),
        "sysinfo":   lambda: cmd_sysinfo(),
        "modules":   lambda: cmd_modules(),
        "clear":     lambda: os.system("clear" if platform.system() != "Windows" else "cls"),
        "echo":      lambda: print(" ".join(args)),
        "run":       lambda: cmd_run(args),
        "voice":     lambda: voice_engine.trigger() if voice_engine else Printer.error("Voice Engine not initialized"),
        "shell":     lambda: cmd_shell(args),
        "history":   lambda: cmd_history(history),
        "knowledge": lambda: cmd_knowledge(brain=brain),
        "learn":     lambda: cmd_learn(args, brain=brain),
        "plan":      lambda: cmd_plan(args, brain=brain),
        "read":      lambda: cmd_read_file(args, brain=brain),
        "skills":    lambda: cmd_skills(args, brain=brain),
        "skill":     lambda: cmd_skills(args, brain=brain),
        "task":      lambda: cmd_task(args, brain=brain),
        "models":    lambda: cmd_models(args),
        "model":     lambda: cmd_models(args),
        "router":    lambda: cmd_router(brain=brain),
        "ask":       lambda: cmd_ask(args, brain=brain),
        "research":  lambda: cmd_research(args, brain=brain),
        "websearch": lambda: cmd_research(args, brain=brain),
        "look":      lambda: cmd_look(args, brain=brain),
        "see":       lambda: cmd_look(args, brain=brain),
        "place":     lambda: cmd_place(args, brain=brain),
        "places":    lambda: cmd_place(["list"], brain=brain),
        "awareness": lambda: cmd_awareness(args, brain=brain),
        "worldstate":lambda: cmd_awareness(["status"], brain=brain),
        "exit":      None,
        "quit":      None,
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

    voice_engine = None
    try:
        voice_engine = VoiceEngine(
            command_callback=lambda text:
                parse_and_dispatch(text, session_history, voice_engine, brain=brain, speaker=speaker, is_voice=True)
        )
        voice_engine.start()
    except Exception as e:
        Printer.warn(f"Voice engine unavailable ({e}) — running in text-only mode.")

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
    if voice_engine is not None:
        voice_engine.stop()


if __name__ == "__main__":
    main()