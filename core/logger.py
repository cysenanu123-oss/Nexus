"""
NEXUS — core/logger.py
Centralized logging system for NEXUS.

Features:
  - Color-coded terminal output (DEBUG/INFO/WARN/ERROR/CRITICAL)
  - Rotating file logs (max size + backup count from config)
  - Single call to setup_logging() wires everything up
  - Every NEXUS module just does: log = logging.getLogger("nexus.module")

Usage:
    # In main.py — call once at startup:
    from core.logger import setup_logging
    setup_logging()

    # In any module:
    import logging
    log = logging.getLogger("nexus.voice.listener")
    log.info("Microphone opened.")
    log.warning("Overflow detected.")
    log.error("Stream failed: %s", exc)
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import datetime
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────
#  ANSI COLOR MAP  (matches main.py Color class palette)
# ─────────────────────────────────────────────────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"

_LEVEL_COLORS = {
    "DEBUG":    "\033[36m",        # cyan
    "INFO":     "\033[92m",        # bright green
    "WARNING":  "\033[93m",        # bright yellow
    "ERROR":    "\033[91m",        # bright red
    "CRITICAL": "\033[1m\033[91m", # bold bright red
}

_LEVEL_LABELS = {
    "DEBUG":    "DEBUG",
    "INFO":     " INFO",
    "WARNING":  " WARN",
    "ERROR":    " FAIL",
    "CRITICAL": " CRIT",
}

# Logger name → short tag shown in terminal
_NAME_TAGS: dict[str, str] = {
    "nexus":                "NEXUS",
    "nexus.brain":          "BRAIN",
    "nexus.voice":          "VOICE",
    "nexus.voice.listener": "  MIC",
    "nexus.voice.wakeword": " WAKE",
    "nexus.voice.stt":      "  STT",
    "nexus.voice.tts":      "  TTS",
    "nexus.voice.engine":   "  ENG",
    "nexus.intent":         " INTNT",
    "nexus.dispatcher":     " DISP",
    "nexus.reasoning":      "  RSN",
    "nexus.planner":        " PLAN",
    "nexus.conversation":   " CONV",
    "nexus.memory":         "  MEM",
    "nexus.config":         "  CFG",
    "nexus.plugins":        " PLUG",
    "nexus.cyber":          "CYBER",
    "nexus.vision":         "  VIS",
}


def _get_tag(name: str) -> str:
    """Return the short tag for a logger name, falling back to the last segment."""
    if name in _NAME_TAGS:
        return _NAME_TAGS[name]
    # Walk up the hierarchy
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in _NAME_TAGS:
            return _NAME_TAGS[candidate]
    # Fall back to last segment, max 6 chars
    return parts[-1].upper()[:6].rjust(6)


# ─────────────────────────────────────────────────────────────
#  COLOR TERMINAL FORMATTER
# ─────────────────────────────────────────────────────────────

class NexusTerminalFormatter(logging.Formatter):
    """
    Colorized single-line formatter for terminal output.

    Format:
        HH:MM:SS  [LEVEL]  TAG  — message
    """

    _colors_enabled: bool = True

    def __init__(self, colors: bool = True):
        super().__init__()
        self._colors_enabled = colors

    def format(self, record: logging.LogRecord) -> str:
        ts    = datetime.datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level = record.levelname
        tag   = _get_tag(record.name)
        msg   = record.getMessage()

        # Append exception info if present
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)

        if self._colors_enabled:
            color  = _LEVEL_COLORS.get(level, "")
            label  = _LEVEL_LABELS.get(level, level[:5].rjust(5))
            return (
                f"{_DIM}{ts}{_RESET} "
                f"{color}[{label}]{_RESET} "
                f"{_DIM}{tag:>6}{_RESET} "
                f"{_DIM}—{_RESET} "
                f"{msg}"
            )
        else:
            label = _LEVEL_LABELS.get(level, level[:5].rjust(5))
            return f"{ts} [{label}] {tag:>6} — {msg}"


# ─────────────────────────────────────────────────────────────
#  FILE FORMATTER  (plain text, no ANSI)
# ─────────────────────────────────────────────────────────────

class NexusFileFormatter(logging.Formatter):
    """
    Plain-text formatter for log files.

    Format:
        2026-05-09 18:27:14  INFO    nexus.voice.listener — Microphone opened.
    """

    def format(self, record: logging.LogRecord) -> str:
        ts    = datetime.datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname
        name  = record.name
        msg   = record.getMessage()

        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)

        return f"{ts}  {level:<8} {name} — {msg}"


# ─────────────────────────────────────────────────────────────
#  SETUP FUNCTION  — call once in main.py
# ─────────────────────────────────────────────────────────────

def setup_logging(
    log_level:        Optional[str]  = None,
    log_to_file:      Optional[bool] = None,
    log_dir:          Optional[str]  = None,
    max_bytes:        Optional[int]  = None,
    backup_count:     Optional[int]  = None,
    suppress_noisy:   bool           = True,
) -> logging.Logger:
    """
    Configure the NEXUS logging system.

    Reads defaults from core/config.py (cfg) if available.
    All parameters are optional overrides.

    Parameters
    ----------
    log_level      — "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL"
    log_to_file    — write rotating log file to logs/nexus.log
    log_dir        — directory for log files (default: logs/)
    max_bytes      — max size per log file in bytes
    backup_count   — number of rotated backups to keep
    suppress_noisy — silence chatty third-party loggers (httpx, faster_whisper, etc.)

    Returns
    -------
    The root nexus logger.
    """

    # ── Load defaults from config ──────────────────────────────
    cfg_level        = "INFO"
    cfg_to_file      = True
    cfg_max_mb       = 10
    cfg_backup_count = 5

    try:
        from core.config import cfg
        cfg_level        = cfg.get("system.log_level",        "INFO")
        cfg_to_file      = cfg.get("system.log_to_file",      True)
        cfg_max_mb       = cfg.get("system.log_max_size_mb",  10)
        cfg_backup_count = cfg.get("system.log_backup_count", 5)
    except Exception:
        pass   # config not available yet — use hardcoded defaults

    # Apply parameter overrides
    level        = (log_level    or cfg_level).upper()
    to_file      = log_to_file   if log_to_file   is not None else cfg_to_file
    max_b        = max_bytes     or (cfg_max_mb * 1024 * 1024)
    n_backups    = backup_count  if backup_count  is not None else cfg_backup_count

    numeric_level = getattr(logging, level, logging.INFO)

    # ── Root NEXUS logger ──────────────────────────────────────
    root = logging.getLogger("nexus")
    root.setLevel(logging.DEBUG)   # capture everything; handlers filter
    root.handlers.clear()          # avoid duplicate handlers on re-init
    root.propagate = False         # don't bubble up to Python root logger

    # ── Terminal handler ───────────────────────────────────────
    colors = _supports_color()
    term_handler = logging.StreamHandler(sys.stderr)
    term_handler.setLevel(numeric_level)
    term_handler.setFormatter(NexusTerminalFormatter(colors=colors))
    root.addHandler(term_handler)

    # ── File handler (rotating) ────────────────────────────────
    if to_file:
        _log_dir = Path(log_dir) if log_dir else _resolve_log_dir()
        _log_dir.mkdir(parents=True, exist_ok=True)
        log_path = _log_dir / "nexus.log"

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=max_b,
            backupCount=n_backups,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)   # always write DEBUG to file
        file_handler.setFormatter(NexusFileFormatter())
        root.addHandler(file_handler)

        root.debug("Log file: %s  (max=%dMB, backups=%d)", log_path, max_b // 1024**2, n_backups)

    # ── Suppress noisy third-party loggers ────────────────────
    if suppress_noisy:
        _quiet = [
            "httpx", "httpcore", "urllib3", "requests",
            "faster_whisper", "ctranslate2",
            "onnxruntime", "openwakeword",
            "speechbrain", "torch", "tensorflow",
        ]
        for name in _quiet:
            logging.getLogger(name).setLevel(logging.WARNING)

    root.info(
        "Logging initialized — level=%s  file=%s  colors=%s",
        level, to_file, colors,
    )

    return root


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def _supports_color() -> bool:
    """True if the terminal likely supports ANSI color codes."""
    if not hasattr(sys.stderr, "isatty") or not sys.stderr.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return True


def _resolve_log_dir() -> Path:
    """
    Find the logs/ directory relative to the project root.
    Works whether called from main.py or from a subdirectory.
    """
    # Walk up from this file to find the project root (contains main.py)
    here = Path(__file__).resolve().parent
    for candidate in [here.parent, here.parent.parent, Path.cwd()]:
        if (candidate / "main.py").exists():
            return candidate / "logs"
    return Path("logs")


def get_logger(name: str) -> logging.Logger:
    """
    Convenience wrapper — ensures the name is always prefixed with 'nexus.'.

    Usage:
        from core.logger import get_logger
        log = get_logger("voice.listener")   # → logging.getLogger("nexus.voice.listener")
    """
    if not name.startswith("nexus"):
        name = f"nexus.{name}"
    return logging.getLogger(name)


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST — python core/logger.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  NEXUS — core/logger.py test\n")

    log = setup_logging(log_level="DEBUG", log_to_file=True)

    log.debug("This is a DEBUG message")
    log.info("This is an INFO message")
    log.warning("This is a WARNING message")
    log.error("This is an ERROR message")
    log.critical("This is a CRITICAL message")

    # Test child loggers
    logging.getLogger("nexus.voice.listener").info("Microphone opened at 44100 Hz")
    logging.getLogger("nexus.brain").info("Brain initialized")
    logging.getLogger("nexus.voice.stt").warning("Low confidence transcription: 0.31")
    logging.getLogger("nexus.dispatcher").error("Application 'burpsuite' not found on PATH")

    print("\n  Check logs/nexus.log for the file output.\n")