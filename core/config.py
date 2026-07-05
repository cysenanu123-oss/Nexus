"""
NEXUS — core/config.py
Configuration loader, validator, and live-access system.

Usage anywhere in NEXUS:
    from core.config import cfg
    name = cfg.get("identity.assistant_name")   # "NEXUS"
    cfg.set("system.debug_mode", False)
    cfg.save()
"""

import json
import os
import copy
import datetime
from pathlib import Path
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────
#  SCHEMA — defines every key, its type, and constraints
#  If a loaded value fails its check, the default is used.
# ─────────────────────────────────────────────────────────────

SCHEMA: dict[str, dict] = {
    # identity
    "identity.assistant_name":       {"type": str,   "default": "NEXUS"},
    "identity.owner_name":           {"type": str,   "default": "User"},
    "identity.wake_word":            {"type": str,   "default": "hey nexus"},
    "identity.version":              {"type": str,   "default": "0.1.0-alpha"},
    "identity.personality":          {"type": str,   "default": "precise",
                                      "choices": ["precise", "friendly", "terse"]},

    # system
    "system.debug_mode":             {"type": bool,  "default": True},
    "system.log_level":              {"type": str,   "default": "INFO",
                                      "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
    "system.log_to_file":            {"type": bool,  "default": True},
    "system.log_max_size_mb":        {"type": int,   "default": 10,    "min": 1,  "max": 500},
    "system.log_backup_count":       {"type": int,   "default": 5,     "min": 1,  "max": 20},
    "system.startup_animation":      {"type": bool,  "default": True},
    "system.clear_on_start":         {"type": bool,  "default": True},
    "system.command_history_limit":  {"type": int,   "default": 500,   "min": 10, "max": 10000},

    # voice
    "voice.enabled":                 {"type": bool,  "default": False},
    "voice.engine":                  {"type": str,   "default": "faster-whisper",
                                      "choices": ["faster-whisper", "vosk", "whisper"]},
    "voice.model_size":              {"type": str,   "default": "base",
                                      "choices": ["tiny", "base", "small", "medium", "large"]},
    "voice.language":                {"type": str,   "default": "en"},
    "voice.sample_rate":             {"type": int,   "default": 16000, "min": 8000, "max": 48000},
    "voice.noise_reduction":         {"type": bool,  "default": True},
    "voice.wake_word_sensitivity":   {"type": float, "default": 0.7,   "min": 0.0, "max": 1.0},
    "voice.silence_timeout_sec":     {"type": float, "default": 2.0,   "min": 0.5, "max": 10.0},
    "voice.max_listen_sec":          {"type": int,   "default": 15,    "min": 3,   "max": 120},

    # speaker_id
    "speaker_id.enabled":            {"type": bool,  "default": False},
    "speaker_id.confidence_threshold":{"type": float,"default": 0.85,  "min": 0.0, "max": 1.0},

    # vision
    "vision.enabled":                {"type": bool,  "default": False},
    "vision.screen_capture":         {"type": bool,  "default": False},
    "vision.ocr_engine":             {"type": str,   "default": "tesseract",
                                      "choices": ["tesseract", "easyocr"]},
    "vision.capture_interval_sec":   {"type": float, "default": 2.0,   "min": 0.5, "max": 60.0},
    "vision.save_screenshots":       {"type": bool,  "default": False},

    # llm
    "llm.provider":                  {"type": str,   "default": "ollama",
                                      "choices": ["ollama", "anthropic", "openai", "local"]},
    "llm.host":                      {"type": str,   "default": "http://localhost:11434"},
    "llm.model":                     {"type": str,   "default": "mistral"},
    "llm.temperature":               {"type": float, "default": 0.7,   "min": 0.0, "max": 2.0},
    "llm.max_tokens":                {"type": int,   "default": 2048,  "min": 64,  "max": 32768},
    "llm.stream_responses":          {"type": bool,  "default": True},
    "llm.api_timeout_sec":           {"type": int,   "default": 60,    "min": 5,   "max": 300},
    # tiered brain router — cloud escalation (privacy-first: off by default)
    "llm.tiered_routing":            {"type": bool,  "default": False},
    "llm.allow_cloud":               {"type": bool,  "default": False},
    "llm.cloud_confirm":             {"type": bool,  "default": True},
    "llm.cloud_provider":            {"type": str,   "default": "anthropic",
                                      "choices": ["anthropic", "openai"]},
    "llm.cloud_model":               {"type": str,   "default": "claude-sonnet-5"},

    # memory
    "memory.enabled":                {"type": bool,  "default": False},
    "memory.short_term_limit":       {"type": int,   "default": 50,    "min": 5,   "max": 1000},
    "memory.auto_summarize":         {"type": bool,  "default": True},
    "memory.summarize_after_turns":  {"type": int,   "default": 20,    "min": 5,   "max": 200},

    # security
    "security.voice_auth_required":  {"type": bool,  "default": False},
    "security.restrict_shell_commands": {"type": bool, "default": True},
    "security.activity_logging":     {"type": bool,  "default": True},
    "security.sandbox_enabled":      {"type": bool,  "default": False},

    # interface
    "interface.terminal_theme":      {"type": str,   "default": "dark",
                                      "choices": ["dark", "light"]},
    "interface.color_output":        {"type": bool,  "default": True},
    "interface.show_timestamps":     {"type": bool,  "default": True},
    "interface.banner_on_start":     {"type": bool,  "default": True},
    "interface.typing_effect":       {"type": bool,  "default": False},

    # plugins
    "plugins.enabled":               {"type": bool,  "default": True},
    "plugins.auto_load":             {"type": bool,  "default": True},
    "plugins.safe_mode":             {"type": bool,  "default": True},
}


# ─────────────────────────────────────────────────────────────
#  CONFIG CLASS
# ─────────────────────────────────────────────────────────────

class NexusConfig:
    """
    Central configuration manager for NEXUS.

    Loads settings.json, validates every value against the schema,
    falls back to safe defaults on any bad value, and exposes a
    dot-path API for reading and writing settings at runtime.
    """

    _DEFAULT_PATH = Path(__file__).parent.parent / "config" / "settings.json"

    def __init__(self, config_path: Optional[Path] = None):
        self._path   = Path(config_path) if config_path else self._DEFAULT_PATH
        self._data: dict = {}
        self._warnings: list[str] = []
        self.load()

    # ── public API ────────────────────────────────────────────

    def load(self) -> None:
        """Load and validate settings.json. Falls back to defaults on error."""
        self._warnings = []

        if not self._path.exists():
            self._warnings.append(f"Config file not found at {self._path}. Using all defaults.")
            self._data = self._build_defaults()
            self._write_file(self._data)
            return

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self._warnings.append(f"settings.json is malformed: {exc}. Using all defaults.")
            self._data = self._build_defaults()
            return

        self._data = self._validate(raw)

    def save(self) -> None:
        """Persist current in-memory config back to settings.json."""
        self._data["_meta"]["last_updated"] = datetime.date.today().isoformat()
        self._write_file(self._data)

    def get(self, dot_path: str, fallback: Any = None) -> Any:
        """
        Read a config value using dot notation.
        e.g.  cfg.get("voice.enabled")  →  False
        """
        keys = dot_path.split(".")
        node = self._data
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return fallback
            node = node[k]
        return node

    def set(self, dot_path: str, value: Any) -> bool:
        """
        Update a config value in memory (does not auto-save).
        Validates the new value against the schema first.
        Returns True if accepted, False if rejected.
        """
        rule = SCHEMA.get(dot_path)
        if rule:
            ok, reason, value = self._check_value(dot_path, value, rule)
            if not ok:
                self._warnings.append(f"set('{dot_path}', ...) rejected: {reason}")
                return False

        keys  = dot_path.split(".")
        node  = self._data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
        return True

    def reload(self) -> None:
        """Re-read settings.json from disk (picks up manual edits)."""
        self.load()

    def as_dict(self) -> dict:
        """Return a deep copy of the full config dict."""
        return copy.deepcopy(self._data)

    def warnings(self) -> list[str]:
        """Return any validation warnings generated during last load."""
        return list(self._warnings)

    def summary(self) -> str:
        """Return a human-readable one-liner of key settings."""
        return (
            f"name={self.get('identity.assistant_name')} | "
            f"wake_word='{self.get('identity.wake_word')}' | "
            f"voice={self.get('voice.enabled')} | "
            f"vision={self.get('vision.enabled')} | "
            f"llm={self.get('llm.provider')}/{self.get('llm.model')} | "
            f"debug={self.get('system.debug_mode')}"
        )

    # ── internal ──────────────────────────────────────────────

    def _validate(self, raw: dict) -> dict:
        """Walk the schema, validate each key, replace bad values with defaults."""
        data = copy.deepcopy(raw)

        for dot_path, rule in SCHEMA.items():
            keys    = dot_path.split(".")
            node    = data
            missing = False

            # Navigate to the parent dict
            for k in keys[:-1]:
                if not isinstance(node, dict) or k not in node:
                    missing = True
                    break
                node = node[k]

            leaf = keys[-1]

            if missing or not isinstance(node, dict) or leaf not in node:
                # Key absent — silently inject default
                self._deep_set(data, keys, rule["default"])
                continue

            raw_val = node[leaf]
            ok, reason, coerced = self._check_value(dot_path, raw_val, rule)
            if not ok:
                self._warnings.append(
                    f"'{dot_path}': {reason}. Using default: {rule['default']!r}"
                )
                self._deep_set(data, keys, rule["default"])
            elif coerced != raw_val:
                self._deep_set(data, keys, coerced)

        return data

    @staticmethod
    def _check_value(dot_path: str, value: Any, rule: dict) -> tuple[bool, str, Any]:
        """
        Validate a single value against its schema rule.
        Returns (is_ok, reason_string, coerced_value).
        """
        expected = rule["type"]

        # Allow None for nullable fields (e.g. microphone_index)
        if value is None:
            return True, "", value

        # Type coercion: bool must come before int (bool is subclass of int)
        if expected is bool:
            if isinstance(value, bool):
                return True, "", value
            return False, f"expected bool, got {type(value).__name__}", value

        if expected is int:
            if isinstance(value, float) and value == int(value):
                value = int(value)
            if not isinstance(value, int):
                return False, f"expected int, got {type(value).__name__}", value

        if expected is float:
            if isinstance(value, int):
                value = float(value)
            if not isinstance(value, float):
                return False, f"expected float, got {type(value).__name__}", value

        if expected is str and not isinstance(value, str):
            return False, f"expected str, got {type(value).__name__}", value

        # Range checks
        if "min" in rule and value < rule["min"]:
            return False, f"{value} is below minimum {rule['min']}", value
        if "max" in rule and value > rule["max"]:
            return False, f"{value} is above maximum {rule['max']}", value

        # Choices check
        if "choices" in rule and value not in rule["choices"]:
            return False, f"'{value}' not in allowed choices {rule['choices']}", value

        return True, "", value

    @staticmethod
    def _deep_set(data: dict, keys: list[str], value: Any) -> None:
        node = data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

    def _build_defaults(self) -> dict:
        data: dict = {}
        for dot_path, rule in SCHEMA.items():
            self._deep_set(data, dot_path.split("."), rule["default"])
        return data

    def _write_file(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# ─────────────────────────────────────────────────────────────
#  SINGLETON — import this anywhere in NEXUS
# ─────────────────────────────────────────────────────────────

cfg = NexusConfig()


# ─────────────────────────────────────────────────────────────
#  CLI — run directly to inspect config
#  Usage: python core/config.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("\n  NEXUS Config Inspector\n")
    print(f"  File   : {cfg._path}")
    print(f"  Status : {'OK' if not cfg.warnings() else 'WARNINGS'}")
    print(f"  Summary: {cfg.summary()}")

    if cfg.warnings():
        print("\n  Warnings:")
        for w in cfg.warnings():
            print(f"    ⚠  {w}")

    if "--full" in sys.argv:
        print("\n  Full config:\n")
        print(json.dumps(cfg.as_dict(), indent=4))

    if "--validate" in sys.argv:
        # Write a fresh validated settings.json
        cfg.save()
        print("\n  Config saved (validated + normalised).")

    print()