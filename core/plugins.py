"""
core/plugins.py
NEXUS Plugin Architecture — auto-loads Python plugins from plugins/ folder.

Closes Phase 1 foundation task: "Create modular plugin architecture"

Plugin contract
───────────────
Every plugin is a Python file in plugins/ that defines a class named Plugin
inheriting from PluginBase. The class must implement at minimum:

    class Plugin(PluginBase):
        NAME        = "my_plugin"      # unique snake_case ID
        VERSION     = "1.0.0"
        DESCRIPTION = "What it does"

        def activate(self) -> bool:    # called on load — return True if ready
            ...
            return True

        def deactivate(self) -> None:  # called on unload / NEXUS shutdown
            ...

Optional hooks — implement any you need:

        def on_intent(self, intent, text: str) -> str | None:
            # Called for every user input before Brain routes it.
            # Return a string to short-circuit normal routing, or None to pass through.
            ...

        def on_response(self, text: str, response: str) -> str:
            # Called after Brain produces a response.
            # Can modify and return the response string.
            return response

        def on_startup(self) -> None:
            # Called once after all plugins are loaded.
            ...

        def on_shutdown(self) -> None:
            # Called once before NEXUS exits.
            ...

        def commands(self) -> dict[str, callable]:
            # Return extra CLI commands this plugin contributes.
            # {"weather": self.handle_weather, ...}
            return {}

Example plugin (plugins/hello_plugin.py):
─────────────────────────────────────────
    from core.plugins import PluginBase

    class Plugin(PluginBase):
        NAME        = "hello_plugin"
        VERSION     = "1.0.0"
        DESCRIPTION = "Greets the user on startup"

        def activate(self) -> bool:
            self.log.info("Hello plugin active.")
            return True

        def deactivate(self) -> None:
            pass

        def on_startup(self) -> None:
            print("[PLUGIN] Hello from hello_plugin!")

Usage in NEXUS:
───────────────
    from core.plugins import PluginManager
    pm = PluginManager()
    pm.load_all()

    # Call hooks
    modified_response = pm.hook_response(user_text, nexus_response)
    extra_commands    = pm.all_commands()

    pm.shutdown()
"""

from __future__ import annotations

import os
import sys
import importlib
import importlib.util
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any

log = logging.getLogger("nexus.plugins")


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

PLUGINS_DIR  = Path("plugins")
PLUGIN_CLASS = "Plugin"       # the class name every plugin must define


# ─────────────────────────────────────────────────────────────
#  PLUGIN BASE — the contract every plugin must fulfil
# ─────────────────────────────────────────────────────────────

class PluginBase(ABC):
    """
    Base class for all NEXUS plugins.

    Subclass this in your plugin file and name the subclass 'Plugin'.
    Override the hooks you need — everything except activate() and
    deactivate() is optional.
    """

    NAME:        str = "unnamed_plugin"
    VERSION:     str = "0.0.0"
    DESCRIPTION: str = ""
    AUTHOR:      str = ""
    SAFE:        bool = True   # False = won't load in safe mode

    def __init__(self):
        self.log    = logging.getLogger(f"nexus.plugins.{self.NAME}")
        self._active = False

    # ── Required ──────────────────────────────────────────────

    @abstractmethod
    def activate(self) -> bool:
        """
        Initialize the plugin and return True if it's ready.
        Return False to cancel loading.
        """
        ...

    @abstractmethod
    def deactivate(self) -> None:
        """Clean up resources when the plugin is unloaded."""
        ...

    # ── Optional hooks ────────────────────────────────────────

    def on_intent(self, intent: Any, text: str) -> Optional[str]:
        """
        Called before Brain routes each input.
        Return a response string to intercept, or None to pass through.
        """
        return None

    def on_response(self, text: str, response: str) -> str:
        """
        Called after Brain produces a response.
        Can modify and return the response.
        """
        return response

    def on_startup(self) -> None:
        """Called once after all plugins are loaded."""
        pass

    def on_shutdown(self) -> None:
        """Called once before NEXUS exits."""
        pass

    def commands(self) -> dict[str, callable]:
        """
        Return extra CLI commands this plugin contributes.
        Keys become available commands in the REPL.
        """
        return {}

    # ── Internal state ────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self._active

    def __repr__(self) -> str:
        status = "active" if self._active else "inactive"
        return f"Plugin({self.NAME!r} v{self.VERSION} [{status}])"


# ─────────────────────────────────────────────────────────────
#  PLUGIN RECORD — wraps a loaded plugin with metadata
# ─────────────────────────────────────────────────────────────

class PluginRecord:
    """Internal record tracking a single loaded plugin."""

    def __init__(self, plugin: PluginBase, path: Path, load_time: float):
        self.plugin    = plugin
        self.path      = path
        self.load_time = load_time
        self.error:    Optional[str] = None

    @property
    def name(self) -> str:
        return self.plugin.NAME

    def __repr__(self) -> str:
        return f"PluginRecord({self.name!r}, path={self.path.name!r})"


# ─────────────────────────────────────────────────────────────
#  PLUGIN MANAGER
# ─────────────────────────────────────────────────────────────

class PluginManager:
    """
    Discovers, loads, and manages the lifecycle of NEXUS plugins.

    On load_all():
        - Scans plugins/ for *_plugin.py files
        - Imports each module
        - Instantiates the Plugin class
        - Calls plugin.activate()
        - Registers successfully activated plugins

    After loading:
        - hook_intent(intent, text)  → let plugins intercept commands
        - hook_response(text, resp)  → let plugins modify responses
        - all_commands()             → collect extra REPL commands
        - shutdown()                 → gracefully deactivate all

    Usage:
        pm = PluginManager()
        pm.load_all()

        # In the Brain's _route():
        intercepted = pm.hook_intent(intent, text)
        if intercepted:
            return intercepted

        # After getting Brain response:
        response = pm.hook_response(user_text, response)

        # Extra commands for the REPL:
        for cmd, fn in pm.all_commands().items():
            dispatch[cmd] = fn

        # On exit:
        pm.shutdown()
    """

    def __init__(
        self,
        plugins_dir: Path = PLUGINS_DIR,
        safe_mode:   bool = True,
        auto_load:   bool = True,
    ):
        self.plugins_dir = Path(plugins_dir)
        self.safe_mode   = safe_mode
        self.auto_load   = auto_load

        self._records: dict[str, PluginRecord] = {}   # name → record
        self._load_order: list[str] = []               # tracks insertion order

        log.info(
            "PluginManager initialized — dir=%s, safe_mode=%s, auto_load=%s",
            self.plugins_dir, safe_mode, auto_load,
        )

    # ── Discovery & loading ───────────────────────────────────

    def load_all(self) -> int:
        """
        Scan plugins_dir and load every valid plugin file.

        Returns
        -------
        int — number of successfully loaded plugins
        """
        if not self.plugins_dir.exists():
            self.plugins_dir.mkdir(parents=True, exist_ok=True)
            log.info("Created plugins directory: %s", self.plugins_dir)

        plugin_files = sorted(self.plugins_dir.glob("*.py"))
        plugin_files = [f for f in plugin_files if not f.name.startswith("_")]

        if not plugin_files:
            log.info("No plugins found in %s", self.plugins_dir)
            return 0

        log.info("Found %d plugin file(s) in %s", len(plugin_files), self.plugins_dir)

        loaded = 0
        for path in plugin_files:
            if self.load(path):
                loaded += 1

        log.info("Loaded %d / %d plugin(s).", loaded, len(plugin_files))

        # Fire on_startup for all loaded plugins
        for name in self._load_order:
            record = self._records[name]
            try:
                record.plugin.on_startup()
            except Exception as e:
                log.error("Plugin %r on_startup failed: %s", name, e)

        return loaded

    def load(self, path: Path) -> bool:
        """
        Load a single plugin file.

        Parameters
        ----------
        path : Path to the plugin .py file

        Returns
        -------
        bool — True if activated successfully
        """
        path = Path(path)
        t0   = time.time()

        log.debug("Loading plugin: %s", path.name)

        # ── Import module ──────────────────────────────────────
        try:
            module = self._import_module(path)
        except Exception as e:
            log.error("Failed to import %s: %s", path.name, e)
            return False

        # ── Get Plugin class ───────────────────────────────────
        if not hasattr(module, PLUGIN_CLASS):
            log.warning("%s has no 'Plugin' class — skipped.", path.name)
            return False

        cls = getattr(module, PLUGIN_CLASS)

        if not (isinstance(cls, type) and issubclass(cls, PluginBase)):
            log.warning(
                "%s.Plugin doesn't inherit PluginBase — skipped.", path.name
            )
            return False

        # ── Instantiate ────────────────────────────────────────
        try:
            instance: PluginBase = cls()
        except Exception as e:
            log.error("Failed to instantiate %s.Plugin: %s", path.name, e)
            return False

        # ── Safe mode check ────────────────────────────────────
        if self.safe_mode and not instance.SAFE:
            log.warning(
                "Plugin %r marked SAFE=False — skipped (safe_mode=True).",
                instance.NAME,
            )
            return False

        # ── Duplicate check ────────────────────────────────────
        if instance.NAME in self._records:
            log.warning(
                "Duplicate plugin name %r from %s — skipped.",
                instance.NAME, path.name,
            )
            return False

        # ── Activate ───────────────────────────────────────────
        try:
            ready = instance.activate()
        except Exception as e:
            log.error("Plugin %r activate() raised: %s", instance.NAME, e)
            return False

        if not ready:
            log.warning("Plugin %r activate() returned False — not loaded.", instance.NAME)
            return False

        instance._active = True
        record = PluginRecord(
            plugin    = instance,
            path      = path,
            load_time = time.time() - t0,
        )
        self._records[instance.NAME]   = record
        self._load_order.append(instance.NAME)

        log.info(
            "Plugin loaded: %r v%s — %s (%.2fs)",
            instance.NAME, instance.VERSION,
            instance.DESCRIPTION or "no description",
            record.load_time,
        )
        return True

    def unload(self, name: str) -> bool:
        """
        Deactivate and remove a plugin by name.

        Returns True if the plugin was found and unloaded.
        """
        if name not in self._records:
            log.warning("Plugin %r not found.", name)
            return False

        record = self._records[name]
        try:
            record.plugin.on_shutdown()
            record.plugin.deactivate()
            record.plugin._active = False
        except Exception as e:
            log.error("Plugin %r deactivate() raised: %s", name, e)

        del self._records[name]
        if name in self._load_order:
            self._load_order.remove(name)

        log.info("Plugin unloaded: %r", name)
        return True

    def reload(self, name: str) -> bool:
        """Unload then re-load a plugin (picks up file changes)."""
        if name not in self._records:
            return False
        path = self._records[name].path
        self.unload(name)
        return self.load(path)

    # ── Hooks ─────────────────────────────────────────────────

    def hook_intent(self, intent: Any, text: str) -> Optional[str]:
        """
        Give each active plugin a chance to intercept the input.

        Returns
        -------
        str   — if any plugin handles it (first match wins)
        None  — pass through to normal Brain routing
        """
        for name in self._load_order:
            plugin = self._records[name].plugin
            try:
                result = plugin.on_intent(intent, text)
                if result is not None:
                    log.debug("Plugin %r intercepted input.", name)
                    return result
            except Exception as e:
                log.error("Plugin %r on_intent raised: %s", name, e)
        return None

    def hook_response(self, text: str, response: str) -> str:
        """
        Let each active plugin post-process the Brain's response.

        Plugins are called in load order; each sees the previous plugin's output.

        Returns
        -------
        str — final (possibly modified) response
        """
        for name in self._load_order:
            plugin = self._records[name].plugin
            try:
                response = plugin.on_response(text, response)
            except Exception as e:
                log.error("Plugin %r on_response raised: %s", name, e)
        return response

    def all_commands(self) -> dict[str, callable]:
        """
        Collect extra CLI commands contributed by all plugins.

        Returns
        -------
        dict[str, callable] — merged command table from all plugins
        """
        commands: dict[str, callable] = {}
        for name in self._load_order:
            plugin = self._records[name].plugin
            try:
                for cmd, fn in plugin.commands().items():
                    if cmd in commands:
                        log.warning(
                            "Command %r from plugin %r conflicts with existing command — skipped.",
                            cmd, name,
                        )
                    else:
                        commands[cmd] = fn
            except Exception as e:
                log.error("Plugin %r commands() raised: %s", name, e)
        return commands

    def shutdown(self) -> None:
        """Gracefully deactivate all plugins (call before NEXUS exits)."""
        log.info("Shutting down %d plugin(s)...", len(self._records))
        for name in reversed(self._load_order):
            record = self._records[name]
            try:
                record.plugin.on_shutdown()
                record.plugin.deactivate()
                record.plugin._active = False
            except Exception as e:
                log.error("Plugin %r shutdown raised: %s", name, e)
        self._records.clear()
        self._load_order.clear()
        log.info("All plugins shut down.")

    # ── Introspection ─────────────────────────────────────────

    @property
    def loaded(self) -> list[PluginBase]:
        """Return list of all active plugin instances."""
        return [self._records[n].plugin for n in self._load_order]

    def get(self, name: str) -> Optional[PluginBase]:
        """Return a plugin instance by name, or None."""
        r = self._records.get(name)
        return r.plugin if r else None

    def is_loaded(self, name: str) -> bool:
        return name in self._records

    def count(self) -> int:
        return len(self._records)

    def status_table(self) -> list[dict]:
        """Return a list of dicts for display (used by main.py 'modules' command)."""
        rows = []
        for name in self._load_order:
            r = self._records[name]
            rows.append({
                "name":        r.name,
                "version":     r.plugin.VERSION,
                "description": r.plugin.DESCRIPTION,
                "path":        r.path.name,
                "load_time":   f"{r.load_time*1000:.0f}ms",
                "safe":        r.plugin.SAFE,
            })
        return rows

    def print_status(self) -> None:
        """Print a formatted plugin status table."""
        rows = self.status_table()
        if not rows:
            print(f"\n  No plugins loaded.\n  Drop .py files into: {self.plugins_dir.resolve()}\n")
            return

        print(f"\n  NEXUS Plugin Status — {len(rows)} loaded\n")
        print(f"  {'NAME':<22} {'VER':<8} {'LOAD':<8} {'FILE':<30} DESCRIPTION")
        print("  " + "─" * 80)
        for r in rows:
            print(
                f"  \033[92m{r['name']:<22}\033[0m"
                f"\033[2m{r['version']:<8}\033[0m"
                f"\033[2m{r['load_time']:<8}\033[0m"
                f"\033[2m{r['path']:<30}\033[0m"
                f"{r['description']}"
            )
        print()

    # ── Internal ──────────────────────────────────────────────

    def _import_module(self, path: Path):
        """Import a plugin file as a module without polluting sys.modules permanently."""
        module_name = f"nexus_plugin_{path.stem}"
        spec   = importlib.util.spec_from_file_location(module_name, str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


# ─────────────────────────────────────────────────────────────
#  SINGLETON — shared across Brain and main.py
# ─────────────────────────────────────────────────────────────

_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Return the shared PluginManager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        try:
            from core.config import cfg
            safe_mode = cfg.get("plugins.safe_mode", True)
            auto_load = cfg.get("plugins.auto_load", True)
            plugins_dir = cfg.get("plugins.plugins_dir", "plugins")
        except Exception:
            safe_mode, auto_load, plugins_dir = True, True, "plugins"

        _plugin_manager = PluginManager(
            plugins_dir = Path(plugins_dir),
            safe_mode   = safe_mode,
            auto_load   = auto_load,
        )
    return _plugin_manager


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST — python core/plugins.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("\n  NEXUS Plugin System Test\n")

    pm = PluginManager(plugins_dir=Path("plugins"), safe_mode=True)
    n  = pm.load_all()

    print(f"  Loaded {n} plugin(s).\n")
    pm.print_status()

    cmds = pm.all_commands()
    if cmds:
        print(f"  Extra commands contributed by plugins: {list(cmds.keys())}\n")

    if "--shutdown" in sys.argv:
        pm.shutdown()
        print("  All plugins shut down.\n")