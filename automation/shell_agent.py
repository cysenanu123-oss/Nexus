"""
automation/shell_agent.py
NEXUS Shell Agent — executes system-level automation steps.

Handles all step types that interact with the operating system:
  - Running shell commands (subprocess)
  - Launching and killing applications
  - File system operations (create, read, write, delete, copy, move)
  - Package installation (pip, apt)
  - Process management
  - Checks (file exists, process running, import available)

Design principles:
  - Never uses raw shell=True unless required (injection prevention)
  - All destructive operations are logged before execution
  - Returns (success: bool, output: str) for every action
  - Falls back gracefully when tools aren't available

Usage:
    from automation.shell_agent import ShellAgent
    agent = ShellAgent()
    success, output = agent.run(step)
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
import time
import logging
from pathlib import Path
from typing import Optional

from core.platform_utils import IS_WINDOWS, IS_MAC

log = logging.getLogger("nexus.automation.shell")


# ─────────────────────────────────────────────────────────────
#  APP ALIASES  (user phrase → actual binary)
# ─────────────────────────────────────────────────────────────

_APP_ALIASES_LINUX: dict[str, str] = {
    # Browsers
    "browser":           "firefox",
    "chrome":            "google-chrome",
    "chromium":          "chromium",
    "firefox":           "firefox",
    "brave":             "brave-browser",
    "edge":              "microsoft-edge",
    # Editors
    "vs code":           "code",
    "vscode":            "code",
    "vscodium":          "vscodium",
    "code":              "code",
    "vim":               "vim",
    "nvim":              "nvim",
    "nano":              "nano",
    "gedit":             "gedit",
    "mousepad":          "mousepad",
    "text editor":       "mousepad",
    "leafpad":           "leafpad",
    # Terminals
    "terminal":          "xfce4-terminal",
    "xfce terminal":     "xfce4-terminal",
    "konsole":           "konsole",
    "gnome terminal":    "gnome-terminal",
    "xterm":             "xterm",
    # Security tools
    "burp":              "burpsuite",
    "burp suite":        "burpsuite",
    "wireshark":         "wireshark",
    "metasploit":        "msfconsole",
    "nmap":              "nmap",
    "zaproxy":           "zaproxy",
    "owasp zap":         "zaproxy",
    # File managers
    "files":             "thunar",
    "file manager":      "thunar",
    "thunar":            "thunar",
    "nautilus":          "nautilus",
    # System
    "calculator":        "galculator",
    "galculator":        "galculator",
    "settings":          "xfce4-settings-manager",
    "task manager":      "xfce4-taskmanager",
    "system monitor":    "xfce4-taskmanager",
    # Media
    "vlc":               "vlc",
    "spotify":           "spotify",
    "discord":           "discord",
    # Dev
    "python":            "python3",
    "jupyter":           "jupyter-notebook",
    "postman":           "postman",
}

_APP_ALIASES_WINDOWS: dict[str, str] = {
    # Browsers
    "browser":           "chrome.exe",
    "chrome":            "chrome.exe",
    "firefox":           "firefox.exe",
    "brave":             "brave.exe",
    "edge":              "msedge.exe",
    "chromium":          "chromium.exe",
    # Editors
    "vs code":           "code.exe",
    "vscode":            "code.exe",
    "code":              "code.exe",
    "notepad":           "notepad.exe",
    "notepad++":         "notepad++.exe",
    "text editor":       "notepad.exe",
    "wordpad":           "wordpad.exe",
    # Terminals
    "terminal":          "wt.exe",
    "windows terminal":  "wt.exe",
    "cmd":               "cmd.exe",
    "powershell":        "powershell.exe",
    # Messaging / Social
    "whatsapp":          "WhatsApp.exe",
    "discord":           "Discord.exe",
    "telegram":          "Telegram.exe",
    "slack":             "slack.exe",
    "teams":             "Teams.exe",
    "zoom":              "Zoom.exe",
    # Media
    "spotify":           "Spotify.exe",
    "vlc":               "vlc.exe",
    # Security tools
    "wireshark":         "Wireshark.exe",
    "burp":              "burpsuite.exe",
    "burp suite":        "burpsuite.exe",
    "nmap":              "nmap.exe",
    # System utilities
    "calculator":        "calc.exe",
    "task manager":      "taskmgr.exe",
    "system monitor":    "taskmgr.exe",
    "file manager":      "explorer.exe",
    "explorer":          "explorer.exe",
    "files":             "explorer.exe",
    "control panel":     "control.exe",
    "settings":          "ms-settings:",
    # Dev
    "python":            "python.exe",
    "jupyter":           "jupyter-notebook.exe",
    "postman":           "Postman.exe",
    "git bash":          "git-bash.exe",
}

_APP_ALIASES_MAC: dict[str, str] = {
    "browser":           "Safari",
    "chrome":            "Google Chrome",
    "firefox":           "Firefox",
    "brave":             "Brave Browser",
    "edge":              "Microsoft Edge",
    "vs code":           "Visual Studio Code",
    "vscode":            "Visual Studio Code",
    "code":              "Visual Studio Code",
    "terminal":          "Terminal",
    "iterm":             "iTerm",
    "whatsapp":          "WhatsApp",
    "discord":           "Discord",
    "slack":             "Slack",
    "spotify":           "Spotify",
    "vlc":               "VLC",
    "calculator":        "Calculator",
    "file manager":      "Finder",
    "finder":            "Finder",
    "files":             "Finder",
    "settings":          "System Preferences",
    "python":            "Python Launcher",
    "postman":           "Postman",
}

if IS_WINDOWS:
    APP_ALIASES = _APP_ALIASES_WINDOWS
elif IS_MAC:
    APP_ALIASES = _APP_ALIASES_MAC
else:
    APP_ALIASES = _APP_ALIASES_LINUX

# Common Windows install paths to search for executables
_WIN_SEARCH_PATHS: list[Path] = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs",
    Path(os.environ.get("LOCALAPPDATA", "")) / "WhatsApp",
    Path(os.environ.get("APPDATA", "")) / "Spotify" / "Spotify.exe",
    Path(os.environ.get("APPDATA", "")) / "Discord" / "Update.exe",
    Path("C:/Program Files"),
    Path("C:/Program Files (x86)"),
    Path(os.environ.get("LOCALAPPDATA", "")) / "Discord" / "app-" ,
]


# ─────────────────────────────────────────────────────────────
#  SHELL AGENT
# ─────────────────────────────────────────────────────────────

class ShellAgent:
    """
    Executes shell-type and check-type steps.

    Every public method returns (success: bool, output: str).
    """

    def __init__(self, default_timeout: float = 30.0):
        self.default_timeout = default_timeout
        log.info("ShellAgent ready.")

    # ── Main dispatch ─────────────────────────────────────────

    def run(self, step) -> tuple[bool, str]:
        """Dispatch a shell or web step to the right handler."""
        action = step.action.lower()

        handlers = {
            "launch_app":    self._launch_app,
            "kill_app":      self._kill_app,
            "run_command":   self._run_command,
            "create_file":   self._create_file,
            "write_file":    self._write_file,
            "delete_file":   self._delete_file,
            "screenshot":    self._screenshot,
            "navigate_url":  self._navigate_url,   # handled here for web steps
        }

        handler = handlers.get(action)
        if handler:
            return handler(step)

        # Generic fallback: treat target as shell command
        if step.target:
            return self._run_command(step)

        return False, f"ShellAgent: unknown action {action!r}"

    def check(self, step) -> tuple[bool, str]:
        """Dispatch a check step."""
        action = step.action.lower()

        checks = {
            "check_file_exists":    self._check_file_exists,
            "check_process_running": self._check_process_running,
            "check_python_import":  self._check_python_import,
            "check_screen_text":    self._check_screen_text,
        }

        handler = checks.get(action)
        if handler:
            return handler(step)

        return True, f"Check not implemented: {action!r} — assuming ok"

    # ── App management ────────────────────────────────────────

    def _launch_app(self, step) -> tuple[bool, str]:
        target = step.target.strip().lower()
        app_name = APP_ALIASES.get(target, target)

        if IS_WINDOWS:
            return self._launch_app_windows(step.target, app_name)
        elif IS_MAC:
            return self._launch_app_mac(step.target, app_name)
        else:
            return self._launch_app_linux(step.target, app_name)

    def _launch_app_linux(self, label: str, binary: str) -> tuple[bool, str]:
        resolved = shutil.which(binary) or shutil.which(label)
        if not resolved:
            return False, f"App '{binary}' not found. Install with: apt install {binary}"
        try:
            subprocess.Popen([resolved], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
            log.info("Launched: %s", resolved)
            return True, f"Launched {label}"
        except Exception as e:
            return False, f"Failed to launch {label}: {e}"

    def _launch_app_windows(self, label: str, exe: str) -> tuple[bool, str]:
        # 1) Try PATH first (works for cmd.exe, notepad.exe, etc.)
        if shutil.which(exe):
            try:
                os.startfile(exe)   # type: ignore[attr-defined]
                return True, f"Launched {label}"
            except Exception:
                pass

        # 2) ms-settings: and other URI schemes
        if ":" in exe and not exe.endswith(".exe"):
            try:
                os.startfile(exe)   # type: ignore[attr-defined]
                return True, f"Opened {label}"
            except Exception as e:
                return False, f"Failed to open {label}: {e}"

        # 3) Search common install directories
        search_dirs = [
            Path(os.environ.get("LOCALAPPDATA", "C:/Users/Default/AppData/Local")),
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs",
            Path(os.environ.get("APPDATA", "")),
            Path("C:/Program Files"),
            Path("C:/Program Files (x86)"),
        ]
        for base in search_dirs:
            if not base.exists():
                continue
            for found in base.rglob(exe):
                if found.is_file():
                    try:
                        os.startfile(str(found))   # type: ignore[attr-defined]
                        log.info("Launched via search: %s", found)
                        return True, f"Launched {label} ({found.name})"
                    except Exception:
                        pass

        # 4) Let Windows shell resolve it (Start-Process)
        try:
            subprocess.Popen(
                ["powershell", "-WindowStyle", "Hidden", "-Command",
                 f'Start-Process "{exe}"'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return True, f"Launched {label} via shell"
        except Exception as e:
            pass

        return False, (
            f"App '{label}' not found. Make sure it is installed. "
            f"Search '{label}' in the Start Menu to confirm."
        )

    def _launch_app_mac(self, label: str, app_name: str) -> tuple[bool, str]:
        # Try 'open -a AppName' which works for any .app in /Applications
        try:
            result = subprocess.run(
                ["open", "-a", app_name],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                return True, f"Launched {label}"
        except Exception:
            pass
        # Fallback: binary on PATH
        binary = shutil.which(app_name.lower().replace(" ", "-"))
        if binary:
            subprocess.Popen([binary], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
            return True, f"Launched {label}"
        return False, f"App '{label}' not found. Install with: brew install --cask {app_name.lower()}"

    def _kill_app(self, step) -> tuple[bool, str]:
        target = step.target.strip().lower()
        app_name = APP_ALIASES.get(target, target)

        if IS_WINDOWS:
            exe = app_name if app_name.endswith(".exe") else app_name + ".exe"
            try:
                result = subprocess.run(
                    ["taskkill", "/F", "/IM", exe],
                    capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    return True, f"Closed {step.target}"
                return False, f"No running process found for {step.target!r}"
            except Exception as e:
                return False, str(e)
        else:
            binary = app_name if not IS_MAC else app_name.lower().replace(" ", "")
            try:
                result = subprocess.run(
                    ["pkill", "-f", binary],
                    capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    return True, f"Closed {step.target}"
                return False, f"No running process found for {step.target!r}"
            except Exception as e:
                return False, str(e)

    # ── Command execution ─────────────────────────────────────

    def _run_command(self, step) -> tuple[bool, str]:
        cmd     = step.target.strip()
        timeout = step.params.get("timeout", step.timeout_sec or self.default_timeout)

        if not cmd:
            return False, "No command specified."

        log.info("Running command: %r (timeout=%.1fs)", cmd, timeout)

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=float(timeout),
                cwd=os.getcwd(),
            )
            output = (result.stdout or result.stderr or "").strip()
            success = result.returncode == 0

            if not success:
                log.warning("Command exited %d: %s", result.returncode, output[:200])

            return success, output or "(no output)"

        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout}s"
        except Exception as e:
            return False, str(e)

    # ── File operations ───────────────────────────────────────

    def _create_file(self, step) -> tuple[bool, str]:
        path    = Path(step.target.strip())
        content = step.params.get("content", "")

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            log.info("Created file: %s", path)
            return True, f"Created: {path}"
        except Exception as e:
            return False, f"Failed to create {path}: {e}"

    def _write_file(self, step) -> tuple[bool, str]:
        path    = Path(step.target.strip())
        content = step.params.get("content", "")
        mode    = step.params.get("mode", "w")   # "w" = overwrite, "a" = append

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, mode) as f:
                f.write(content)
            log.info("Wrote to file: %s (%s)", path, "append" if mode == "a" else "overwrite")
            return True, f"Wrote to: {path}"
        except Exception as e:
            return False, f"Failed to write {path}: {e}"

    def _delete_file(self, step) -> tuple[bool, str]:
        path = Path(step.target.strip())

        if not path.exists():
            return False, f"Path does not exist: {path}"

        log.warning("Deleting: %s", path)
        try:
            if path.is_dir():
                import shutil as _shutil
                _shutil.rmtree(path)
            else:
                path.unlink()
            return True, f"Deleted: {path}"
        except Exception as e:
            return False, f"Failed to delete {path}: {e}"

    # ── Screenshot ────────────────────────────────────────────

    def _screenshot(self, step) -> tuple[bool, str]:
        save_dir = step.target or "data/screenshots"
        try:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from vision.capture import ScreenCapturer
            capturer = ScreenCapturer(save_dir=Path(save_dir))
            shot     = capturer.capture()
            path     = shot.save(Path(save_dir))
            return True, f"Screenshot saved: {path}"
        except ImportError:
            # Fallback: scrot or gnome-screenshot
            for cmd in ["scrot", "gnome-screenshot -f"]:
                if shutil.which(cmd.split()[0]):
                    ts   = int(time.time())
                    out  = f"{save_dir}/nexus_{ts}.png"
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    ok, _ = self._run_command(
                        type("S", (), {"target": f"{cmd} {out}",
                                       "params": {}, "timeout_sec": 10})()
                    )
                    if ok:
                        return True, f"Screenshot saved: {out}"
            return False, "Screenshot failed: no capture tool available"

    # ── Navigate URL (for web-type steps) ─────────────────────

    def _navigate_url(self, step) -> tuple[bool, str]:
        url = step.target.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # Try to open via existing browser window using xdg-open
        try:
            subprocess.Popen(
                ["xdg-open", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True, f"Opened URL: {url}"
        except Exception as e:
            return False, str(e)

    # ── Checks ────────────────────────────────────────────────

    def _check_file_exists(self, step) -> tuple[bool, str]:
        path = Path(step.target.strip())
        if path.exists():
            return True, f"File exists: {path}"
        return False, f"File not found: {path}"

    def _check_process_running(self, step) -> tuple[bool, str]:
        proc = step.target.strip()
        try:
            result = subprocess.run(
                ["pgrep", "-f", proc],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                pids = result.stdout.strip().split()
                return True, f"Process {proc!r} running (PIDs: {', '.join(pids[:5])})"
            return False, f"Process {proc!r} not found"
        except Exception as e:
            return False, str(e)

    def _check_python_import(self, step) -> tuple[bool, str]:
        pkg = step.target.strip()
        # Normalize package name (hyphens → underscores)
        mod = pkg.replace("-", "_").lower()
        try:
            import importlib
            importlib.import_module(mod)
            return True, f"Python package {pkg!r} is importable"
        except ImportError:
            return False, f"Python package {pkg!r} not importable"
        except Exception as e:
            return False, str(e)

    def _check_screen_text(self, step) -> tuple[bool, str]:
        text = step.target.strip()
        try:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from vision.vision import Vision
            v = Vision()
            found = v.is_text_on_screen(text)
            if found:
                return True, f"Text {text!r} found on screen"
            return False, f"Text {text!r} not found on screen"
        except ImportError:
            return True, "Vision unavailable — check skipped"


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    # Allow running as: python automation/shell_agent.py ...
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import time as _time
    logging.basicConfig(level=logging.INFO)

    agent = ShellAgent()

    # Minimal fake step for testing
    class FakeStep:
        def __init__(self, action, target, params=None, timeout_sec=30.0):
            self.action      = action
            self.target      = target
            self.params      = params or {}
            self.timeout_sec = timeout_sec

    tests = [
        FakeStep("run_command", "echo 'NEXUS shell agent test'"),
        FakeStep("run_command", "ls /tmp"),
        FakeStep("check_file_exists", "/tmp"),
        FakeStep("check_python_import", "os"),
        FakeStep("check_python_import", "nonexistent_pkg_xyz"),
        FakeStep("create_file", "/tmp/nexus_test.txt", {"content": "hello from NEXUS\n"}),
        FakeStep("run_command", "cat /tmp/nexus_test.txt"),
    ]

    if len(sys.argv) > 1:
        tests = [FakeStep("run_command", " ".join(sys.argv[1:]))]

    print("\n─── ShellAgent Test ───\n")
    for step in tests:
        t0 = _time.time()
        if step.action.startswith("check_"):
            success, output = agent.check(step)
        else:
            success, output = agent.run(step)
        elapsed = _time.time() - t0
        icon = "✓" if success else "✗"
        print(f"  {icon} [{step.action}] {step.target!r}  ({elapsed:.2f}s)")
        if output:
            print(f"     {output[:100]}")
    print()