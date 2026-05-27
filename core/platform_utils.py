"""
core/platform_utils.py
NEXUS Cross-Platform Utilities

Central hub for all OS-conditional behaviour. Every module that needs
to do something differently on Windows vs Linux imports from here instead
of scattering `if platform.system() == "Windows"` checks everywhere.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path
from typing import Optional

log = logging.getLogger("nexus.platform")

# ── OS detection ─────────────────────────────────────────────────────────────

IS_WINDOWS = sys.platform == "win32"
IS_LINUX   = sys.platform.startswith("linux")
IS_MAC     = sys.platform == "darwin"
IS_WSL     = IS_LINUX and "microsoft" in platform.uname().release.lower()

SYSTEM = "windows" if IS_WINDOWS else ("mac" if IS_MAC else "linux")

# ── Temp files ───────────────────────────────────────────────────────────────

def temp_path(suffix: str = ".tmp") -> str:
    """Return a cross-platform temporary file path (not yet created)."""
    return os.path.join(tempfile.gettempdir(), f"nexus_{os.getpid()}{suffix}")


# ── Desktop notifications ────────────────────────────────────────────────────

def notify(title: str, message: str, urgency: str = "normal") -> bool:
    """
    Send a desktop notification. Returns True on success.
    Linux: notify-send
    Windows: win10toast → plyer → tkinter fallback
    macOS: osascript
    """
    try:
        if IS_WINDOWS:
            return _notify_windows(title, message)
        elif IS_MAC:
            subprocess.Popen(
                ["osascript", "-e",
                 f'display notification "{message}" with title "{title}"'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return True
        else:
            # Linux / WSL2
            if shutil.which("notify-send"):
                urgency_map = {"low": "low", "normal": "normal",
                               "high": "critical", "critical": "critical"}
                u = urgency_map.get(urgency, "normal")
                subprocess.Popen(
                    ["notify-send", "-u", u, title, message],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                return True
    except Exception as e:
        log.debug("notify failed: %s", e)
    return False


def _notify_windows(title: str, message: str) -> bool:
    # 1. win10toast (best looking)
    try:
        from win10toast import ToastNotifier  # type: ignore
        ToastNotifier().show_toast(title, message, duration=5, threaded=True)
        return True
    except ImportError:
        pass
    # 2. plyer
    try:
        from plyer import notification  # type: ignore
        notification.notify(title=title, message=message, timeout=5)
        return True
    except ImportError:
        pass
    # 3. PowerShell toast (no extra deps)
    try:
        ps = (
            f"Add-Type -AssemblyName System.Windows.Forms; "
            f"[System.Windows.Forms.MessageBox]::Show('{message}', '{title}')"
        )
        subprocess.Popen(
            ["powershell", "-WindowStyle", "Hidden", "-Command", ps],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        pass
    return False


# ── Open URL / file ──────────────────────────────────────────────────────────

def open_url(url: str) -> bool:
    """Open a URL in the default browser, cross-platform."""
    try:
        if IS_WINDOWS:
            os.startfile(url)  # type: ignore[attr-defined]
        elif IS_MAC:
            subprocess.Popen(["open", url],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            if shutil.which("xdg-open"):
                subprocess.Popen(["xdg-open", url],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                webbrowser.open(url)
        return True
    except Exception as e:
        log.debug("open_url failed: %s", e)
        webbrowser.open(url)
        return True


def open_file(path: str) -> bool:
    """Open a file with its default application."""
    try:
        if IS_WINDOWS:
            os.startfile(path)  # type: ignore[attr-defined]
        elif IS_MAC:
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return True
    except Exception as e:
        log.warning("open_file failed: %s", e)
        return False


# ── Clipboard ────────────────────────────────────────────────────────────────

def clipboard_copy(text: str) -> bool:
    """Copy text to system clipboard, cross-platform."""
    # 1. pyperclip (best cross-platform, works everywhere with pip)
    try:
        import pyperclip  # type: ignore
        pyperclip.copy(text)
        return True
    except ImportError:
        pass
    # 2. Platform-native fallbacks
    try:
        if IS_WINDOWS:
            subprocess.run(["clip"], input=text.encode("utf-8"),
                           capture_output=True, check=True)
        elif IS_MAC:
            subprocess.run(["pbcopy"], input=text.encode("utf-8"),
                           capture_output=True, check=True)
        else:
            if shutil.which("xclip"):
                subprocess.run(["xclip", "-selection", "clipboard"],
                               input=text.encode("utf-8"),
                               capture_output=True, check=True)
            elif shutil.which("xsel"):
                subprocess.run(["xsel", "--clipboard", "--input"],
                               input=text.encode("utf-8"),
                               capture_output=True, check=True)
            elif shutil.which("wl-copy"):
                subprocess.run(["wl-copy"],
                               input=text.encode("utf-8"),
                               capture_output=True, check=True)
            else:
                return False
        return True
    except Exception as e:
        log.debug("clipboard_copy failed: %s", e)
        return False


# ── Ping ─────────────────────────────────────────────────────────────────────

def ping_cmd(ip: str, count: int = 1, timeout_sec: int = 1) -> list[str]:
    """Return the correct ping command for the current OS."""
    if IS_WINDOWS:
        return ["ping", "-n", str(count), "-w", str(timeout_sec * 1000), ip]
    else:
        return ["ping", "-c", str(count), "-W", str(timeout_sec), ip]


# ── System information ───────────────────────────────────────────────────────

def system_info_cmd() -> str:
    """Return a shell command that prints basic system info."""
    if IS_WINDOWS:
        return "systeminfo | findstr /C:\"OS\" /C:\"Memory\" /C:\"Processor\""
    elif IS_MAC:
        return "uname -a && vm_stat && df -h"
    else:
        return "uname -a && free -h && df -h"


def kill_process_cmd(proc_name: str) -> str:
    """Return OS-appropriate command to kill a process by name."""
    if IS_WINDOWS:
        return f"taskkill /F /IM \"{proc_name}\" /T"
    else:
        return f"pkill -f '{proc_name}'"


def network_info_cmd() -> str:
    """Return OS-appropriate command to show network interfaces + routes."""
    if IS_WINDOWS:
        return "ipconfig /all"
    elif IS_MAC:
        return "ifconfig && netstat -rn"
    else:
        return "ip addr show && ip route"


def brightness_cmd(action: str, level: Optional[str] = None) -> str:
    """Return an OS-appropriate screen brightness command."""
    if IS_WINDOWS:
        # PowerShell WMI brightness control
        if level and level.isdigit():
            pct = max(0, min(100, int(level)))
            return (
                f"powershell -Command \"(Get-WmiObject -Namespace root/WMI "
                f"-Class WmiMonitorBrightnessMethods).WmiSetBrightness(1, {pct})\""
            )
        inc = "10" if action == "up" else "-10"
        return (
            f"powershell -Command \"$b=(Get-WmiObject -Namespace root/WMI "
            f"-Class WmiMonitorBrightness).CurrentBrightness; "
            f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods)"
            f".WmiSetBrightness(1, [math]::Max(0,[math]::Min(100,$b+{inc})))\""
        )
    elif IS_MAC:
        if level and level.isdigit():
            val = int(level) / 100
            return f"brightness {val:.2f}"
        return "brightness +0.1" if action == "up" else "brightness -0.1"
    else:
        if level and level.isdigit():
            return (
                f"xrandr --output $(xrandr | grep ' connected' | head -1 | "
                f"cut -d' ' -f1) --brightness {int(level)/100:.1f}"
            )
        return "xbacklight -inc 10" if action == "up" else "xbacklight -dec 10"


# ── Default interface detection (Windows-aware) ──────────────────────────────

def get_default_gateway_iface() -> Optional[str]:
    """
    Return the name of the default gateway interface.
    Works on Linux (ip route), Windows (route print parse), macOS (netstat).
    """
    try:
        if IS_WINDOWS:
            proc = subprocess.run(
                ["powershell", "-Command",
                 "(Get-NetRoute -DestinationPrefix '0.0.0.0/0' | "
                 "Sort-Object RouteMetric | Select-Object -First 1).InterfaceAlias"],
                capture_output=True, text=True, timeout=10,
            )
            name = proc.stdout.strip()
            return name if name else None
        elif IS_MAC:
            proc = subprocess.run(
                ["route", "-n", "get", "default"],
                capture_output=True, text=True, timeout=5,
            )
            for line in proc.stdout.splitlines():
                if "interface:" in line:
                    return line.split(":")[-1].strip()
        else:
            proc = subprocess.run(
                ["ip", "route", "show", "default"],
                capture_output=True, text=True, timeout=5,
            )
            import re
            m = re.search(r"dev (\S+)", proc.stdout)
            return m.group(1) if m else None
    except Exception:
        return None


# ── ARP table ────────────────────────────────────────────────────────────────

def arp_table() -> list[dict]:
    """
    Return ARP cache entries as list of {ip, mac, interface}.
    Handles different arp output formats across platforms.
    """
    import re
    entries = []
    try:
        if IS_WINDOWS:
            proc = subprocess.run(["arp", "-a"], capture_output=True, text=True)
            for line in proc.stdout.splitlines():
                # Windows arp -a:  192.168.1.1     00-11-22-33-44-55   dynamic
                m = re.match(
                    r"\s+(\d+\.\d+\.\d+\.\d+)\s+([0-9a-f\-]+)\s+(\w+)", line
                )
                if m:
                    mac = m.group(2).replace("-", ":").lower()
                    entries.append({"ip": m.group(1), "mac": mac,
                                    "type": m.group(3), "interface": ""})
        else:
            proc = subprocess.run(["arp", "-n"], capture_output=True, text=True)
            for line in proc.stdout.splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 3 and parts[2] not in ("HWaddress", "incomplete"):
                    entries.append({
                        "ip": parts[0],
                        "mac": parts[2],
                        "interface": parts[-1] if len(parts) > 3 else "",
                    })
    except Exception as e:
        log.debug("arp_table failed: %s", e)
    return entries


# ── Routing table ─────────────────────────────────────────────────────────────

def routing_table() -> list[dict]:
    """Return routing table entries as plain dicts."""
    routes = []
    try:
        if IS_WINDOWS:
            proc = subprocess.run(
                ["powershell", "-Command",
                 "Get-NetRoute | Select-Object DestinationPrefix,NextHop,"
                 "InterfaceAlias,RouteMetric | ConvertTo-Csv -NoTypeInformation"],
                capture_output=True, text=True, timeout=15,
            )
            import csv, io
            reader = csv.DictReader(io.StringIO(proc.stdout))
            for row in reader:
                routes.append({
                    "route":     f"{row.get('DestinationPrefix','')} via {row.get('NextHop','')}",
                    "interface": row.get("InterfaceAlias", ""),
                    "metric":    row.get("RouteMetric", ""),
                })
        else:
            proc = subprocess.run(["ip", "route"], capture_output=True, text=True)
            for line in proc.stdout.splitlines():
                routes.append({"route": line.strip()})
    except Exception as e:
        log.debug("routing_table failed: %s", e)
    return routes


# ── Windows log paths ─────────────────────────────────────────────────────────

WINDOWS_LOG_PATHS = {
    "security": [r"C:\Windows\System32\winevt\Logs\Security.evtx"],
    "system":   [r"C:\Windows\System32\winevt\Logs\System.evtx"],
    "application": [r"C:\Windows\System32\winevt\Logs\Application.evtx"],
}
