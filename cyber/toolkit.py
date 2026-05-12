"""
NEXUS ToolKit — Smart Tool Registry
------------------------------------
Knows which tool fits which job.
Checks if installed. Auto-installs if missing.
"""

import subprocess
import shutil
import sys
import json
import os
from typing import Optional


# ─────────────────────────────────────────────
#  MASTER TOOL REGISTRY
#  Each entry: what it does, how to check, how to install
# ─────────────────────────────────────────────
TOOL_REGISTRY = {
    # ── Network Scanning ──────────────────────
    "nmap": {
        "category": "network_scan",
        "description": "Network mapper — port scanning, OS detection, service versioning",
        "use_cases": ["port scan", "open ports", "service detection", "os detection", "network scan"],
        "check_cmd": ["nmap", "--version"],
        "install_apt": "nmap",
        "install_pip": None,
        "python_pkg": None,
        "priority": 10,
    },
    "masscan": {
        "category": "network_scan",
        "description": "Ultra-fast port scanner — scans entire internet in minutes",
        "use_cases": ["fast scan", "large network", "bulk port scan"],
        "check_cmd": ["masscan", "--version"],
        "install_apt": "masscan",
        "install_pip": None,
        "python_pkg": None,
        "priority": 7,
    },
    "python-nmap": {
        "category": "network_scan",
        "description": "Python wrapper for nmap — programmatic scanning",
        "use_cases": ["python scan", "automated scan", "scripted scan"],
        "check_cmd": None,
        "install_apt": None,
        "install_pip": "python-nmap",
        "python_pkg": "nmap",
        "priority": 8,
    },

    # ── Network Discovery ─────────────────────
    "arp-scan": {
        "category": "network_discovery",
        "description": "ARP scanner — discovers all live hosts on local network",
        "use_cases": ["devices on network", "arp scan", "host discovery", "who is on my network"],
        "check_cmd": ["arp-scan", "--version"],
        "install_apt": "arp-scan",
        "install_pip": None,
        "python_pkg": None,
        "priority": 10,
    },
    "scapy": {
        "category": "network_discovery",
        "description": "Python packet manipulation — ARP, sniffing, custom packets",
        "use_cases": ["packet crafting", "arp discovery", "sniffing", "custom packets"],
        "check_cmd": None,
        "install_apt": None,
        "install_pip": "scapy",
        "python_pkg": "scapy",
        "priority": 9,
    },
    "netdiscover": {
        "category": "network_discovery",
        "description": "Active/passive ARP reconnaissance",
        "use_cases": ["passive discovery", "arp recon"],
        "check_cmd": ["netdiscover", "-h"],
        "install_apt": "netdiscover",
        "install_pip": None,
        "python_pkg": None,
        "priority": 7,
    },

    # ── Vulnerability Scanning ────────────────
    "nikto": {
        "category": "vuln_scan",
        "description": "Web server vulnerability scanner",
        "use_cases": ["web vuln", "web scan", "http vulnerabilities", "nikto"],
        "check_cmd": ["nikto", "-Version"],
        "install_apt": "nikto",
        "install_pip": None,
        "python_pkg": None,
        "priority": 9,
    },
    "openvas": {
        "category": "vuln_scan",
        "description": "Full vulnerability assessment system",
        "use_cases": ["full vuln scan", "vulnerability assessment", "cve scan"],
        "check_cmd": ["openvas", "--version"],
        "install_apt": "openvas",
        "install_pip": None,
        "python_pkg": None,
        "priority": 8,
    },

    # ── Log Analysis ──────────────────────────
    "logwatch": {
        "category": "log_analysis",
        "description": "System log analyzer — summarizes suspicious activity",
        "use_cases": ["analyze logs", "suspicious activity", "log summary", "system events"],
        "check_cmd": ["logwatch", "--version"],
        "install_apt": "logwatch",
        "install_pip": None,
        "python_pkg": None,
        "priority": 8,
    },
    "fail2ban": {
        "category": "log_analysis",
        "description": "Intrusion prevention — detects brute force from logs",
        "use_cases": ["brute force", "ssh attacks", "login attempts", "intrusion detection"],
        "check_cmd": ["fail2ban-client", "--version"],
        "install_apt": "fail2ban",
        "install_pip": None,
        "python_pkg": None,
        "priority": 9,
    },

    # ── Password & Crypto ─────────────────────
    "hashcat": {
        "category": "password",
        "description": "World's fastest password cracker — GPU-accelerated",
        "use_cases": ["crack hash", "password recovery", "hash cracking"],
        "check_cmd": ["hashcat", "--version"],
        "install_apt": "hashcat",
        "install_pip": None,
        "python_pkg": None,
        "priority": 9,
    },
    "john": {
        "category": "password",
        "description": "John the Ripper — password strength auditing",
        "use_cases": ["crack password", "audit password", "john the ripper"],
        "check_cmd": ["john", "--version"],
        "install_apt": "john",
        "install_pip": None,
        "python_pkg": None,
        "priority": 8,
    },

    # ── Traffic Analysis ──────────────────────
    "wireshark": {
        "category": "traffic_analysis",
        "description": "Network protocol analyzer — deep packet inspection",
        "use_cases": ["capture traffic", "packet analysis", "protocol analysis", "wireshark"],
        "check_cmd": ["wireshark", "--version"],
        "install_apt": "wireshark",
        "install_pip": None,
        "python_pkg": None,
        "priority": 9,
    },
    "tcpdump": {
        "category": "traffic_analysis",
        "description": "CLI packet capture and analysis",
        "use_cases": ["capture packets", "dump traffic", "network traffic"],
        "check_cmd": ["tcpdump", "--version"],
        "install_apt": "tcpdump",
        "install_pip": None,
        "python_pkg": None,
        "priority": 10,
    },
    "pyshark": {
        "category": "traffic_analysis",
        "description": "Python wrapper for tshark/wireshark — programmatic packet analysis",
        "use_cases": ["python packet analysis", "automated traffic analysis"],
        "check_cmd": None,
        "install_apt": None,
        "install_pip": "pyshark",
        "python_pkg": "pyshark",
        "priority": 7,
    },

    # ── Web Exploitation ─────────────────────
    "sqlmap": {
        "category": "web_exploit",
        "description": "Automatic SQL injection detection and exploitation",
        "use_cases": ["sql injection", "database exploit", "sqlmap"],
        "check_cmd": ["sqlmap", "--version"],
        "install_apt": "sqlmap",
        "install_pip": None,
        "python_pkg": None,
        "priority": 9,
    },

    # ── Wireless ─────────────────────────────
    "aircrack-ng": {
        "category": "wireless",
        "description": "WiFi network security auditing suite",
        "use_cases": ["wifi audit", "wireless security", "wpa crack", "aircrack"],
        "check_cmd": ["aircrack-ng", "--help"],
        "install_apt": "aircrack-ng",
        "install_pip": None,
        "python_pkg": None,
        "priority": 9,
    },

    # ── OSINT ─────────────────────────────────
    "theHarvester": {
        "category": "osint",
        "description": "Email, subdomain, host, port, banner and employee name harvesting",
        "use_cases": ["osint", "email harvesting", "subdomain enum", "recon"],
        "check_cmd": ["theHarvester", "-h"],
        "install_apt": None,
        "install_pip": "theHarvester",
        "python_pkg": None,
        "priority": 8,
    },
    "shodan": {
        "category": "osint",
        "description": "Python Shodan API client — search internet-connected devices",
        "use_cases": ["shodan", "internet recon", "exposed devices"],
        "check_cmd": None,
        "install_apt": None,
        "install_pip": "shodan",
        "python_pkg": "shodan",
        "priority": 8,
    },

    # ── Utilities ─────────────────────────────
    "whois": {
        "category": "recon",
        "description": "Domain registration lookup",
        "use_cases": ["domain info", "whois", "ip owner", "registrar"],
        "check_cmd": ["whois", "--version"],
        "install_apt": "whois",
        "install_pip": None,
        "python_pkg": None,
        "priority": 8,
    },
    "dig": {
        "category": "recon",
        "description": "DNS lookup utility",
        "use_cases": ["dns lookup", "dns records", "resolve domain"],
        "check_cmd": ["dig", "-v"],
        "install_apt": "dnsutils",
        "install_pip": None,
        "python_pkg": None,
        "priority": 9,
    },
    "netstat": {
        "category": "recon",
        "description": "Network connections, routing tables, interface stats",
        "use_cases": ["active connections", "listening ports", "network stats"],
        "check_cmd": ["netstat", "--version"],
        "install_apt": "net-tools",
        "install_pip": None,
        "python_pkg": None,
        "priority": 10,
    },
    "ipinfo": {
        "category": "recon",
        "description": "Python IP geolocation and info lookup",
        "use_cases": ["ip info", "geolocate ip", "ip lookup"],
        "check_cmd": None,
        "install_apt": None,
        "install_pip": "ipinfo",
        "python_pkg": "ipinfo",
        "priority": 7,
    },
    "requests": {
        "category": "utility",
        "description": "HTTP library for Python — web requests",
        "use_cases": ["http request", "web request", "api call"],
        "check_cmd": None,
        "install_apt": None,
        "install_pip": "requests",
        "python_pkg": "requests",
        "priority": 6,
    },
    "psutil": {
        "category": "utility",
        "description": "Process and system utilities — CPU, memory, network interfaces",
        "use_cases": ["system info", "process list", "network interfaces", "cpu usage"],
        "check_cmd": None,
        "install_apt": None,
        "install_pip": "psutil",
        "python_pkg": "psutil",
        "priority": 8,
    },
}


class ToolKit:
    """
    Smart tool manager for NEXUS cyber operations.
    - Recommends the best tool for any given task
    - Checks installation status
    - Auto-installs missing tools
    - Provides usage guidance
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._status_cache: dict[str, bool] = {}

    # ─────────────────────────────────────────
    #  TOOL SELECTION
    # ─────────────────────────────────────────

    def best_tool_for(self, task: str) -> list[dict]:
        """
        Given a task description, return ranked list of suitable tools.
        Returns top matches sorted by relevance score × priority.
        """
        task_lower = task.lower()
        matches = []

        for tool_name, info in TOOL_REGISTRY.items():
            score = 0
            # Check use_case keyword matches
            for use_case in info["use_cases"]:
                if use_case in task_lower:
                    score += 10
                else:
                    # Partial word match
                    for word in use_case.split():
                        if word in task_lower:
                            score += 3

            # Category match bonus
            if info["category"].replace("_", " ") in task_lower:
                score += 5

            if score > 0:
                matches.append({
                    "name": tool_name,
                    "score": score * info["priority"],
                    "info": info,
                    "installed": self.is_installed(tool_name),
                })

        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:5]  # Top 5

    def recommend(self, task: str) -> Optional[dict]:
        """Return the single best tool for a task, preferring already-installed ones."""
        candidates = self.best_tool_for(task)
        if not candidates:
            return None

        # Prefer installed tools with high score
        installed = [c for c in candidates if c["installed"]]
        if installed:
            return installed[0]
        return candidates[0]

    # ─────────────────────────────────────────
    #  INSTALLATION CHECK
    # ─────────────────────────────────────────

    def is_installed(self, tool_name: str) -> bool:
        """Check if a tool is available on this machine."""
        if tool_name in self._status_cache:
            return self._status_cache[tool_name]

        info = TOOL_REGISTRY.get(tool_name)
        if not info:
            result = shutil.which(tool_name) is not None
            self._status_cache[tool_name] = result
            return result

        # Check via python import
        if info.get("python_pkg"):
            try:
                __import__(info["python_pkg"])
                self._status_cache[tool_name] = True
                return True
            except ImportError:
                pass

        # Check via command
        if info.get("check_cmd"):
            try:
                subprocess.run(
                    info["check_cmd"],
                    capture_output=True,
                    timeout=5
                )
                self._status_cache[tool_name] = True
                return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        # Fallback: shutil which
        binary = tool_name.split("/")[-1]
        result = shutil.which(binary) is not None
        self._status_cache[tool_name] = result
        return result

    def scan_installed(self) -> dict[str, bool]:
        """Check all registered tools and return installation status."""
        results = {}
        for tool_name in TOOL_REGISTRY:
            results[tool_name] = self.is_installed(tool_name)
        self._status_cache.update(results)
        return results

    # ─────────────────────────────────────────
    #  AUTO-INSTALL
    # ─────────────────────────────────────────

    def install(self, tool_name: str) -> dict:
        """
        Try to install a tool. Returns result dict with success, method, output.
        Tries pip first (faster, no sudo), then apt.
        """
        info = TOOL_REGISTRY.get(tool_name)
        if not info:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        if self.is_installed(tool_name):
            return {"success": True, "method": "already_installed", "tool": tool_name}

        result = {"tool": tool_name, "success": False, "attempts": []}

        # ── Try pip install ───────────────────
        if info.get("install_pip"):
            pkg = info["install_pip"]
            if self.verbose:
                print(f"[NEXUS] Installing {tool_name} via pip ({pkg})...")
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                    capture_output=True, text=True, timeout=120
                )
                if proc.returncode == 0:
                    # Invalidate cache
                    self._status_cache.pop(tool_name, None)
                    if self.verbose:
                        print(f"[NEXUS] ✓ {tool_name} installed via pip")
                    return {"success": True, "method": "pip", "tool": tool_name, "pkg": pkg}
                else:
                    result["attempts"].append({"method": "pip", "error": proc.stderr[:300]})
            except subprocess.TimeoutExpired:
                result["attempts"].append({"method": "pip", "error": "timeout"})

        # ── Try apt-get install ───────────────
        if info.get("install_apt"):
            pkg = info["install_apt"]
            if self.verbose:
                print(f"[NEXUS] Installing {tool_name} via apt ({pkg})...")
            try:
                # Try without sudo first (if already root)
                for cmd in [
                    ["apt-get", "install", "-y", pkg],
                    ["sudo", "apt-get", "install", "-y", pkg],
                ]:
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=180
                    )
                    if proc.returncode == 0:
                        self._status_cache.pop(tool_name, None)
                        if self.verbose:
                            print(f"[NEXUS] ✓ {tool_name} installed via apt")
                        return {"success": True, "method": "apt", "tool": tool_name, "pkg": pkg}
                    result["attempts"].append({"method": " ".join(cmd[:2]), "error": proc.stderr[:200]})
            except subprocess.TimeoutExpired:
                result["attempts"].append({"method": "apt", "error": "timeout"})

        result["error"] = "All install methods failed"
        result["manual_hint"] = self._manual_install_hint(tool_name)
        return result

    def ensure(self, tool_name: str) -> bool:
        """
        Ensure a tool is available. Install if missing.
        Returns True if tool is ready to use.
        """
        if self.is_installed(tool_name):
            return True
        result = self.install(tool_name)
        return result.get("success", False)

    # ─────────────────────────────────────────
    #  UTILITIES
    # ─────────────────────────────────────────

    def list_by_category(self) -> dict[str, list]:
        """Return all tools grouped by category."""
        categories: dict[str, list] = {}
        for name, info in TOOL_REGISTRY.items():
            cat = info["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({"name": name, "description": info["description"]})
        return categories

    def status_report(self) -> str:
        """Pretty-print all tools and their install status."""
        statuses = self.scan_installed()
        lines = ["\n╔══════════════════════════════════════════╗",
                 "║       NEXUS CYBER TOOLKIT STATUS          ║",
                 "╚══════════════════════════════════════════╝\n"]

        by_cat = self.list_by_category()
        for category, tools in sorted(by_cat.items()):
            lines.append(f"  [{category.upper().replace('_',' ')}]")
            for t in tools:
                name = t["name"]
                icon = "✓" if statuses.get(name) else "✗"
                status = "installed" if statuses.get(name) else "not found"
                lines.append(f"    {icon}  {name:<20} — {t['description'][:45]}")
            lines.append("")

        installed_count = sum(1 for v in statuses.values() if v)
        lines.append(f"  {installed_count}/{len(statuses)} tools installed")
        return "\n".join(lines)

    def _manual_install_hint(self, tool_name: str) -> str:
        info = TOOL_REGISTRY.get(tool_name, {})
        hints = []
        if info.get("install_pip"):
            hints.append(f"pip install {info['install_pip']}")
        if info.get("install_apt"):
            hints.append(f"sudo apt-get install {info['install_apt']}")
        return " | ".join(hints) if hints else f"Search: how to install {tool_name}"