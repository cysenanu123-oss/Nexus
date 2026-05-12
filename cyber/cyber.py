"""
NEXUS CyberBrain — Natural Language Cyber Interface
-----------------------------------------------------
Understands what you want to do and picks the right tool.

  "scan my network"              → ARP discovery + port scan
  "what ports are open on X.X.X.X" → targeted nmap
  "check for suspicious activity"  → log analysis + process check
  "what devices are on my network" → host discovery
  "show toolkit status"            → tool inventory
  "install nmap"                   → auto-install
"""

import re
import socket
from typing import Optional
from .toolkit import ToolKit
from .scanner import PortScanner
from .network import NetworkIntel
from .analyzer import LogAnalyzer


# ─────────────────────────────────────────────
#  INTENT PATTERNS
# ─────────────────────────────────────────────
INTENT_PATTERNS = [
    # Network scanning
    {
        "intents": ["scan my network", "scan network", "network scan", "scan local"],
        "action": "full_network_scan",
    },
    {
        "intents": ["what devices", "who is on", "devices on", "hosts on", "find devices",
                    "discover hosts", "who's on", "what's on my network"],
        "action": "discover_devices",
    },
    {
        "intents": ["open ports on", "ports on", "scan ports", "check ports", "port scan",
                    "what ports", "scan host", "scan ip"],
        "action": "port_scan_target",
        "needs_target": True,
    },
    {
        "intents": ["quick scan", "fast scan"],
        "action": "quick_scan",
        "needs_target": True,
    },
    {
        "intents": ["full scan", "deep scan", "detailed scan", "service scan"],
        "action": "full_scan",
        "needs_target": True,
    },
    {
        "intents": ["stealth scan", "silent scan", "syn scan"],
        "action": "stealth_scan",
        "needs_target": True,
    },

    # Network info
    {
        "intents": ["my ip", "local ip", "ip address", "what is my ip", "show interfaces",
                    "network interfaces", "interface info"],
        "action": "show_interfaces",
    },
    {
        "intents": ["external ip", "public ip", "my public ip", "internet ip"],
        "action": "external_ip",
    },
    {
        "intents": ["arp table", "arp cache", "local arp"],
        "action": "show_arp",
    },
    {
        "intents": ["active connections", "open connections", "network connections",
                    "established connections"],
        "action": "show_connections",
    },
    {
        "intents": ["routing table", "routes", "ip routes"],
        "action": "show_routes",
    },
    {
        "intents": ["my subnet", "local subnet", "network range", "cidr"],
        "action": "show_subnet",
    },

    # Log analysis
    {
        "intents": ["suspicious activity", "check logs", "analyze logs", "security scan",
                    "intrusion", "threats", "attacks", "log analysis", "security report"],
        "action": "analyze_logs",
    },
    {
        "intents": ["failed login", "brute force", "ssh attacks", "login attempts"],
        "action": "check_logins",
    },
    {
        "intents": ["suspicious process", "malicious process", "check processes"],
        "action": "check_processes",
    },
    {
        "intents": ["active sessions", "logged in", "who is logged", "current users"],
        "action": "active_sessions",
    },
    {
        "intents": ["listening ports", "open ports local", "services running"],
        "action": "check_listening",
    },

    # Toolkit management
    {
        "intents": ["toolkit status", "tool status", "what tools", "available tools",
                    "installed tools", "show tools"],
        "action": "toolkit_status",
    },
    {
        "intents": ["install ", "download "],
        "action": "install_tool",
        "needs_target": True,
    },
    {
        "intents": ["best tool for", "recommend tool", "which tool"],
        "action": "recommend_tool",
        "needs_target": True,
    },
    {
        "intents": ["help", "what can you do", "commands", "cyber help"],
        "action": "show_help",
    },
]


class CyberBrain:
    """
    The intelligent interface for NEXUS cybersecurity operations.
    Parses natural language → dispatches to right module → formats results.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.tk = ToolKit(verbose=verbose)
        self.scanner = PortScanner(toolkit=self.tk)
        self.network = NetworkIntel(toolkit=self.tk)
        self.analyzer = LogAnalyzer(toolkit=self.tk)

    # ─────────────────────────────────────────
    #  MAIN ENTRY POINT
    # ─────────────────────────────────────────

    def run(self, command: str) -> str:
        """
        Parse a natural language command and execute the right operation.

        Args:
            command: Natural language instruction

        Returns:
            Formatted string result
        """
        print(f"\n[NEXUS Cyber] → {command}")
        cmd_lower = command.lower().strip()

        intent = self._parse_intent(cmd_lower)
        if not intent:
            return self._suggest_similar(cmd_lower)

        action = intent["action"]
        target = self._extract_target(cmd_lower) if intent.get("needs_target") else None

        # Dispatch
        return self._dispatch(action, target, command)

    # ─────────────────────────────────────────
    #  INTENT PARSING
    # ─────────────────────────────────────────

    def _parse_intent(self, cmd: str) -> Optional[dict]:
        """Match command to best intent."""
        best_match = None
        best_score = 0

        for pattern in INTENT_PATTERNS:
            for trigger in pattern["intents"]:
                if trigger in cmd:
                    score = len(trigger)  # Longer match = more specific
                    if score > best_score:
                        best_score = score
                        best_match = pattern

        return best_match

    def _extract_target(self, cmd: str) -> Optional[str]:
        """Extract IP, hostname, CIDR, or tool name from command."""
        # IP address
        ip_match = re.search(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?:/\d{1,2})?)\b", cmd)
        if ip_match:
            return ip_match.group(1)

        # CIDR range
        cidr_match = re.search(r"\b(\d+\.\d+\.\d+\.\d+/\d+)\b", cmd)
        if cidr_match:
            return cidr_match.group(1)

        # Hostname (simple heuristic)
        hostname_match = re.search(
            r"(?:on|scan|check|for)\s+([a-zA-Z0-9][a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,})\b", cmd
        )
        if hostname_match:
            return hostname_match.group(1)

        # Tool name (for install / recommend)
        tool_match = re.search(r"(?:install|download|for)\s+([a-zA-Z0-9_\-]+)", cmd)
        if tool_match:
            return tool_match.group(1)

        return None

    # ─────────────────────────────────────────
    #  ACTION DISPATCHER
    # ─────────────────────────────────────────

    def _dispatch(self, action: str, target: Optional[str], original: str) -> str:
        """Route action to the correct handler."""

        # ── Network Scanning ──────────────────
        if action == "full_network_scan":
            return self._do_full_network_scan()

        if action == "discover_devices":
            subnet = target or self.network.get_local_subnet()
            hosts = self.network.discover_hosts(subnet)
            return self.network.format_host_table(hosts)

        if action == "port_scan_target":
            if not target:
                return "[!] Please specify a target IP or hostname.\n    Example: scan ports on 192.168.1.1"
            results = self.scanner.scan_target(target)
            return self.scanner.format_results(results)

        if action == "quick_scan":
            tgt = target or self.network.get_local_subnet()
            if not tgt:
                return "[!] No target. Try: quick scan 192.168.1.1"
            results = self.scanner.quick_scan(tgt)
            return self.scanner.format_results(results)

        if action == "full_scan":
            if not target:
                return "[!] Please specify a target for full scan.\n    Example: full scan 192.168.1.1"
            results = self.scanner.full_scan(target)
            return self.scanner.format_results(results)

        if action == "stealth_scan":
            if not target:
                return "[!] Stealth scan needs a target IP."
            results = self.scanner.stealth_scan(target)
            return self.scanner.format_results(results)

        # ── Network Info ──────────────────────
        if action == "show_interfaces":
            ifaces = self.network.get_interfaces()
            return self.network.format_interfaces(ifaces)

        if action == "external_ip":
            ip = self.network.external_ip()
            return f"\n  Your external/public IP: {ip or 'Could not determine'}\n"

        if action == "show_arp":
            table = self.network.get_arp_table()
            if not table:
                return "  ARP table is empty or not accessible."
            lines = ["\n  ARP TABLE", "  " + "─" * 40]
            for entry in table:
                lines.append(f"  {entry['ip']:<18} {entry['mac']:<20} {entry.get('interface', '')}")
            return "\n".join(lines)

        if action == "show_connections":
            conns = self.network.get_active_connections()
            if not conns:
                return "  No active connections found."
            lines = ["\n  ACTIVE CONNECTIONS", "  " + "─" * 50,
                     f"  {'LOCAL':<25} {'REMOTE':<25} {'STATUS'}"]
            for c in conns[:30]:
                lines.append(f"  {c.get('local',''):<25} {c.get('remote',''):<25} {c.get('status','')}")
            return "\n".join(lines)

        if action == "show_routes":
            routes = self.network.get_routing_table()
            lines = ["\n  ROUTING TABLE", "  " + "─" * 50]
            for r in routes:
                lines.append(f"  {r['route']}")
            return "\n".join(lines)

        if action == "show_subnet":
            subnet = self.network.get_local_subnet()
            return f"\n  Local subnet: {subnet or 'Could not determine'}\n"

        # ── Log Analysis ──────────────────────
        if action == "analyze_logs":
            analysis = self.analyzer.analyze(hours=24)
            return self.analyzer.format_report(analysis)

        if action == "check_logins":
            result = self.analyzer.check_failed_logins()
            if "error" in result:
                return f"[!] {result['error']}"
            lines = ["\n  FAILED LOGIN ANALYSIS", "  " + "─" * 40]
            sources = result.get("failed_login_sources", [])
            if not sources:
                lines.append("  ✅ No failed logins found")
            else:
                lines.append(f"  Total attacking IPs: {result['total_ips']}")
                lines.append(f"  Total attempts: {result['total_attempts']}\n")
                for s in sources[:15]:
                    lines.append(f"  [{s['severity']:<8}] {s['ip']:<18} {s['attempts']:>5} attempts")
            return "\n".join(lines)

        if action == "check_processes":
            procs = self.analyzer.check_suspicious_processes()
            if not procs:
                return "\n  ✅ No suspicious processes detected\n"
            lines = ["\n  ⚠️  SUSPICIOUS PROCESSES DETECTED", "  " + "─" * 50]
            for p in procs:
                lines.append(f"  PID {p['pid']:<7} {p['name']:<20} [{p['flag']}]")
                lines.append(f"    User: {p.get('user','?')}  CMD: {p.get('cmdline','')[:60]}")
            return "\n".join(lines)

        if action == "active_sessions":
            sessions = self.analyzer.check_active_sessions()
            if not sessions:
                return "\n  No active user sessions found\n"
            lines = ["\n  ACTIVE SESSIONS", "  " + "─" * 50,
                     f"  {'USER':<15} {'TERMINAL':<12} {'TIME':<18} {'FROM'}"]
            for s in sessions:
                lines.append(f"  {s['user']:<15} {s['terminal']:<12} {s['login_time']:<18} {s['from']}")
            return "\n".join(lines)

        if action == "check_listening":
            result = self.analyzer.check_open_ports_vs_services()
            if "error" in result:
                return f"[!] {result['error']}"
            ports = result.get("listening_ports", [])
            lines = ["\n  LISTENING PORTS", "  " + "─" * 40,
                     f"  {'PORT':<8} {'PROCESS':<20} {'PID'}  {'NOTE'}"]
            for p in sorted(ports, key=lambda x: x["port"]):
                note = "⚠️  Suspicious" if p.get("suspicious") else ""
                lines.append(f"  {p['port']:<8} {p['process']:<20} {str(p['pid']):<6} {note}")
            return "\n".join(lines)

        # ── Toolkit Management ────────────────
        if action == "toolkit_status":
            return self.tk.status_report()

        if action == "install_tool":
            if not target:
                return "[!] Specify a tool to install. Example: install nmap"
            result = self.tk.install(target)
            if result.get("success"):
                return f"\n  ✅ {target} installed successfully via {result.get('method')}\n"
            return (f"\n  ✗ Could not install {target}\n"
                    f"  Manual install: {result.get('manual_hint', 'see docs')}\n")

        if action == "recommend_tool":
            query = target or original
            candidates = self.tk.best_tool_for(query)
            if not candidates:
                return f"\n  No tool recommendations found for: {query}\n"
            lines = [f"\n  TOOL RECOMMENDATIONS for: {query}", "  " + "─" * 40]
            for c in candidates[:5]:
                status = "✓ installed" if c["installed"] else "✗ not installed"
                lines.append(f"\n  [{c['info']['category']}] {c['name']} — {status}")
                lines.append(f"    {c['info']['description']}")
                if not c["installed"]:
                    lines.append(f"    Install: {self.tk._manual_install_hint(c['name'])}")
            return "\n".join(lines)

        if action == "show_help":
            return self._help_text()

        return f"[!] Unknown action: {action}"

    # ─────────────────────────────────────────
    #  COMPOUND OPERATIONS
    # ─────────────────────────────────────────

    def _do_full_network_scan(self) -> str:
        """Full network scan: interface info + host discovery + quick port scan."""
        output = []

        # 1. Show local interfaces
        ifaces = self.network.get_interfaces()
        output.append(self.network.format_interfaces(ifaces))

        # 2. Detect subnet
        subnet = self.network.get_local_subnet()
        if not subnet:
            output.append("\n[!] Could not determine local subnet for host discovery")
            return "\n".join(output)

        output.append(f"\n  Scanning subnet: {subnet}")

        # 3. Discover hosts
        hosts = self.network.discover_hosts(subnet)
        output.append(self.network.format_host_table(hosts))

        # 4. Quick port scan on discovered hosts (up to 5)
        if hosts:
            output.append("\n  Running quick port scan on discovered hosts...\n")
            for host in hosts[:5]:
                ip = host.get("ip", "")
                if ip and not ip.startswith("127."):
                    result = self.scanner.quick_scan(ip)
                    output.append(self.scanner.format_results(result))

        return "\n".join(output)

    # ─────────────────────────────────────────
    #  UTILITIES
    # ─────────────────────────────────────────

    def _suggest_similar(self, cmd: str) -> str:
        """Suggest similar commands when no match found."""
        return (
            f"\n  [?] I didn't understand: '{cmd}'\n"
            f"  Type 'help' to see available commands.\n"
            f"\n  Quick examples:\n"
            f"    scan my network\n"
            f"    what ports are open on 192.168.1.1\n"
            f"    check for suspicious activity\n"
            f"    show toolkit status\n"
        )

    def _help_text(self) -> str:
        return """
╔═══════════════════════════════════════════════════════╗
║            NEXUS CYBERSECURITY COMMANDS               ║
╚═══════════════════════════════════════════════════════╝

  NETWORK SCANNING
  ─────────────────────────────────────────────────────
  scan my network                 Full scan: interfaces + hosts + ports
  what devices are on my network  ARP host discovery only
  scan ports on 192.168.1.1       Targeted port scan
  quick scan 192.168.1.1          Fast top-100 ports scan
  full scan 192.168.1.1           Deep scan, all ports + services
  stealth scan 192.168.1.1        SYN stealth scan (needs root)

  NETWORK INTELLIGENCE
  ─────────────────────────────────────────────────────
  show interfaces                 All network interfaces + IPs
  my external ip                  Public IP address
  show arp table                  Current ARP cache
  active connections              Established TCP/UDP connections
  show routing table              IP routes
  my subnet                       Local CIDR range

  LOG ANALYSIS & THREAT DETECTION
  ─────────────────────────────────────────────────────
  check for suspicious activity   Full 24h security analysis
  show failed logins              Brute force + login attempts
  check suspicious processes      Scan for reverse shells, miners
  active sessions                 Who is currently logged in
  check listening ports           What services are listening

  TOOLKIT MANAGEMENT
  ─────────────────────────────────────────────────────
  show toolkit status             All tools + install status
  install nmap                    Auto-install a tool
  best tool for port scanning     Get tool recommendations

═══════════════════════════════════════════════════════
"""