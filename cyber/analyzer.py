"""
NEXUS LogAnalyzer — Threat Detection & Log Analysis
-----------------------------------------------------
Parses system logs for suspicious patterns:
- Failed logins / brute force attempts
- Privilege escalation
- Port scans detected
- Unusual process activity
- Known malicious IPs
"""

import os
import re
import subprocess
import glob
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional
from .toolkit import ToolKit


# ─────────────────────────────────────────────
#  THREAT SIGNATURES
# ─────────────────────────────────────────────
THREAT_PATTERNS = {
    "brute_force_ssh": {
        "pattern": r"(Failed password|Invalid user|authentication failure).*from (\d+\.\d+\.\d+\.\d+)",
        "severity": "HIGH",
        "description": "SSH brute force attempt",
        "ip_group": 2,
    },
    "sudo_abuse": {
        "pattern": r"sudo.*COMMAND=.*(?:chmod|passwd|shadow|visudo|/bin/sh|/bin/bash)",
        "severity": "HIGH",
        "description": "Suspicious sudo command execution",
    },
    "root_login": {
        "pattern": r"ROOT LOGIN|Accepted.*root@",
        "severity": "CRITICAL",
        "description": "Direct root login detected",
    },
    "port_scan_detected": {
        "pattern": r"(kernel.*SFW2|Firewall|IPTABLES|DROP|REJECT).*DPT=",
        "severity": "MEDIUM",
        "description": "Firewall blocked connection (possible port scan)",
    },
    "new_user_added": {
        "pattern": r"(useradd|adduser|new user|new account)",
        "severity": "MEDIUM",
        "description": "New user account created",
    },
    "service_crash": {
        "pattern": r"(segfault|core dumped|killed|OOM killer)",
        "severity": "MEDIUM",
        "description": "Service crash or OOM kill",
    },
    "cron_modification": {
        "pattern": r"(cron|crontab).*(?:REPLACE|ADD|DELETE|EDIT)",
        "severity": "MEDIUM",
        "description": "Cron job modification",
    },
    "ssh_success_unusual": {
        "pattern": r"Accepted (password|publickey) for .* from (\d+\.\d+\.\d+\.\d+)",
        "severity": "LOW",
        "description": "Successful SSH login",
        "ip_group": 2,
    },
    "connection_refused": {
        "pattern": r"connection refused|ECONNREFUSED",
        "severity": "LOW",
        "description": "Connection refused (possible scan)",
    },
}

# Common log file locations
LOG_PATHS = {
    "auth":    ["/var/log/auth.log", "/var/log/secure"],
    "syslog":  ["/var/log/syslog", "/var/log/messages"],
    "kern":    ["/var/log/kern.log"],
    "daemon":  ["/var/log/daemon.log"],
    "nginx":   ["/var/log/nginx/access.log", "/var/log/nginx/error.log"],
    "apache":  ["/var/log/apache2/access.log", "/var/log/apache2/error.log"],
    "fail2ban":["/var/log/fail2ban.log"],
    "dpkg":    ["/var/log/dpkg.log"],
}


class LogAnalyzer:
    """
    Analyzes system logs for security threats and anomalies.
    Intelligently locates available log files and parses them.
    """

    def __init__(self, toolkit: Optional[ToolKit] = None):
        self.tk = toolkit or ToolKit(verbose=False)

    # ─────────────────────────────────────────
    #  MAIN ANALYSIS
    # ─────────────────────────────────────────

    def analyze(self, hours: int = 24, log_types: Optional[list] = None) -> dict:
        """
        Full security analysis of system logs.

        Args:
            hours: How many hours back to analyze
            log_types: List of log types to check (default: all available)

        Returns:
            dict with threats, stats, recommendations
        """
        print(f"[Analyzer] Scanning logs from last {hours}h...")

        available_logs = self._find_available_logs(log_types)
        if not available_logs:
            return {
                "error": "No readable log files found",
                "hint": "Try running with sudo for /var/log access",
                "logs_checked": [],
            }

        all_threats = []
        stats = defaultdict(int)
        ip_frequency = defaultdict(int)

        cutoff = datetime.now() - timedelta(hours=hours)

        for log_type, log_path in available_logs.items():
            print(f"[Analyzer]   → {log_path}")
            entries = self._read_log(log_path)
            threats = self._scan_entries(entries, cutoff, ip_frequency)
            all_threats.extend(threats)
            stats[f"lines_{log_type}"] += len(entries)

        # Sort by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        all_threats.sort(key=lambda x: (severity_order.get(x["severity"], 9), x.get("timestamp", "")))

        # Top attacking IPs
        top_ips = sorted(ip_frequency.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "threats": all_threats,
            "threat_count": len(all_threats),
            "logs_analyzed": list(available_logs.values()),
            "stats": dict(stats),
            "top_attacker_ips": [{"ip": ip, "attempts": cnt} for ip, cnt in top_ips],
            "recommendations": self._generate_recommendations(all_threats, top_ips),
            "analysis_window_hours": hours,
        }

    def check_failed_logins(self) -> dict:
        """Specifically analyze failed login attempts."""
        auth_log = self._find_log("auth")
        if not auth_log:
            return {"error": "Auth log not accessible"}

        entries = self._read_log(auth_log)
        failures = defaultdict(list)

        for entry in entries:
            match = re.search(
                r"(Failed password|Invalid user).*from (\d+\.\d+\.\d+\.\d+)",
                entry
            )
            if match:
                ip = match.group(2)
                failures[ip].append(entry[:100])

        results = []
        for ip, attempts in sorted(failures.items(), key=lambda x: -len(x[1])):
            results.append({
                "ip": ip,
                "attempts": len(attempts),
                "severity": "CRITICAL" if len(attempts) > 50
                            else "HIGH" if len(attempts) > 10
                            else "MEDIUM",
            })

        return {
            "failed_login_sources": results,
            "total_ips": len(results),
            "total_attempts": sum(r["attempts"] for r in results),
        }

    def check_active_sessions(self) -> list[dict]:
        """Return currently logged-in users."""
        sessions = []
        try:
            proc = subprocess.run(["who"], capture_output=True, text=True)
            for line in proc.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    sessions.append({
                        "user": parts[0],
                        "terminal": parts[1],
                        "login_time": " ".join(parts[2:4]) if len(parts) > 3 else "",
                        "from": parts[4] if len(parts) > 4 else "local",
                    })
        except Exception:
            pass
        return sessions

    def check_suspicious_processes(self) -> list[dict]:
        """Look for suspicious running processes."""
        suspicious = []
        suspicious_keywords = [
            "nc", "netcat", "ncat", "socat",
            "meterpreter", "reverse_shell", "bind_shell",
            "cryptominer", "xmrig", "minerd",
            "ngrok", "frpc", "chisel",
        ]

        if self.tk.ensure("psutil"):
            try:
                import psutil
                for proc in psutil.process_iter(["pid", "name", "cmdline", "username"]):
                    try:
                        info = proc.info
                        name = info.get("name", "").lower()
                        cmdline = " ".join(info.get("cmdline", [])).lower()
                        for keyword in suspicious_keywords:
                            if keyword in name or keyword in cmdline:
                                suspicious.append({
                                    "pid": info["pid"],
                                    "name": info["name"],
                                    "user": info.get("username", ""),
                                    "cmdline": cmdline[:100],
                                    "flag": keyword,
                                    "severity": "HIGH",
                                })
                                break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except Exception as e:
                print(f"[Analyzer] psutil process check failed: {e}")

        return suspicious

    def check_open_ports_vs_services(self) -> dict:
        """Compare listening ports against known services for anomalies."""
        if not self.tk.ensure("psutil"):
            return {"error": "psutil not available"}

        try:
            import psutil
            listening = []
            for conn in psutil.net_connections(kind="inet"):
                if conn.status == "LISTEN":
                    port = conn.laddr.port
                    try:
                        proc = psutil.Process(conn.pid) if conn.pid else None
                        pname = proc.name() if proc else "unknown"
                    except Exception:
                        pname = "unknown"
                    listening.append({
                        "port": port,
                        "process": pname,
                        "pid": conn.pid,
                        "suspicious": port > 1024 and pname in ("python", "python3", "sh", "bash"),
                    })
            return {"listening_ports": listening}
        except Exception as e:
            return {"error": str(e)}

    # ─────────────────────────────────────────
    #  LOG DISCOVERY & READING
    # ─────────────────────────────────────────

    def _find_available_logs(self, log_types: Optional[list] = None) -> dict[str, str]:
        """Find all readable log files."""
        available = {}
        check_types = log_types or list(LOG_PATHS.keys())
        for log_type in check_types:
            path = self._find_log(log_type)
            if path:
                available[log_type] = path
        return available

    def _find_log(self, log_type: str) -> Optional[str]:
        """Find first readable log file of given type."""
        for path in LOG_PATHS.get(log_type, []):
            if os.path.exists(path) and os.access(path, os.R_OK):
                return path
        return None

    def _read_log(self, path: str, max_lines: int = 50000) -> list[str]:
        """Read log file, handle rotation."""
        lines = []
        try:
            with open(path, "r", errors="ignore") as f:
                lines = f.readlines()[-max_lines:]
        except PermissionError:
            print(f"[Analyzer] Permission denied: {path}")
        except Exception as e:
            print(f"[Analyzer] Error reading {path}: {e}")
        return [l.strip() for l in lines if l.strip()]

    # ─────────────────────────────────────────
    #  THREAT SCANNING
    # ─────────────────────────────────────────

    def _scan_entries(
        self, entries: list[str],
        cutoff: datetime,
        ip_freq: dict
    ) -> list[dict]:
        """Scan log entries against all threat patterns."""
        threats = []
        for entry in entries:
            for threat_name, sig in THREAT_PATTERNS.items():
                match = re.search(sig["pattern"], entry, re.IGNORECASE)
                if match:
                    threat = {
                        "type": threat_name,
                        "severity": sig["severity"],
                        "description": sig["description"],
                        "raw": entry[:200],
                        "timestamp": self._extract_timestamp(entry),
                    }
                    # Extract IP if pattern specifies
                    if "ip_group" in sig:
                        try:
                            ip = match.group(sig["ip_group"])
                            threat["source_ip"] = ip
                            ip_freq[ip] += 1
                        except IndexError:
                            pass
                    threats.append(threat)
                    break  # One threat per line
        return threats

    def _extract_timestamp(self, entry: str) -> str:
        """Extract timestamp from log entry."""
        # Common formats: "Jan 15 10:30:00", "2024-01-15T10:30:00"
        patterns = [
            r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})",
            r"(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})",
        ]
        for pat in patterns:
            match = re.search(pat, entry)
            if match:
                return match.group(1)
        return ""

    def _generate_recommendations(self, threats: list, top_ips: list) -> list[str]:
        """Generate actionable security recommendations."""
        recs = []
        threat_types = {t["type"] for t in threats}

        if "brute_force_ssh" in threat_types:
            recs.append("🔴 Install fail2ban to auto-block brute force: sudo apt install fail2ban")
            recs.append("🔴 Disable password auth for SSH, use key-based auth only")
            if top_ips:
                top_ip = top_ips[0][0]
                recs.append(f"🔴 Block top attacker: sudo iptables -A INPUT -s {top_ip} -j DROP")

        if "root_login" in threat_types:
            recs.append("🚨 CRITICAL: Direct root SSH detected — disable PermitRootLogin in /etc/ssh/sshd_config")

        if "sudo_abuse" in threat_types:
            recs.append("⚠️  Review sudoers file: sudo visudo — restrict unnecessary sudo permissions")

        if "port_scan_detected" in threat_types:
            recs.append("ℹ️  Port scan activity detected — consider enabling port knocking or moving SSH port")

        if not recs:
            recs.append("✅ No critical threats detected in the analysis window")

        return recs

    # ─────────────────────────────────────────
    #  FORMATTING
    # ─────────────────────────────────────────

    def format_report(self, analysis: dict) -> str:
        """Format analysis results as a security report."""
        if "error" in analysis:
            return f"\n[!] Analysis failed: {analysis['error']}\n    {analysis.get('hint', '')}"

        lines = [
            f"\n{'═'*60}",
            f"  NEXUS SECURITY ANALYSIS REPORT",
            f"  Window: Last {analysis.get('analysis_window_hours', 24)} hours",
            f"{'═'*60}",
        ]

        threats = analysis.get("threats", [])
        if not threats:
            lines.append("\n  ✅ No threats detected in analyzed logs\n")
        else:
            # Group by severity
            by_sev = defaultdict(list)
            for t in threats:
                by_sev[t["severity"]].append(t)

            for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                sev_threats = by_sev.get(sev, [])
                if not sev_threats:
                    continue
                icon = {"CRITICAL": "🚨", "HIGH": "🔴", "MEDIUM": "⚠️ ", "LOW": "ℹ️ "}.get(sev, "•")
                lines.append(f"\n  {icon} {sev} ({len(sev_threats)} events)")
                lines.append(f"  {'─'*40}")
                # Show unique types
                shown = {}
                for t in sev_threats:
                    key = t["type"]
                    if key not in shown:
                        shown[key] = {"desc": t["description"], "count": 0,
                                      "ips": set(), "sample": t["raw"]}
                    shown[key]["count"] += 1
                    if "source_ip" in t:
                        shown[key]["ips"].add(t["source_ip"])
                for key, info in shown.items():
                    lines.append(f"    • {info['desc']} — {info['count']}x")
                    if info["ips"]:
                        lines.append(f"      Sources: {', '.join(list(info['ips'])[:5])}")

        # Top attacking IPs
        top_ips = analysis.get("top_attacker_ips", [])
        if top_ips:
            lines.append(f"\n  TOP ATTACK SOURCES")
            lines.append(f"  {'─'*40}")
            for entry in top_ips[:5]:
                lines.append(f"    {entry['ip']:<18} {entry['attempts']:>5} attempts")

        # Recommendations
        recs = analysis.get("recommendations", [])
        if recs:
            lines.append(f"\n  RECOMMENDATIONS")
            lines.append(f"  {'─'*40}")
            for rec in recs:
                lines.append(f"  {rec}")

        # Stats
        lines.append(f"\n  Logs analyzed: {len(analysis.get('logs_analyzed', []))}")
        lines.append(f"{'═'*60}\n")
        return "\n".join(lines)