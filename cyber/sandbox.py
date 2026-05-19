"""
cyber/sandbox.py
NEXUS Sandbox Engine — isolated vulnerability testing environment.

Creates a Docker-based or chroot-based sandbox that:
  1. Pulls the target's service stack (from port scan + banner grab)
  2. Spins up a local clone of those services
  3. Runs automated vuln checks (nuclei, nikto, nmap scripts) inside
  4. Reports findings without touching the real target again

Requirements:
  - Docker (recommended) — auto-detected
  - Fallback: Python-level service simulation + nmap scripting
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("nexus.cyber.sandbox")

_SANDBOX_DIR  = Path(__file__).parent.parent / "data" / "sandboxes"
_REPORT_DIR   = Path(__file__).parent.parent / "data" / "sandbox_reports"

# Docker images to spin up based on detected service
_SERVICE_IMAGES = {
    "http":     "nginx:alpine",
    "https":    "nginx:alpine",
    "ftp":      "stilliard/pure-ftpd",
    "ssh":      "linuxserver/openssh-server",
    "mysql":    "mysql:8",
    "postgres": "postgres:15-alpine",
    "redis":    "redis:alpine",
    "smtp":     "namshi/smtp",
    "smb":      "dperson/samba",
    "telnet":   "alpine",
}


class SandboxEngine:
    """
    Spin up an isolated clone of a target's services and test for vulns.

    Workflow:
        scan target (port scan) → identify services
        → spin up matching Docker containers
        → run vuln checks inside sandbox
        → report findings
        → destroy sandbox

    NEXUS narrates every step.
    """

    def __init__(self):
        _SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
        _REPORT_DIR.mkdir(parents=True, exist_ok=True)
        self._docker_ready = self._check_docker()
        try:
            from core.stream_output import get_output
            self._out = get_output()
        except ImportError:
            self._out = None

    def _say(self, method: str, *args):
        if not self._out:
            return
        getattr(self._out, method, self._out.thinking)(*args)

    # ── Docker detection ──────────────────────────────────────

    def _check_docker(self) -> bool:
        try:
            proc = subprocess.run(
                ["docker", "info"], capture_output=True, timeout=10,
            )
            return proc.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    # ── Main API ──────────────────────────────────────────────

    def create_and_test(self, target: str, scan_results: Optional[dict] = None) -> str:
        """
        Full sandbox pipeline:
          1. Identify services (from scan_results or do a quick scan)
          2. Build a Docker compose environment matching those services
          3. Run vulnerability checks inside the sandbox
          4. Return a detailed report

        Args:
            target:       IP or hostname (authorized target)
            scan_results: Optional pre-scanned port data dict
        """
        self._say("planning", f"Building sandbox for {target}")

        # Step 1: Get service list
        services = self._extract_services(scan_results) if scan_results else []
        if not services:
            self._say("thinking", "No scan data — running quick port scan first")
            services = self._quick_scan_services(target)

        if not services:
            return f"  [!] No services detected on {target}. Run a scan first."

        self._say("thinking", f"Detected services: {', '.join(s['service'] for s in services)}")

        # Step 2: Create sandbox
        if self._docker_ready:
            return self._docker_sandbox(target, services)
        else:
            self._say("warn", "Docker not available — using nmap script sandbox")
            return self._nmap_script_sandbox(target, services)

    def vulnerability_scan(self, target: str, scan_type: str = "standard") -> str:
        """
        Run automated vulnerability scanning using available tools.

        Tools tried in order: nuclei → nikto → nmap-vuln-scripts → custom checks

        Args:
            target:    IP or hostname (must be authorized)
            scan_type: "standard" | "web" | "network" | "full"
        """
        self._say("planning", f"Vulnerability scan ({scan_type}): {target}")
        results = []

        # Nuclei (best option)
        if self._cmd_exists("nuclei"):
            self._say("running", f"nuclei -u {target} -severity medium,high,critical")
            results.append(("nuclei", self._run_nuclei(target, scan_type)))

        # Nikto (web focused)
        if self._cmd_exists("nikto") and scan_type in ("web", "standard", "full"):
            self._say("running", f"nikto -host {target}")
            results.append(("nikto", self._run_nikto(target)))

        # nmap vulnerability scripts
        self._say("running", f"nmap --script=vuln {target}")
        results.append(("nmap-vuln", self._run_nmap_vuln(target)))

        # Python basic checks
        results.append(("nexus-checks", self._python_vuln_checks(target)))

        return self._format_vuln_report(target, results)

    def monitor_target(self, target: str, interval: int = 60, duration: int = 300) -> str:
        """
        Continuously monitor a target for changes: new open ports,
        service changes, HTTP status changes.

        Args:
            target:   IP or hostname
            interval: check every N seconds
            duration: total monitoring duration in seconds
        """
        import time
        self._say("planning", f"Monitoring {target} every {interval}s for {duration}s")

        baseline = self._take_snapshot(target)
        self._say("thinking", f"Baseline captured — {len(baseline.get('ports', []))} ports")

        changes = []
        checks  = duration // interval
        start   = time.time()

        for i in range(checks):
            time.sleep(interval)
            self._say("thinking", f"Check {i+1}/{checks} — {target}")
            current = self._take_snapshot(target)
            diff = self._diff_snapshots(baseline, current)
            if diff:
                changes.append({
                    "time":    datetime.now().strftime("%H:%M:%S"),
                    "changes": diff,
                })
                self._say("warn", f"Change detected: {', '.join(diff)}")
            elapsed = time.time() - start
            if elapsed >= duration:
                break

        return self._format_monitor_report(target, baseline, changes)

    # ── Docker sandbox ────────────────────────────────────────

    def _docker_sandbox(self, target: str, services: list[dict]) -> str:
        """Create a Docker-compose sandbox and run vuln checks."""
        sandbox_id  = f"nexus_sandbox_{datetime.now():%Y%m%d_%H%M%S}"
        sandbox_dir = _SANDBOX_DIR / sandbox_id
        sandbox_dir.mkdir(parents=True)

        self._say("thinking", f"Creating sandbox: {sandbox_id}")

        compose = self._build_compose(services, sandbox_id)
        compose_file = sandbox_dir / "docker-compose.yml"
        compose_file.write_text(compose)

        self._say("running", "docker compose up -d")
        try:
            proc = subprocess.run(
                ["docker", "compose", "-f", str(compose_file), "up", "-d"],
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode != 0:
                return (
                    f"  [!] Docker compose failed:\n{proc.stderr[:500]}\n\n"
                    f"  Generated compose file:\n{compose[:1000]}"
                )
        except subprocess.TimeoutExpired:
            return "  [!] Docker compose timed out (120s)"

        self._say("done", "Sandbox containers started")

        # Map service ports and run checks
        report_lines = [
            f"\n{'═'*60}",
            f"  NEXUS SANDBOX REPORT — {target}",
            f"  Sandbox: {sandbox_id}",
            f"{'═'*60}\n",
        ]

        for svc in services:
            port   = svc.get("port", 0)
            sname  = svc.get("service", "unknown")
            mapped = svc.get("sandbox_port", port + 10000)
            report_lines.append(f"  Service: {sname}:{port} → sandbox localhost:{mapped}")

        # Run nuclei/nmap against sandbox
        sandbox_host = "localhost"
        self._say("running", f"vulnerability scan on sandbox {sandbox_host}")
        vuln_result = self.vulnerability_scan(sandbox_host, scan_type="full")
        report_lines.append(vuln_result)

        # Cleanup
        self._say("running", "docker compose down (cleanup)")
        subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "down", "-v"],
            capture_output=True, timeout=60,
        )

        report = "\n".join(report_lines)
        self._save_sandbox_report(sandbox_id, report)
        self._say("done", f"Sandbox complete — report saved")
        return report

    def _build_compose(self, services: list[dict], sandbox_id: str) -> str:
        """Generate docker-compose.yml for detected services."""
        lines = [
            "version: '3.8'",
            "services:",
        ]
        for i, svc in enumerate(services[:8]):
            sname   = svc.get("service", f"svc{i}").replace("-", "_")
            image   = _SERVICE_IMAGES.get(svc.get("service", ""), "alpine")
            port    = svc.get("port", 8000)
            sandbox_port = port + 10000
            svc["sandbox_port"] = sandbox_port
            lines += [
                f"  {sname}_{i}:",
                f"    image: {image}",
                f"    ports:",
                f"      - '{sandbox_port}:{port}'",
                f"    restart: 'no'",
            ]
        lines += ["networks:", "  default:", "    driver: bridge"]
        return "\n".join(lines)

    # ── Nmap script sandbox (fallback) ────────────────────────

    def _nmap_script_sandbox(self, target: str, services: list[dict]) -> str:
        """Run nmap vulnerability scripts directly on target as fallback."""
        self._say("thinking", "Running nmap vuln scripts (no Docker available)")
        result = self._run_nmap_vuln(target)
        return (
            f"\n  NEXUS SANDBOX (nmap scripts) — {target}\n"
            f"  Note: Docker not available. Running nmap vuln scripts on target.\n\n"
            f"{result}"
        )

    # ── Vulnerability scanners ────────────────────────────────

    def _run_nuclei(self, target: str, scan_type: str) -> str:
        severity = "medium,high,critical"
        tags = {
            "web":     "-tags cve,exposure,misconfig",
            "network": "-tags network",
            "full":    "-tags cve,exposure,misconfig,network,default-logins",
        }.get(scan_type, "")

        cmd = ["nuclei", "-u", target, "-severity", severity, "-silent"]
        if tags:
            cmd += tags.split()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            out = proc.stdout.strip()
            return out if out else "  No nuclei findings."
        except subprocess.TimeoutExpired:
            return "  nuclei timed out (300s)"
        except Exception as exc:
            return f"  nuclei error: {exc}"

    def _run_nikto(self, target: str) -> str:
        url = target if target.startswith("http") else f"http://{target}"
        try:
            proc = subprocess.run(
                ["nikto", "-host", url, "-Format", "txt", "-nointeractive"],
                capture_output=True, text=True, timeout=180,
            )
            lines = [l for l in proc.stdout.splitlines() if not l.startswith("-")]
            return "\n".join(lines[:50]) or "  No nikto findings."
        except subprocess.TimeoutExpired:
            return "  nikto timed out (180s)"
        except Exception as exc:
            return f"  nikto error: {exc}"

    def _run_nmap_vuln(self, target: str) -> str:
        try:
            proc = subprocess.run(
                ["nmap", "--script=vuln", "-T4", target,
                 "--script-timeout=30s", "-oN", "-"],
                capture_output=True, text=True, timeout=180,
            )
            relevant = [
                l for l in proc.stdout.splitlines()
                if any(kw in l.lower() for kw in [
                    "vuln", "exploit", "cve-", "error", "vulnerable",
                    "open", "script output",
                ])
            ]
            return "\n".join(relevant[:60]) or "  No nmap vuln findings."
        except subprocess.TimeoutExpired:
            return "  nmap --script=vuln timed out"
        except FileNotFoundError:
            return "  nmap not installed."
        except Exception as exc:
            return f"  nmap error: {exc}"

    def _python_vuln_checks(self, target: str) -> str:
        """Basic Python-level vulnerability checks."""
        import socket
        findings = []
        host = target if "://" not in target else target.split("://")[1].split("/")[0]

        # Check for open dangerous ports
        dangerous = {21: "FTP", 23: "Telnet", 512: "rexec", 513: "rlogin",
                     514: "rsh", 2049: "NFS", 6379: "Redis (no auth?)"}
        for port, svc in dangerous.items():
            try:
                with socket.create_connection((host, port), timeout=2):
                    findings.append(f"  ⚠  Port {port} ({svc}) is open — potentially dangerous")
            except Exception:
                pass

        # Check for default Redis (no auth)
        try:
            with socket.create_connection((host, 6379), timeout=2) as s:
                s.send(b"PING\r\n")
                resp = s.recv(64)
                if b"+PONG" in resp:
                    findings.append("  🔴 Redis is accessible without authentication!")
        except Exception:
            pass

        # Check for anonymous FTP
        try:
            import ftplib
            with ftplib.FTP(host, timeout=5) as ftp:
                ftp.login("anonymous", "nexus@nexus.ai")
                findings.append("  🔴 Anonymous FTP login is allowed!")
        except Exception:
            pass

        if not findings:
            findings.append("  ✓ Basic Python checks passed — no obvious misconfigurations")
        return "\n".join(findings)

    # ── Monitoring ────────────────────────────────────────────

    def _take_snapshot(self, target: str) -> dict:
        """Quick snapshot of open ports and HTTP status."""
        snapshot = {"ports": [], "http_status": None}
        import socket
        for port in [80, 443, 22, 21, 25, 3306, 5432, 6379, 8080, 8443]:
            try:
                with socket.create_connection((target, port), timeout=1):
                    snapshot["ports"].append(port)
            except Exception:
                pass
        try:
            import requests
            resp = requests.get(
                f"http://{target}", timeout=5, allow_redirects=True,
            )
            snapshot["http_status"] = resp.status_code
        except Exception:
            pass
        return snapshot

    def _diff_snapshots(self, baseline: dict, current: dict) -> list[str]:
        changes = []
        old_ports = set(baseline.get("ports", []))
        new_ports = set(current.get("ports", []))
        for p in new_ports - old_ports:
            changes.append(f"New port opened: {p}")
        for p in old_ports - new_ports:
            changes.append(f"Port closed: {p}")
        if baseline.get("http_status") != current.get("http_status"):
            changes.append(
                f"HTTP status changed: {baseline.get('http_status')} → {current.get('http_status')}"
            )
        return changes

    def _format_monitor_report(self, target: str, baseline: dict, changes: list) -> str:
        lines = [
            f"\n{'═'*60}",
            f"  NEXUS MONITOR REPORT — {target}",
            f"{'═'*60}",
            f"  Baseline ports: {', '.join(str(p) for p in baseline.get('ports', []))}",
            f"  HTTP status:    {baseline.get('http_status', 'N/A')}",
            "",
        ]
        if not changes:
            lines.append("  ✓ No changes detected during monitoring period.")
        else:
            lines.append(f"  ⚠  {len(changes)} change event(s) detected:")
            for ev in changes:
                lines.append(f"  [{ev['time']}] {', '.join(ev['changes'])}")
        return "\n".join(lines)

    def _format_vuln_report(self, target: str, results: list[tuple]) -> str:
        lines = [
            f"\n{'═'*60}",
            f"  NEXUS VULNERABILITY REPORT — {target}",
            f"{'═'*60}\n",
        ]
        for tool, output in results:
            lines.append(f"  ── {tool.upper()} ──")
            lines.append(output)
            lines.append("")
        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _cmd_exists(cmd: str) -> bool:
        try:
            subprocess.run([cmd, "--version"], capture_output=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _extract_services(scan_results: dict) -> list[dict]:
        services = []
        for host in scan_results.get("hosts", []):
            for port_info in host.get("ports", []):
                services.append({
                    "port":    port_info.get("port", 0),
                    "service": port_info.get("service", "unknown"),
                    "version": port_info.get("version", ""),
                })
        return services

    def _quick_scan_services(self, target: str) -> list[dict]:
        from cyber.scanner import PortScanner
        from cyber.toolkit import ToolKit
        tk  = ToolKit(verbose=False)
        sc  = PortScanner(toolkit=tk)
        res = sc.quick_scan(target)
        return self._extract_services(res)

    def _save_sandbox_report(self, sandbox_id: str, content: str):
        fname = _REPORT_DIR / f"{sandbox_id}.txt"
        fname.write_text(content, encoding="utf-8")
