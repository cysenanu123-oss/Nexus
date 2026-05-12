"""
NEXUS CyberScanner — Port Scanning Module
------------------------------------------
Uses nmap (system) or python-nmap (pip) automatically.
Falls back to raw socket scanning if neither available.
"""

import subprocess
import socket
import ipaddress
import json
import re
from typing import Optional
from .toolkit import ToolKit


class PortScanner:
    """
    Port scanning with automatic tool detection and fallback.
    Priority: nmap system binary → python-nmap → raw sockets
    """

    def __init__(self, toolkit: Optional[ToolKit] = None):
        self.tk = toolkit or ToolKit(verbose=True)
        self._nmap_ready = False
        self._nm = None  # python-nmap object if available
        self._init_scanner()

    def _init_scanner(self):
        """Detect and prepare the best available scanner."""
        # Try system nmap
        if self.tk.is_installed("nmap"):
            self._nmap_ready = True
            print("[Scanner] nmap detected — full scanning available")
            return

        # Try python-nmap
        if self.tk.ensure("python-nmap"):
            try:
                import nmap as nmap_lib
                self._nm = nmap_lib.PortScanner()
                self._nmap_ready = True
                print("[Scanner] python-nmap loaded successfully")
                return
            except Exception as e:
                print(f"[Scanner] python-nmap load failed: {e}")

        print("[Scanner] nmap not available — using raw socket fallback")

    # ─────────────────────────────────────────
    #  PUBLIC API
    # ─────────────────────────────────────────

    def scan_target(self, target: str, ports: str = "1-1024", scan_type: str = "basic") -> dict:
        """
        Scan a target host or subnet.

        Args:
            target: IP, hostname, or CIDR (e.g. "192.168.1.1", "192.168.1.0/24")
            ports:  Port range string e.g. "22,80,443" or "1-1024" or "top100"
            scan_type: "basic" | "full" | "quick" | "service" | "stealth"

        Returns:
            dict with hosts, open ports, services
        """
        print(f"[Scanner] Scanning {target} | ports={ports} | type={scan_type}")

        if self._nmap_ready:
            return self._nmap_scan(target, ports, scan_type)
        else:
            return self._socket_scan(target, ports)

    def quick_scan(self, target: str) -> dict:
        """Fast scan of most common ports (top 100)."""
        return self.scan_target(target, ports="top100", scan_type="quick")

    def full_scan(self, target: str) -> dict:
        """Comprehensive scan — all ports, service detection."""
        return self.scan_target(target, ports="1-65535", scan_type="full")

    def service_scan(self, target: str, ports: str = "1-1024") -> dict:
        """Scan with service/version detection."""
        return self.scan_target(target, ports=ports, scan_type="service")

    def stealth_scan(self, target: str, ports: str = "1-1024") -> dict:
        """SYN stealth scan (requires root)."""
        return self.scan_target(target, ports=ports, scan_type="stealth")

    # ─────────────────────────────────────────
    #  NMAP BACKEND
    # ─────────────────────────────────────────

    def _nmap_scan(self, target: str, ports: str, scan_type: str) -> dict:
        """Execute nmap and parse results."""
        args = self._build_nmap_args(scan_type, ports)

        # Use python-nmap if loaded
        if self._nm:
            return self._pynmap_scan(target, args)

        # Use system nmap binary
        return self._system_nmap_scan(target, args)

    def _build_nmap_args(self, scan_type: str, ports: str) -> str:
        """Build nmap argument string for scan type."""
        base_args = {
            "quick":   "-T4 --top-ports 100",
            "basic":   "-T4",
            "full":    "-T4 -A -sV",
            "service": "-sV -sC",
            "stealth": "-sS -T2",
            "os":      "-O -sV",
        }.get(scan_type, "-T4")

        port_arg = ""
        if ports == "top100":
            port_arg = "--top-ports 100"
        elif ports == "top1000":
            port_arg = "--top-ports 1000"
        else:
            port_arg = f"-p {ports}"

        return f"{base_args} {port_arg}".strip()

    def _system_nmap_scan(self, target: str, args: str) -> dict:
        """Run system nmap binary, parse XML output."""
        cmd = ["nmap", "-oX", "-"] + args.split() + [target]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            if proc.returncode not in (0, 1):
                return {"error": proc.stderr[:500], "target": target}
            return self._parse_nmap_xml(proc.stdout, target)
        except subprocess.TimeoutExpired:
            return {"error": "Scan timed out", "target": target}
        except FileNotFoundError:
            return {"error": "nmap binary not found", "target": target}

    def _pynmap_scan(self, target: str, args: str) -> dict:
        """Use python-nmap library to scan."""
        try:
            self._nm.scan(hosts=target, arguments=args)
            return self._parse_pynmap_results(target)
        except Exception as e:
            return {"error": str(e), "target": target}

    def _parse_nmap_xml(self, xml_output: str, target: str) -> dict:
        """Parse nmap XML output into clean dict."""
        result = {"target": target, "hosts": [], "summary": ""}
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_output)

            for host in root.findall("host"):
                host_data = {"ip": "", "hostname": "", "state": "", "ports": []}

                addr = host.find("address[@addrtype='ipv4']")
                if addr is not None:
                    host_data["ip"] = addr.get("addr", "")

                hostname_el = host.find("hostnames/hostname")
                if hostname_el is not None:
                    host_data["hostname"] = hostname_el.get("name", "")

                status = host.find("status")
                if status is not None:
                    host_data["state"] = status.get("state", "")

                for port in host.findall("ports/port"):
                    state = port.find("state")
                    if state is not None and state.get("state") == "open":
                        service = port.find("service")
                        port_data = {
                            "port": int(port.get("portid", 0)),
                            "protocol": port.get("protocol", "tcp"),
                            "state": "open",
                            "service": service.get("name", "unknown") if service is not None else "unknown",
                            "version": service.get("version", "") if service is not None else "",
                        }
                        host_data["ports"].append(port_data)

                result["hosts"].append(host_data)

            runstats = root.find("runstats/finished")
            if runstats is not None:
                result["summary"] = runstats.get("summary", "")

        except Exception as e:
            result["parse_error"] = str(e)

        return result

    def _parse_pynmap_results(self, target: str) -> dict:
        """Parse python-nmap results."""
        result = {"target": target, "hosts": []}
        for host in self._nm.all_hosts():
            host_data = {
                "ip": host,
                "state": self._nm[host].state(),
                "ports": []
            }
            for proto in self._nm[host].all_protocols():
                for port in self._nm[host][proto].keys():
                    port_info = self._nm[host][proto][port]
                    if port_info["state"] == "open":
                        host_data["ports"].append({
                            "port": port,
                            "protocol": proto,
                            "state": "open",
                            "service": port_info.get("name", "unknown"),
                            "version": port_info.get("version", ""),
                        })
            result["hosts"].append(host_data)
        return result

    # ─────────────────────────────────────────
    #  SOCKET FALLBACK SCANNER
    # ─────────────────────────────────────────

    def _socket_scan(self, target: str, ports: str) -> dict:
        """Pure Python socket-based port scanner — works without nmap."""
        print("[Scanner] Using socket fallback scanner...")
        port_list = self._parse_port_range(ports)
        open_ports = []

        try:
            ip = socket.gethostbyname(target)
        except socket.gaierror:
            return {"error": f"Cannot resolve host: {target}", "target": target}

        socket.setdefaulttimeout(1)
        for port in port_list:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    result = s.connect_ex((ip, port))
                    if result == 0:
                        service = self._guess_service(port)
                        open_ports.append({
                            "port": port,
                            "protocol": "tcp",
                            "state": "open",
                            "service": service,
                        })
            except Exception:
                continue

        return {
            "target": target,
            "method": "socket_fallback",
            "hosts": [{
                "ip": ip,
                "state": "up",
                "ports": open_ports
            }],
            "note": "Install nmap for more accurate results: sudo apt-get install nmap"
        }

    def _parse_port_range(self, ports: str) -> list[int]:
        """Parse port range string into list of ints."""
        if ports == "top100":
            return [21,22,23,25,53,80,110,111,135,139,143,443,445,993,995,
                    1723,3306,3389,5900,8080,8443,8888,9090,10000]
        if "-" in ports and "," not in ports:
            start, end = ports.split("-")
            return list(range(int(start), min(int(end) + 1, 65536)))
        if "," in ports:
            return [int(p.strip()) for p in ports.split(",")]
        try:
            return [int(ports)]
        except ValueError:
            return list(range(1, 1025))  # default

    def _guess_service(self, port: int) -> str:
        """Guess service name from well-known port."""
        common = {
            21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
            53: "dns", 80: "http", 110: "pop3", 143: "imap",
            443: "https", 445: "smb", 3306: "mysql",
            3389: "rdp", 5432: "postgresql", 5900: "vnc",
            6379: "redis", 8080: "http-alt", 8443: "https-alt",
            27017: "mongodb",
        }
        return common.get(port, "unknown")

    # ─────────────────────────────────────────
    #  FORMATTING
    # ─────────────────────────────────────────

    def format_results(self, results: dict) -> str:
        """Pretty-print scan results."""
        if "error" in results:
            return f"[!] Scan error: {results['error']}"

        lines = [f"\n{'═'*50}",
                 f"  NEXUS SCAN RESULTS — {results.get('target', 'unknown')}",
                 f"{'═'*50}"]

        hosts = results.get("hosts", [])
        if not hosts:
            lines.append("  No hosts found.")
        else:
            for host in hosts:
                ip = host.get("ip", "?")
                hostname = f" ({host['hostname']})" if host.get("hostname") else ""
                state = host.get("state", "unknown")
                lines.append(f"\n  Host: {ip}{hostname} [{state}]")

                ports = host.get("ports", [])
                if not ports:
                    lines.append("    No open ports found.")
                else:
                    lines.append(f"    {'PORT':<8} {'PROTO':<6} {'SERVICE':<15} {'VERSION'}")
                    lines.append(f"    {'─'*50}")
                    for p in sorted(ports, key=lambda x: x["port"]):
                        ver = p.get("version", "")[:25]
                        lines.append(
                            f"    {p['port']:<8} {p['protocol']:<6} {p['service']:<15} {ver}"
                        )

        if results.get("summary"):
            lines.append(f"\n  {results['summary']}")
        if results.get("note"):
            lines.append(f"\n  ℹ  {results['note']}")
        lines.append("═" * 50)
        return "\n".join(lines)