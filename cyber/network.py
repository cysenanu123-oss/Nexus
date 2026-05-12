"""
NEXUS NetworkIntel — Network Discovery & Intelligence
------------------------------------------------------
ARP scanning, interface enumeration, local subnet detection.
Uses: scapy → arp-scan → arping → fallback (ping sweep)
"""

import subprocess
import socket
import struct
import fcntl
import ipaddress
import re
import os
from typing import Optional
from .toolkit import ToolKit


class NetworkIntel:
    """
    Discovers devices, maps networks, enumerates interfaces.
    Intelligently picks the best available tool for each job.
    """

    def __init__(self, toolkit: Optional[ToolKit] = None):
        self.tk = toolkit or ToolKit(verbose=True)

    # ─────────────────────────────────────────
    #  INTERFACE INFORMATION
    # ─────────────────────────────────────────

    def get_interfaces(self) -> list[dict]:
        """Return all network interfaces with IP, MAC, status."""
        interfaces = []

        # Try psutil first (most reliable)
        if self.tk.ensure("psutil"):
            try:
                import psutil
                addrs = psutil.net_if_addrs()
                stats = psutil.net_if_stats()
                for name, addr_list in addrs.items():
                    iface = {"name": name, "ipv4": [], "ipv6": [], "mac": "", "up": False}
                    if name in stats:
                        iface["up"] = stats[name].isup
                        iface["speed"] = stats[name].speed
                    for addr in addr_list:
                        import psutil
                        if addr.family == socket.AF_INET:
                            iface["ipv4"].append({
                                "address": addr.address,
                                "netmask": addr.netmask,
                                "broadcast": addr.broadcast,
                            })
                        elif addr.family == socket.AF_INET6:
                            iface["ipv6"].append(addr.address)
                        elif addr.family == psutil.AF_LINK:
                            iface["mac"] = addr.address
                    interfaces.append(iface)
                return interfaces
            except Exception as e:
                print(f"[Network] psutil interface read failed: {e}")

        # Fallback: parse ip addr output
        try:
            proc = subprocess.run(["ip", "addr"], capture_output=True, text=True)
            return self._parse_ip_addr(proc.stdout)
        except FileNotFoundError:
            pass

        # Last resort: ifconfig
        try:
            proc = subprocess.run(["ifconfig", "-a"], capture_output=True, text=True)
            return self._parse_ifconfig(proc.stdout)
        except FileNotFoundError:
            return []

    def get_default_interface(self) -> Optional[dict]:
        """Return the default network interface (the one with internet access)."""
        try:
            proc = subprocess.run(
                ["ip", "route", "show", "default"],
                capture_output=True, text=True
            )
            match = re.search(r"dev (\S+)", proc.stdout)
            if match:
                iface_name = match.group(1)
                for iface in self.get_interfaces():
                    if iface["name"] == iface_name:
                        return iface
        except Exception:
            pass
        return None

    def get_local_subnet(self) -> Optional[str]:
        """
        Detect local subnet in CIDR notation (e.g. 192.168.1.0/24).
        """
        iface = self.get_default_interface()
        if iface and iface.get("ipv4"):
            for addr_info in iface["ipv4"]:
                ip = addr_info.get("address", "")
                mask = addr_info.get("netmask", "255.255.255.0")
                if ip and not ip.startswith("127."):
                    try:
                        network = ipaddress.IPv4Network(f"{ip}/{mask}", strict=False)
                        return str(network)
                    except ValueError:
                        continue

        # Fallback: use socket to find local IP
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
            parts = local_ip.split(".")
            subnet = f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
            return subnet
        except Exception:
            return None

    # ─────────────────────────────────────────
    #  ARP DISCOVERY
    # ─────────────────────────────────────────

    def discover_hosts(self, subnet: Optional[str] = None) -> list[dict]:
        """
        Discover all live hosts on the network using the best available method.
        Auto-detects subnet if not provided.
        """
        if not subnet:
            subnet = self.get_local_subnet()
            if not subnet:
                print("[Network] Could not determine local subnet")
                return []

        print(f"[Network] Discovering hosts on {subnet}...")

        # Priority order: scapy → arp-scan → nmap → ping sweep
        if self.tk.ensure("scapy"):
            result = self._scapy_arp_scan(subnet)
            if result:
                return result

        if self.tk.is_installed("arp-scan"):
            result = self._arp_scan_tool(subnet)
            if result:
                return result

        if self.tk.is_installed("nmap"):
            result = self._nmap_discovery(subnet)
            if result:
                return result

        # Pure Python ping sweep as last resort
        return self._ping_sweep(subnet)

    def _scapy_arp_scan(self, subnet: str) -> list[dict]:
        """ARP scan using scapy — most reliable for local discovery."""
        try:
            from scapy.all import ARP, Ether, srp
            arp = ARP(pdst=subnet)
            ether = Ether(dst="ff:ff:ff:ff:ff:ff")
            packet = ether / arp
            result = srp(packet, timeout=3, verbose=False)[0]
            hosts = []
            for sent, received in result:
                hosts.append({
                    "ip": received.psrc,
                    "mac": received.hwsrc,
                    "method": "arp",
                    "hostname": self._resolve_hostname(received.psrc),
                })
            print(f"[Network] scapy ARP found {len(hosts)} hosts")
            return hosts
        except Exception as e:
            print(f"[Network] scapy ARP failed: {e}")
            return []

    def _arp_scan_tool(self, subnet: str) -> list[dict]:
        """Use arp-scan system tool."""
        try:
            proc = subprocess.run(
                ["arp-scan", "--localnet", "--quiet"],
                capture_output=True, text=True, timeout=60
            )
            return self._parse_arp_scan_output(proc.stdout)
        except Exception as e:
            print(f"[Network] arp-scan failed: {e}")
            return []

    def _nmap_discovery(self, subnet: str) -> list[dict]:
        """Use nmap -sn (ping scan) for host discovery."""
        try:
            proc = subprocess.run(
                ["nmap", "-sn", "-T4", subnet],
                capture_output=True, text=True, timeout=120
            )
            return self._parse_nmap_discovery(proc.stdout)
        except Exception as e:
            print(f"[Network] nmap discovery failed: {e}")
            return []

    def _ping_sweep(self, subnet: str) -> list[dict]:
        """Pure Python ping sweep — slowest but always works."""
        print("[Network] Running ping sweep (may be slow)...")
        hosts = []
        try:
            network = ipaddress.IPv4Network(subnet, strict=False)
            # Limit to /24 for speed
            host_list = list(network.hosts())[:254]
            for ip in host_list:
                ip_str = str(ip)
                proc = subprocess.run(
                    ["ping", "-c", "1", "-W", "1", ip_str],
                    capture_output=True, timeout=3
                )
                if proc.returncode == 0:
                    hosts.append({
                        "ip": ip_str,
                        "mac": "unknown",
                        "method": "ping",
                        "hostname": self._resolve_hostname(ip_str),
                    })
        except Exception as e:
            print(f"[Network] ping sweep failed: {e}")
        return hosts

    # ─────────────────────────────────────────
    #  NETWORK INFO
    # ─────────────────────────────────────────

    def get_routing_table(self) -> list[dict]:
        """Return the system routing table."""
        routes = []
        try:
            proc = subprocess.run(["ip", "route"], capture_output=True, text=True)
            for line in proc.stdout.splitlines():
                routes.append({"route": line.strip()})
        except Exception:
            pass
        return routes

    def get_arp_table(self) -> list[dict]:
        """Return current ARP cache."""
        entries = []
        try:
            proc = subprocess.run(["arp", "-n"], capture_output=True, text=True)
            for line in proc.stdout.splitlines()[1:]:  # skip header
                parts = line.split()
                if len(parts) >= 3:
                    entries.append({
                        "ip": parts[0],
                        "mac": parts[2],
                        "interface": parts[-1] if len(parts) > 3 else "",
                    })
        except Exception:
            pass
        return entries

    def get_active_connections(self) -> list[dict]:
        """Return active network connections."""
        connections = []

        if self.tk.ensure("psutil"):
            try:
                import psutil
                for conn in psutil.net_connections(kind="inet"):
                    if conn.status in ("ESTABLISHED", "LISTEN"):
                        connections.append({
                            "local": f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else "",
                            "remote": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "",
                            "status": conn.status,
                            "pid": conn.pid,
                        })
                return connections
            except Exception:
                pass

        # Fallback: ss command
        try:
            proc = subprocess.run(["ss", "-tunp"], capture_output=True, text=True)
            return self._parse_ss_output(proc.stdout)
        except Exception:
            return []

    def external_ip(self) -> Optional[str]:
        """Get external/public IP address."""
        try:
            import urllib.request
            with urllib.request.urlopen("https://api.ipify.org", timeout=5) as f:
                return f.read().decode().strip()
        except Exception:
            return None

    # ─────────────────────────────────────────
    #  PARSERS
    # ─────────────────────────────────────────

    def _parse_arp_scan_output(self, output: str) -> list[dict]:
        """Parse arp-scan output."""
        hosts = []
        for line in output.splitlines():
            match = re.match(r"(\d+\.\d+\.\d+\.\d+)\s+([0-9a-f:]+)\s*(.*)", line)
            if match:
                hosts.append({
                    "ip": match.group(1),
                    "mac": match.group(2),
                    "vendor": match.group(3).strip(),
                    "method": "arp-scan",
                })
        return hosts

    def _parse_nmap_discovery(self, output: str) -> list[dict]:
        """Parse nmap -sn output."""
        hosts = []
        current = {}
        for line in output.splitlines():
            if "Nmap scan report" in line:
                if current:
                    hosts.append(current)
                ip_match = re.search(r"(\d+\.\d+\.\d+\.\d+)", line)
                hostname_match = re.search(r"for ([^\(]+)", line)
                current = {
                    "ip": ip_match.group(1) if ip_match else "",
                    "hostname": hostname_match.group(1).strip() if hostname_match else "",
                    "method": "nmap-ping",
                }
        if current:
            hosts.append(current)
        return hosts

    def _parse_ip_addr(self, output: str) -> list[dict]:
        """Parse ip addr output into interface list."""
        interfaces = []
        current = None
        for line in output.splitlines():
            iface_match = re.match(r"\d+: (\S+):", line)
            if iface_match:
                if current:
                    interfaces.append(current)
                name = iface_match.group(1).rstrip("@")
                current = {"name": name, "ipv4": [], "ipv6": [], "mac": "",
                           "up": "UP" in line}
            elif current:
                mac_match = re.search(r"link/ether (\S+)", line)
                if mac_match:
                    current["mac"] = mac_match.group(1)
                ip_match = re.search(r"inet (\d+\.\d+\.\d+\.\d+)/(\d+)", line)
                if ip_match:
                    current["ipv4"].append({
                        "address": ip_match.group(1),
                        "prefix": ip_match.group(2),
                    })
        if current:
            interfaces.append(current)
        return interfaces

    def _parse_ifconfig(self, output: str) -> list[dict]:
        """Parse ifconfig output."""
        interfaces = []
        for block in re.split(r"\n(?=\S)", output):
            if not block.strip():
                continue
            name_match = re.match(r"(\S+)", block)
            if not name_match:
                continue
            iface = {"name": name_match.group(1), "ipv4": [], "ipv6": [], "mac": ""}
            ip_match = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", block)
            if ip_match:
                iface["ipv4"].append({"address": ip_match.group(1)})
            mac_match = re.search(r"ether ([0-9a-f:]+)", block)
            if mac_match:
                iface["mac"] = mac_match.group(1)
            interfaces.append(iface)
        return interfaces

    def _parse_ss_output(self, output: str) -> list[dict]:
        """Parse ss -tunp output."""
        connections = []
        for line in output.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 5:
                connections.append({
                    "proto": parts[0],
                    "local": parts[4],
                    "remote": parts[5] if len(parts) > 5 else "",
                    "status": parts[1],
                })
        return connections

    def _resolve_hostname(self, ip: str) -> str:
        """Reverse DNS lookup for IP."""
        try:
            return socket.gethostbyaddr(ip)[0]
        except Exception:
            return ""

    # ─────────────────────────────────────────
    #  FORMATTING
    # ─────────────────────────────────────────

    def format_host_table(self, hosts: list[dict]) -> str:
        """Pretty-print host discovery results."""
        if not hosts:
            return "  No hosts discovered."
        lines = [f"\n{'═'*60}",
                 f"  NEXUS NETWORK MAP — {len(hosts)} host(s) found",
                 f"{'═'*60}",
                 f"  {'IP ADDRESS':<18} {'MAC ADDRESS':<20} {'HOSTNAME'}"]
        lines.append(f"  {'─'*58}")
        for h in sorted(hosts, key=lambda x: socket.inet_aton(x["ip"]) if self._valid_ip(x["ip"]) else b""):
            ip = h.get("ip", "")
            mac = h.get("mac", "—")
            hostname = h.get("hostname", "") or h.get("vendor", "")
            lines.append(f"  {ip:<18} {mac:<20} {hostname}")
        lines.append(f"{'═'*60}")
        return "\n".join(lines)

    def format_interfaces(self, interfaces: list[dict]) -> str:
        """Pretty-print interface information."""
        lines = ["\n  NETWORK INTERFACES\n  " + "─" * 40]
        for iface in interfaces:
            status = "UP" if iface.get("up") else "DOWN"
            lines.append(f"\n  [{iface['name']}]  {status}")
            if iface.get("mac"):
                lines.append(f"    MAC:  {iface['mac']}")
            for addr in iface.get("ipv4", []):
                lines.append(f"    IPv4: {addr.get('address', '')}  mask: {addr.get('netmask', addr.get('prefix', ''))}")
        return "\n".join(lines)

    @staticmethod
    def _valid_ip(ip: str) -> bool:
        try:
            socket.inet_aton(ip)
            return True
        except Exception:
            return False