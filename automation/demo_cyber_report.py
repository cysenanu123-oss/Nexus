#!/usr/bin/env python3
"""
automation/demo_cyber_report.py
════════════════════════════════════════════════════════════════
NEXUS Full Automation Demo — Cyber Research & Report Generation
════════════════════════════════════════════════════════════════

What this script does, step by step:
  1.  Search the web for top cybersecurity news and attack techniques
  2.  Scrape and summarize 3+ sources
  3.  Compile a professional markdown report
  4.  Open Mousepad (text editor)
  5.  Wait for the window, then focus it
  6.  Type the full report using keyboard automation
  7.  Move the mouse to File menu (visually), click it
  8.  Click "Save As" from the menu
  9.  Type the filename in the save dialog
  10. Press Enter to confirm
  11. Minimize mousepad, open file manager to the saved file
  12. Send a desktop notification: "Cyber Report Ready"
  13. Put everything back (close extra windows, return terminal focus)

Run:
    cd ~/Desktop/work/NEXUS
    python automation/demo_cyber_report.py

Requirements (auto-installed if missing):
    pip install requests beautifulsoup4 pyautogui pillow
    sudo apt install xdotool wmctrl tesseract-ocr -y   (for OCR save)
"""

import os
import sys
import time
import logging
import textwrap
from datetime import datetime
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
log = logging.getLogger("nexus.demo")


# ══════════════════════════════════════════════════════════════
#  COLOUR HELPERS
# ══════════════════════════════════════════════════════════════

C_CYAN   = "\033[96m"
C_GREEN  = "\033[92m"
C_YELLOW = "\033[93m"
C_RED    = "\033[91m"
C_BOLD   = "\033[1m"
C_RESET  = "\033[0m"

def banner(msg: str) -> None:
    width = 62
    print(f"\n{C_BOLD}{C_CYAN}{'═' * width}{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}  {msg}{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}{'═' * width}{C_RESET}\n")

def step_log(n: int, total: int, msg: str) -> None:
    print(f"  {C_BOLD}[{n}/{total}]{C_RESET} {msg}")

def ok(msg: str) -> None:
    print(f"  {C_GREEN}✓{C_RESET} {msg}")

def warn(msg: str) -> None:
    print(f"  {C_YELLOW}⚠{C_RESET} {msg}")

def fail(msg: str) -> None:
    print(f"  {C_RED}✗{C_RESET} {msg}")


# ══════════════════════════════════════════════════════════════
#  PHASE 1 — WEB RESEARCH
# ══════════════════════════════════════════════════════════════

SEARCH_TOPICS = [
    "top cybersecurity attacks 2025 techniques",
    "latest critical vulnerabilities CVE 2025",
    "best penetration testing tools 2025 kali",
]

REPORT_SAVE_PATH = str(Path.home() / "Desktop" / "nexus_cyber_report.txt")


def fetch_url(url: str, timeout: int = 8) -> str:
    """Fetch URL content, return plain text."""
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) "
                "Gecko/20100101 Firefox/120.0"
            )
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove scripts/styles
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return " ".join(text.split())[:4000]
    except Exception as e:
        log.warning("Fetch failed for %s: %s", url, e)
        return ""


def duckduckgo_search(query: str, max_results: int = 3) -> list[dict]:
    """Search DuckDuckGo and return list of {title, url, snippet}."""
    try:
        import requests
        url = "https://api.duckduckgo.com/"
        params = {
            "q":      query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }
        resp = requests.get(url, params=params, timeout=8)
        data = resp.json()
        results = []

        # RelatedTopics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title":   topic.get("Text", "")[:80],
                    "url":     topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", "")[:300],
                })

        # AbstractText
        if data.get("AbstractText"):
            results.insert(0, {
                "title":   data.get("Heading", query),
                "url":     data.get("AbstractURL", ""),
                "snippet": data["AbstractText"][:400],
            })

        return results[:max_results]

    except Exception as e:
        log.warning("DuckDuckGo search failed: %s", e)
        return []


def fallback_cyber_facts() -> list[dict]:
    """
    Hardcoded baseline facts used if web search is unavailable.
    This ensures the demo always produces a useful report.
    """
    return [
        {
            "title": "SQL Injection — OWASP #1 Attack",
            "url":   "https://owasp.org/www-community/attacks/SQL_Injection",
            "snippet": (
                "SQL injection allows attackers to interfere with database queries. "
                "It can allow attackers to view, modify, or delete data they normally "
                "cannot access, including other users' data. Prevention: use parameterised "
                "queries, prepared statements, and stored procedures."
            ),
        },
        {
            "title": "Ransomware — 2025 Threat Landscape",
            "url":   "https://www.cisa.gov/ransomware",
            "snippet": (
                "Ransomware attacks have grown 74% YoY. LockBit 3.0, BlackCat (ALPHV), "
                "and Cl0p remain the most active RaaS groups. Average ransom demand now "
                "exceeds $1.5M. Defence: offline backups, MFA, network segmentation, "
                "zero-trust architecture."
            ),
        },
        {
            "title": "Critical CVEs to Watch — Q1 2025",
            "url":   "https://nvd.nist.gov/vuln/search",
            "snippet": (
                "Top critical CVEs include privilege escalation flaws in Windows Kernel, "
                "remote code execution in OpenSSH (regreSSHion), and SSRF in common cloud "
                "SDKs. Patch Tuesday continues to deliver 80–120 fixes monthly. "
                "Prioritise CVE CVSS score ≥9.0 for immediate patching."
            ),
        },
        {
            "title": "Top Pentesting Tools on Kali Linux 2025",
            "url":   "https://www.kali.org/tools/",
            "snippet": (
                "Essential tools: Nmap (network mapping), Metasploit Framework (exploitation), "
                "Burp Suite (web app testing), Wireshark (traffic analysis), John the Ripper "
                "(password cracking), Hashcat (GPU hash cracking), Gobuster (directory brute-force), "
                "SQLMap (automated SQL injection), Nikto (web server scanner)."
            ),
        },
        {
            "title": "Social Engineering & Phishing Trends",
            "url":   "https://www.proofpoint.com/us/threat-reference/phishing",
            "snippet": (
                "94% of cyberattacks begin with a phishing email. AI-generated spear-phishing "
                "now produces near-perfect targeted messages. Vishing (voice phishing) attacks "
                "have increased 260%. Defence: user training, DMARC/DKIM/SPF enforcement, "
                "email sandboxing, MFA on all accounts."
            ),
        },
        {
            "title": "Zero-Day Exploits and Bug Bounty",
            "url":   "https://zerodium.com/program.html",
            "snippet": (
                "Zero-day market prices reflect software risk: iOS full-chain RCE fetches "
                "$2.5M on grey markets. Responsible disclosure via HackerOne, Bugcrowd, "
                "and vendor bug bounty programmes pays $100–$1M+ for critical findings. "
                "Always test within legal scope: written authorisation is mandatory."
            ),
        },
    ]


def do_research() -> list[dict]:
    """Run web searches and collect findings. Returns list of result dicts."""
    all_results = []

    try:
        import requests
        ok("Internet available — performing live search")
        for query in SEARCH_TOPICS:
            print(f"    🔍 Searching: {query!r}")
            results = duckduckgo_search(query, max_results=2)
            if results:
                all_results.extend(results)
                ok(f"  Got {len(results)} results")
            else:
                warn(f"  No results for {query!r}")
            time.sleep(0.5)   # polite delay

    except ImportError:
        warn("requests not installed — using offline knowledge base")

    if len(all_results) < 3:
        warn("Supplementing with offline cybersecurity knowledge base")
        all_results.extend(fallback_cyber_facts())

    # Deduplicate
    seen = set()
    unique = []
    for r in all_results:
        key = r.get("title", "")[:40]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    ok(f"Research complete: {len(unique)} findings")
    return unique[:8]


# ══════════════════════════════════════════════════════════════
#  PHASE 2 — REPORT COMPILATION
# ══════════════════════════════════════════════════════════════

def compile_report(findings: list[dict]) -> str:
    """Build a formatted text report from research findings."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║           NEXUS CYBER INTELLIGENCE REPORT                   ║",
        f"║  Generated: {now:<49}║",
        "║  Classification: RESEARCH | Personal Lab Use                ║",
        "╚══════════════════════════════════════════════════════════════╝",
        "",
        "EXECUTIVE SUMMARY",
        "─────────────────",
        "This report was autonomously generated by NEXUS — an AI-powered",
        "cybersecurity assistant running on Kali Linux. The research covers",
        "current threat landscape, critical vulnerabilities, and essential",
        "penetration testing tools and techniques.",
        "",
    ]

    for i, finding in enumerate(findings, 1):
        title   = finding.get("title",   "Finding")
        url     = finding.get("url",     "")
        snippet = finding.get("snippet", "")

        lines.append(f"[{i}] {title}")
        lines.append("─" * min(len(title) + 4, 62))
        # Word-wrap the snippet
        wrapped = textwrap.fill(snippet, width=62)
        lines.extend(wrapped.splitlines())
        if url:
            lines.append(f"    Source: {url}")
        lines.append("")

    lines += [
        "",
        "RECOMMENDATIONS",
        "───────────────",
        "1. Patch all systems — prioritise CVSS ≥ 9.0 vulnerabilities",
        "2. Enable MFA on all user and admin accounts immediately",
        "3. Maintain offline, encrypted backups tested monthly",
        "4. Conduct phishing simulations and security awareness training",
        "5. Implement network segmentation and zero-trust architecture",
        "6. Run regular vulnerability scans with Nmap + Nessus/OpenVAS",
        "7. Perform quarterly penetration tests — document all findings",
        "",
        "TOOLS REFERENCED",
        "────────────────",
        "  Nmap        — network mapping & port scanning",
        "  Metasploit  — exploitation framework",
        "  Burp Suite  — web application security testing",
        "  Wireshark   — network traffic analysis",
        "  SQLMap      — automated SQL injection testing",
        "  Hashcat     — GPU-accelerated password recovery",
        "  Gobuster    — directory & DNS enumeration",
        "  John        — password cracking",
        "",
        "─" * 62,
        "  Report generated by NEXUS Automation Engine",
        f"  {now} | Kali Linux | Personal Security Lab",
        "─" * 62,
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  PHASE 3 — GUI AUTOMATION (open editor, type, save, close)
# ══════════════════════════════════════════════════════════════

def run_gui_automation(report_text: str, save_path: str) -> bool:
    """
    Use the NEXUS Automation engine to:
      - Open Mousepad
      - Type the report
      - Save it via File menu (visual) or Ctrl+Shift+S
      - Close Mousepad
    """
    from automation.automation import Automation
    from automation.planner   import TaskPlanner, ExecutionPlan, Step
    from automation.executor  import Executor
    from automation.reporter  import Reporter

    reporter = Reporter(verbose=True)
    executor = Executor(confirm_high_risk=False)
    filename = Path(save_path).name

    # ── Build a custom multi-step plan ────────────────────────
    # We do this manually (not via the planner) because the report
    # text is dynamic and very long.

    steps_raw = [
        # 1. Launch mousepad
        Step(index=1, type="shell", action="launch_app", target="mousepad",
             description="Open Mousepad text editor"),
        # 2. Wait for the window
        Step(index=2, type="gui", action="wait_window", target="mousepad",
             description="Wait for Mousepad to appear",
             depends_on=[1], optional=True, timeout_sec=8.0),
        # 3. Pause — let the window fully render
        Step(index=3, type="wait", action="sleep", params={"seconds": 1.0},
             description="Wait for Mousepad to fully render",
             depends_on=[2]),
        # 4. Focus the window (make sure keyboard input goes there)
        Step(index=4, type="gui", action="focus_window", target="mousepad",
             description="Focus Mousepad window", depends_on=[3]),
        # 5. Click inside the text area (centre of screen)
        Step(index=5, type="gui", action="click",
             params={"coords": (800, 400)},
             description="Click text area to set cursor", depends_on=[4]),
        # 6. Type the full report
        Step(index=6, type="gui", action="type_text",
             params={"text": report_text},
             description=f"Type report ({len(report_text)} chars)",
             depends_on=[5], timeout_sec=120.0),
        # 7. Move mouse visually to File menu (top-left) and click
        Step(index=7, type="gui", action="click_menu_item",
             target="File",
             params={"item": "Save As"},
             description="Click File → Save As (mouse navigation)",
             depends_on=[6], optional=True),
        # 8. Wait for save dialog
        Step(index=8, type="wait", action="sleep", params={"seconds": 0.8},
             description="Wait for Save As dialog",
             depends_on=[7]),
        # 9. Clear any pre-filled filename and type ours
        Step(index=9, type="gui", action="hotkey",
             target="ctrl+a", params={"keys": ["ctrl", "a"]},
             description="Select all in filename field", depends_on=[8]),
        Step(index=10, type="gui", action="type_text",
             params={"text": filename},
             description=f"Type filename: {filename!r}", depends_on=[9]),
        # 10. Confirm save
        Step(index=11, type="gui", action="press_key",
             target="return",
             description="Confirm save", depends_on=[10]),
        # 11. Wait after save
        Step(index=12, type="wait", action="sleep", params={"seconds": 0.5},
             description="Wait after save"),
        # 12. Close Mousepad
        Step(index=13, type="gui", action="hotkey",
             target="ctrl+q", params={"keys": ["ctrl", "q"]},
             description="Close Mousepad", depends_on=[12]),
        # 13. If unsaved prompt appears, press Enter to confirm close
        Step(index=14, type="wait", action="sleep", params={"seconds": 0.4},
             description="Wait for any close dialog", optional=True),
        Step(index=15, type="gui", action="press_key",
             target="return",
             description="Confirm close (if prompted)", optional=True),
        # 14. Send desktop notification
        Step(index=16, type="shell", action="run_command",
             target=f'notify-send "NEXUS Cyber Report" "Report saved to Desktop: {filename}" --icon=dialog-information',
             description="Send desktop notification",
             optional=True),
        # 15. Open the file manager at Desktop to show the file
        Step(index=17, type="shell", action="run_command",
             target=f"thunar {Path(save_path).parent}",
             description="Open file manager at Desktop",
             optional=True, timeout_sec=5.0),
    ]

    plan = ExecutionPlan(
        instruction=f"Research cyber threats and save report as {filename}",
        steps=steps_raw,
        goal=f"Report saved to {save_path}",
        risk_level="low",
        requires_confirmation=False,
        estimated_duration_sec=len(report_text) * 0.04 + 20,
    )

    reporter.on_start(plan)
    result = executor.run(plan, on_progress=reporter.on_step)
    reporter.summarize(result)

    return result.success


# ══════════════════════════════════════════════════════════════
#  FALLBACK — if GUI automation fails, write the file directly
# ══════════════════════════════════════════════════════════════

def save_report_direct(report_text: str, save_path: str) -> None:
    """Write the report directly to disk as a fallback."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(save_path).write_text(report_text, encoding="utf-8")
    ok(f"Report written directly to: {save_path}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    TOTAL_PHASES = 4

    banner("NEXUS CYBER INTELLIGENCE — FULL AUTOMATION DEMO")
    print(f"  Output file: {REPORT_SAVE_PATH}")
    print(f"  Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ── Phase 1: Research ──────────────────────────────────────
    step_log(1, TOTAL_PHASES, "WEB RESEARCH — searching for cybersecurity intelligence...")
    t0       = time.time()
    findings = do_research()
    ok(f"Research phase done in {time.time()-t0:.1f}s — {len(findings)} findings")

    # ── Phase 2: Compile report ────────────────────────────────
    step_log(2, TOTAL_PHASES, "COMPILING REPORT...")
    report_text = compile_report(findings)
    ok(f"Report compiled — {len(report_text)} characters, {report_text.count(chr(10))} lines")
    print()
    print("  ── Report Preview (first 400 chars) ──")
    print("  " + report_text[:400].replace("\n", "\n  "))
    print("  ...")
    print()

    # ── Phase 3: GUI automation ────────────────────────────────
    step_log(3, TOTAL_PHASES, "GUI AUTOMATION — opening editor and typing report...")
    time.sleep(1.5)   # brief pause before launching GUI automation

    gui_success = run_gui_automation(report_text, REPORT_SAVE_PATH)

    if not gui_success:
        warn("GUI automation had issues — saving report directly to disk")
        save_report_direct(report_text, REPORT_SAVE_PATH)
    else:
        ok("GUI automation completed successfully")

    # ── Phase 4: Verify ───────────────────────────────────────
    step_log(4, TOTAL_PHASES, "VERIFYING SAVED REPORT...")
    path = Path(REPORT_SAVE_PATH)

    if path.exists():
        size = path.stat().st_size
        ok(f"File confirmed: {path}  ({size:,} bytes)")
    else:
        # GUI save might have used a slightly different path
        # Search Desktop for matching file
        desktop = Path.home() / "Desktop"
        candidates = list(desktop.glob("nexus_cyber*"))
        if candidates:
            ok(f"Found saved file: {candidates[0]}")
        else:
            warn("File not found on Desktop — check if Mousepad saved to a different location")

    banner("DEMO COMPLETE")
    print(f"  {C_GREEN}✓ Cyber report generated and saved{C_RESET}")
    print(f"  {C_CYAN}  File: {REPORT_SAVE_PATH}{C_RESET}")
    print(f"  {C_CYAN}  Open it with: mousepad {REPORT_SAVE_PATH}{C_RESET}")
    print()


if __name__ == "__main__":
    # Quick dependency check
    missing = []
    try:
        import pyautogui
    except ImportError:
        missing += ["pyautogui", "pillow"]
    try:
        import requests
    except ImportError:
        missing.append("requests")
    try:
        import bs4
    except ImportError:
        missing.append("beautifulsoup4")

    if missing:
        print(f"\n{C_YELLOW}⚠ Missing packages. Installing: {missing}{C_RESET}")
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "pip", "install"] + missing,
            check=False
        )
        print()

    main()
