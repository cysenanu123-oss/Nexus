#!/usr/bin/env python3
"""
run.py — NEXUS launcher.

Picks the right front-end for the situation:

    python run.py          # Jarvis HUD if PyQt + a display are available, else CLI
    python run.py --hud    # force the HUD (errors out if unavailable)
    python run.py --cli    # force the terminal interface
    python run.py --check  # print what's available and exit

Works the same on Linux, macOS and Windows.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _has_display() -> bool:
    """Is there a GUI display we can open a window on?"""
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    # Linux/other: need X or Wayland.
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _has_pyqt() -> bool:
    try:
        import PyQt5  # noqa: F401
        return True
    except Exception:
        return False


def run_hud() -> int:
    from ui.nexus_voice_app import main as hud_main
    hud_main()
    return 0


def run_cli() -> int:
    from main import main as cli_main
    cli_main()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Launch NEXUS.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--hud", action="store_true", help="force the Jarvis HUD")
    g.add_argument("--cli", action="store_true", help="force the terminal interface")
    ap.add_argument("--check", action="store_true", help="report availability and exit")
    args = ap.parse_args()

    hud_ok = _has_pyqt() and _has_display()

    if args.check:
        print(f"  display : {'yes' if _has_display() else 'no'}")
        print(f"  PyQt5   : {'yes' if _has_pyqt() else 'no'}")
        print(f"  HUD     : {'available' if hud_ok else 'unavailable → CLI'}")
        return 0

    if args.cli:
        return run_cli()

    if args.hud:
        if not hud_ok:
            print("HUD unavailable (need PyQt5 + a display). "
                  "Install with: pip install PyQt5", file=sys.stderr)
            return 2
        return run_hud()

    # Auto: prefer the HUD, fall back to the CLI.
    if hud_ok:
        try:
            return run_hud()
        except Exception as e:
            print(f"HUD failed to start ({e}); falling back to CLI.", file=sys.stderr)
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
