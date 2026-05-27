"""
automation/gui_agent.py
NEXUS GUI Agent — mouse, keyboard, and window automation.

Handles all GUI-type steps:
  - Mouse clicks (left, right, double)
  - Keyboard input (type text, press keys, hotkeys)
  - Window management (focus, maximize, minimize, close)
  - URL navigation via address bar
  - Scroll
  - Screenshots

Tools used (in priority order):
  1. pyautogui  — mouse/keyboard (pip install pyautogui)
  2. xdotool    — window management (sudo apt install xdotool)
  3. wmctrl     — window control fallback (sudo apt install wmctrl)

Usage:
    from automation.gui_agent import GUIAgent
    agent = GUIAgent()
    success, output = agent.run(step)
"""

from __future__ import annotations

import subprocess
import shutil
import sys
import time
import logging
from typing import Optional

log = logging.getLogger("nexus.automation.gui")

_IS_WINDOWS = sys.platform == "win32"


def _win32_focus(title_fragment: str) -> bool:
    """Focus a window by partial title on Windows using pygetwindow or win32gui."""
    # 1. pygetwindow
    try:
        import pygetwindow as gw  # type: ignore
        wins = gw.getWindowsWithTitle(title_fragment)
        if wins:
            wins[0].activate()
            return True
    except ImportError:
        pass
    # 2. win32gui
    try:
        import win32gui  # type: ignore
        import win32con  # type: ignore
        def _cb(hwnd, results):
            if title_fragment.lower() in win32gui.GetWindowText(hwnd).lower():
                results.append(hwnd)
        found = []
        win32gui.EnumWindows(_cb, found)
        if found:
            win32gui.ShowWindow(found[0], win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(found[0])
            return True
    except ImportError:
        pass
    return False


def _win32_maximize() -> bool:
    """Maximize the foreground window on Windows."""
    try:
        import win32gui, win32con  # type: ignore
        hwnd = win32gui.GetForegroundWindow()
        win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
        return True
    except ImportError:
        pass
    try:
        import pygetwindow as gw  # type: ignore
        w = gw.getActiveWindow()
        if w:
            w.maximize()
            return True
    except ImportError:
        pass
    return False


def _win32_minimize() -> bool:
    """Minimize the foreground window on Windows."""
    try:
        import win32gui, win32con  # type: ignore
        hwnd = win32gui.GetForegroundWindow()
        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
        return True
    except ImportError:
        pass
    try:
        import pygetwindow as gw  # type: ignore
        w = gw.getActiveWindow()
        if w:
            w.minimize()
            return True
    except ImportError:
        pass
    return False


# ─────────────────────────────────────────────────────────────
#  KEY NAME ALIASES  (spoken name → pyautogui key)
# ─────────────────────────────────────────────────────────────

KEY_ALIASES: dict[str, str] = {
    "enter":      "return",
    "return":     "return",
    "space":      "space",
    "tab":        "tab",
    "escape":     "escape",
    "esc":        "escape",
    "backspace":  "backspace",
    "delete":     "delete",
    "del":        "delete",
    "up":         "up",
    "down":       "down",
    "left":       "left",
    "right":      "right",
    "home":       "home",
    "end":        "end",
    "page up":    "pageup",
    "page down":  "pagedown",
    "f1":  "f1",  "f2":  "f2",  "f3":  "f3",  "f4":  "f4",
    "f5":  "f5",  "f6":  "f6",  "f7":  "f7",  "f8":  "f8",
    "f9":  "f9",  "f10": "f10", "f11": "f11", "f12": "f12",
}


# ─────────────────────────────────────────────────────────────
#  GUI AGENT
# ─────────────────────────────────────────────────────────────

class GUIAgent:
    """
    Executes GUI automation steps using pyautogui + xdotool.

    Every public method returns (success: bool, output: str).
    """

    def __init__(self):
        self._pyautogui = None
        self._has_xdotool = shutil.which("xdotool") is not None
        self._has_wmctrl  = shutil.which("wmctrl") is not None
        self._init_pyautogui()
        log.info(
            "GUIAgent ready — pyautogui=%s, xdotool=%s, wmctrl=%s",
            self._pyautogui is not None,
            self._has_xdotool,
            self._has_wmctrl,
        )

    def _init_pyautogui(self):
        """Lazy-import pyautogui and configure it."""
        try:
            import pyautogui
            pyautogui.FAILSAFE  = True    # move mouse to corner to abort
            pyautogui.PAUSE     = 0.05    # small pause between actions
            self._pyautogui = pyautogui
        except ImportError:
            log.warning(
                "pyautogui not installed. "
                "Install with: pip install pyautogui pillow"
            )
        except Exception as e:
            log.warning("pyautogui init error: %s", e)

    # ── Main dispatch ─────────────────────────────────────────

    def run(self, step) -> tuple[bool, str]:
        """Dispatch a GUI step to the right handler."""
        action = step.action.lower()

        handlers = {
            # Mouse
            "click":               self._click,
            "double_click":        self._double_click,
            "right_click":         self._right_click,
            "drag_drop":           self._drag_drop,
            # Screen text clicking (OCR-based)
            "click_screen_text":   self._click_screen_text,
            "click_menu_item":     self._click_menu_item,
            # Keyboard
            "type_text":           self._type_text,
            "press_key":           self._press_key,
            "hotkey":              self._hotkey,
            # Scroll
            "scroll":              self._scroll,
            # Window management
            "focus_window":        self._focus_window,
            "wait_window":         self._wait_window,
            "maximize_window":     self._maximize_window,
            "minimize_window":     self._minimize_window,
            "close_window":        self._close_window,
            "focus_active_window": self._focus_active_window,
            # URL navigation
            "navigate_url":        self._navigate_url,
            # Screenshot
            "screenshot":          self._screenshot,
        }

        handler = handlers.get(action)
        if handler:
            return handler(step)

        return False, f"GUIAgent: unknown action {action!r}"

    # ── Mouse actions ─────────────────────────────────────────

    def _click(self, step) -> tuple[bool, str]:
        target = step.target.strip()
        coords = step.params.get("coords")   # (x, y) if known

        if coords:
            return self._click_at(*coords)
        if target:
            # Try to find and click by image/text
            pos = self._find_target(target)
            if pos:
                return self._click_at(*pos)
            # Fall back to xdotool key search
            return self._xdotool_click_window(target)
        return False, "No click target specified."

    def _double_click(self, step) -> tuple[bool, str]:
        target = step.target.strip()
        coords = step.params.get("coords")

        if coords:
            return self._click_at(*coords, clicks=2)
        if target:
            pos = self._find_target(target)
            if pos:
                return self._click_at(*pos, clicks=2)
        return False, "No double-click target specified."

    def _right_click(self, step) -> tuple[bool, str]:
        target = step.target.strip()
        coords = step.params.get("coords")

        if coords:
            return self._click_at(*coords, button="right")
        if target:
            pos = self._find_target(target)
            if pos:
                return self._click_at(*pos, button="right")
        return False, "No right-click target specified."

    def _click_at(
        self, x: int, y: int,
        clicks: int = 1, button: str = "left"
    ) -> tuple[bool, str]:
        if not self._pyautogui:
            return self._xdotool_click_xy(x, y, clicks, button)
        try:
            self._pyautogui.click(x, y, clicks=clicks, button=button)
            return True, f"Clicked ({x}, {y}) {clicks}x with {button} button"
        except Exception as e:
            return False, str(e)

    def _drag_drop(self, step) -> tuple[bool, str]:
        src = step.params.get("from", (0, 0))
        dst = step.params.get("to",   (0, 0))
        if not self._pyautogui:
            return False, "pyautogui required for drag_drop"
        try:
            self._pyautogui.moveTo(*src, duration=0.3)
            self._pyautogui.dragTo(*dst, duration=0.5)
            return True, f"Dragged from {src} to {dst}"
        except Exception as e:
            return False, str(e)

    # ── Keyboard actions ──────────────────────────────────────

    def _type_text(self, step) -> tuple[bool, str]:
        text = step.params.get("text") or step.target or ""
        if not text:
            return False, "No text specified."

        if self._pyautogui:
            try:
                self._pyautogui.write(text, interval=0.03)
                return True, f"Typed {len(text)} characters"
            except Exception as e:
                return False, str(e)

        if self._has_xdotool:
            return self._xdotool_type(text)

        return False, "No typing tool available (install pyautogui or xdotool)"

    def _press_key(self, step) -> tuple[bool, str]:
        raw_key = step.target.strip().lower()
        key     = KEY_ALIASES.get(raw_key, raw_key)

        if self._pyautogui:
            try:
                self._pyautogui.press(key)
                return True, f"Pressed key: {key}"
            except Exception as e:
                return False, str(e)

        if self._has_xdotool:
            try:
                subprocess.run(
                    ["xdotool", "key", key],
                    capture_output=True, timeout=5
                )
                return True, f"Pressed key: {key}"
            except Exception as e:
                return False, str(e)

        return False, "No key press tool available"

    def _hotkey(self, step) -> tuple[bool, str]:
        combo = step.target.strip()
        keys  = step.params.get("keys") or combo.split("+")
        keys  = [k.strip().lower() for k in keys]
        keys  = [KEY_ALIASES.get(k, k) for k in keys]

        if self._pyautogui:
            try:
                self._pyautogui.hotkey(*keys)
                return True, f"Pressed hotkey: {'+'.join(keys)}"
            except Exception as e:
                return False, str(e)

        if self._has_xdotool:
            try:
                xdotool_combo = "+".join(keys)
                subprocess.run(
                    ["xdotool", "key", xdotool_combo],
                    capture_output=True, timeout=5
                )
                return True, f"Pressed hotkey: {xdotool_combo}"
            except Exception as e:
                return False, str(e)

        return False, "No hotkey tool available"

    # ── Scroll ────────────────────────────────────────────────

    def _scroll(self, step) -> tuple[bool, str]:
        direction = step.params.get("direction", step.target or "down").lower()
        amount    = int(step.params.get("amount", 3))

        # pyautogui scroll: positive = up, negative = down
        scroll_map = {"up": amount, "down": -amount, "left": 0, "right": 0}

        if self._pyautogui:
            try:
                dy = scroll_map.get(direction, -amount)
                self._pyautogui.scroll(dy)
                return True, f"Scrolled {direction} {amount} clicks"
            except Exception as e:
                return False, str(e)

        if self._has_xdotool:
            btn = {"up": 4, "down": 5, "left": 6, "right": 7}.get(direction, 5)
            try:
                for _ in range(amount):
                    subprocess.run(
                        ["xdotool", "click", str(btn)],
                        capture_output=True, timeout=5
                    )
                return True, f"Scrolled {direction} {amount}x via xdotool"
            except Exception as e:
                return False, str(e)

        return False, "No scroll tool available"

    # ── Window management ─────────────────────────────────────

    def _focus_window(self, step) -> tuple[bool, str]:
        target = step.target.strip()
        if not target:
            return False, "No window target specified."

        if _IS_WINDOWS:
            ok = _win32_focus(target)
            return (True, f"Focused window: {target!r}") if ok else \
                   (False, "Window not found (install pygetwindow or pywin32)")

        if self._has_xdotool:
            try:
                result = subprocess.run(
                    ["xdotool", "search", "--name", target],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    wid = result.stdout.strip().split()[0]
                    subprocess.run(
                        ["xdotool", "windowfocus", "--sync", wid],
                        capture_output=True, timeout=5
                    )
                    return True, f"Focused window: {target!r}"
            except Exception as e:
                log.debug("xdotool focus failed: %s", e)

        if self._has_wmctrl:
            try:
                subprocess.run(
                    ["wmctrl", "-a", target],
                    capture_output=True, timeout=5
                )
                return True, f"Focused window: {target!r} (wmctrl)"
            except Exception as e:
                return False, str(e)

        return False, "xdotool or wmctrl required for window focus"

    def _wait_window(self, step) -> tuple[bool, str]:
        target  = step.target.strip()
        timeout = step.timeout_sec or 10.0

        if self._has_xdotool:
            try:
                deadline = time.time() + timeout
                while time.time() < deadline:
                    result = subprocess.run(
                        ["xdotool", "search", "--name", target],
                        capture_output=True, text=True, timeout=3
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        return True, f"Window appeared: {target!r}"
                    time.sleep(0.5)
                return False, f"Window {target!r} did not appear within {timeout}s"
            except Exception as e:
                log.debug("wait_window error: %s", e)

        # Dumb fallback: just wait a bit
        time.sleep(min(timeout * 0.3, 3.0))
        return True, f"Waited ~{timeout*0.3:.0f}s for window {target!r}"

    def _maximize_window(self, step) -> tuple[bool, str]:
        if _IS_WINDOWS:
            return (True, "Maximized active window") if _win32_maximize() else \
                   (False, "Install pygetwindow or pywin32 for window control")
        if self._has_xdotool:
            try:
                subprocess.run(
                    ["xdotool", "getactivewindow", "windowmaximize"],
                    capture_output=True, timeout=5
                )
                return True, "Maximized active window"
            except Exception as e:
                return False, str(e)
        if self._has_wmctrl:
            try:
                subprocess.run(
                    ["wmctrl", "-r", ":ACTIVE:", "-b", "add,maximized_vert,maximized_horz"],
                    capture_output=True, timeout=5
                )
                return True, "Maximized active window (wmctrl)"
            except Exception as e:
                return False, str(e)
        return False, "xdotool or wmctrl required for maximize"

    def _minimize_window(self, step) -> tuple[bool, str]:
        if _IS_WINDOWS:
            return (True, "Minimized active window") if _win32_minimize() else \
                   (False, "Install pygetwindow or pywin32 for window control")
        if self._has_xdotool:
            try:
                subprocess.run(
                    ["xdotool", "getactivewindow", "windowminimize"],
                    capture_output=True, timeout=5
                )
                return True, "Minimized active window"
            except Exception as e:
                return False, str(e)
        return False, "xdotool required for minimize"

    def _close_window(self, step) -> tuple[bool, str]:
        if self._has_xdotool:
            try:
                subprocess.run(
                    ["xdotool", "getactivewindow", "windowclose"],
                    capture_output=True, timeout=5
                )
                return True, "Closed active window"
            except Exception as e:
                return False, str(e)
        if self._pyautogui:
            try:
                self._pyautogui.hotkey("alt", "F4")
                return True, "Closed window via Alt+F4"
            except Exception as e:
                return False, str(e)
        return False, "No close-window tool available"

    def _focus_active_window(self, step) -> tuple[bool, str]:
        if self._has_xdotool:
            try:
                subprocess.run(
                    ["xdotool", "getactivewindow", "windowfocus"],
                    capture_output=True, timeout=5
                )
                return True, "Focused active window"
            except Exception as e:
                log.debug("focus_active_window error: %s", e)
        time.sleep(0.1)
        return True, "Window focus assumed"

    # ── URL navigation ────────────────────────────────────────

    def _navigate_url(self, step) -> tuple[bool, str]:
        """
        Navigate an open browser to a URL.
        Strategy: focus address bar (Ctrl+L), type URL, press Enter.
        """
        url = step.target.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        if self._pyautogui:
            try:
                time.sleep(0.3)
                self._pyautogui.hotkey("ctrl", "l")      # focus address bar
                time.sleep(0.2)
                self._pyautogui.hotkey("ctrl", "a")      # select all
                self._pyautogui.write(url, interval=0.03)
                self._pyautogui.press("return")
                return True, f"Navigated to {url}"
            except Exception as e:
                log.warning("navigate_url pyautogui error: %s", e)

        if self._has_xdotool:
            try:
                subprocess.run(["xdotool", "key", "ctrl+l"], capture_output=True, timeout=3)
                time.sleep(0.2)
                subprocess.run(["xdotool", "type", "--clearmodifiers", url],
                               capture_output=True, timeout=5)
                subprocess.run(["xdotool", "key", "Return"], capture_output=True, timeout=3)
                return True, f"Navigated to {url} (xdotool)"
            except Exception as e:
                return False, str(e)

        return False, "pyautogui or xdotool required for URL navigation"

    # ── Screenshot ────────────────────────────────────────────

    def _screenshot(self, step) -> tuple[bool, str]:
        if self._pyautogui:
            try:
                from pathlib import Path
                save_dir = step.target or "data/screenshots"
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                ts   = int(time.time())
                path = f"{save_dir}/nexus_{ts}.png"
                img  = self._pyautogui.screenshot()
                img.save(path)
                return True, f"Screenshot saved: {path}"
            except Exception as e:
                return False, str(e)
        return False, "pyautogui required for screenshot"

    # ── OCR-based screen text finder ──────────────────────────

    def find_text_on_screen(self, text: str) -> Optional[tuple[int, int]]:
        """
        Find text on screen using OCR. Returns (x, y) center or None.
        Requires: pip install pytesseract pillow  +  sudo apt install tesseract-ocr
        """
        if not self._pyautogui:
            return None
        try:
            import pytesseract
            screenshot = self._pyautogui.screenshot()
            # Get per-word bounding boxes
            data = pytesseract.image_to_data(
                screenshot,
                output_type=pytesseract.Output.DICT,
                config="--psm 11",
            )
            target = text.lower().strip()
            for i, word in enumerate(data["text"]):
                if word and target in word.lower():
                    x = data["left"][i] + data["width"][i]  // 2
                    y = data["top"][i]  + data["height"][i] // 2
                    conf = int(data["conf"][i])
                    if conf > 40:   # ignore low-confidence noise
                        log.info("Found %r at (%d, %d) conf=%d", text, x, y, conf)
                        return x, y
        except ImportError:
            log.debug("pytesseract not installed — cannot find screen text")
        except Exception as e:
            log.debug("OCR search failed: %s", e)
        return None

    def smooth_move_and_click(
        self, x: int, y: int,
        button: str = "left",
        duration: float = 0.4,
    ) -> tuple[bool, str]:
        """Move mouse visibly to (x, y) then click."""
        if self._pyautogui:
            try:
                self._pyautogui.moveTo(x, y, duration=duration)
                time.sleep(0.1)
                self._pyautogui.click(x, y, button=button)
                return True, f"Moved and clicked ({x}, {y})"
            except Exception as e:
                return False, str(e)
        if self._has_xdotool:
            try:
                subprocess.run(
                    ["xdotool", "mousemove", "--sync", str(x), str(y)],
                    capture_output=True, timeout=5
                )
                time.sleep(0.15)
                subprocess.run(
                    ["xdotool", "click", "1"],
                    capture_output=True, timeout=5
                )
                return True, f"Moved and clicked ({x}, {y}) via xdotool"
            except Exception as e:
                return False, str(e)
        return False, "No mouse tool available"

    def _click_screen_text(self, step) -> tuple[bool, str]:
        """
        Find text on screen via OCR and click it.
        step.target = text to find and click
        step.params["button"] = "left" | "right" (default left)
        """
        text   = step.target.strip()
        button = step.params.get("button", "left")

        if not text:
            return False, "No text specified for click_screen_text"

        pos = self.find_text_on_screen(text)
        if pos:
            return self.smooth_move_and_click(pos[0], pos[1], button=button)

        # Fallback: try xdotool image search
        return False, f"Text {text!r} not found on screen (install tesseract for OCR)"

    def _click_menu_item(self, step) -> tuple[bool, str]:
        """
        Click a menu item by navigating the menu bar visually.
        step.target = "File" | "Edit" | "View" etc.
        step.params["item"]  = submenu item text (e.g. "Save As")
        """
        menu_name = step.target.strip()
        item_name = step.params.get("item", "").strip()

        # Step 1: Find and click the top-level menu
        pos = self.find_text_on_screen(menu_name)
        if not pos:
            # Many apps have File menu in top-left area — try a fixed fallback
            log.warning("Could not find menu %r via OCR — trying fixed position", menu_name)
            pos = (50, 25)   # rough guess for XFCE apps

        ok, out = self.smooth_move_and_click(pos[0], pos[1])
        if not ok:
            return False, f"Could not click menu {menu_name!r}: {out}"

        if not item_name:
            return True, f"Clicked menu: {menu_name!r}"

        # Step 2: Wait for menu to open, then find and click the item
        time.sleep(0.4)
        item_pos = self.find_text_on_screen(item_name)
        if not item_pos:
            # Try pressing Escape to close menu and report failure
            if self._pyautogui:
                self._pyautogui.press("escape")
            return False, f"Menu item {item_name!r} not found on screen"

        return self.smooth_move_and_click(item_pos[0], item_pos[1])

    # ── Target locator ────────────────────────────────────────

    def _find_target(self, target: str) -> Optional[tuple[int, int]]:
        """Try to find a target on screen. Returns (x, y) or None."""
        if not self._pyautogui:
            return None
        # Try image search (if .png path given)
        if target.endswith(".png") and __import__("os").path.exists(target):
            try:
                loc = self._pyautogui.locateCenterOnScreen(target, confidence=0.8)
                if loc:
                    return loc.x, loc.y
            except Exception:
                pass
        return None

    # ── xdotool helpers ───────────────────────────────────────

    def _xdotool_click_xy(self, x, y, clicks=1, button="left") -> tuple[bool, str]:
        btn_map = {"left": "1", "middle": "2", "right": "3"}
        btn = btn_map.get(button, "1")
        try:
            for _ in range(clicks):
                subprocess.run(
                    ["xdotool", "mousemove", str(x), str(y), "click", btn],
                    capture_output=True, timeout=5
                )
            return True, f"Clicked ({x}, {y}) via xdotool"
        except Exception as e:
            return False, str(e)

    def _xdotool_click_window(self, window_name: str) -> tuple[bool, str]:
        if not self._has_xdotool:
            return False, "xdotool not installed"
        try:
            result = subprocess.run(
                ["xdotool", "search", "--name", window_name],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                wid = result.stdout.strip().split()[0]
                subprocess.run(
                    ["xdotool", "windowfocus", wid, "click", "1"],
                    capture_output=True, timeout=5
                )
                return True, f"Clicked window: {window_name!r}"
        except Exception as e:
            log.debug("xdotool window click failed: %s", e)
        return False, f"Could not find/click window: {window_name!r}"

    def _xdotool_type(self, text: str) -> tuple[bool, str]:
        try:
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", "30", text],
                capture_output=True, timeout=30
            )
            return True, f"Typed {len(text)} characters via xdotool"
        except Exception as e:
            return False, str(e)


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    # Allow running as: python automation/gui_agent.py ...
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    logging.basicConfig(level=logging.INFO)

    agent = GUIAgent()

    print(f"\n─── GUIAgent Status ───")
    print(f"  pyautogui : {'✓' if agent._pyautogui else '✗ (pip install pyautogui pillow)'}")
    print(f"  xdotool   : {'✓' if agent._has_xdotool else '✗ (sudo apt install xdotool)'}")
    print(f"  wmctrl    : {'✓' if agent._has_wmctrl  else '✗ (sudo apt install wmctrl)'}")

    class FakeStep:
        def __init__(self, action, target="", params=None, timeout_sec=10.0):
            self.action      = action
            self.target      = target
            self.params      = params or {}
            self.timeout_sec = timeout_sec

    if "--press" in sys.argv:
        step = FakeStep("press_key", "return")
        ok, out = agent.run(step)
        print(f"\n  {'✓' if ok else '✗'} press_key: {out}")

    elif "--type" in sys.argv:
        idx  = sys.argv.index("--type")
        text = " ".join(sys.argv[idx + 1:]) if len(sys.argv) > idx + 1 else "hello NEXUS"
        step = FakeStep("type_text", params={"text": text})
        ok, out = agent.run(step)
        print(f"\n  {'✓' if ok else '✗'} type_text: {out}")

    else:
        print("\n  Usage:")
        print("    python automation/gui_agent.py --press")
        print("    python automation/gui_agent.py --type hello world")
