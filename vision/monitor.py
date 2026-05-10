"""
NEXUS — vision/monitor.py
Live screen monitoring — watches the screen for changes and events.

Sits above the OCR layer:

    [SCREEN]
         ↓
    capture.py
         ↓
    ocr.py
         ↓
    monitor.py         ← YOU ARE HERE
         ↓
    vision.py (brain)

Responsibilities:
  - Watch the screen continuously at a configurable interval
  - Detect when the screen changes significantly
  - Fire callbacks when specific text appears or disappears
  - Detect active window / application changes
  - Provide a diff between two screenshots

Dependencies:
    pip install mss pillow opencv-python pytesseract
"""

from __future__ import annotations

import time
import threading
import logging
from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("nexus.vision.monitor")

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

DEFAULT_INTERVAL_SEC  = 2.0    # seconds between captures
DEFAULT_CHANGE_THRESH = 0.03   # fraction of pixels that must change to trigger event
                                # 0.03 = 3% of pixels changed


# ─────────────────────────────────────────────────────────────
#  SCREEN EVENT
# ─────────────────────────────────────────────────────────────

@dataclass
class ScreenEvent:
    """
    Fired when the screen monitor detects something noteworthy.

    Attributes
    ----------
    event_type   : "change" | "text_found" | "text_lost" | "idle" | "active"
    change_ratio : fraction of pixels that changed (0.0 – 1.0)
    screenshot   : the current Screenshot at time of event
    ocr_result   : OCR result if text scanning is enabled
    trigger_text : the text that triggered the event (for text_found/lost)
    timestamp    : time.time() of event
    """
    event_type:   str
    change_ratio: float
    screenshot:   object          # vision.capture.Screenshot
    ocr_result:   Optional[object] = None   # vision.ocr.OCRResult
    trigger_text: str             = ""
    timestamp:    float           = field(default_factory=time.time)

    def __repr__(self) -> str:
        return (
            f"ScreenEvent(type={self.event_type!r}, "
            f"change={self.change_ratio:.1%}, "
            f"trigger={self.trigger_text!r})"
        )


# ─────────────────────────────────────────────────────────────
#  SCREEN DIFF
# ─────────────────────────────────────────────────────────────

def _compute_change_ratio(
    prev: np.ndarray,
    curr: np.ndarray,
    threshold: int = 30,
) -> float:
    """
    Compute the fraction of pixels that changed between two frames.

    Parameters
    ----------
    prev, curr  : numpy arrays (H, W, 3) BGR
    threshold   : per-channel difference to count as "changed"

    Returns
    -------
    float in [0.0, 1.0]
    """
    if prev.shape != curr.shape:
        return 1.0   # shapes differ — treat as full change

    diff        = np.abs(prev.astype(np.int16) - curr.astype(np.int16))
    changed     = np.any(diff > threshold, axis=2)
    return float(changed.sum()) / changed.size


def _resize_for_diff(array: np.ndarray, scale: float = 0.25) -> np.ndarray:
    """Downscale array for fast change detection (avoids full-res comparison)."""
    if not _CV2_AVAILABLE:
        # Fallback: simple slice-based downsample
        step = max(1, int(1 / scale))
        return array[::step, ::step]
    h, w  = array.shape[:2]
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    return cv2.resize(array, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ─────────────────────────────────────────────────────────────
#  SCREEN MONITOR
# ─────────────────────────────────────────────────────────────

class ScreenMonitor:
    """
    Continuously watches the screen and fires events on changes.

    Usage — callback mode (runs in background):
        def on_change(event):
            print(f"Screen changed: {event.change_ratio:.1%}")

        monitor = ScreenMonitor(on_change=on_change)
        monitor.start()
        # ... do other things ...
        monitor.stop()

    Usage — text watching:
        monitor = ScreenMonitor(
            watch_for=["error", "warning", "exception"],
            on_text_found=lambda e: print(f"Found: {e.trigger_text}")
        )
        monitor.start()

    Usage — blocking one-shot:
        monitor = ScreenMonitor()
        event = monitor.wait_for_change(timeout=30.0)
        if event:
            print("Screen changed!")
    """

    def __init__(
        self,
        monitor_index: int = 1,
        interval_sec: float = DEFAULT_INTERVAL_SEC,
        change_threshold: float = DEFAULT_CHANGE_THRESH,
        ocr_enabled: bool = False,
        watch_for: Optional[list[str]] = None,
        on_change: Optional[Callable[[ScreenEvent], None]] = None,
        on_text_found: Optional[Callable[[ScreenEvent], None]] = None,
        on_text_lost: Optional[Callable[[ScreenEvent], None]] = None,
        save_on_change: bool = False,
    ):
        self.monitor_index    = monitor_index
        self.interval_sec     = interval_sec
        self.change_threshold = change_threshold
        self.ocr_enabled      = ocr_enabled
        self.watch_for        = [t.lower() for t in (watch_for or [])]
        self.on_change        = on_change
        self.on_text_found    = on_text_found
        self.on_text_lost     = on_text_lost
        self.save_on_change   = save_on_change

        self._running         = False
        self._thread: Optional[threading.Thread] = None
        self._change_event    = threading.Event()
        self._last_event: Optional[ScreenEvent] = None

        # State tracking
        self._prev_array: Optional[np.ndarray] = None
        self._active_texts: set[str] = set()

        # Lazy-load capturer and OCR
        self._capturer = None
        self._ocr      = None

    # ── lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="nexus-screen-monitor",
        )
        self._thread.start()
        logger.info(
            "ScreenMonitor started — interval=%.1fs, ocr=%s, watching=%s",
            self.interval_sec, self.ocr_enabled, self.watch_for,
        )

    def stop(self) -> None:
        """Stop the monitoring thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.interval_sec + 2)
        logger.info("ScreenMonitor stopped.")

    def __enter__(self) -> "ScreenMonitor":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ── blocking API ──────────────────────────────────────────

    def wait_for_change(
        self,
        timeout: Optional[float] = None,
    ) -> Optional[ScreenEvent]:
        """
        Block until the screen changes or timeout expires.

        Returns ScreenEvent on change, None on timeout.
        """
        if not self._running:
            self.start()
        self._change_event.clear()
        fired = self._change_event.wait(timeout=timeout)
        if fired:
            return self._last_event
        return None

    def snapshot(self) -> "ScreenEvent":
        """
        Take a single screenshot + optional OCR right now.
        Does not require the monitor to be running.
        """
        capturer = self._get_capturer()
        shot     = capturer.capture(self.monitor_index)
        ocr_res  = None

        if self.ocr_enabled:
            ocr_engine = self._get_ocr()
            ocr_res    = ocr_engine.extract(shot)

        return ScreenEvent(
            event_type   = "snapshot",
            change_ratio = 0.0,
            screenshot   = shot,
            ocr_result   = ocr_res,
        )

    def read_screen_text(self) -> str:
        """
        Convenience: capture screen and return extracted text immediately.
        """
        capturer   = self._get_capturer()
        ocr_engine = self._get_ocr()
        shot       = capturer.capture(self.monitor_index)
        result     = ocr_engine.extract(shot)
        return result.text

    # ── main loop ─────────────────────────────────────────────

    def _monitor_loop(self) -> None:
        capturer   = self._get_capturer()
        ocr_engine = self._get_ocr() if self.ocr_enabled or self.watch_for else None

        while self._running:
            try:
                shot    = capturer.capture(self.monitor_index)
                small   = _resize_for_diff(shot.array)

                # ── Change detection ──────────────────────────
                change_ratio = 0.0
                if self._prev_array is not None:
                    change_ratio = _compute_change_ratio(
                        self._prev_array, small
                    )

                self._prev_array = small.copy()

                changed = change_ratio >= self.change_threshold

                # ── OCR + text watching ────────────────────────
                ocr_result = None
                if ocr_engine and (changed or self.watch_for):
                    ocr_result = ocr_engine.extract(shot)
                    self._check_text_events(shot, ocr_result, change_ratio)

                # ── Fire change event ──────────────────────────
                if changed:
                    event = ScreenEvent(
                        event_type   = "change",
                        change_ratio = change_ratio,
                        screenshot   = shot,
                        ocr_result   = ocr_result,
                    )
                    self._last_event = event
                    self._change_event.set()

                    if self.save_on_change:
                        shot.save()

                    if self.on_change:
                        try:
                            self.on_change(event)
                        except Exception as exc:
                            logger.error("on_change callback error: %s", exc)

                    logger.debug(
                        "Screen change detected: %.1f%%", change_ratio * 100
                    )

            except Exception as exc:
                logger.error("Monitor loop error: %s", exc)

            time.sleep(self.interval_sec)

    def _check_text_events(self, shot, ocr_result, change_ratio: float) -> None:
        """Check if watched text has appeared or disappeared."""
        if not self.watch_for or ocr_result is None:
            return

        current_texts = set()
        for term in self.watch_for:
            if ocr_result.contains(term):
                current_texts.add(term)

        # Newly appeared
        appeared = current_texts - self._active_texts
        for term in appeared:
            logger.info("Watched text appeared on screen: %r", term)
            if self.on_text_found:
                event = ScreenEvent(
                    event_type   = "text_found",
                    change_ratio = change_ratio,
                    screenshot   = shot,
                    ocr_result   = ocr_result,
                    trigger_text = term,
                )
                try:
                    self.on_text_found(event)
                except Exception as exc:
                    logger.error("on_text_found callback error: %s", exc)

        # Disappeared
        lost = self._active_texts - current_texts
        for term in lost:
            logger.info("Watched text disappeared from screen: %r", term)
            if self.on_text_lost:
                event = ScreenEvent(
                    event_type   = "text_lost",
                    change_ratio = change_ratio,
                    screenshot   = shot,
                    ocr_result   = ocr_result,
                    trigger_text = term,
                )
                try:
                    self.on_text_lost(event)
                except Exception as exc:
                    logger.error("on_text_lost callback error: %s", exc)

        self._active_texts = current_texts

    # ── lazy loaders ──────────────────────────────────────────

    def _get_capturer(self):
        if self._capturer is None:
            try:
                from vision.capture import ScreenCapturer
            except ModuleNotFoundError:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from vision.capture import ScreenCapturer
            self._capturer = ScreenCapturer()
        return self._capturer

    def _get_ocr(self):
        if self._ocr is None:
            try:
                from vision.ocr import OCREngine
            except ModuleNotFoundError:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from vision.ocr import OCREngine
            self._ocr = OCREngine()
        return self._ocr


# ─────────────────────────────────────────────────────────────
#  STANDALONE ENTRY — python vision/monitor.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n  NEXUS — vision/monitor.py")
    print("  " + "─" * 40)

    # ── --watch: detect screen changes ───────────────────────
    if "--watch" in sys.argv:
        duration = 30
        if "--for" in sys.argv:
            idx      = sys.argv.index("--for")
            duration = int(sys.argv[idx + 1])

        print(f"\n  Watching for screen changes for {duration}s...\n")
        changes = []

        def on_change(event: ScreenEvent):
            changes.append(event)
            print(
                f"  \033[93m⚡ Change detected\033[0m  "
                f"{event.change_ratio:.1%} pixels changed  "
                f"({time.strftime('%H:%M:%S')})"
            )

        monitor = ScreenMonitor(on_change=on_change, interval_sec=1.0)
        monitor.start()
        time.sleep(duration)
        monitor.stop()
        print(f"\n  Total changes detected: {len(changes)}\n")

    # ── --find: watch for specific text ──────────────────────
    elif "--find" in sys.argv:
        idx   = sys.argv.index("--find")
        terms = sys.argv[idx + 1:]
        print(f"\n  Watching for text: {terms}")
        print(f"  Press Ctrl-C to stop.\n")

        def on_found(event: ScreenEvent):
            print(f"  \033[92m✓ Found:\033[0m '{event.trigger_text}'")

        def on_lost(event: ScreenEvent):
            print(f"  \033[91m✗ Lost:\033[0m '{event.trigger_text}'")

        monitor = ScreenMonitor(
            watch_for     = terms,
            on_text_found = on_found,
            on_text_lost  = on_lost,
            ocr_enabled   = True,
            interval_sec  = 2.0,
        )

        try:
            monitor.start()
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  Stopped.")
            monitor.stop()

    # ── --read: one-shot read screen text ─────────────────────
    elif "--read" in sys.argv:
        print("\n  Reading screen text...\n")
        monitor = ScreenMonitor(ocr_enabled=True)
        text    = monitor.read_screen_text()
        lines   = [l for l in text.splitlines() if l.strip()]
        for line in lines[:30]:
            print(f"  {line}")
        if len(lines) > 30:
            print(f"  ... ({len(lines) - 30} more lines)")
        print()

    else:
        print()
        print("  Usage:")
        print("    python vision/monitor.py --watch           # detect screen changes (30s)")
        print("    python vision/monitor.py --watch --for 60  # watch for 60 seconds")
        print("    python vision/monitor.py --find error warn # alert when text appears")
        print("    python vision/monitor.py --read            # read current screen text")
        print()