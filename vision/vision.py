"""
vision/vision.py
NEXUS Vision Brain — the main interface for all screen intelligence.

Ties together:
    capture.py  → screenshots
    ocr.py      → text extraction
    monitor.py  → live screen monitoring

This is what the rest of NEXUS imports when it needs eyes.

Usage:
    from vision.vision import Vision

    v = Vision()
    text = v.read_screen()
    v.watch(on_change=lambda e: print(e))
    v.find_text("error", timeout=30)
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("nexus.vision")


class Vision:
    """
    Single entry point for all NEXUS vision capabilities.

    Responsibilities:
        - Read text from the screen (one-shot or continuous)
        - Watch for screen changes and fire callbacks
        - Find specific text on screen and alert when found/lost
        - Feed screen context to the Brain on demand
        - Save screenshots with metadata

    Example:
        vision = Vision()

        # One-shot read
        text = vision.read_screen()
        print(text)

        # Watch for changes
        vision.watch(on_change=lambda e: print(f"Screen changed: {e.change_ratio:.1%}"))

        # Find text
        found = vision.find_text("hackthebox", timeout=60)
        if found:
            print("HackTheBox is on screen!")

        # Stop watching
        vision.stop()
    """

    def __init__(
        self,
        monitor_index: int = 1,
        ocr_enabled: bool = True,
        auto_save: bool = False,
        save_dir: Path = Path("data/screenshots"),
    ):
        self.monitor_index = monitor_index
        self.ocr_enabled   = ocr_enabled
        self.auto_save     = auto_save
        self.save_dir      = save_dir

        self._capturer  = None
        self._ocr       = None
        self._monitor   = None
        self._watching  = False

        log.info(
            "Vision initialized — monitor=%d, ocr=%s, auto_save=%s",
            monitor_index, ocr_enabled, auto_save,
        )

    # ─────────────────────────────────────────────
    # One-shot operations
    # ─────────────────────────────────────────────

    def read_screen(self, monitor: Optional[int] = None) -> str:
        """
        Capture the screen and return all visible text.

        Returns
        -------
        str — extracted text, empty string if OCR fails
        """
        try:
            capturer = self._get_capturer()
            shot     = capturer.capture(monitor or self.monitor_index)
            result   = self._get_ocr().extract(shot)

            if self.auto_save:
                shot.save(self.save_dir)

            log.info("read_screen: %d words extracted", result.word_count)
            return result.text

        except Exception as e:
            log.error("read_screen failed: %s", e)
            return ""

    def screenshot(self, save: bool = False) -> Optional[object]:
        """
        Take a screenshot and optionally save it.

        Returns
        -------
        Screenshot object (from vision.capture) or None on failure
        """
        try:
            capturer = self._get_capturer()
            shot     = capturer.capture(self.monitor_index)
            if save or self.auto_save:
                path = shot.save(self.save_dir)
                log.info("Screenshot saved: %s", path)
            return shot
        except Exception as e:
            log.error("screenshot failed: %s", e)
            return None

    def is_text_on_screen(self, text: str) -> bool:
        """
        Check if specific text is currently visible on screen.

        Parameters
        ----------
        text : str — text to search for (case-insensitive)

        Returns
        -------
        bool
        """
        screen_text = self.read_screen()
        return text.lower() in screen_text.lower()

    def find_text(
        self,
        text: str,
        timeout: float = 30.0,
        interval: float = 2.0,
    ) -> bool:
        """
        Poll the screen until text appears or timeout expires.

        Parameters
        ----------
        text     : str   — text to search for
        timeout  : float — max seconds to wait
        interval : float — seconds between checks

        Returns
        -------
        bool — True if found before timeout
        """
        log.info("Waiting for text %r on screen (timeout=%.1fs)...", text, timeout)
        start = time.time()

        while time.time() - start < timeout:
            if self.is_text_on_screen(text):
                log.info("Text %r found on screen.", text)
                return True
            time.sleep(interval)

        log.info("Text %r not found within %.1fs.", text, timeout)
        return False

    def get_context(self, max_words: int = 200) -> str:
        """
        Get a concise summary of what's currently on screen.
        Used by the Brain to understand what the user is doing.

        Returns
        -------
        str — first `max_words` words of screen text
        """
        text  = self.read_screen()
        words = text.split()

        if not words:
            return ""

        if len(words) <= max_words:
            return text

        summary = " ".join(words[:max_words])
        return summary + f" ... [{len(words) - max_words} more words]"

    def list_monitors(self) -> list[dict]:
        """Return available monitors."""
        return self._get_capturer().list_monitors()

    # ─────────────────────────────────────────────
    # Continuous monitoring
    # ─────────────────────────────────────────────

    def watch(
        self,
        on_change:     Optional[Callable] = None,
        on_text_found: Optional[Callable] = None,
        on_text_lost:  Optional[Callable] = None,
        watch_for:     Optional[list[str]] = None,
        interval:      float = 2.0,
        change_threshold: float = 0.03,
    ) -> None:
        """
        Start watching the screen in the background.

        Parameters
        ----------
        on_change     : called whenever the screen changes significantly
        on_text_found : called when a watched text string appears
        on_text_lost  : called when a watched text string disappears
        watch_for     : list of text strings to watch for
        interval      : seconds between screen checks
        change_threshold : fraction of pixels that must change (0.03 = 3%)
        """
        if self._watching:
            log.warning("Vision is already watching. Call stop() first.")
            return

        from vision.monitor import ScreenMonitor

        self._monitor = ScreenMonitor(
            monitor_index    = self.monitor_index,
            interval_sec     = interval,
            change_threshold = change_threshold,
            ocr_enabled      = bool(watch_for or on_text_found or on_text_lost),
            watch_for        = watch_for or [],
            on_change        = on_change,
            on_text_found    = on_text_found,
            on_text_lost     = on_text_lost,
            save_on_change   = self.auto_save,
        )
        self._monitor.start()
        self._watching = True
        log.info("Vision watching started.")

    def stop(self) -> None:
        """Stop background screen monitoring."""
        if self._monitor and self._watching:
            self._monitor.stop()
            self._watching = False
            log.info("Vision watching stopped.")

    def wait_for_change(self, timeout: float = 30.0) -> Optional[object]:
        """
        Block until the screen changes or timeout expires.

        Returns ScreenEvent or None.
        """
        if not self._watching:
            self.watch()

        return self._monitor.wait_for_change(timeout=timeout)

    @property
    def is_watching(self) -> bool:
        return self._watching

    # ─────────────────────────────────────────────
    # Brain integration
    # ─────────────────────────────────────────────

    def answer_about_screen(self, question: str) -> str:
        """
        Given a question about what's on screen, return a relevant answer.
        Uses OCR to get screen text, then does basic keyword matching.

        For smarter answers, the Brain should call get_context() and
        pass it to the LLM.

        Examples:
            vision.answer_about_screen("what app is open")
            vision.answer_about_screen("is there an error on screen")
        """
        text = self.read_screen()
        q    = question.lower()

        if not text:
            return "I can't see anything on the screen right now."

        # Error detection
        if any(w in q for w in ["error", "fail", "crash", "exception"]):
            lines = [l for l in text.splitlines()
                     if any(w in l.lower() for w in ["error", "fail", "exception", "traceback"])]
            if lines:
                return "I see errors on screen:\n" + "\n".join(lines[:5])
            return "No errors visible on screen."

        # What's open
        if any(w in q for w in ["what", "open", "app", "application", "window"]):
            first_lines = [l.strip() for l in text.splitlines() if l.strip()][:3]
            if first_lines:
                return "Screen shows: " + " | ".join(first_lines)
            return "Screen content is unclear."

        # Contains search
        if "is there" in q or "can you see" in q or "do you see" in q:
            # Extract what they're looking for
            for phrase in ["is there", "can you see", "do you see"]:
                if phrase in q:
                    target = q.split(phrase)[-1].strip().strip("?")
                    if target and target in text.lower():
                        return f"Yes, I can see '{target}' on screen."
                    elif target:
                        return f"No, I don't see '{target}' on screen."

        # Fallback: return a snippet
        words   = text.split()
        snippet = " ".join(words[:50])
        return f"Screen contains: {snippet}..."

    # ─────────────────────────────────────────────
    # Lazy loaders
    # ─────────────────────────────────────────────

    def _get_capturer(self):
        if self._capturer is None:
            from vision.capture import ScreenCapturer
            self._capturer = ScreenCapturer(
                default_monitor=self.monitor_index,
                auto_save=self.auto_save,
                save_dir=self.save_dir,
            )
        return self._capturer

    def _get_ocr(self):
        if self._ocr is None:
            from vision.ocr import OCREngine
            self._ocr = OCREngine()
        return self._ocr

    # ─────────────────────────────────────────────
    # Context manager
    # ─────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()


# ─────────────────────────────────────────────
# Module-level convenience
# ─────────────────────────────────────────────

_default_vision: Optional[Vision] = None


def get_vision() -> Vision:
    """Return the shared Vision instance (created on first call)."""
    global _default_vision
    if _default_vision is None:
        _default_vision = Vision()
    return _default_vision


# ─────────────────────────────────────────────
# CLI test — python vision/vision.py
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Fix module shadowing when running standalone
    current_dir = str(Path(__file__).resolve().parent)
    if current_dir in sys.path:
        sys.path.remove(current_dir)
    if "" in sys.path:
        sys.path.remove("")
    root_dir = str(Path(__file__).resolve().parent.parent)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    logging.basicConfig(level=logging.INFO)

    print("\n─── NEXUS Vision Test ───\n")
    v = Vision()

    # ── --read ────────────────────────────────────────────────
    if "--read" in sys.argv:
        print("Reading screen...\n")
        text = v.read_screen()
        lines = [l for l in text.splitlines() if l.strip()]
        for line in lines[:20]:
            print(f"  {line}")
        if len(lines) > 20:
            print(f"  ... ({len(lines) - 20} more lines)")

    # ── --find <text> ─────────────────────────────────────────
    elif "--find" in sys.argv:
        idx  = sys.argv.index("--find")
        term = " ".join(sys.argv[idx + 1:])
        print(f"Looking for: {term!r}\n")
        found = v.find_text(term, timeout=30)
        print("✓ Found!" if found else "✗ Not found.")

    # ── --watch ───────────────────────────────────────────────
    elif "--watch" in sys.argv:
        print("Watching for screen changes... (Ctrl+C to stop)\n")

        def on_change(event):
            print(f"  ⚡ Change: {event.change_ratio:.1%} pixels")

        v.watch(on_change=on_change, interval=1.5)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            v.stop()
            print("\nStopped.")

    # ── --screenshot ──────────────────────────────────────────
    elif "--screenshot" in sys.argv:
        print("Taking screenshot...\n")
        shot = v.screenshot(save=True)
        if shot:
            print(f"  Saved: {shot.saved_path}")
            print(f"  Size : {shot.width}×{shot.height}")

    # ── --context ─────────────────────────────────────────────
    elif "--context" in sys.argv:
        print("Getting screen context for Brain...\n")
        ctx = v.get_context()
        print(f"  {ctx[:500]}...")

    # ── --ask ─────────────────────────────────────────────────
    elif "--ask" in sys.argv:
        idx = sys.argv.index("--ask")
        q   = " ".join(sys.argv[idx + 1:])
        print(f"Question: {q!r}\n")
        answer = v.answer_about_screen(q)
        print(f"  {answer}")

    # ── default ───────────────────────────────────────────────
    else:
        print("Usage:")
        print("  python vision/vision.py --read                    # read screen text")
        print("  python vision/vision.py --find <text>             # wait for text")
        print("  python vision/vision.py --watch                   # watch for changes")
        print("  python vision/vision.py --screenshot              # take and save screenshot")
        print("  python vision/vision.py --context                 # get screen context")
        print("  python vision/vision.py --ask is there an error   # ask about screen")