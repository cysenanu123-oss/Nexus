"""
NEXUS — vision/capture.py
Screen capture layer — the eyes of NEXUS.

Sits at the bottom of the vision pipeline:

    [SCREEN]
         ↓
    capture.py         ← YOU ARE HERE
         ↓
    ocr.py
         ↓
    monitor.py
         ↓
    vision.py (brain)

Responsibilities:
  - Capture the full screen or a specific region
  - Capture a specific monitor in multi-monitor setups
  - Save screenshots to disk optionally
  - Return numpy arrays or PIL images for upstream processing
  - Expose a clean interface to the layers above

Dependencies:
    pip install mss pillow opencv-python
"""

from __future__ import annotations

import time
import logging
import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("nexus.vision.capture")

# ── lazy imports ──────────────────────────────────────────────
try:
    import mss
    import mss.tools
    _MSS_AVAILABLE = True
except ImportError:
    _MSS_AVAILABLE = False

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

SCREENSHOTS_DIR = Path("data/screenshots")
DEFAULT_MONITOR = 1        # 1 = primary monitor (0 = all monitors combined)


# ─────────────────────────────────────────────────────────────
#  SCREENSHOT RESULT
# ─────────────────────────────────────────────────────────────

class Screenshot:
    """
    Result of a screen capture operation.

    Attributes
    ----------
    image      : PIL Image (RGB)
    array      : numpy uint8 array (H, W, 3) in BGR for OpenCV
    width      : int
    height     : int
    monitor    : int — which monitor was captured
    timestamp  : float — time.time() of capture
    saved_path : Path | None — path if saved to disk
    """

    def __init__(
        self,
        image: "Image.Image",
        array: np.ndarray,
        monitor: int,
    ):
        self.image      = image
        self.array      = array
        self.width      = image.width
        self.height     = image.height
        self.monitor    = monitor
        self.timestamp  = time.time()
        self.saved_path: Optional[Path] = None

    def save(self, directory: Path = SCREENSHOTS_DIR, prefix: str = "nexus") -> Path:
        """Save screenshot to disk with a timestamped filename."""
        directory.mkdir(parents=True, exist_ok=True)
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = directory / f"{prefix}_{ts}.png"
        self.image.save(str(path))
        self.saved_path = path
        logger.debug("Screenshot saved: %s", path)
        return path

    def to_gray(self) -> np.ndarray:
        """Return grayscale numpy array (H, W) — useful for OCR."""
        if _CV2_AVAILABLE:
            return cv2.cvtColor(self.array, cv2.COLOR_BGR2GRAY)
        return np.array(self.image.convert("L"))

    def crop(self, x: int, y: int, w: int, h: int) -> "Screenshot":
        """Return a new Screenshot cropped to the given region."""
        region  = self.image.crop((x, y, x + w, y + h))
        arr     = np.array(region)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1].copy()  # RGB → BGR
        return Screenshot(region, arr, self.monitor)

    def __repr__(self) -> str:
        return (
            f"Screenshot(monitor={self.monitor}, "
            f"{self.width}×{self.height}, "
            f"saved={self.saved_path is not None})"
        )


# ─────────────────────────────────────────────────────────────
#  SCREEN CAPTURER
# ─────────────────────────────────────────────────────────────

class ScreenCapturer:
    """
    Captures screenshots using MSS (cross-platform, fast).

    Usage:
        capturer = ScreenCapturer()

        # Capture primary monitor
        shot = capturer.capture()

        # Capture specific monitor
        shot = capturer.capture(monitor=2)

        # Capture a region (x, y, width, height)
        shot = capturer.capture_region(100, 100, 800, 600)

        # List available monitors
        capturer.list_monitors()
    """

    def __init__(
        self,
        default_monitor: int = DEFAULT_MONITOR,
        auto_save: bool = False,
        save_dir: Path = SCREENSHOTS_DIR,
    ):
        if not _MSS_AVAILABLE:
            raise ImportError(
                "mss not installed. Run: pip install mss"
            )
        if not _PIL_AVAILABLE:
            raise ImportError(
                "Pillow not installed. Run: pip install pillow"
            )

        self.default_monitor = default_monitor
        self.auto_save       = auto_save
        self.save_dir        = save_dir

    def capture(self, monitor: Optional[int] = None) -> Screenshot:
        """
        Capture a full monitor screenshot.

        Parameters
        ----------
        monitor : int | None
            Monitor index (1 = primary). None uses default_monitor.

        Returns
        -------
        Screenshot object
        """
        mon_idx = monitor if monitor is not None else self.default_monitor

        with mss.mss() as sct:
            monitors = sct.monitors
            if mon_idx >= len(monitors):
                logger.warning(
                    "Monitor %d not found (%d available). Using monitor 1.",
                    mon_idx, len(monitors) - 1,
                )
                mon_idx = 1

            mon     = monitors[mon_idx]
            raw     = sct.grab(mon)
            image   = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
            array   = np.array(image)[:, :, ::-1].copy()   # RGB → BGR for OpenCV

        shot = Screenshot(image, array, mon_idx)

        if self.auto_save:
            shot.save(self.save_dir)

        logger.debug(
            "Captured monitor %d: %d×%d px", mon_idx, shot.width, shot.height
        )
        return shot

    def capture_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        monitor: Optional[int] = None,
    ) -> Screenshot:
        """
        Capture a specific rectangular region of a monitor.

        Parameters
        ----------
        x, y         : top-left corner of the region
        width, height: size of the region
        monitor      : monitor index (None = default)
        """
        mon_idx = monitor if monitor is not None else self.default_monitor

        region = {"top": y, "left": x, "width": width, "height": height,
                  "mon": mon_idx}

        with mss.mss() as sct:
            raw   = sct.grab(region)
            image = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
            array = np.array(image)[:, :, ::-1].copy()

        shot = Screenshot(image, array, mon_idx)

        if self.auto_save:
            shot.save(self.save_dir)

        logger.debug(
            "Captured region (%d,%d) %d×%d px", x, y, width, height
        )
        return shot

    def capture_window_title(self, title: str) -> Optional[Screenshot]:
        """
        Try to find and capture a window by its title.
        Falls back to full screen if window not found.
        Requires xdotool on Linux.
        """
        import subprocess, shutil

        if not shutil.which("xdotool"):
            logger.warning("xdotool not found — capturing full screen instead.")
            return self.capture()

        try:
            # Get window geometry
            result = subprocess.run(
                ["xdotool", "search", "--name", title, "getwindowgeometry"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode != 0:
                logger.warning("Window '%s' not found.", title)
                return self.capture()

            lines = result.stdout.strip().splitlines()
            pos_line  = next((l for l in lines if "Position" in l), None)
            size_line = next((l for l in lines if "Geometry" in l), None)

            if not pos_line or not size_line:
                return self.capture()

            x, y = map(int, pos_line.split(":")[1].strip().split(","))
            w, h = map(int, size_line.split(":")[1].strip().split("x"))

            return self.capture_region(x, y, w, h)

        except Exception as exc:
            logger.error("Window capture failed: %s", exc)
            return self.capture()

    def list_monitors(self) -> list[dict]:
        """Return a list of available monitors with their dimensions."""
        with mss.mss() as sct:
            monitors = []
            for i, mon in enumerate(sct.monitors):
                if i == 0:
                    continue  # skip virtual "all monitors" entry
                monitors.append({ 
                    "index":  i,
                    "left":   mon["left"],
                    "top":    mon["top"],
                    "width":  mon["width"],
                    "height": mon["height"],
                })
        return monitors

    def print_monitors(self) -> None:
        """Print available monitors to terminal."""
        monitors = self.list_monitors()
        print(f"\n  {'IDX':<5} {'RESOLUTION':<16} {'POSITION'}")
        print("  " + "─" * 40)
        for m in monitors:
            print(
                f"  {m['index']:<5} "
                f"{m['width']}×{m['height']:<10} "
                f"({m['left']}, {m['top']})"
            )
        print()


# ─────────────────────────────────────────────────────────────
#  CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────

_default_capturer: Optional[ScreenCapturer] = None


def capture(monitor: int = DEFAULT_MONITOR, save: bool = False) -> Screenshot:
    """
    One-shot screen capture.

    Usage:
        from vision.capture import capture
        shot = capture()
        shot.save()
    """
    global _default_capturer
    if _default_capturer is None:
        _default_capturer = ScreenCapturer(auto_save=save)
    return _default_capturer.capture(monitor)


# ─────────────────────────────────────────────────────────────
#  STANDALONE ENTRY — python vision/capture.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n  NEXUS — vision/capture.py")
    print("  " + "─" * 40)

    if not _MSS_AVAILABLE:
        print("  ✗ mss not installed. Run: pip install mss")
        sys.exit(1)

    capturer = ScreenCapturer()

    if "--monitors" in sys.argv:
        print("\n  Available monitors:")
        capturer.print_monitors()

    elif "--capture" in sys.argv:
        print("\n  Capturing primary monitor...")
        shot = capturer.capture()
        path = shot.save()
        print(f"  ✓ Screenshot saved: {path}")
        print(f"  Size: {shot.width}×{shot.height} px")

    elif "--region" in sys.argv:
        # Usage: --region x y w h
        idx = sys.argv.index("--region")
        x, y, w, h = int(sys.argv[idx+1]), int(sys.argv[idx+2]), \
                     int(sys.argv[idx+3]), int(sys.argv[idx+4])
        print(f"\n  Capturing region ({x},{y}) {w}×{h}...")
        shot = capturer.capture_region(x, y, w, h)
        path = shot.save()
        print(f"  ✓ Region saved: {path}")

    else:
        # Default: capture and show info
        print("\n  Capturing screen...")
        shot = capturer.capture()
        print(f"  Monitor : {shot.monitor}")
        print(f"  Size    : {shot.width}×{shot.height} px")
        print(f"  Array   : {shot.array.shape} {shot.array.dtype}")
        print()
        print("  Usage:")
        print("    python vision/capture.py --monitors       # list monitors")
        print("    python vision/capture.py --capture        # capture + save")
        print("    python vision/capture.py --region x y w h # capture region")
        print()