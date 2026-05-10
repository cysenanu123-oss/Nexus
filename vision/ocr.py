"""
NEXUS — vision/ocr.py
OCR layer — extracts text from screenshots.

Sits above the capture layer:

    [SCREEN]
         ↓
    capture.py
         ↓
    ocr.py             ← YOU ARE HERE
         ↓
    monitor.py
         ↓
    vision.py (brain)

Responsibilities:
  - Extract text from Screenshot objects or image files
  - Return structured OCRResult with text, confidence, word boxes
  - Support full-image and region-specific OCR
  - Pre-process images for better accuracy (threshold, denoise)

Dependencies:
    pip install pytesseract pillow opencv-python
    sudo apt install tesseract-ocr -y
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("nexus.vision.ocr")

# ── lazy imports ──────────────────────────────────────────────
try:
    import pytesseract
    _TESS_AVAILABLE = True
except ImportError:
    _TESS_AVAILABLE = False

try:
    from PIL import Image, ImageFilter, ImageEnhance
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
#  OCR RESULT
# ─────────────────────────────────────────────────────────────

@dataclass
class OCRResult:
    """
    Result of an OCR extraction.

    Attributes
    ----------
    text       : full extracted text (stripped)
    lines      : list of non-empty text lines
    words      : list of individual words
    confidence : mean confidence score 0–100 (Tesseract)
    raw        : raw Tesseract output string
    """
    text:       str         = ""
    lines:      list[str]   = field(default_factory=list)
    words:      list[str]   = field(default_factory=list)
    confidence: float       = 0.0
    raw:        str         = ""

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) == 0

    @property
    def word_count(self) -> int:
        return len(self.words)

    def contains(self, query: str, case_sensitive: bool = False) -> bool:
        """Check if the extracted text contains a query string."""
        if case_sensitive:
            return query in self.text
        return query.lower() in self.text.lower()

    def find_pattern(self, pattern: str) -> list[str]:
        """Find all regex matches in the extracted text."""
        return re.findall(pattern, self.text)

    def __str__(self) -> str:
        if self.is_empty:
            return "[no text detected]"
        preview = self.text[:100].replace("\n", " ")
        if len(self.text) > 100:
            preview += "..."
        return f'OCRResult({self.word_count} words, conf={self.confidence:.0f}%): "{preview}"'


# ─────────────────────────────────────────────────────────────
#  IMAGE PRE-PROCESSING
# ─────────────────────────────────────────────────────────────

def _preprocess_for_ocr(image: "Image.Image", mode: str = "auto") -> "Image.Image":
    """
    Pre-process a PIL image to improve OCR accuracy.

    Modes:
        auto     — auto-select based on image content
        thresh   — binary threshold (good for dark text on light bg)
        denoise  — gaussian blur then threshold
        enhance  — contrast + sharpness boost
        raw      — no processing
    """
    # Convert to grayscale first
    gray = image.convert("L")

    if mode == "raw":
        return gray

    if mode == "enhance" or mode == "auto":
        # Boost contrast
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(2.0)
        # Boost sharpness
        enhancer = ImageEnhance.Sharpness(gray)
        gray = enhancer.enhance(2.0)

    if mode == "thresh":
        # Apply threshold using numpy
        arr = np.array(gray)
        arr = np.where(arr > 128, 255, 0).astype(np.uint8)
        gray = Image.fromarray(arr)

    if mode == "denoise" and _CV2_AVAILABLE:
        arr = np.array(gray)
        arr = cv2.GaussianBlur(arr, (3, 3), 0)
        _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gray = Image.fromarray(arr)

    return gray


# ─────────────────────────────────────────────────────────────
#  OCR ENGINE
# ─────────────────────────────────────────────────────────────

class OCREngine:
    """
    Tesseract-based OCR engine for NEXUS.

    Usage:
        ocr = OCREngine()

        # Extract text from a Screenshot
        from vision.capture import capture
        shot = capture()
        result = ocr.extract(shot)
        print(result.text)

        # Extract from a file
        result = ocr.extract_file("data/screenshots/nexus_xxx.png")

        # Extract from a specific region of a screenshot
        result = ocr.extract_region(shot, x=100, y=50, w=800, h=200)
    """

    def __init__(
        self,
        lang: str = "eng",
        preprocess: str = "auto",
        tesseract_config: str = "--oem 3 --psm 3",
    ):
        if not _TESS_AVAILABLE:
            raise ImportError(
                "pytesseract not installed.\n"
                "Run: pip install pytesseract && sudo apt install tesseract-ocr -y"
            )
        if not _PIL_AVAILABLE:
            raise ImportError("Pillow not installed. Run: pip install pillow")

        self.lang             = lang
        self.preprocess       = preprocess
        self.tesseract_config = tesseract_config

        # Verify tesseract binary is available
        try:
            ver = pytesseract.get_tesseract_version()
            logger.info("Tesseract version: %s", ver)
        except Exception as exc:
            raise RuntimeError(
                f"Tesseract not found: {exc}\n"
                "Run: sudo apt install tesseract-ocr -y"
            )

    # ── main extraction methods ───────────────────────────────

    def extract(self, screenshot) -> OCRResult:
        """
        Extract text from a Screenshot object.

        Parameters
        ----------
        screenshot : vision.capture.Screenshot

        Returns
        -------
        OCRResult
        """
        return self._run_ocr(screenshot.image)

    def extract_image(self, image: "Image.Image") -> OCRResult:
        """Extract text from a PIL Image directly."""
        return self._run_ocr(image)

    def extract_array(self, array: np.ndarray) -> OCRResult:
        """Extract text from a numpy array (BGR or RGB)."""
        if array.shape[2] == 3:
            # Assume BGR from OpenCV — convert to RGB for PIL
            rgb = array[:, :, ::-1]
        else:
            rgb = array
        image = Image.fromarray(rgb.astype(np.uint8))
        return self._run_ocr(image)

    def extract_file(self, path: str | Path) -> OCRResult:
        """Extract text from an image file on disk."""
        image = Image.open(str(path))
        return self._run_ocr(image)

    def extract_region(
        self,
        screenshot,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> OCRResult:
        """
        Extract text from a specific region of a Screenshot.

        Parameters
        ----------
        screenshot : vision.capture.Screenshot
        x, y       : top-left corner
        w, h       : width and height of region
        """
        cropped = screenshot.image.crop((x, y, x + w, y + h))
        return self._run_ocr(cropped)

    # ── internal ──────────────────────────────────────────────

    def _run_ocr(self, image: "Image.Image") -> OCRResult:
        """Core OCR execution."""
        try:
            # Pre-process
            processed = _preprocess_for_ocr(image, mode=self.preprocess)

            # Run Tesseract — get both text and detailed data
            raw_text = pytesseract.image_to_string(
                processed,
                lang=self.lang,
                config=self.tesseract_config,
            )

            # Get confidence data
            try:
                data = pytesseract.image_to_data(
                    processed,
                    lang=self.lang,
                    config=self.tesseract_config,
                    output_type=pytesseract.Output.DICT,
                )
                confidences = [
                    int(c) for c in data["conf"]
                    if str(c).lstrip("-").isdigit() and int(c) >= 0
                ]
                mean_conf = float(np.mean(confidences)) if confidences else 0.0
            except Exception:
                mean_conf = 0.0

            # Parse text into lines and words
            text  = raw_text.strip()
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            words = text.split()

            result = OCRResult(
                text       = text,
                lines      = lines,
                words      = words,
                confidence = mean_conf,
                raw        = raw_text,
            )

            logger.debug(
                "OCR: %d words, confidence=%.1f%%",
                len(words), mean_conf,
            )

            return result

        except Exception as exc:
            logger.error("OCR failed: %s", exc)
            return OCRResult()

    def read_screen(self, monitor: int = 1) -> OCRResult:
        """
        Convenience: capture the screen and extract all text in one call.

        Usage:
            ocr = OCREngine()
            result = ocr.read_screen()
            print(result.text)
        """
        try:
            from vision.capture import ScreenCapturer
        except ModuleNotFoundError:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from vision.capture import ScreenCapturer

        capturer = ScreenCapturer()
        shot     = capturer.capture(monitor)
        return self.extract(shot)


# ─────────────────────────────────────────────────────────────
#  CONVENIENCE FUNCTION
# ─────────────────────────────────────────────────────────────

_default_engine: Optional[OCREngine] = None


def read_screen(monitor: int = 1) -> OCRResult:
    """
    One-shot: capture screen and extract all text.

    Usage:
        from vision.ocr import read_screen
        result = read_screen()
        print(result.text)
    """
    global _default_engine
    if _default_engine is None:
        _default_engine = OCREngine()
    return _default_engine.read_screen(monitor)


# ─────────────────────────────────────────────────────────────
#  STANDALONE ENTRY — python vision/ocr.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n  NEXUS — vision/ocr.py")
    print("  " + "─" * 40)

    if not _TESS_AVAILABLE:
        print("  ✗ pytesseract not installed.")
        print("    Run: pip install pytesseract && sudo apt install tesseract-ocr -y")
        sys.exit(1)

    ocr = OCREngine()

    # ── --screen: read live screen ────────────────────────────
    if "--screen" in sys.argv:
        print("\n  Reading screen text...\n")
        result = ocr.read_screen()
        print(f"  Words      : {result.word_count}")
        print(f"  Confidence : {result.confidence:.1f}%")
        print(f"\n  Extracted text:\n")
        print("  " + "\n  ".join(result.lines[:30]))
        if len(result.lines) > 30:
            print(f"  ... ({len(result.lines) - 30} more lines)")

    # ── --file: read from image file ──────────────────────────
    elif "--file" in sys.argv:
        idx  = sys.argv.index("--file")
        path = sys.argv[idx + 1]
        print(f"\n  Reading text from: {path}\n")
        result = ocr.extract_file(path)
        print(f"  Words      : {result.word_count}")
        print(f"  Confidence : {result.confidence:.1f}%")
        print(f"\n  Text:\n")
        for line in result.lines:
            print(f"  {line}")

    # ── --find: search for text on screen ─────────────────────
    elif "--find" in sys.argv:
        idx   = sys.argv.index("--find")
        query = " ".join(sys.argv[idx + 1:])
        print(f"\n  Searching screen for: '{query}'\n")
        result = ocr.read_screen()
        if result.contains(query):
            print(f"  \033[92m✓ Found '{query}' on screen\033[0m")
        else:
            print(f"  \033[91m✗ '{query}' not found on screen\033[0m")

    else:
        print()
        print("  Usage:")
        print("    python vision/ocr.py --screen              # read all text on screen")
        print("    python vision/ocr.py --file image.png      # read text from file")
        print("    python vision/ocr.py --find <text>         # search for text on screen")
        print()