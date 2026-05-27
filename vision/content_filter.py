"""
vision/content_filter.py
NEXUS Content Safety Filter — NSFW image/frame detection using opennsfw2.

Adapted from Deep-Live-Cam's content safety checks (predicter.py).
Provides a fast single-frame check and a multi-frame video scan.

Requires: pip install opennsfw2

Usage:
    cf = get_content_filter()
    result = cf.check_frame(frame)        # {"safe": bool, "score": float}
    result = cf.check_file("/path/img")   # same, from file path
    safe   = cf.is_safe(frame)            # quick bool
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional, Union

import numpy as np

log = logging.getLogger("nexus.vision.content_filter")

_NSFW_THRESHOLD  = 0.85   # probability above which content is flagged
_MODEL_LOCK      = threading.Lock()


class ContentFilter:
    """
    NSFW content detection wrapper around opennsfw2.
    Thread-safe singleton — model is loaded once.
    """

    def __init__(self, threshold: float = _NSFW_THRESHOLD):
        self.threshold = threshold
        self._model    = None
        self._load_model()

    def _load_model(self):
        try:
            import opennsfw2 as n2
            self._model = n2
            log.info("ContentFilter ready (opennsfw2, threshold=%.2f).", self.threshold)
        except ImportError:
            log.warning("opennsfw2 not installed — content filter disabled. "
                        "Install with: pip install opennsfw2")
        except Exception as e:
            log.warning("ContentFilter init failed: %s", e)

    @property
    def ready(self) -> bool:
        return self._model is not None

    def check_frame(self, frame: np.ndarray) -> dict:
        """
        Check a single BGR (OpenCV) frame.
        Returns: {"safe": bool, "score": float, "label": str}
        """
        if not self.ready:
            return {"safe": True, "score": 0.0, "label": "unknown",
                    "error": "opennsfw2 not installed"}
        try:
            from PIL import Image
            rgb = frame[:, :, ::-1]   # BGR → RGB
            pil_img = Image.fromarray(rgb)
            with _MODEL_LOCK:
                score = self._model.predict_image(pil_img)
            label = "NSFW" if score >= self.threshold else "SAFE"
            return {"safe": score < self.threshold, "score": float(score), "label": label}
        except Exception as e:
            log.warning("Content check failed: %s", e)
            return {"safe": True, "score": 0.0, "label": "error", "error": str(e)}

    def check_file(self, path: Union[str, Path]) -> dict:
        """
        Check an image file by path.
        Returns same dict as check_frame.
        """
        if not self.ready:
            return {"safe": True, "score": 0.0, "label": "unknown",
                    "error": "opennsfw2 not installed"}
        try:
            from PIL import Image
            pil_img = Image.open(str(path)).convert("RGB")
            with _MODEL_LOCK:
                score = self._model.predict_image(pil_img)
            label = "NSFW" if score >= self.threshold else "SAFE"
            return {"safe": score < self.threshold, "score": float(score),
                    "label": label, "path": str(path)}
        except Exception as e:
            log.warning("Content file check failed for %s: %s", path, e)
            return {"safe": True, "score": 0.0, "label": "error",
                    "error": str(e), "path": str(path)}

    def is_safe(self, frame: np.ndarray) -> bool:
        """Quick bool — True if the frame is safe."""
        return self.check_frame(frame)["safe"]

    def scan_directory(self, directory: Union[str, Path],
                       extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                       ) -> list[dict]:
        """
        Scan all images in a directory.
        Returns list of results sorted by score descending (worst first).
        """
        path = Path(directory)
        results = []
        for f in path.rglob("*"):
            if f.suffix.lower() in extensions:
                result = self.check_file(f)
                results.append(result)
        results.sort(key=lambda r: r["score"], reverse=True)
        return results


# ── Singleton ──────────────────────────────────────────────────

_filter: Optional[ContentFilter] = None
_filter_lock = threading.Lock()


def get_content_filter(threshold: float = _NSFW_THRESHOLD) -> ContentFilter:
    global _filter
    with _filter_lock:
        if _filter is None:
            _filter = ContentFilter(threshold=threshold)
    return _filter


if __name__ == "__main__":
    import sys, cv2
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s — %(message)s")

    cf = get_content_filter()
    if not cf.ready:
        print("opennsfw2 not available.")
        sys.exit(1)

    if len(sys.argv) > 1:
        result = cf.check_file(sys.argv[1])
        print(result)
    else:
        cap = cv2.VideoCapture(0)
        print("Webcam content filter — press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = cf.check_frame(frame)
            label  = f"{result['label']} ({result['score']:.2f})"
            color  = (0, 0, 255) if not result["safe"] else (0, 255, 0)
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("NEXUS Content Filter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
