"""
vision/face_detector.py
NEXUS Face Detection — thread-safe face recognition using insightface.

Adapted from Deep-Live-Cam's face detection pipeline:
  - buffalo_l model (106-point landmarks, ArcFace embeddings)
  - Thread-safe singleton (one model load, many callers)
  - Owner enrollment + recognition for access gating

Requires: pip install insightface onnxruntime

Usage:
    fd = get_face_detector()
    faces = fd.detect(frame)           # list of Face objects
    match = fd.recognize(frame)        # True if owner face found
    fd.enroll(frame)                   # save owner embedding
"""

from __future__ import annotations

import logging
import pickle
import threading
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("nexus.vision.face_detector")

_PROFILE_PATH = Path("data/face_profile.pkl")
_SIMILARITY_THRESHOLD = 0.45   # cosine distance — lower = stricter


class FaceDetector:
    """
    Thread-safe insightface wrapper.
    Handles detection, embedding extraction, and owner verification.
    """

    def __init__(self):
        self._lock   = threading.Lock()
        self._app    = None
        self._owner_embedding: Optional[np.ndarray] = None
        self._load_model()
        self._load_profile()

    def _load_model(self):
        try:
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(name="buffalo_l",
                                     providers=["CPUExecutionProvider"])
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            log.info("FaceDetector ready (buffalo_l, CPU).")
        except ImportError:
            log.warning("insightface not installed — face detection disabled. "
                        "Install with: pip install insightface onnxruntime")
        except Exception as e:
            log.warning("FaceDetector model load failed: %s", e)

    def _load_profile(self):
        if _PROFILE_PATH.exists():
            try:
                with open(_PROFILE_PATH, "rb") as f:
                    self._owner_embedding = pickle.load(f)
                log.info("Owner face profile loaded.")
            except Exception as e:
                log.warning("Could not load face profile: %s", e)

    @property
    def ready(self) -> bool:
        return self._app is not None

    @property
    def is_enrolled(self) -> bool:
        return self._owner_embedding is not None

    def detect(self, frame: np.ndarray) -> list:
        """
        Detect all faces in a BGR frame.
        Returns list of insightface Face objects (each has .bbox, .embedding, .kps).
        """
        if not self.ready:
            return []
        with self._lock:
            return self._app.get(frame)

    def enroll(self, frame: np.ndarray) -> bool:
        """
        Enroll the dominant face in the frame as the owner profile.
        """
        faces = self.detect(frame)
        if not faces:
            log.warning("No face detected for enrollment.")
            return False

        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        self._owner_embedding = face.embedding

        _PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_PROFILE_PATH, "wb") as f:
            pickle.dump(self._owner_embedding, f)

        log.info("Owner face enrolled and saved.")
        return True

    def recognize(self, frame: np.ndarray) -> dict:
        """
        Check if the owner's face appears in the frame.
        Returns: {"accepted": bool, "score": float, "face_count": int}
        """
        if not self.is_enrolled:
            return {"accepted": False, "score": 0.0, "face_count": 0,
                    "error": "No owner profile enrolled"}

        faces = self.detect(frame)
        if not faces:
            return {"accepted": False, "score": 0.0, "face_count": 0}

        best_score = 0.0
        for face in faces:
            score = self._cosine_similarity(face.embedding, self._owner_embedding)
            if score > best_score:
                best_score = score

        accepted = best_score >= _SIMILARITY_THRESHOLD
        return {"accepted": accepted, "score": float(best_score),
                "face_count": len(faces)}

    def count_faces(self, frame: np.ndarray) -> int:
        return len(self.detect(frame))

    def get_embeddings(self, frame: np.ndarray) -> list[np.ndarray]:
        faces = self.detect(frame)
        return [f.embedding for f in faces if f.embedding is not None]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# ── Singleton ──────────────────────────────────────────────────

_detector: Optional[FaceDetector] = None
_detector_lock = threading.Lock()


def get_face_detector() -> FaceDetector:
    global _detector
    with _detector_lock:
        if _detector is None:
            _detector = FaceDetector()
    return _detector


if __name__ == "__main__":
    import cv2, sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s — %(message)s")

    fd = get_face_detector()
    if not fd.ready:
        print("insightface not available.")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam found.")
        sys.exit(1)

    print("Press 'e' to enroll owner face, 'r' to recognize, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = fd.detect(frame)
        for face in faces:
            x1, y1, x2, y2 = [int(c) for c in face.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("NEXUS Face Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("e"):
            if fd.enroll(frame):
                print("Owner enrolled.")
            else:
                print("Enrollment failed — no face detected.")
        elif key == ord("r"):
            result = fd.recognize(frame)
            print(f"Recognition: {result}")

    cap.release()
    cv2.destroyAllWindows()
