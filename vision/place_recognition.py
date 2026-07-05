"""
vision/place_recognition.py
NEXUS — "I know where I am." Camera-based place recognition.

This is the visual twin of voice/speaker_id.py: instead of turning a voice into
an embedding and matching it to an enrolled owner, it turns a camera frame into
an image embedding (CLIP) and matches it to enrolled *places*. Enroll a few
photos of each room once ("office", "kitchen"), and afterwards NEXUS can tell
which place it's looking at — and announce when you move from one to another.

Design (mirrors speaker_id):
  * ImageEmbedder → L2-normalized vector for a frame (CLIP, lazy-loaded).
  * PlaceProfileStore → per-place mean embedding saved as data/place_profiles/<name>.npy
  * PlaceRecognizer → enroll(name, frames), identify(frame), update(frame) for
    change detection.

The embedder is injectable, so the matching/threshold/change-detection logic is
unit-testable with fake embeddings — no camera, no torch required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("nexus.place_recognition")

PROFILE_DIR = Path(__file__).parent.parent / "data" / "place_profiles"
DEFAULT_THRESHOLD = 0.75          # cosine similarity — above = recognized


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ─────────────────────────────────────────────────────────────
#  Embedder
# ─────────────────────────────────────────────────────────────

class ImageEmbedder:
    """Interface: embed(frame) -> L2-normalized np.ndarray."""

    def available(self) -> bool:
        raise NotImplementedError

    def embed(self, frame) -> np.ndarray:
        raise NotImplementedError


class CLIPEmbedder(ImageEmbedder):
    """CLIP image embeddings via open_clip (lazy). Degrades gracefully if the
    heavy deps (torch/open_clip) aren't installed."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._torch = None

    def _ensure(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            import open_clip
            from PIL import Image  # noqa: F401
        except Exception as e:
            log.warning("CLIP unavailable (pip install open_clip_torch torch): %s", e)
            return False
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained)
            model.eval()
            self._model, self._preprocess, self._torch = model, preprocess, torch
            return True
        except Exception as e:
            log.warning("CLIP model load failed: %s", e)
            return False

    def available(self) -> bool:
        return self._ensure()

    def embed(self, frame) -> np.ndarray:
        if not self._ensure():
            raise RuntimeError("CLIP embedder unavailable — install torch + open_clip_torch")
        from PIL import Image
        if isinstance(frame, (str, Path)):
            img = Image.open(frame).convert("RGB")
        elif isinstance(frame, np.ndarray):
            img = Image.fromarray(frame).convert("RGB")
        else:
            img = frame.convert("RGB")
        tensor = self._preprocess(img).unsqueeze(0)
        with self._torch.no_grad():
            vec = self._model.encode_image(tensor)[0].cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


# ─────────────────────────────────────────────────────────────
#  Profile store
# ─────────────────────────────────────────────────────────────

class PlaceProfileStore:
    """Persists each place's mean embedding as data/place_profiles/<name>.npy."""

    def __init__(self, directory: Path = PROFILE_DIR):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, np.ndarray] = {}
        self._load()

    def _path(self, name: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.lower())
        return self.dir / f"{safe}.npy"

    def _load(self) -> None:
        for p in self.dir.glob("*.npy"):
            try:
                self._cache[p.stem] = np.load(str(p))
            except Exception as e:
                log.warning("Failed to load place profile %s: %s", p, e)

    def add(self, name: str, embedding: np.ndarray) -> None:
        key = name.lower()
        self._cache[self._path(name).stem] = embedding
        np.save(str(self._path(name)), embedding)
        log.info("Saved place profile %r", name)

    def get(self, name: str) -> Optional[np.ndarray]:
        return self._cache.get(self._path(name).stem)

    def remove(self, name: str) -> bool:
        p = self._path(name)
        self._cache.pop(p.stem, None)
        if p.exists():
            p.unlink()
            return True
        return False

    def names(self) -> list[str]:
        return sorted(self._cache.keys())

    def all(self) -> dict[str, np.ndarray]:
        return dict(self._cache)


# ─────────────────────────────────────────────────────────────
#  Recognizer
# ─────────────────────────────────────────────────────────────

@dataclass
class PlaceResult:
    place: str            # recognized place name, or "unknown"
    score: float
    known: bool


@dataclass
class PlaceChange:
    from_place: Optional[str]
    to_place: str
    score: float

    def message(self) -> str:
        if self.from_place and self.from_place != "unknown":
            return f"You've moved from the {self.from_place} to the {self.to_place}."
        return f"You're now in the {self.to_place}."


class PlaceRecognizer:
    def __init__(self, embedder: ImageEmbedder,
                 store: Optional[PlaceProfileStore] = None,
                 threshold: float = DEFAULT_THRESHOLD):
        self.embedder = embedder
        self.store = store or PlaceProfileStore()
        self.threshold = threshold
        self._current: Optional[str] = None

    def enroll(self, name: str, frames: list) -> int:
        """Enroll a place from one or more frames (mean of their embeddings).
        Returns the number of frames successfully embedded."""
        if not frames:
            raise ValueError("need at least one frame to enroll a place")
        embs = []
        for f in frames:
            try:
                embs.append(self.embedder.embed(f))
            except Exception as e:
                log.warning("Skipping a frame during enrollment: %s", e)
        if not embs:
            raise RuntimeError("no frame could be embedded")
        mean = np.mean(np.stack(embs), axis=0)
        norm = np.linalg.norm(mean)
        if norm:
            mean = mean / norm
        self.store.add(name, mean)
        return len(embs)

    def identify(self, frame) -> PlaceResult:
        profiles = self.store.all()
        if not profiles:
            return PlaceResult("unknown", 0.0, False)
        emb = self.embedder.embed(frame)
        best_name, best_score = "unknown", -1.0
        for name, ref in profiles.items():
            s = _cosine(emb, ref)
            if s > best_score:
                best_name, best_score = name, s
        if best_score >= self.threshold:
            return PlaceResult(best_name, best_score, True)
        return PlaceResult("unknown", max(best_score, 0.0), False)

    def update(self, frame) -> Optional[PlaceChange]:
        """Identify the current frame; return a PlaceChange iff the recognized
        place differs from the last known one."""
        result = self.identify(frame)
        if not result.known:
            return None
        if result.place != self._current:
            change = PlaceChange(self._current, result.place, result.score)
            self._current = result.place
            return change
        return None

    @property
    def current_place(self) -> Optional[str]:
        return self._current
