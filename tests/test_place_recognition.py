"""Tests for vision/place_recognition.py — enroll/identify/change-detection."""

import numpy as np

from vision.place_recognition import (
    PlaceRecognizer, PlaceProfileStore, ImageEmbedder,
)


class FakeEmbedder(ImageEmbedder):
    """Maps a frame label (ignoring trailing digits/space) to a one-hot vector,
    so same-place frames are identical (cos=1) and different places orthogonal."""
    def __init__(self):
        self.vocab: dict[str, np.ndarray] = {}

    def available(self):
        return True

    def _base(self, frame):
        return str(frame).rstrip("0123456789 ")

    def embed(self, frame):
        base = self._base(frame)
        if base not in self.vocab:
            v = np.zeros(16, dtype=np.float32)
            v[len(self.vocab)] = 1.0
            self.vocab[base] = v
        return self.vocab[base]


def recognizer(tmp_path, threshold=0.75):
    store = PlaceProfileStore(directory=tmp_path / "places")
    return PlaceRecognizer(FakeEmbedder(), store=store, threshold=threshold)


def test_enroll_and_identify_same_place(tmp_path):
    r = recognizer(tmp_path)
    n = r.enroll("office", ["office1", "office2", "office3"])
    assert n == 3
    res = r.identify("office9")
    assert res.known and res.place == "office"
    assert res.score > 0.99


def test_unknown_place_below_threshold(tmp_path):
    r = recognizer(tmp_path)
    r.enroll("office", ["office1"])
    res = r.identify("kitchen1")     # orthogonal → cos 0
    assert not res.known
    assert res.place == "unknown"


def test_nearest_of_multiple_places_wins(tmp_path):
    r = recognizer(tmp_path)
    r.enroll("office", ["office1", "office2"])
    r.enroll("kitchen", ["kitchen1", "kitchen2"])
    assert r.identify("kitchen3").place == "kitchen"
    assert r.identify("office3").place == "office"


def test_no_profiles_returns_unknown(tmp_path):
    r = recognizer(tmp_path)
    assert not r.identify("anywhere").known


def test_update_emits_change_only_on_transition(tmp_path):
    r = recognizer(tmp_path)
    r.enroll("office", ["office1"])
    r.enroll("kitchen", ["kitchen1"])

    first = r.update("office2")
    assert first is not None and first.to_place == "office"
    assert first.from_place is None

    assert r.update("office3") is None          # same place → no event

    moved = r.update("kitchen2")
    assert moved is not None
    assert moved.from_place == "office" and moved.to_place == "kitchen"
    assert "office" in moved.message() and "kitchen" in moved.message()


def test_profiles_persist_across_instances(tmp_path):
    store_dir = tmp_path / "places"
    r1 = PlaceRecognizer(FakeEmbedder(), store=PlaceProfileStore(store_dir))
    r1.enroll("office", ["office1", "office2"])

    # New store/recognizer reading the same directory sees the enrolled place.
    r2 = PlaceRecognizer(FakeEmbedder(), store=PlaceProfileStore(store_dir))
    assert "office" in r2.store.names()
    assert r2.identify("office5").place == "office"


def test_remove_place(tmp_path):
    store = PlaceProfileStore(tmp_path / "places")
    r = PlaceRecognizer(FakeEmbedder(), store=store)
    r.enroll("office", ["office1"])
    assert store.remove("office")
    assert "office" not in store.names()
