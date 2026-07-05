"""Tests for vision/scene_describer.py — VLM selection + consent-gated download."""

from vision.scene_describer import SceneDescriber


class Spec:
    def __init__(self, name):
        self.name = name


class FakeMM:
    def __init__(self, rec="moondream", installed=()):
        self._rec = rec
        self._installed = set(installed)
        self.ensured = []

    def recommend(self, task):
        return Spec(self._rec) if (task == "vision" and self._rec) else None

    def is_installed(self, name):
        return name in self._installed

    def ensure(self, name, confirm=None, on_progress=None):
        class R:
            pass
        r = R()
        if confirm and confirm("download?"):
            self._installed.add(name)
            self.ensured.append(name)
            r.ok, r.message = True, "downloaded"
        else:
            r.ok, r.message = False, "declined"
        return r


def describer(rec="moondream", installed=(), reply="a tidy desk with a laptop"):
    calls = {}
    def fake_describe(image, prompt, model):
        calls["model"] = model
        calls["prompt"] = prompt
        return reply
    d = SceneDescriber(model_manager=FakeMM(rec, installed), describe_fn=fake_describe)
    return d, calls


def test_pick_model_uses_manager_recommendation():
    d, _ = describer(rec="llava:13b")
    assert d.pick_model() == "llava:13b"


def test_describe_happy_path_when_installed():
    d, calls = describer(rec="moondream", installed=["moondream"])
    res = d.describe("photo.jpg")
    assert res.ok
    assert res.text == "a tidy desk with a laptop"
    assert res.model == "moondream"
    assert calls["model"] == "moondream"


def test_not_installed_without_confirm_asks_user():
    d, calls = describer(rec="llava:7b", installed=[])
    res = d.describe("photo.jpg")            # confirm=None
    assert not res.ok
    assert "models get llava:7b" in res.text
    assert "model" not in calls              # never called the VLM


def test_not_installed_with_confirm_downloads_then_describes():
    mm = FakeMM(rec="moondream", installed=[])
    seen = {}
    d = SceneDescriber(model_manager=mm,
                       describe_fn=lambda img, p, m: seen.setdefault("m", m) or "desc")
    res = d.describe("photo.jpg", confirm=lambda msg: True)
    assert res.ok
    assert mm.ensured == ["moondream"]
    assert seen["m"] == "moondream"


def test_no_vision_model_available():
    d, _ = describer(rec=None)
    res = d.describe("photo.jpg")
    assert not res.ok
    assert "no vision model" in res.text.lower()


def test_vlm_error_reported_as_failure():
    d, _ = describer(rec="moondream", installed=["moondream"], reply="LLM error: boom")
    res = d.describe("photo.jpg")
    assert not res.ok
