"""
Tests for core/model_manager.py — capability filtering and the consent gate.
Downloads and Ollama calls are stubbed; nothing is fetched for real.
"""

import pytest

from core.hardware import HardwareProfile, _STANDARD_SIZES_B
from core.model_manager import ModelManager, EnsureResult, MODEL_REGISTRY


def profile(ram=8.0, vram=None, accel="none"):
    p = HardwareProfile(
        os="Linux", arch="x86_64", cpu_cores=8, ram_gb=ram,
        gpu_vendor="t", gpu_name="t", vram_gb=vram,
        accelerator=accel, disk_free_gb=500.0,
    )
    p.runnable_sizes_b = [s for s in _STANDARD_SIZES_B if p.can_run(s)]
    return p


def manager(ram=8.0, vram=None, accel="none", installed=None):
    m = ModelManager(profile=profile(ram, vram, accel))
    m.installed = lambda: list(installed or [])   # stub discovery
    return m


# ── capability filtering ─────────────────────────────────────────────

def test_small_device_only_sees_small_models():
    m = manager(ram=6.0, accel="none")            # ~3.6 GB usable
    names = {s.name for s in m.available_for_device()}
    assert "qwen2.5:1.5b" in names
    assert "llama3.1:70b" not in names
    assert "qwen2.5:32b" not in names


def test_big_gpu_sees_large_models():
    m = manager(ram=64.0, vram=48.0, accel="cuda")
    names = {s.name for s in m.available_for_device()}
    assert "llama3.1:70b" in names


def test_recommend_prefers_strongest_runnable():
    m = manager(ram=64.0, vram=24.0, accel="cuda")
    rec = m.recommend("reasoning")
    assert rec is not None
    # strongest reasoning model that fits 24 GB VRAM (not the 70B)
    assert rec.name == "qwen2.5:32b"


def test_recommend_returns_none_when_nothing_fits():
    m = manager(ram=2.0, accel="none")
    assert m.recommend("reasoning") is None or m.recommend("reasoning").params_b <= 3


def test_recommend_upgrade_skips_already_installed():
    m = manager(ram=64.0, vram=24.0, accel="cuda", installed=["qwen2.5:32b"])
    # best reasoning model is installed → no upgrade offered
    assert m.recommend_upgrade("reasoning") is None


# ── the consent gate ─────────────────────────────────────────────────

def test_ensure_never_downloads_without_confirm():
    m = manager(ram=16.0, vram=16.0, accel="cuda")
    called = {"download": False}
    m._download_ollama = lambda spec, cb: called.__setitem__("download", True)
    res = m.ensure("qwen2.5:7b", confirm=None)   # no confirm callback
    assert not res.ok
    assert not called["download"], "must not download without consent"


def test_ensure_declined_when_confirm_returns_false():
    m = manager(ram=16.0, vram=16.0, accel="cuda")
    m._download_ollama = lambda spec, cb: EnsureResult(True, "downloaded")
    res = m.ensure("qwen2.5:7b", confirm=lambda msg: False)
    assert not res.ok and "declined" in res.message.lower()


def test_ensure_downloads_when_confirmed():
    m = manager(ram=16.0, vram=16.0, accel="cuda")
    seen = {}
    def fake_dl(spec, cb):
        seen["spec"] = spec.name
        return EnsureResult(True, "downloaded", spec.name)
    m._download_ollama = fake_dl
    res = m.ensure("qwen2.5:7b", confirm=lambda msg: True)
    assert res.ok
    assert seen["spec"] == "qwen2.5:7b"


def test_ensure_refuses_model_too_big_for_device():
    m = manager(ram=4.0, accel="none")
    hit = {"download": False}
    m._download_ollama = lambda spec, cb: hit.__setitem__("download", True)
    res = m.ensure("llama3.1:70b", confirm=lambda msg: True)
    assert not res.ok
    assert not hit["download"], "must not attempt an impossible download"
    assert "can't run" in res.message.lower() or "too" in res.message.lower()


def test_ensure_unknown_model():
    m = manager()
    assert not m.ensure("not-a-real-model", confirm=lambda msg: True).ok


def test_ensure_already_installed_is_noop():
    m = manager(ram=16.0, vram=16.0, accel="cuda", installed=["qwen2.5:7b"])
    res = m.ensure("qwen2.5:7b", confirm=lambda msg: True)
    assert res.ok and "already" in res.message.lower()


def test_registry_specs_are_wellformed():
    for name, spec in MODEL_REGISTRY.items():
        assert spec.name == name
        assert spec.backend in ("ollama", "gguf")
        assert spec.params_b > 0 and spec.size_gb > 0
        assert spec.tags
