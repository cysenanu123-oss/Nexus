"""Tests for core/hardware.py — capability math and safe detection."""

from core.hardware import HardwareProfile, detect, _STANDARD_SIZES_B


def make(ram=16.0, vram=None, accel="none"):
    p = HardwareProfile(
        os="Linux", arch="x86_64", cpu_cores=8, ram_gb=ram,
        gpu_vendor="test", gpu_name="test", vram_gb=vram,
        accelerator=accel, disk_free_gb=500.0,
    )
    p.runnable_sizes_b = [s for s in _STANDARD_SIZES_B if p.can_run(s)]
    return p


def test_cuda_uses_vram_not_ram():
    p = make(ram=8.0, vram=24.0, accel="cuda")
    assert p.effective_accel_gb == 24.0
    assert p.can_run(13)          # 24 GB VRAM easily fits a 13B q4
    assert p.max_model_params_b >= 34


def test_apple_unified_memory_tracks_ram():
    p = make(ram=32.0, vram=None, accel="mps")
    # ~70% of 32 GB usable
    assert 20.0 <= p.effective_accel_gb <= 24.0
    assert p.can_run(13)


def test_cpu_only_is_conservative():
    p = make(ram=8.0, accel="none")
    assert p.effective_accel_gb < p.ram_gb
    assert not p.can_run(13)      # no way an 8 GB CPU box runs a 13B


def test_tiers_scale_with_capability():
    assert make(ram=4.0, accel="none").tier in ("minimal", "light")
    assert make(ram=8.0, vram=8.0, accel="cuda").tier in ("standard", "mid")
    assert make(ram=8.0, vram=48.0, accel="cuda").tier == "workstation"


def test_can_run_respects_quantization():
    p = make(ram=8.0, vram=8.0, accel="cuda")
    # 13B is ~7.8 GB at q4 (fits 8 GB*0.85=6.8? no) — check f16 is heavier than q4
    assert p.can_run(7, quant="q4")
    assert not p.can_run(7, quant="f16")   # f16 needs ~14.7 GB


def test_detect_never_raises_and_is_sane():
    p = detect()
    assert p.os
    assert p.cpu_cores >= 1
    assert p.ram_gb >= 0
    assert isinstance(p.runnable_sizes_b, list)
    assert p.tier in ("minimal", "light", "standard", "mid", "high", "workstation")
