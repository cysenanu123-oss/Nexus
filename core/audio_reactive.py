"""
core/audio_reactive.py
NEXUS — audio-reactive envelope for the HUD.

The Jarvis HUD should pulse to the actual voice — swell when NEXUS speaks or when
you talk to it — instead of a fixed animation. This module is the pure math for
that: convert a stream of raw audio levels into a smooth 0..1 "energy" the UI can
map to sphere expansion, glow, ring speed, etc.

Kept dependency-light (numpy optional) and free of any GUI/audio I/O so the
mapping is unit-testable without a screen or a microphone.
"""

from __future__ import annotations

import math


def rms(samples) -> float:
    """Root-mean-square level of an audio buffer (list or numpy array)."""
    try:
        import numpy as np
        arr = np.asarray(samples, dtype="float64")
        if arr.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(arr * arr)))
    except Exception:
        vals = list(samples)
        if not vals:
            return 0.0
        return math.sqrt(sum(v * v for v in vals) / len(vals))


class AmplitudeEnvelope:
    """Attack/decay smoother that turns jittery levels into a stable 0..1 energy.

    Fast attack (jump up quickly when sound starts) + slower decay (fall off
    gently) reads as a lively, natural pulse rather than a strobe.
    """

    def __init__(self, attack: float = 0.5, decay: float = 0.12, gain: float = 6.0):
        # attack/decay are per-update lerp factors in (0, 1].
        self.attack = attack
        self.decay = decay
        self.gain = gain
        self._value = 0.0

    @property
    def value(self) -> float:
        return self._value

    def push(self, level: float) -> float:
        """Feed a new raw level (e.g. an RMS in ~0..0.3); returns smoothed 0..1."""
        target = _clamp01(max(0.0, level) * self.gain)
        k = self.attack if target > self._value else self.decay
        self._value += (target - self._value) * k
        self._value = _clamp01(self._value)
        return self._value

    def decay_step(self) -> float:
        """Advance one frame with no new audio (envelope relaxes toward 0)."""
        self._value += (0.0 - self._value) * self.decay
        if self._value < 1e-4:
            self._value = 0.0
        return self._value

    def reset(self) -> None:
        self._value = 0.0


def burst_from_amplitude(energy: float, base: float = 1.0, span: float = 0.9) -> float:
    """Map 0..1 energy to a sphere-expansion multiplier for the HUD."""
    return base + span * _clamp01(energy)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
