"""Tests for core/audio_reactive.py — the HUD's audio-reactive envelope."""

from core.audio_reactive import (
    rms, AmplitudeEnvelope, burst_from_amplitude,
)


def test_rms_of_silence_is_zero():
    assert rms([0, 0, 0, 0]) == 0.0
    assert rms([]) == 0.0


def test_rms_of_constant_signal():
    assert abs(rms([0.5, -0.5, 0.5, -0.5]) - 0.5) < 1e-9


def test_envelope_attacks_fast_and_decays_slow():
    env = AmplitudeEnvelope(attack=0.5, decay=0.1, gain=1.0)
    up = env.push(1.0)
    assert up > 0.4                      # jumped up quickly
    # With no more sound it should fall, but slower than it rose.
    d1 = env.decay_step()
    assert d1 < up


def test_envelope_clamped_0_1():
    env = AmplitudeEnvelope(gain=100.0)
    for _ in range(10):
        v = env.push(1.0)
        assert 0.0 <= v <= 1.0
    assert env.value <= 1.0


def test_envelope_relaxes_to_zero():
    env = AmplitudeEnvelope(decay=0.5)
    env.push(1.0)
    for _ in range(100):
        env.decay_step()
    assert env.value == 0.0


def test_silence_keeps_envelope_low():
    env = AmplitudeEnvelope()
    for _ in range(20):
        v = env.push(0.0)
    assert v == 0.0


def test_burst_mapping_bounds():
    assert burst_from_amplitude(0.0) == 1.0
    assert burst_from_amplitude(1.0) == 1.9
    # clamps out-of-range energy
    assert burst_from_amplitude(5.0) == 1.9
    assert burst_from_amplitude(-1.0) == 1.0


def test_louder_input_gives_bigger_burst():
    env = AmplitudeEnvelope(gain=4.0)
    quiet = burst_from_amplitude(env.push(0.02))
    env.reset()
    loud = burst_from_amplitude(env.push(0.25))
    assert loud > quiet
