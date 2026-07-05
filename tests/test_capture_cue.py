"""
Test the "still listening" cue path in MicrophoneListener.capture_phrase,
driving the real method with a scripted fake mic (no audio hardware).
"""

import numpy as np

from voice.listener import MicrophoneListener


def _make_listener(script: str):
    """script: 'S'/'.' per chunk. Returns (listener, statuses_list)."""
    lis = MicrophoneListener()          # __init__ doesn't touch the sound device
    lis._running = True
    lis._capture_rate = 16000
    lis._chunk_frames = 320             # 20 ms chunks → pause hint at 400ms = 20 chunks
    lis.vad_min_ms = 40                 # ~2 chunks of speech is "enough"
    lis.end_silence_ms = 1000           # end-of-turn = 50 chunks (> 20 hint)
    lis.max_phrase_sec = 30

    flags = iter([c == "S" for c in script])
    consumed = {"n": 0}

    def fake_read_chunk(timeout=0.5):
        consumed["n"] += 1
        if consumed["n"] > len(script):
            lis._running = False        # stop after the script ends
            return None
        fake_read_chunk._last = next(flags, False)
        return np.zeros(320, dtype=np.int16)

    def fake_is_speech(chunk):
        return getattr(fake_read_chunk, "_last", False)

    lis.read_chunk = fake_read_chunk
    lis.is_speech = fake_is_speech
    return lis


def test_pause_cue_fires_during_a_thinking_pause():
    # 5 chunks speech, then a 30-chunk pause (> 20-chunk hint) → cue fires;
    # not long enough (< 50) to end the turn, then more speech, then real end.
    script = "S" * 5 + "." * 30 + "S" * 5 + "." * 55
    lis = _make_listener(script)
    statuses = []
    lis.capture_phrase(on_status=statuses.append)

    assert "pausing" in statuses, "the still-listening cue should fire on a pause"
    assert statuses[0] == "speaking"


def test_no_pause_cue_for_continuous_speech_then_stop():
    # Straight speech then a single end-of-turn gap — the pause cue may fire as
    # the end gap grows, but speech must be reported first.
    script = "S" * 20 + "." * 55
    lis = _make_listener(script)
    statuses = []
    lis.capture_phrase(on_status=statuses.append)
    assert statuses and statuses[0] == "speaking"


def test_cue_deduplicates_repeats():
    # The cue should not spam: consecutive identical states collapse to one.
    script = "S" * 5 + "." * 60
    lis = _make_listener(script)
    statuses = []
    lis.capture_phrase(on_status=statuses.append)
    # No two consecutive statuses are identical.
    assert all(a != b for a, b in zip(statuses, statuses[1:]))
