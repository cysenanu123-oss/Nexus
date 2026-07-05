"""
Tests for voice/endpointing.py — the pause-tolerant end-of-turn detector.

Notation for readability: each character is one audio chunk fed to the
endpointer — 'S' = speech, '.' = silence.
"""

from voice.endpointing import Endpointer, ms_to_chunks


def run(pattern: str, end_silence_chunks=5, min_speech_chunks=2, max_chunks=1000):
    """Feed a pattern of S/. chunks; return (final_status, chunks_consumed)."""
    ep = Endpointer(end_silence_chunks, min_speech_chunks, max_chunks)
    for i, ch in enumerate(pattern, 1):
        status = ep.update(ch == "S")
        if status != Endpointer.LISTENING:
            return status, i
    return Endpointer.LISTENING, len(pattern)


def test_finishes_after_end_of_turn_silence():
    # speech, then 5 chunks of silence → done exactly at the 5th silence.
    status, n = run("SSSSS.....", end_silence_chunks=5)
    assert status == Endpointer.DONE
    assert n == 10                       # 5 speech + 5 silence


def test_short_pause_does_not_end_the_turn():
    # A 3-chunk thinking pause (< 5) must NOT finish; the turn continues and
    # only ends after the real 5-chunk end-of-turn gap.
    status, n = run("SSS...SSS.....", end_silence_chunks=5)
    assert status == Endpointer.DONE
    # It should finish at the FINAL long gap, not the middle short pause.
    assert n == len("SSS...SSS.....")


def test_multiple_thinking_pauses_kept_as_one_turn():
    pattern = "SS..SS..SS..SS....."     # several short pauses, then a long one
    status, n = run(pattern, end_silence_chunks=5)
    assert status == Endpointer.DONE
    assert n == len(pattern)             # consumed the whole thing


def test_does_not_finish_without_enough_speech():
    # A single speech blip then silence — below min_speech_chunks=2 → keeps
    # listening (treated as noise, not a real utterance).
    status, _ = run("S.........", end_silence_chunks=3, min_speech_chunks=2)
    assert status == Endpointer.LISTENING


def test_leading_silence_is_ignored():
    # Silence before any speech must not count toward the end-of-turn gap.
    status, n = run(".....SSS.....", end_silence_chunks=5, min_speech_chunks=2)
    assert status == Endpointer.DONE
    assert n == len(".....SSS.....")


def test_hard_cap_stops_runaway_turn():
    ep = Endpointer(end_silence_chunks=5, min_speech_chunks=2, max_chunks=6)
    status = Endpointer.LISTENING
    for _ in range(6):
        status = ep.update(True)         # nonstop speech, never pauses
    assert status == Endpointer.TOO_LONG


def test_reset_clears_state():
    ep = Endpointer(3, 2, 100)
    for ch in "SSS...":
        ep.update(ch == "S")
    ep.reset()
    assert ep.total == 0 and ep.speech_chunks == 0 and not ep.started


def test_ms_to_chunks():
    # 1200 ms at 16 kHz with 320-frame chunks (20 ms) → 60 chunks.
    assert ms_to_chunks(1200, 16000, 320) == 60
    assert ms_to_chunks(0, 16000, 320) == 1        # floored at 1
    assert ms_to_chunks(1000, 16000, 0) == 1       # guards bad input


def test_longer_end_silence_is_more_pause_tolerant():
    # With a longer end-of-turn threshold, a pause that would end a short-gap
    # turn is now tolerated.
    short = run("SS....SS.", end_silence_chunks=3)   # 3-silence gap ends it early
    long_ = run("SS....SS.", end_silence_chunks=8)   # tolerant → keeps listening
    assert short[0] == Endpointer.DONE
    assert long_[0] == Endpointer.LISTENING
