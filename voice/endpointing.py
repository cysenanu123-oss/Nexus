"""
voice/endpointing.py
NEXUS — utterance endpointing that tolerates thinking pauses.

The old capture stopped at the FIRST silence gap (~300 ms), so if you paused to
think mid-sentence it ended your turn and dropped the rest. This decides when a
spoken turn is actually finished: it keeps listening through short pauses and
only finalizes after a longer *end-of-turn* silence — so "speaking… thinking…
speaking…" is captured as one utterance.

Pure logic (no audio/mic/numpy), so the endpointing behavior is unit-testable:
feed it one `is_speech` flag per audio chunk and it tells you to keep listening,
finish, or stop because the hard cap was hit.
"""

from __future__ import annotations


def ms_to_chunks(ms: float, capture_rate: int, frames_per_chunk: int) -> int:
    """How many audio chunks correspond to `ms` milliseconds."""
    if frames_per_chunk <= 0 or capture_rate <= 0:
        return 1
    return max(1, int((ms / 1000.0) * capture_rate / frames_per_chunk))


class Endpointer:
    """Streaming end-of-turn detector.

    Parameters
    ----------
    end_silence_chunks : trailing silence (in chunks) that ends the turn. This is
        the key knob — set it to ~1–1.5 s worth so ordinary thinking pauses do
        NOT end the turn.
    min_speech_chunks  : require at least this much speech before finalizing, so
        a stray noise or cough doesn't count as a whole utterance.
    max_chunks         : hard safety cap on total turn length.
    """

    LISTENING = "listening"
    DONE = "done"
    TOO_LONG = "too_long"

    def __init__(self, end_silence_chunks: int, min_speech_chunks: int, max_chunks: int):
        self.end_silence_chunks = max(1, int(end_silence_chunks))
        self.min_speech_chunks = max(1, int(min_speech_chunks))
        self.max_chunks = max(2, int(max_chunks))
        self.reset()

    def reset(self) -> None:
        self.total = 0
        self.speech_chunks = 0
        self.trailing_silence = 0
        self.started = False

    @property
    def had_enough_speech(self) -> bool:
        return self.speech_chunks >= self.min_speech_chunks

    def update(self, is_speech: bool) -> str:
        """Advance one chunk. Returns LISTENING / DONE / TOO_LONG."""
        self.total += 1

        if is_speech:
            self.speech_chunks += 1
            self.trailing_silence = 0
            self.started = True
        elif self.started:
            # Only count silence once the user has actually started talking, so
            # leading quiet doesn't inflate the trailing-silence run.
            self.trailing_silence += 1

        if self.total >= self.max_chunks:
            return self.TOO_LONG

        # Finish only when we've heard enough speech AND a full end-of-turn gap.
        if (self.started
                and self.had_enough_speech
                and self.trailing_silence >= self.end_silence_chunks):
            return self.DONE

        return self.LISTENING
