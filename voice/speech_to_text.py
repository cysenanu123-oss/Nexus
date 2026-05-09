"""
NEXUS — voice/speech_to_text.py
Speech-to-Text layer — the voice-to-words converter of NEXUS.

Sits directly above the wake-word layer in the voice pipeline:

    [MICROPHONE]
         ↓
    listener.py
         ↓
    wakeword.py
         ↓
    speech_to_text.py    ← YOU ARE HERE
         ↓
    command parser

Responsibilities:
  - Receive a float32 audio array (16 kHz, from capture_phrase())
  - Run it through a Whisper model (via faster-whisper)
  - Return a clean TranscriptionResult with text, language, confidence
  - Expose a simple transcribe(audio) function for single-shot use
  - Provide a Transcriber class for persistent use (model loaded once)

Backends:
  - faster-whisper  (primary)  — pip install faster-whisper
    Local ONNX/CTranslate2 runtime, no API key, works offline.
    Default model: "tiny" for low latency; swap to "base" or "small"
    for better accuracy.

Dependencies:
    pip install faster-whisper
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("nexus.voice.stt")

# ── lazy import ───────────────────────────────────────────────
try:
    from faster_whisper import WhisperModel as _WhisperModel
    _FW_AVAILABLE = True
except ImportError:
    _WhisperModel = None   # type: ignore
    _FW_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
#  CONSTANTS & DEFAULTS
# ─────────────────────────────────────────────────────────────

SAMPLE_RATE     = 16_000     # Hz — must match listener.py

# Model size trade-off:
#   tiny   → ~39 MB,  ~30× real-time on CPU  — great for command detection
#   base   → ~74 MB,  ~16× real-time on CPU  — better accuracy
#   small  → ~244 MB, ~6× real-time on CPU   — even better
#   medium → ~769 MB, ~2× real-time on CPU   — near human-level
# For wake-word-triggered short commands, "tiny" is the sweet spot.
DEFAULT_MODEL   = "tiny"
DEFAULT_DEVICE  = "cpu"       # "cuda" if you have a GPU
DEFAULT_COMPUTE = "int8"      # quantised — fast on CPU, minimal accuracy loss
DEFAULT_LANG    = None        # None = auto-detect per utterance

# Whisper VAD filter: skip silent audio segments automatically
VAD_FILTER      = True

# No-speech probability threshold: above this → treat as silence, return ""
NO_SPEECH_PROB_THRESHOLD = 0.6


# ─────────────────────────────────────────────────────────────
#  TRANSCRIPTION RESULT
# ─────────────────────────────────────────────────────────────

@dataclass
class TranscriptionResult:
    """
    Output of a single transcription call.

    Attributes
    ----------
    text          — normalised transcribed text (stripped, lowercased)
    raw_text      — original text as returned by Whisper (un-lowercased)
    language      — detected language code, e.g. "en"
    language_prob — confidence of language detection  0.0 – 1.0
    avg_logprob   — average log-probability of the transcription
                    (higher = more confident, typically > -0.5 is good)
    no_speech_prob — probability that the audio contained no speech
    duration_sec  — duration of the audio in seconds
    elapsed_sec   — wall-clock time taken to transcribe
    segments      — list of raw segment dicts (word-level detail)
    """
    text:           str   = ""
    raw_text:       str   = ""
    language:       str   = "?"
    language_prob:  float = 0.0
    avg_logprob:    float = 0.0
    no_speech_prob: float = 0.0
    duration_sec:   float = 0.0
    elapsed_sec:    float = 0.0
    segments:       list  = field(default_factory=list)

    @property
    def is_speech(self) -> bool:
        """True if the result contains likely speech (not silence/noise)."""
        return (
            bool(self.text)
            and self.no_speech_prob < NO_SPEECH_PROB_THRESHOLD
        )

    @property
    def confidence(self) -> float:
        """
        Normalised confidence score 0.0 – 1.0.
        Derived from avg_logprob: logprob of -0.0 → 1.0, -1.0 → ~0.37.
        """
        import math
        return max(0.0, min(1.0, math.exp(self.avg_logprob)))

    def __str__(self) -> str:
        if not self.text:
            return "[silence]"
        return (
            f'"{self.text}"  '
            f'[lang={self.language} p={self.language_prob:.2f}  '
            f'conf={self.confidence:.2f}  '
            f'no_speech={self.no_speech_prob:.2f}  '
            f'rt={self.elapsed_sec:.2f}s]'
        )


# ─────────────────────────────────────────────────────────────
#  TRANSCRIBER CLASS
# ─────────────────────────────────────────────────────────────

class Transcriber:
    """
    Persistent Whisper-based speech-to-text engine.

    Loads the model once, then accepts repeated .transcribe() calls
    with minimal latency (no model reload between calls).

    Usage:
        stt = Transcriber()           # loads model on construction
        result = stt.transcribe(audio)
        print(result.text)

    Or as a context manager:
        with Transcriber() as stt:
            result = stt.transcribe(audio)

    Parameters
    ----------
    model_size   — "tiny" | "base" | "small" | "medium" | "large-v3"
    device       — "cpu" | "cuda"
    compute_type — "int8" | "float16" | "float32"
    language     — ISO 639-1 code ("en") or None for auto-detect
    vad_filter   — strip silent segments before transcription
    """

    def __init__(
        self,
        model_size:   str           = DEFAULT_MODEL,
        device:       str           = DEFAULT_DEVICE,
        compute_type: str           = DEFAULT_COMPUTE,
        language:     Optional[str] = DEFAULT_LANG,
        vad_filter:   bool          = VAD_FILTER,
    ):
        if not _FW_AVAILABLE:
            raise ImportError(
                "faster-whisper is not installed.\n"
                "Run:  pip install faster-whisper"
            )

        self.model_size   = model_size
        self.device       = device
        self.compute_type = compute_type
        self.language     = language
        self.vad_filter   = vad_filter

        self._model: Optional[_WhisperModel] = None
        self._load_model()

    # ── context manager ───────────────────────────────────────

    def __enter__(self) -> "Transcriber":
        return self

    def __exit__(self, *_) -> None:
        self.unload()

    # ── lifecycle ─────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the Whisper model into memory (runs once)."""
        logger.info(
            "Loading faster-whisper model '%s' on %s (%s) …",
            self.model_size, self.device, self.compute_type,
        )
        t0 = time.time()
        self._model = _WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        elapsed = time.time() - t0
        logger.info(
            "faster-whisper model '%s' loaded in %.2fs.",
            self.model_size, elapsed,
        )

    def unload(self) -> None:
        """Release the model from memory."""
        self._model = None
        logger.info("Transcriber model unloaded.")

    # ── main API ──────────────────────────────────────────────

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe a float32 audio array to text.

        Parameters
        ----------
        audio          — float32 ndarray at 16 kHz (from capture_phrase())
        language       — override per-call language (None = auto)
        initial_prompt — optional hint text to guide transcription style

        Returns
        -------
        TranscriptionResult
        """
        if self._model is None:
            raise RuntimeError("Model is not loaded. Call _load_model() first.")

        if audio is None or len(audio) == 0:
            logger.debug("transcribe() called with empty audio — returning empty result.")
            return TranscriptionResult()

        # Ensure float32 in [-1, 1]
        audio = _ensure_float32(audio)
        duration_sec = len(audio) / SAMPLE_RATE

        t0 = time.time()

        try:
            segments_gen, info = self._model.transcribe(
                audio,
                language=language or self.language,
                vad_filter=self.vad_filter,
                initial_prompt=initial_prompt,
                without_timestamps=True,
                beam_size=5,
                no_speech_threshold=NO_SPEECH_PROB_THRESHOLD,
            )

            # Materialise the generator — segments are lazy in faster-whisper
            raw_segments = list(segments_gen)

        except Exception as exc:
            logger.error("Transcription failed: %s", exc)
            return TranscriptionResult(duration_sec=duration_sec)

        elapsed = time.time() - t0

        # ── Assemble result ───────────────────────────────────
        raw_text = " ".join(s.text.strip() for s in raw_segments).strip()
        text     = raw_text.lower().strip()

        avg_logprob    = float(np.mean([s.avg_logprob    for s in raw_segments])) if raw_segments else 0.0
        no_speech_prob = float(np.mean([s.no_speech_prob for s in raw_segments])) if raw_segments else 1.0

        result = TranscriptionResult(
            text           = text,
            raw_text       = raw_text,
            language       = info.language,
            language_prob  = float(info.language_probability),
            avg_logprob    = avg_logprob,
            no_speech_prob = no_speech_prob,
            duration_sec   = duration_sec,
            elapsed_sec    = elapsed,
            segments       = [
                {
                    "text":          s.text.strip(),
                    "start":         s.start,
                    "end":           s.end,
                    "avg_logprob":   s.avg_logprob,
                    "no_speech_prob": s.no_speech_prob,
                }
                for s in raw_segments
            ],
        )

        logger.info(
            "Transcribed %.2fs of audio in %.2fs → %s",
            duration_sec, elapsed,
            f'"{result.text}"' if result.text else "[silence]",
        )

        return result

    def is_ready(self) -> bool:
        """True if the model is loaded and ready."""
        return self._model is not None


# ─────────────────────────────────────────────────────────────
#  MODULE-LEVEL CONVENIENCE FUNCTION
# ─────────────────────────────────────────────────────────────

_default_transcriber: Optional[Transcriber] = None


def transcribe(
    audio: np.ndarray,
    model_size: str = DEFAULT_MODEL,
    language: Optional[str] = None,
) -> TranscriptionResult:
    """
    One-shot convenience wrapper — loads a shared Transcriber on first call
    and reuses it for subsequent calls (model stays in memory).

    Example:
        from voice.speech_to_text import transcribe
        result = transcribe(audio)
        print(result.text)
    """
    global _default_transcriber
    if _default_transcriber is None or _default_transcriber.model_size != model_size:
        _default_transcriber = Transcriber(model_size=model_size)
    return _default_transcriber.transcribe(audio, language=language)


# ─────────────────────────────────────────────────────────────
#  AUDIO HELPERS
# ─────────────────────────────────────────────────────────────

def _ensure_float32(audio: np.ndarray) -> np.ndarray:
    """
    Ensure the audio array is float32 in the range [-1.0, 1.0].
    Handles int16 input gracefully.
    """
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    # Clip to safe range in case of floating-point drift
    return np.clip(audio, -1.0, 1.0)


# ─────────────────────────────────────────────────────────────
#  STANDALONE ENTRY — python voice/speech_to_text.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n  NEXUS — voice/speech_to_text.py")
    print("  " + "─" * 40)

    if not _FW_AVAILABLE:
        print("  ✗  faster-whisper not installed.")
        print("     Run:  pip install faster-whisper")
        sys.exit(1)

    # ── --file mode: transcribe a WAV file ────────────────────
    if "--file" in sys.argv:
        idx = sys.argv.index("--file")
        wav_path = sys.argv[idx + 1]
        print(f"\n  Transcribing file: {wav_path}\n")

        try:
            from faster_whisper import decode_audio
            audio = decode_audio(wav_path)
        except Exception as e:
            print(f"  ✗  Could not load audio file: {e}")
            sys.exit(1)

        model_size = "base"
        for i, arg in enumerate(sys.argv):
            if arg == "--model" and i + 1 < len(sys.argv):
                model_size = sys.argv[i + 1]

        stt = Transcriber(model_size=model_size)
        result = stt.transcribe(audio)

        print(f"\n  Result   : {result}")
        print(f"  Text     : {result.raw_text!r}")
        print(f"  Language : {result.language} (p={result.language_prob:.2f})")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  No-speech : {result.no_speech_prob:.2f}")
        print(f"  RT factor : {result.elapsed_sec / max(result.duration_sec, 0.001):.2f}× real-time")
        sys.exit(0)

    # ── --listen mode: capture from mic and transcribe ────────
    if "--listen" in sys.argv:
        # Resolve listener import regardless of run context
        try:
            from voice.listener import MicrophoneListener
        except ModuleNotFoundError:
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from voice.listener import MicrophoneListener

        model_size = "tiny"
        for i, arg in enumerate(sys.argv):
            if arg == "--model" and i + 1 < len(sys.argv):
                model_size = sys.argv[i + 1]

        print(f"\n  Speak now — NEXUS will transcribe each phrase.")
        print(f"  Model: {model_size}   Ctrl-C to stop.\n")

        stt = Transcriber(model_size=model_size)
        listener = MicrophoneListener()

        try:
            listener.start()
            while True:
                print("  \033[2m[listening...]\033[0m", flush=True)
                audio = listener.capture_phrase(verbose=False)
                if audio is None:
                    continue

                result = stt.transcribe(audio)

                if result.is_speech:
                    print(
                        f"\n  \033[92m▶  {result.raw_text}\033[0m"
                        f"\033[2m  (conf={result.confidence:.2f}"
                        f"  lang={result.language}"
                        f"  {result.elapsed_sec:.2f}s)\033[0m\n"
                    )
                else:
                    print(f"  \033[2m[no speech detected]\033[0m")

        except KeyboardInterrupt:
            print("\n  Stopped.\n")
        finally:
            listener.stop()

        sys.exit(0)

    # ── --bench mode: benchmark model load + transcription ────
    if "--bench" in sys.argv:
        print("\n  Benchmarking faster-whisper model load times...\n")
        test_audio = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)  # 2s silence

        for size in ["tiny", "base", "small"]:
            t0 = time.time()
            try:
                stt = Transcriber(model_size=size)
                load_t = time.time() - t0

                t1 = time.time()
                stt.transcribe(test_audio)
                infer_t = time.time() - t1

                print(f"  {size:<8} load={load_t:.2f}s  infer(2s audio)={infer_t:.2f}s")
            except Exception as e:
                print(f"  {size:<8} FAILED: {e}")

        print()
        sys.exit(0)

    # ── default: show usage ───────────────────────────────────
    print(
        "\n  Usage:\n"
        "    python voice/speech_to_text.py --listen              # live mic transcription\n"
        "    python voice/speech_to_text.py --listen --model base # use a larger model\n"
        "    python voice/speech_to_text.py --file audio.wav      # transcribe a file\n"
        "    python voice/speech_to_text.py --bench               # benchmark model sizes\n"
    )
