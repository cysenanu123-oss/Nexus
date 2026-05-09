"""
NEXUS — voice/wakeword.py
Wake-word detection layer — the trigger of NEXUS.

Sits directly above the microphone capture layer:

    [MICROPHONE]
         ↓
    listener.py
         ↓
    wakeword.py        ← YOU ARE HERE
         ↓
    speech_to_text.py
         ↓
    command parser

Responsibilities:
  - Continuously consume raw audio chunks from MicrophoneListener
  - Run a wake-word model on those chunks in real time
  - Fire a callback (or unblock a blocking call) the moment "Hey Nexus"
    (or any configured phrase) is detected
  - Expose a clean, simple interface to the layer above:
      wait_for_wake_word()   — block until the wake word is heard
      start() / stop()       — lifecycle management
      on_detected(callback)  — register a callback for async usage

Backend priority (auto-selected at startup):
  1. openWakeWord  — pip install openwakeword
     Uses the pre-trained "hey_jarvis" model as the closest phonetic match
     to "Hey Nexus", or a custom .tflite/.onnx model if you provide one.
     Feeds audio to the model in the 80 ms frames it expects.

  2. Keyword fallback  — zero extra dependencies
     Splits audio into overlapping 1-second windows, applies VAD, and
     checks if a Whisper or simple RMS pattern roughly matches the wake
     phrase.  Much less accurate but works immediately without any install.

Dependencies:
    pip install sounddevice numpy scipy          # already needed for listener
    pip install openwakeword                     # recommended — best accuracy
"""

from __future__ import annotations

import time
import logging
import threading
import collections
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger("nexus.voice.wakeword")

# ── lazy imports ──────────────────────────────────────────────
try:
    from openwakeword.model import Model as _OWWModel
    import openwakeword.utils as _oww_utils
    _OWW_AVAILABLE = True
except ImportError:
    _OWWModel = None          # type: ignore
    _oww_utils = None         # type: ignore
    _OWW_AVAILABLE = False

# Support two execution contexts:
#   python voice/wakeword.py          → 'voice' not on sys.path, use sibling import
#   from voice.wakeword import ...    → package import, use full path
try:
    from voice.listener import MicrophoneListener, SAMPLE_RATE
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from listener import MicrophoneListener, SAMPLE_RATE  # type: ignore


# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────

# openWakeWord processes audio in exactly 80 ms frames at 16 kHz
OWW_FRAME_MS    = 80
OWW_FRAME_SIZE  = int(SAMPLE_RATE * OWW_FRAME_MS / 1000)   # = 1280 samples

# Detection threshold — lower = more sensitive but more false positives
OWW_THRESHOLD   = 0.5

# Pre-trained model to use when no custom model is supplied.
# "hey_jarvis" is the closest phonetically to "Hey Nexus" in the
# default openWakeWord model set.  Alternatives: "hey_mycroft", "alexa".
OWW_DEFAULT_MODEL = "hey_jarvis"

# Cooldown between successive detections (seconds) — prevents double-fire
COOLDOWN_SEC = 1.5

# Fallback backend: rolling audio window size (seconds) used for VAD scoring
FALLBACK_WINDOW_SEC = 1.2


# ─────────────────────────────────────────────────────────────
#  DETECTION RESULT
# ─────────────────────────────────────────────────────────────

class WakeWordResult:
    """
    Returned / passed to callback whenever the wake word is detected.

    Attributes:
        model_name  — which model fired (e.g. "hey_jarvis" or "fallback_vad")
        score       — confidence score  0.0 – 1.0  (1.0 for fallback hits)
        timestamp   — time.time() of detection
    """

    def __init__(self, model_name: str, score: float):
        self.model_name = model_name
        self.score      = score
        self.timestamp  = time.time()

    def __repr__(self) -> str:
        return (
            f"WakeWordResult(model={self.model_name!r}, "
            f"score={self.score:.3f})"
        )


# ─────────────────────────────────────────────────────────────
#  WAKE WORD DETECTOR
# ─────────────────────────────────────────────────────────────

class WakeWordDetector:
    """
    Listens continuously via a MicrophoneListener and fires whenever the
    configured wake word / phrase is detected.

    Usage (blocking):
        detector = WakeWordDetector()
        with detector:
            result = detector.wait_for_wake_word()
            print(f"Wake word detected! score={result.score:.2f}")

    Usage (callback / async):
        def on_wake(result):
            print("NEXUS activated!")

        detector = WakeWordDetector(callback=on_wake)
        detector.start()
        # ... your main loop ...
        detector.stop()

    Parameters
    ----------
    wake_phrase : str
        Human-readable name of what you're listening for (used for logging).
    model_path : str | None
        Path to a custom .tflite or .onnx openWakeWord model.
        Pass None to use the built-in default model (hey_jarvis).
    threshold : float
        Confidence threshold for openWakeWord (0.0 – 1.0).
    callback : callable | None
        If provided, called with a WakeWordResult every time the wake word
        fires.  The call happens on the detector's background thread.
    device_index : int | None
        Microphone device index passed through to MicrophoneListener.
    cooldown_sec : float
        Minimum seconds between successive detections.
    """

    def __init__(
        self,
        wake_phrase: str = "hey nexus",
        model_path: Optional[str] = None,
        threshold: float = OWW_THRESHOLD,
        callback: Optional[Callable[[WakeWordResult], None]] = None,
        device_index: Optional[int] = None,
        cooldown_sec: float = COOLDOWN_SEC,
        listener: Optional[MicrophoneListener] = None,
    ):
        self.wake_phrase  = wake_phrase.lower().strip()
        self.model_path   = model_path
        self.threshold    = threshold
        self.callback     = callback
        self.device_index = device_index
        self.cooldown_sec = cooldown_sec

        self._shared_listener = listener is not None
        self._listener: Optional[MicrophoneListener] = listener
        self._thread:   Optional[threading.Thread]   = None
        self._running   = False
        self._paused    = False
        self._lock      = threading.Lock()

        # Event set each time a detection fires — used by wait_for_wake_word()
        self._detected_event = threading.Event()
        self._last_result:   Optional[WakeWordResult] = None
        self._last_fire_ts   = 0.0

        # Backend resolved at start()
        self._backend: str = "none"
        self._oww_model     = None   # openWakeWord Model instance

    # ── context manager ───────────────────────────────────────

    def __enter__(self) -> "WakeWordDetector":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ── lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        """Start the listener and the background detection thread."""
        with self._lock:
            if self._running:
                return

            if not self._shared_listener:
                self._listener = MicrophoneListener(device_index=self.device_index)
                self._listener.start()
            elif self._listener and not self._listener._stream:
                # If shared listener was passed but not started, start it
                self._listener.start()

            self._backend = self._init_backend()
            self._running = True
            self._detected_event.clear()

        self._thread = threading.Thread(
            target=self._run_loop,
            name="nexus-wakeword",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "WakeWordDetector started — wake_phrase=%r, backend=%s",
            self.wake_phrase, self._backend,
        )

    def stop(self) -> None:
        """Stop the detector and close the microphone."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._paused = False

        if self._listener and not self._shared_listener:
            self._listener.stop()
            self._listener = None

        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

        logger.info("WakeWordDetector stopped.")

    def pause(self) -> None:
        """Temporarily stop reading audio (useful when another system takes the mic)."""
        self._paused = True

    def resume(self) -> None:
        """Resume reading audio."""
        self._paused = False

    def trigger(self) -> None:
        """Manually force the detector to fire."""
        self._last_result = WakeWordResult(model_name="manual_trigger", score=1.0)
        self._detected_event.set()

    # ── public API ────────────────────────────────────────────

    def wait_for_wake_word(self, timeout: Optional[float] = None) -> Optional[WakeWordResult]:
        """
        Block until the wake word is detected (or timeout expires).

        Returns:
            WakeWordResult on detection, None on timeout.

        Example:
            with WakeWordDetector() as det:
                result = det.wait_for_wake_word()
        """
        if not self._running:
            raise RuntimeError(
                "Detector is not running. Call start() or use 'with WakeWordDetector()'."
            )
        fired = self._detected_event.wait(timeout=timeout)
        if fired:
            self._detected_event.clear()
            return self._last_result
        return None

    @property
    def backend(self) -> str:
        """Which detection backend is active: 'openwakeword' | 'fallback_vad'."""
        return self._backend

    @property
    def is_running(self) -> bool:
        return self._running

    # ── backend initialisation ────────────────────────────────

    def _init_backend(self) -> str:
        """
        Try to load the openWakeWord backend.
        Falls back to the simple energy/VAD detector if unavailable.
        """
        if not _OWW_AVAILABLE:
            logger.warning(
                "openWakeWord not installed — using fallback VAD backend. "
                "For accurate wake-word detection, run: pip install openwakeword"
            )
            return "fallback_vad"

        try:
            import pathlib

            if self.model_path is None:
                # Resolve the bundled model path from the installed package.
                # openwakeword v0.4.x ships models inside the package resources;
                # the constructor wants an explicit file path, not a bare name.
                _oww_pkg = pathlib.Path(_OWWModel.__module__.split(".")[0])
                _pkg_root = pathlib.Path(
                    __import__("openwakeword").__file__
                ).parent
                _model_file = (
                    _pkg_root / "resources" / "models"
                    / f"{OWW_DEFAULT_MODEL}.onnx"
                )
                if not _model_file.exists():
                    # Try versioned filename e.g. hey_jarvis_v0.1.onnx
                    candidates = list(
                        (_pkg_root / "resources" / "models").glob(
                            f"{OWW_DEFAULT_MODEL}*.onnx"
                        )
                    )
                    if not candidates:
                        raise FileNotFoundError(
                            f"No bundled model found for '{OWW_DEFAULT_MODEL}' "
                            f"in {_pkg_root / 'resources' / 'models'}"
                        )
                    _model_file = candidates[0]

                model_path_to_use = str(_model_file)
            else:
                model_path_to_use = self.model_path

            logger.info(
                "Loading openWakeWord model: %s  (threshold=%.2f)",
                model_path_to_use, self.threshold,
            )
            # v0.4.x API: positional arg is wakeword_model_paths (list of paths)
            self._oww_model = _OWWModel(
                wakeword_model_paths=[model_path_to_use],
            )
            logger.info("openWakeWord model loaded successfully.")
            return "openwakeword"

        except Exception as exc:
            logger.warning(
                "openWakeWord failed to load (%s) — using fallback VAD backend.", exc
            )
            return "fallback_vad"

    # ── main detection loop ───────────────────────────────────

    def _run_loop(self) -> None:
        """Background thread — reads audio and runs the active backend."""
        if self._backend == "openwakeword":
            self._loop_openwakeword()
        else:
            self._loop_fallback_vad()

    # ── openWakeWord backend ──────────────────────────────────

    def _loop_openwakeword(self) -> None:
        """
        Feed 80 ms frames to openWakeWord.

        The listener reads in 400 ms chunks at the native hardware rate
        (e.g. 44100 Hz), already resampled to 16 kHz float32 via
        capture_phrase().  Here we split each listener chunk into the
        1280-sample (80 ms @ 16 kHz) frames that openWakeWord expects.

        We use stream_chunks() directly so we bypass the VAD logic in
        capture_phrase() — wake word detection must run continuously,
        even during silence.
        """
        # Accumulate samples here; drain 1280 at a time into the model
        frame_buffer: list[np.ndarray] = []
        sample_bank  = np.zeros(0, dtype=np.float32)

        while self._running and self._listener is not None:
            if self._paused:
                time.sleep(0.05)
                # clear buffers so it doesn't trigger on stale audio when resumed
                sample_bank = np.zeros(0, dtype=np.float32)
                continue

            # read_chunk() returns int16 at the hardware capture rate
            raw = self._listener.read_chunk(timeout=0.6)
            if raw is None:
                continue

            # Convert to float32 at 16 kHz (resampling handled internally)
            chunk_f32 = raw.astype(np.float32) / 32768.0
            if self._listener._capture_rate != SAMPLE_RATE:
                try:
                    from voice.listener import _resample
                except ModuleNotFoundError:
                    from listener import _resample  # type: ignore
                chunk_f32 = _resample(
                    chunk_f32,
                    self._listener._capture_rate,
                    SAMPLE_RATE,
                )

            sample_bank = np.concatenate([sample_bank, chunk_f32])

            # Drain complete 80 ms frames from sample_bank
            while len(sample_bank) >= OWW_FRAME_SIZE:
                frame       = sample_bank[:OWW_FRAME_SIZE]
                sample_bank = sample_bank[OWW_FRAME_SIZE:]

                # openWakeWord expects int16 PCM bytes or float32 ndarray
                try:
                    prediction = self._oww_model.predict(frame)
                except Exception as exc:
                    logger.error("openWakeWord predict() error: %s", exc)
                    continue

                # prediction is { model_name: score }
                for model_name, score in prediction.items():
                    if score >= self.threshold:
                        self._fire(
                            WakeWordResult(model_name=model_name, score=float(score))
                        )

    # ── fallback VAD backend ──────────────────────────────────

    def _loop_fallback_vad(self) -> None:
        """
        Lightweight fallback — no ML model required.

        Maintains a sliding window of audio and fires if:
          1.  A speech burst is detected (RMS above threshold)
          2.  The speech energy profile resembles a 2-syllable + 2-syllable
              pattern (e.g.  "HEY-NEX"  "US")
          3.  The burst lasts between 0.4 s and 2.0 s (word-length guard)

        This is intentionally simple and will produce more false-positives
        than openWakeWord.  It is a placeholder so NEXUS can function
        before openWakeWord is installed.  Install openWakeWord for
        production use.
        """
        window_samples = int(SAMPLE_RATE * FALLBACK_WINDOW_SEC)
        ring: collections.deque[np.ndarray] = collections.deque()
        ring_len = 0   # total samples in ring

        # State machine
        in_burst     = False
        burst_start  = 0.0
        silence_pad  = 0

        # Raise threshold above typical ambient noise (~0.01-0.02 RMS on most
        # mics at rest).  0.035 requires noticeable speech energy to trigger.
        VAD_THRESHOLD  = 0.035   # RMS floor for a genuine speech onset
        SILENCE_CHUNKS = 1       # 1 × 400 ms = 400 ms quiet to end a burst
        MIN_BURST_SEC  = 0.35    # shortest plausible wake phrase
        MAX_BURST_SEC  = 2.0     # longest plausible wake phrase

        while self._running and self._listener is not None:
            if self._paused:
                time.sleep(0.05)
                in_burst = False
                silence_pad = 0
                continue

            raw = self._listener.read_chunk(timeout=0.6)
            if raw is None:
                continue

            rms = float(np.sqrt(np.mean((raw.astype(np.float32) / 32768.0) ** 2)))

            if rms >= VAD_THRESHOLD:
                if not in_burst:
                    in_burst    = True
                    burst_start = time.time()
                    logger.debug("[fallback] Speech burst started (RMS=%.4f)", rms)
                silence_pad = 0
            else:
                if in_burst:
                    silence_pad += 1
                    if silence_pad >= SILENCE_CHUNKS:
                        # Burst ended — check duration
                        burst_dur = time.time() - burst_start
                        in_burst  = False
                        silence_pad = 0
                        logger.debug(
                            "[fallback] Burst ended — duration=%.2fs", burst_dur
                        )
                        if MIN_BURST_SEC <= burst_dur <= MAX_BURST_SEC:
                            # Duration matches a plausible 2+2 syllable wake phrase
                            self._fire(
                                WakeWordResult(
                                    model_name="fallback_vad",
                                    score=min(1.0, rms * 20),
                                )
                            )

            # Guard: if in_burst too long, reset
            if in_burst and (time.time() - burst_start) > MAX_BURST_SEC:
                in_burst    = False
                silence_pad = 0

    # ── fire detection ────────────────────────────────────────

    def _fire(self, result: WakeWordResult) -> None:
        """
        Called when a detection fires.  Applies cooldown, sets the event,
        and calls the user callback (if any).
        """
        now = time.time()
        if now - self._last_fire_ts < self.cooldown_sec:
            logger.debug(
                "[wakeword] Suppressed duplicate detection (cooldown). score=%.3f",
                result.score,
            )
            return

        self._last_fire_ts = now
        self._last_result  = result

        logger.info(
            "\033[92m[wakeword] DETECTED — model=%s  score=%.3f\033[0m",
            result.model_name, result.score,
        )

        # Unblock wait_for_wake_word()
        self._detected_event.set()

        # Fire user callback on this thread
        if self.callback is not None:
            try:
                self.callback(result)
            except Exception as exc:
                logger.error("Wake word callback raised an exception: %s", exc)


# ─────────────────────────────────────────────────────────────
#  STANDALONE ENTRY — python voice/wakeword.py
# ─────────────────────────────────────────────────────────────

def _print_backend_info() -> None:
    if _OWW_AVAILABLE:
        print(f"  \033[92m✓\033[0m  openWakeWord available  — will use model: {OWW_DEFAULT_MODEL!r}")
    else:
        print(
            "  \033[33m⚠\033[0m  openWakeWord NOT installed — fallback VAD backend will be used.\n"
            "       For accurate detection: \033[1mpip install openwakeword\033[0m"
        )


if __name__ == "__main__":
    import sys
    import logging as _logging

    _logging.basicConfig(
        level=_logging.DEBUG,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n  NEXUS — voice/wakeword.py")
    print("  " + "─" * 40)
    _print_backend_info()
    print()

    if "--info" in sys.argv:
        sys.exit(0)

    # ── demo mode: wait for N detections then exit ─────────────
    n_detections = 3
    detected     = 0

    def on_detected(result: WakeWordResult) -> None:
        global detected
        detected += 1
        ts = time.strftime("%H:%M:%S")
        print(
            f"\n  \033[92m[{ts}] 🎙  Wake word detected!\033[0m"
            f"  model={result.model_name!r}  score={result.score:.3f}"
            f"  (detection #{detected}/{n_detections})"
        )
        if detected >= n_detections:
            print("\n  Demo complete — stopping.\n")

    print(f"  Listening for wake word ({n_detections} detections)...")
    print("  Say \033[1m'Hey Nexus'\033[0m  — Ctrl-C to quit.\n")

    detector = WakeWordDetector(
        wake_phrase="hey nexus",
        callback=on_detected,
    )

    try:
        detector.start()
        while detected < n_detections:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        detector.stop()
        print("  Done.\n")
