"""
NEXUS — voice/listener.py
Microphone capture layer — the ears of NEXUS.

Sits at the bottom of the voice pipeline:

    [MICROPHONE]
         ↓
    listener.py        ← YOU ARE HERE
         ↓
    wakeword.py
         ↓
    speech_to_text.py
         ↓
    command parser

Responsibilities:
  - Open and manage the microphone stream via sounddevice
  - Capture raw PCM audio in real time
  - Detect voice activity (VAD) so we only pass speech upward
  - Expose clean, simple interfaces for the layers above:
      capture_phrase()   — block until one spoken phrase is captured
      stream_chunks()    — yield raw audio chunks continuously
      list_devices()     — show available microphone devices
      test_microphone()  — sanity-check mic level in terminal

Dependencies (install once on your machine):
    pip install sounddevice numpy scipy
    sudo apt install libportaudio2 portaudio19-dev ffmpeg -y
"""

from __future__ import annotations

import time
import threading
import logging
import numpy as np

# ── lazy import so the rest of NEXUS still loads if sounddevice
#    is missing — lets us show a clean error instead of a crash
try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except OSError:
    _SD_AVAILABLE = False
    sd = None  # type: ignore

from scipy.signal import butter, sosfilt

logger = logging.getLogger("nexus.voice.listener")


# ─────────────────────────────────────────────────────────────
#  CONSTANTS & DEFAULTS
# ─────────────────────────────────────────────────────────────

SAMPLE_RATE     = 16_000   # Hz  — Whisper target rate (we resample to this)
CHANNELS        = 1        # Mono
DTYPE           = "int16"  # 16-bit PCM

# How many milliseconds of audio per read() call.
# Larger = fewer syscalls & less overflow risk.  400 ms is very safe on ALSA.
CHUNK_MS        = 400

# Voice Activity Detection
VAD_ENERGY_THRESHOLD  = 0.02   # RMS threshold — tuned for ALC257 ambient noise
VAD_SPEECH_PAD_MS     = 300    # ms of silence to keep after speech ends
VAD_MIN_SPEECH_MS     = 200    # ignore bursts shorter than this
VAD_END_SILENCE_MS    = 1200   # end-of-turn: how long you must go quiet to finish
                               # a turn. Long enough to survive a thinking pause,
                               # so "speaking… thinking… speaking" stays one turn.
VAD_MAX_PHRASE_SEC    = 30     # hard cut — max length of one captured phrase

# Noise gate / high-pass filter
HIGHPASS_CUTOFF_HZ = 80        # filter out low-freq rumble (AC, fans)

RING_BUFFER_SEC = 60           # ring buffer holds last N seconds of audio


# ─────────────────────────────────────────────────────────────
#  AUDIO UTILITIES
# ─────────────────────────────────────────────────────────────

def _highpass_filter(audio: np.ndarray, cutoff: int = HIGHPASS_CUTOFF_HZ,
                     sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Apply a Butterworth high-pass filter to remove low-frequency noise
    (HVAC rumble, desk vibration, etc.) before VAD analysis.
    """
    sos = butter(4, cutoff, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, audio.astype(np.float32))


def _rms(audio: np.ndarray) -> float:
    """Root-mean-square energy of an audio chunk — used for VAD."""
    samples = audio.astype(np.float32) / 32768.0   # normalise int16 → [-1, 1]
    return float(np.sqrt(np.mean(samples ** 2)))


def _int16_to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert int16 PCM to float32 in [-1.0, 1.0] range (Whisper input format)."""
    return audio.astype(np.float32) / 32768.0


def _frames_to_ms(frames: int, sr: int = SAMPLE_RATE) -> float:
    return (frames / sr) * 1000.0


def _get_device_rate(device_index: int | None) -> int:
    """
    Query sounddevice for the native sample rate of a device.
    Falls back to 44100 if the query fails — covers most built-in cards.
    """
    if not _SD_AVAILABLE:
        return 44_100
    try:
        info = sd.query_devices(device_index, kind="input")
        return int(info["default_samplerate"])
    except Exception:
        return 44_100


def _resample(audio: np.ndarray, from_rate: int, to_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Resample a float32 audio array from `from_rate` to `to_rate` Hz.
    Uses scipy.signal.resample_poly for high-quality integer-ratio resampling.
    No-op if rates are already equal.
    """
    if from_rate == to_rate:
        return audio

    from math import gcd
    from scipy.signal import resample_poly

    g    = gcd(to_rate, from_rate)
    up   = to_rate   // g
    down = from_rate // g
    return resample_poly(audio, up, down).astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  MicrophoneListener  (blocking-read mode — no callback)
# ─────────────────────────────────────────────────────────────

class MicrophoneListener:
    """
    Opens a sounddevice InputStream in **blocking-read** mode and
    reads from the microphone on whichever thread calls read_chunk().

    This avoids the callback/audio-thread model entirely:
      - No real-time callback constraints
      - No queue contention
      - No PortAudio xrun/overflow from slow Python callbacks
      - Works reliably on ALSA without tuning buffer sizes

    Upper layers call:
        capture_phrase()   → np.ndarray of speech audio (float32, 16 kHz)
        stream_chunks()    → generator of raw int16 chunks
        is_speech(chunk)   → bool VAD check on a chunk

    Example:
        listener = MicrophoneListener()
        with listener:
            audio = listener.capture_phrase()
            # pass audio to speech_to_text.transcribe(audio)
    """

    def __init__(
        self,
        device_index: int | None = None,
        sample_rate: int = SAMPLE_RATE,
        chunk_ms: int = CHUNK_MS,
        vad_threshold: float = VAD_ENERGY_THRESHOLD,
        vad_pad_ms: int = VAD_SPEECH_PAD_MS,
        vad_min_ms: int = VAD_MIN_SPEECH_MS,
        end_silence_ms: int = VAD_END_SILENCE_MS,
        max_phrase_sec: float = VAD_MAX_PHRASE_SEC,
        noise_filter: bool = True,
    ):
        self.device_index   = device_index
        self.sample_rate    = sample_rate    # Whisper target rate (16 kHz)
        self.chunk_ms       = chunk_ms
        self.vad_threshold  = vad_threshold
        self.vad_pad_ms     = vad_pad_ms
        self.vad_min_ms     = vad_min_ms
        self.end_silence_ms = end_silence_ms
        self.max_phrase_sec = max_phrase_sec
        self.noise_filter   = noise_filter

        # Let settings.json tune endpointing without editing code.
        try:
            from core.config import cfg
            self.end_silence_ms = int(cfg.get("voice.end_silence_ms", end_silence_ms))
            self.max_phrase_sec = float(cfg.get("voice.max_listen_sec", max_phrase_sec))
        except Exception:
            pass

        # Resolved at .start() — set to the device's actual hardware rate
        self._capture_rate: int = sample_rate
        self._chunk_frames: int = 0       # frames per read(), set at start()

        self._stream: "sd.InputStream | None" = None
        self._running = False
        self._lock = threading.Lock()

    # ── context manager ───────────────────────────────────────

    def __enter__(self) -> "MicrophoneListener":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ── lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        """Open the microphone stream (blocking-read mode — no callback)."""
        if not _SD_AVAILABLE:
            raise RuntimeError(
                "sounddevice is not available. "
                "Run: pip install sounddevice  &&  sudo apt install portaudio19-dev -y"
            )
        if self._running:
            return

        # ── auto-detect native hardware rate ──────────────────
        self._capture_rate = _get_device_rate(self.device_index)

        # Compute frames per chunk from the requested chunk_ms
        self._chunk_frames = int(self._capture_rate * self.chunk_ms / 1000.0)

        logger.info(
            "Opening microphone (device=%s, capture_rate=%d Hz → resample to %d Hz, "
            "chunk=%d frames = %d ms, mode=blocking)",
            self.device_index if self.device_index is not None else "default",
            self._capture_rate,
            self.sample_rate,
            self._chunk_frames,
            self.chunk_ms,
        )

        # Open in blocking mode — NO callback.
        # We call stream.read() ourselves from the caller's thread.
        self._stream = sd.InputStream(
            device=self.device_index,
            samplerate=self._capture_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=0,
            latency="high",
        )
        self._stream.start()
        self._running = True
        logger.info("Microphone stream open at %d Hz (will resample to %d Hz).",
                    self._capture_rate, self.sample_rate)

    def stop(self) -> None:
        """Close the microphone stream cleanly."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Microphone stream closed.")

    # ── public audio access ───────────────────────────────────

    def read_chunk(self, timeout: float = 1.0) -> np.ndarray | None:
        """
        Read one audio chunk directly from the PortAudio stream (blocking).
        Returns int16 numpy array of shape (chunk_frames,) or None if
        the stream isn't running.

        This is called from the main thread — no audio-thread constraints.
        PortAudio's own internal ring buffer feeds us, so there's no
        Python-side callback to bottleneck.
        """
        if not self._running or self._stream is None:
            return None

        try:
            # stream.read() blocks until `_chunk_frames` are available.
            # `overflow` is True if PortAudio's internal buffer overflowed
            # between reads — this is informational only and doesn't affect
            # the returned data.
            data, overflow = self._stream.read(self._chunk_frames)
            if overflow:
                logger.debug("PortAudio reported input overflow (harmless in blocking mode).")
            # data shape: (chunk_frames, channels) — flatten to 1-D mono
            return data[:, 0].copy()
        except sd.PortAudioError as exc:
            logger.error("PortAudio read error: %s", exc)
            return None

    def stream_chunks(self, timeout: float = 1.0):
        """
        Generator — yields raw int16 audio chunks indefinitely.
        Stops when the listener is stopped.

        Usage:
            for chunk in listener.stream_chunks():
                process(chunk)
        """
        while self._running:
            chunk = self.read_chunk(timeout=timeout)
            if chunk is not None:
                yield chunk

    def is_speech(self, chunk: np.ndarray) -> bool:
        """
        Voice Activity Detection — returns True if the chunk
        contains likely speech based on RMS energy.

        Applies a high-pass filter first to ignore low-freq noise.
        The highpass filter works on the native capture rate.
        """
        if self.noise_filter:
            filtered = _highpass_filter(chunk, sr=self._capture_rate)
            # filtered is float32 but still in int16 scale — normalize before RMS
            energy = float(np.sqrt(np.mean((filtered / 32768.0) ** 2)))
        else:
            energy = _rms(chunk)
        return energy >= self.vad_threshold

    def capture_phrase(
        self,
        pre_speech_chunks: int = 4,
        verbose: bool = False,
        on_status=None,
    ) -> np.ndarray | None:
        """
        Block until a complete spoken phrase is captured.

        Returns:
            float32 numpy array at self.sample_rate  (ready for Whisper)
            None if stream is stopped before any speech

        Algorithm:
            1. Wait for speech onset (VAD triggers)
            2. Collect audio until silence for vad_pad_ms
            3. Enforce min/max duration guards
            4. Return float32 audio array
        """
        if not self._running:
            raise RuntimeError("Listener is not running. Use 'with MicrophoneListener()' or call .start() first.")

        # Chunk counts must use _capture_rate (hardware rate) because that's
        # what sets how many frames arrive per read — not the Whisper rate.
        frames_per_chunk = max(1, self._chunk_frames or int(self._capture_rate * self.chunk_ms / 1000.0))

        pad_chunks = max(1, int((self.vad_pad_ms  / 1000.0) * self._capture_rate / frames_per_chunk))
        min_chunks = max(1, int((self.vad_min_ms  / 1000.0) * self._capture_rate / frames_per_chunk))
        max_chunks = max(2, int(self.max_phrase_sec           * self._capture_rate / frames_per_chunk))

        # Rolling pre-speech buffer — keeps last N chunks so we don't
        # clip the very start of speech during VAD onset lag
        pre_buffer: list[np.ndarray] = []

        if verbose:
            print("\033[2m[listener] Waiting for speech...\033[0m", flush=True)

        # ── Phase 1: wait for speech onset ────────────────────
        while self._running:
            chunk = self.read_chunk(timeout=0.5)
            if chunk is None:
                continue

            pre_buffer.append(chunk)
            if len(pre_buffer) > pre_speech_chunks:
                pre_buffer.pop(0)

            if self.is_speech(chunk):
                break
        else:
            return None   # stopped while waiting

        if verbose:
            print("\033[92m[listener] Speech detected — recording...\033[0m", flush=True)

        # ── Phase 2: collect speech until end-of-turn silence ──────────────
        # Uses the Endpointer so short thinking pauses do NOT end the turn —
        # only a sustained end_silence_ms gap does.
        from voice.endpointing import Endpointer, ms_to_chunks
        end_silence_chunks = ms_to_chunks(self.end_silence_ms, self._capture_rate, frames_per_chunk)
        endpointer = Endpointer(
            end_silence_chunks=end_silence_chunks,
            min_speech_chunks=min_chunks,
            max_chunks=max_chunks,
        )

        phrase_chunks: list[np.ndarray] = list(pre_buffer)
        # The chunk that tripped onset detection was speech — seed that.
        endpointer.update(True)

        # Cue thresholds: start hinting "still listening" once a pause exceeds
        # ~400 ms (a real thinking pause) but before the end-of-turn gap.
        pause_hint_chunks = ms_to_chunks(400, self._capture_rate, frames_per_chunk)
        last_cue: str | None = None

        def _cue(kind: str):
            nonlocal last_cue
            if on_status and kind != last_cue:
                last_cue = kind
                try:
                    on_status(kind)
                except Exception:
                    pass

        _cue("speaking")
        while self._running:
            chunk = self.read_chunk(timeout=0.5)
            if chunk is None:
                continue

            phrase_chunks.append(chunk)
            is_sp  = self.is_speech(chunk)
            status = endpointer.update(is_sp)
            if status == Endpointer.DONE:
                break
            if status == Endpointer.TOO_LONG:
                if verbose:
                    print("\033[33m[listener] Max phrase length reached — cutting.\033[0m")
                break

            # UX cue: are we mid-thought (pausing) or actively talking?
            if is_sp:
                _cue("speaking")
            elif endpointer.trailing_silence >= pause_hint_chunks:
                _cue("pausing")   # still capturing — waiting for you to continue

        total_chunks = len(phrase_chunks)

        # ── Phase 3: validate minimum length ──────────────────
        if len(phrase_chunks) < min_chunks:
            logger.debug("Phrase too short (%d chunks < min %d) — discarded.", len(phrase_chunks), min_chunks)
            return None

        # ── Assemble and convert ───────────────────────────────
        audio_int16 = np.concatenate(phrase_chunks)
        audio_float = _int16_to_float32(audio_int16)

        # Resample from hardware capture rate → 16 kHz for Whisper
        if self._capture_rate != self.sample_rate:
            audio_float = _resample(audio_float, self._capture_rate, self.sample_rate)

        duration_sec = len(audio_float) / self.sample_rate
        logger.debug("Captured phrase: %.2f seconds.", duration_sec)

        if verbose:
            print(f"\033[2m[listener] Phrase captured ({duration_sec:.1f}s)\033[0m")

        return audio_float


# ─────────────────────────────────────────────────────────────
#  DEVICE HELPERS
# ─────────────────────────────────────────────────────────────

def list_devices() -> list[dict]:
    """
    Return a list of available audio input devices.

    Each entry: { "index": int, "name": str, "channels": int, "sample_rate": float }
    """
    if not _SD_AVAILABLE:
        raise RuntimeError("sounddevice not available.")

    devices = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append({
                "index":       i,
                "name":        dev["name"],
                "channels":    dev["max_input_channels"],
                "sample_rate": dev["default_samplerate"],
            })
    return devices


def print_devices() -> None:
    """Print a formatted table of available microphone devices."""
    if not _SD_AVAILABLE:
        print("  ✗ sounddevice not installed. Run: pip install sounddevice")
        return

    devices = list_devices()
    if not devices:
        print("  No input devices found.")
        return

    print(f"\n  {'IDX':<5} {'DEVICE NAME':<40} {'CH':<5} {'RATE'}")
    print("  " + "─" * 60)
    for d in devices:
        print(f"  {d['index']:<5} {d['name']:<40} {d['channels']:<5} {int(d['sample_rate'])} Hz")
    print()


def test_microphone(
    duration_sec: float = 3.0,
    device_index: int | None = None,
) -> None:
    """
    Open the microphone for `duration_sec` seconds and print a
    live RMS energy bar — useful for checking mic levels and
    confirming the VAD threshold is calibrated correctly.

    Run from terminal:
        python voice/listener.py
    """
    if not _SD_AVAILABLE:
        print("  ✗ sounddevice not installed.")
        return

    print(f"\n  Testing microphone for {duration_sec:.0f}s — speak or make noise...\n")

    listener = MicrophoneListener(device_index=device_index)
    listener.start()

    native_rate = listener._capture_rate
    print(f"  Device native rate : {native_rate} Hz  →  resampling to {SAMPLE_RATE} Hz for Whisper\n")

    start = time.time()
    bar_width = 40

    try:
        while time.time() - start < duration_sec:
            chunk = listener.read_chunk(timeout=0.5)
            if chunk is None:
                continue

            rms     = _rms(chunk)
            filled  = int(min(rms / 0.1, 1.0) * bar_width)
            bar     = "█" * filled + "░" * (bar_width - filled)
            label   = "SPEECH" if rms >= VAD_ENERGY_THRESHOLD else "silence"
            color   = "\033[92m" if rms >= VAD_ENERGY_THRESHOLD else "\033[2m"
            print(
                f"\r  {color}[{bar}]\033[0m  RMS={rms:.4f}  {color}{label}\033[0m",
                end="",
                flush=True,
            )
    finally:
        listener.stop()
        print("\n\n  Done.\n")


# ─────────────────────────────────────────────────────────────
#  STANDALONE ENTRY — python voice/listener.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("\n  NEXUS — voice/listener.py")
    print("  " + "─" * 40)

    if "--devices" in sys.argv:
        print_devices()

    elif "--test" in sys.argv:
        # Allow: python voice/listener.py --test --device 2
        dev = None
        if "--device" in sys.argv:
            idx = sys.argv.index("--device")
            dev = int(sys.argv[idx + 1])
        test_microphone(duration_sec=5.0, device_index=dev)

    elif "--capture" in sys.argv:
        # Capture one phrase and print its shape
        print("  Capturing one phrase... speak now.\n")
        listener = MicrophoneListener()
        with listener:
            audio = listener.capture_phrase(verbose=True)
            if audio is not None:
                print(f"\n  Captured: shape={audio.shape}, dtype={audio.dtype}")
                print(f"  Duration: {len(audio) / SAMPLE_RATE:.2f}s")
            else:
                print("  No audio captured.")

    else:
        print(
            "\n  Usage:\n"
            "    python voice/listener.py --devices          # list mic devices\n"
            "    python voice/listener.py --test             # live RMS meter\n"
            "    python voice/listener.py --test --device 2  # test specific device\n"
            "    python voice/listener.py --capture          # capture one phrase\n"
        )