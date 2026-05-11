"""
voice/wakeword.py
NEXUS Wake Word Detector — uses your custom trained hey_nexus.onnx
"""

import numpy as np
import sounddevice as sd
import onnxruntime as rt
import logging
import time
import threading
from math import gcd
from scipy.signal import resample_poly
from typing import Callable, Optional

log = logging.getLogger("nexus.voice.wakeword")

MODEL_PATH    = "models/wakeword/hey_nexus.onnx"
SAMPLE_RATE   = 16000    # model input rate
CAPTURE_RATE  = 44100    # mic native rate
N_MELS        = 32
HOP_LENGTH    = 160
WIN_LENGTH    = 400
N_FFT         = 512
TARGET_FRAMES = 101
THRESHOLD     = 0.85   # confidence required to trigger — raise if false positives
COOLDOWN      = 2.0    # seconds before re-triggering


_GCD        = gcd(SAMPLE_RATE, CAPTURE_RATE)
_UP         = SAMPLE_RATE   // _GCD
_DOWN       = CAPTURE_RATE  // _GCD


def _resample(chunk: np.ndarray) -> np.ndarray:
    """Resample from CAPTURE_RATE to SAMPLE_RATE."""
    return resample_poly(chunk, _UP, _DOWN).astype(np.float32)


class WakeWordDetector:
    """
    Listens on the microphone and fires when it hears "Hey Nexus".
    Uses the custom ONNX model trained on your voice.

    Usage:
        detector = WakeWordDetector()
        detector.wait_for_wake_word()   # blocking

        # or callback style:
        detector.start(callback=lambda: print("Wake word!"))
        detector.stop()
    """

    def __init__(self, threshold: float = THRESHOLD):
        self.threshold     = threshold
        self._running      = False
        self._last_trigger = 0.0
        self._thread: Optional[threading.Thread] = None

        # Load ONNX model
        try:
            self._session = rt.InferenceSession(
                MODEL_PATH,
                providers=["CPUExecutionProvider"],
            )
            self._input_name  = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            log.info(f"Wake word model loaded: {MODEL_PATH}")
            log.info(f"Input:  {self._input_name}")
            log.info(f"Output: {self._output_name}")
        except Exception as e:
            log.error(f"Failed to load ONNX model: {e}")
            raise

        # Build mel filterbank (same params used during training)
        # pyrefly: ignore [missing-import]
        import torchaudio.transforms as T
        # pyrefly: ignore [missing-import]
        import torch
        self._mel = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        )
        self._to_db = T.AmplitudeToDB(top_db=80)

    # ── Feature extraction ───────────────────────────────────────────────

    def _audio_to_features(self, audio: np.ndarray) -> np.ndarray:
        import torch
        waveform = torch.FloatTensor(audio).unsqueeze(0)
        mel      = self._mel(waveform)
        mel_db   = self._to_db(mel)

        frames = mel_db.shape[2]
        if frames < TARGET_FRAMES:
            mel_db = torch.nn.functional.pad(mel_db, (0, TARGET_FRAMES - frames))
        else:
            mel_db = mel_db[:, :, :TARGET_FRAMES]

        features = mel_db.numpy()                    # (1, 32, 101)
        features = (features - features.mean()) / (features.std() + 1e-8)
        return features[np.newaxis, :, :, :]         # (1, 1, 32, 101)

    # ── Inference ────────────────────────────────────────────────────────

    def _predict(self, audio: np.ndarray) -> float:
        features = self._audio_to_features(audio)
        output   = self._session.run(
            [self._output_name],
            {self._input_name: features}
        )
        return float(output[0][0])

    # ── Blocking API ─────────────────────────────────────────────────────

    def wait_for_wake_word(self, listener=None) -> None:
        """Block until 'Hey Nexus' is detected. Uses shared listener if provided."""
        log.info("Listening for wake word...")
        window = np.zeros(SAMPLE_RATE, dtype=np.float32)

        while True:
            # Check if we should stop
            if not self._running:
                break
                
            # Use shared listener (no new mic stream opened)
            if listener is not None:
                # If the listener was explicitly stopped, break to avoid infinite loop
                if not getattr(listener, "_running", True):
                    break
                chunk_int16 = listener.read_chunk(timeout=1.0)
                if chunk_int16 is None:
                    # Give up some CPU before continuing
                    time.sleep(0.01)
                    continue
                chunk = chunk_int16.astype(np.float32) / 32768.0
                if listener._capture_rate != SAMPLE_RATE:
                    chunk = _resample(chunk)
            else:
                # Fallback: open own stream only if no listener provided
                capture_hop = int(HOP_LENGTH * CAPTURE_RATE / SAMPLE_RATE)
                with sd.InputStream(samplerate=CAPTURE_RATE, channels=1,
                                    dtype="float32", blocksize=capture_hop) as stream:
                    chunk, _ = stream.read(capture_hop)
                    chunk    = _resample(chunk[:, 0])

            window = np.roll(window, -len(chunk))
            window[-len(chunk):] = chunk

            score = self._predict(window)

            if score >= self.threshold:
                now = time.time()
                if now - self._last_trigger >= COOLDOWN:
                    self._last_trigger = now
                    log.info(f"Wake word detected! (score={score:.3f})")
                    return

    # ── Callback API ─────────────────────────────────────────────────────

    def start(self, callback: Callable) -> None:
        """Start background listening. Calls callback() on each detection."""
        self._running = True
        self._thread  = threading.Thread(
            target=self._listen_loop,
            args=(callback,),
            daemon=True,
        )
        self._thread.start()
        log.info("Wake word detector started (background).")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        log.info("Wake word detector stopped.")

    def _listen_loop(self, callback: Callable, listener=None) -> None:
        window = np.zeros(SAMPLE_RATE, dtype=np.float32)

        while self._running:
            if listener is not None:
                chunk_int16 = listener.read_chunk(timeout=1.0)
                if chunk_int16 is None:
                    continue
                chunk = chunk_int16.astype(np.float32) / 32768.0
                if listener._capture_rate != SAMPLE_RATE:
                    chunk = _resample(chunk)
            else:
                capture_hop = int(HOP_LENGTH * CAPTURE_RATE / SAMPLE_RATE)
                with sd.InputStream(samplerate=CAPTURE_RATE, channels=1,
                                    dtype="float32", blocksize=capture_hop) as stream:
                    chunk, _ = stream.read(capture_hop)
                    chunk    = _resample(chunk[:, 0])

            window = np.roll(window, -len(chunk))
            window[-len(chunk):] = chunk

            score = self._predict(window)

            if score >= self.threshold:
                now = time.time()
                if now - self._last_trigger >= COOLDOWN:
                    self._last_trigger = now
                    log.info(f"Wake word! score={score:.3f}")
                    callback()

    # ── Context manager ──────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()


# ── Quick test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("\n─── NEXUS Wake Word Test ───")
    print(f"Model     : {MODEL_PATH}")
    print(f"Threshold : {THRESHOLD}")
    print("\nSay 'Hey Nexus' ... (Ctrl+C to stop)\n")

    detector = WakeWordDetector()

    try:
        count = 0
        while True:
            detector.wait_for_wake_word()
            count += 1
            print(f"  [{count}] Wake word detected!")
    except KeyboardInterrupt:
        print("\nStopped.")
