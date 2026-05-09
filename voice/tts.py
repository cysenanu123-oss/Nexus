"""
voice/tts.py
NEXUS Text-to-Speech — speaks ActionResult messages back to the user.

Backends (tried in order):
  1. pyttsx3   — fully offline, zero setup, works immediately
  2. Coqui TTS — higher quality, needs: pip install TTS
  3. espeak    — system fallback, always available on Linux

Usage:
    from voice.tts import Speaker
    speaker = Speaker()
    speaker.say("NEXUS online.")
"""

import logging
import subprocess
import threading
import queue
import shutil

log = logging.getLogger("nexus.tts")


# ─────────────────────────────────────────────
# Backend: pyttsx3 (offline, instant setup)
# ─────────────────────────────────────────────

def _try_pyttsx3():
    try:
        import pyttsx3  # type: ignore
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)    # words per minute
        engine.setProperty("volume", 0.9)
        # Try to pick a male voice (more "assistant" feel)
        voices = engine.getProperty("voices")
        for v in voices:
            if "male" in v.name.lower() or "david" in v.name.lower():
                engine.setProperty("voice", v.id)
                break
        log.info("TTS backend: pyttsx3")
        return engine
    except Exception as e:
        log.debug(f"pyttsx3 unavailable: {e}")
        return None


# ─────────────────────────────────────────────
# Backend: espeak (system binary, always works)
# ─────────────────────────────────────────────

from scipy.io import wavfile
import sounddevice as sd

def _espeak_say(text: str):
    if not shutil.which("espeak") and not shutil.which("espeak-ng"):
        raise RuntimeError("espeak not found")
    
    binary = "espeak-ng" if shutil.which("espeak-ng") else "espeak"
    wav_path = "/tmp/nexus_tts.wav"

    # Generate the TTS into a WAV file instead of trying to play it via aplay
    subprocess.run(
        [binary, "-w", wav_path, "-s", "155", "-a", "180", "-v", "en+m3", text],
        check=True,
        capture_output=True,
    )

    # Read and play through sounddevice, matching the ALSA session
    try:
        fs, data = wavfile.read(wav_path)
        
        # Resample to 44100 Hz since the hardware ALSA device doesn't support 22050 Hz natively
        target_fs = 44100
        if fs != target_fs:
            from scipy.signal import resample
            num_samples = int(len(data) * float(target_fs) / fs)
            data = resample(data, num_samples)
            fs = target_fs

        sd.play(data, fs)
        sd.wait()
    except Exception as e:
        log.error(f"Failed to play TTS audio: {e}")


# ─────────────────────────────────────────────
# Speaker class
# ─────────────────────────────────────────────

class Speaker:
    """
    Speaks text using the best available backend.
    Runs speech in a background thread so it never blocks the main loop.

    Usage:
        speaker = Speaker()
        speaker.say("Opening Firefox.")
        speaker.say("What time is it?", block=True)  # wait until done
        speaker.shutdown()
    """

    def __init__(self, backend: str = "auto"):
        self._engine    = None
        self._backend   = None
        self._q: queue.Queue = queue.Queue()
        self._lock      = threading.Lock()
        self._running   = True

        # Prefer espeak over pyttsx3 on Linux due to ALSA/aplay bugs
        if backend in ("auto", "espeak"):
            if shutil.which("espeak-ng") or shutil.which("espeak"):
                self._backend = "espeak"
                log.info("TTS backend: espeak")

        if self._backend is None and backend in ("auto", "pyttsx3"):
            self._engine = _try_pyttsx3()
            if self._engine:
                self._backend = "pyttsx3"

        if self._backend is None:
            log.warning("No TTS backend found — speech disabled. "
                        "Run: pip install pyttsx3  or  sudo apt install espeak-ng")

        # Start background speech worker
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ── Internal worker ─────────────────────────────────────────────────

    def _run(self):
        while self._running:
            try:
                item = self._q.get(timeout=0.5)
                if item is None:
                    break
                text, done_event = item
                self._speak_now(text)
                if done_event:
                    done_event.set()
                self._q.task_done()
            except queue.Empty:
                continue

    def _speak_now(self, text: str):
        if not text or not self._backend:
            return

        # Strip symbols that sound bad when spoken
        clean = (text
                 .replace("✓", "")
                 .replace("✗", "")
                 .replace("→", "")
                 .replace("•", "")
                 .replace("─", "")
                 .strip())

        try:
            if self._backend == "pyttsx3":
                with self._lock:
                    self._engine.say(clean)
                    self._engine.runAndWait()

            elif self._backend == "espeak":
                _espeak_say(clean)

        except Exception as e:
            log.error(f"TTS error: {e}")

    # ── Public API ──────────────────────────────────────────────────────

    def say(self, text: str, block: bool = False):
        """
        Queue text for speech.
        block=True waits until speech finishes before returning.
        """
        if not self._backend:
            return

        done_event = threading.Event() if block else None
        self._q.put((text, done_event))

        if block and done_event:
            done_event.wait()

    def say_result(self, result, block: bool = False):
        """Convenience: accepts an ActionResult or any object with .message"""
        msg = getattr(result, "message", str(result))
        self.say(msg, block=block)

    def is_busy(self) -> bool:
        return not self._q.empty()

    def shutdown(self):
        self._running = False
        self._q.put(None)
        self._worker.join(timeout=2)
        log.info("TTS speaker shut down.")


# ─────────────────────────────────────────────
# CLI test  —  python voice/tts.py
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time

    logging.basicConfig(level=logging.INFO)
    speaker = Speaker()

    phrases = [
        "NEXUS online. All systems ready.",
        "Opening Firefox.",
        "The time is 3:45 PM.",
        "I didn't understand that command. Try rephrasing.",
    ]

    if len(sys.argv) > 1:
        phrases = [" ".join(sys.argv[1:])]

    for phrase in phrases:
        print(f"Speaking: {phrase!r}")
        speaker.say(phrase, block=True)
        time.sleep(0.4)

    speaker.shutdown()
    print("Done.")