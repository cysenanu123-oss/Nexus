"""
NEXUS — voice/engine.py

Central voice orchestration engine for NEXUS.

Pipeline:

    WakeWordDetector
            ↓
    capture_phrase()
            ↓
    Speech-to-Text
            ↓
    Intent / Command Execution
            ↓
    Response

This module connects all foundational voice systems together
into a live conversational runtime loop.
"""

from __future__ import annotations

import time
import logging
import threading
from typing import Callable, Optional

import numpy as np

# Voice systems
from voice.wakeword import WakeWordDetector
from voice.listener import MicrophoneListener
from voice.speech_to_text import Transcriber

logger = logging.getLogger("nexus.voice.engine")

# Auto-end conversation session after this many seconds of voice silence
_VOICE_SESSION_TIMEOUT = 180


# ─────────────────────────────────────────────────────────────
#  VOICE ENGINE
# ─────────────────────────────────────────────────────────────

class VoiceEngine:
    """
    Main live voice runtime for NEXUS.

    Responsibilities:
        - Wait for wake word
        - Record speech
        - Transcribe speech
        - Execute commands
        - Return responses

    Example:
        engine = VoiceEngine(command_callback=my_parser)
        engine.start()
    """

    def __init__(
        self,
        command_callback: Callable[[str], None],
        wake_phrase: str = "hey nexus",
        whisper_model: str = "tiny",
        session_manager=None,
    ):
        self.command_callback = command_callback
        self.wake_phrase      = wake_phrase
        self.whisper_model    = whisper_model
        self._session_mgr     = session_manager  # ConversationSessionManager, injected from Brain

        self._running         = False
        self._continuous_mode = False
        self._thread: Optional[threading.Thread] = None
        self._last_voice_activity = time.time()

        # Core systems
        self.listener    = MicrophoneListener()
        try:
            self.detector = WakeWordDetector(threshold=0.85)
        except Exception:
            logger.warning("Wake word model unavailable — voice engine disabled. Retrain with tools/train_wakeword.py")
            self.detector = None
        self.transcriber = Transcriber(model_size=self.whisper_model)

        # Speaker ID — loads profile if enrolled
        try:
            from voice.speaker_id import SpeakerIdentifier
            self.speaker_id = SpeakerIdentifier()
            if self.speaker_id.is_enrolled():
                logger.info("Speaker ID active — owner profile loaded.")
            else:
                logger.info("Speaker ID loaded — no profile enrolled yet.")
        except Exception as e:
            logger.warning("Speaker ID unavailable: %s", e)
            self.speaker_id = None

    # ─────────────────────────────────────────────────────────
    #  Lifecycle
    # ─────────────────────────────────────────────────────────

    def start(self):
        """Start the voice engine."""
        if self._running:
            return

        logger.info("Starting NEXUS Voice Engine...")

        self.listener.start()
        # WakeWordDetector opens its own mic stream internally

        self._running = True
        self._thread  = threading.Thread(
            target=self._main_loop,
            daemon=True,
            name="nexus-voice-engine",
        )
        self._thread.start()

        print("\n[NEXUS] Voice Engine ONLINE")
        print("[NEXUS] Waiting for wake word...\n")

    def stop(self):
        """Stop all voice systems."""
        if not self._running:
            return

        logger.info("Stopping NEXUS Voice Engine...")
        self._running = False
        self.listener.stop()

        if self._thread:
            self._thread.join(timeout=3)

        print("\n[NEXUS] Voice Engine OFFLINE")

    def trigger(self, continuous: bool = True):
        """
        Manually trigger voice listening — skips wake word detection.
        If continuous=True, NEXUS keeps listening until told to stop.
        """
        if not self._running:
            logger.error("Voice Engine is not running.")
            return
        self._continuous_mode = continuous

    # ─────────────────────────────────────────────────────────
    #  Main Runtime Loop
    # ─────────────────────────────────────────────────────────

    def _main_loop(self):
        """Main voice processing loop."""

        while self._running:

            # ── 1. WAIT FOR WAKE WORD ─────────────────────────
            if not self._continuous_mode:
                if self.detector is None:
                    logger.warning("No wake word model — running in continuous mode.")
                    self._continuous_mode = True
                else:
                    self.detector.wait_for_wake_word(listener=self.listener)
                    print("\n[NEXUS] Wake word detected")

            print("[NEXUS] Listening...\n")

            # Small pause to avoid clipping the first syllable
            time.sleep(0.2)

            # ── 2. RECORD USER SPEECH ─────────────────────────
            audio = self.listener.capture_phrase(verbose=False)

            if audio is None or len(audio) == 0:
                print("[NEXUS] No speech captured.")
                if self._continuous_mode:
                    print("[NEXUS] Exiting continuous voice mode due to silence.\n")
                    self._continuous_mode = False
                continue

            # ── 2b. SPEAKER VERIFICATION ──────────────────────
            if self.speaker_id and self.speaker_id.is_enrolled():
                result = self.speaker_id.verify(audio)
                if not result.accepted:
                    print(f"[NEXUS] Unknown speaker rejected (score={result.score:.2f})")
                    if not self._continuous_mode:
                        print("\n[NEXUS] Waiting for wake word...\n")
                    continue

            # ── 3. TRANSCRIBE ─────────────────────────────────
            print("[NEXUS] Transcribing...")
            transcript = self.transcriber.transcribe(audio)

            if not transcript.is_speech:
                print("[NEXUS] No speech detected.")
                if self._continuous_mode:
                    print("[NEXUS] Exiting continuous voice mode.\n")
                    self._continuous_mode = False
                continue

            text = transcript.text.strip()
            print(f"\n[USER] {text}\n")
            self._last_voice_activity = time.time()

            # Auto-end stale session if silence gap was long
            if self._session_mgr:
                self._session_mgr.auto_end_if_stale()

            # Handle exit words
            lower_text = text.lower().strip(".?!,;")
            if lower_text in ["stop", "exit", "quit", "cancel",
                              "nevermind", "stop listening"]:
                print("[NEXUS] Exiting voice mode.\n")
                if self._session_mgr:
                    self._session_mgr.end_session()
                self._continuous_mode = False
                if lower_text not in ["exit", "quit"]:
                    continue

            # ── 4. EXECUTE COMMAND ────────────────────────────
            try:
                alive = self.command_callback(text)
                if alive is False:
                    self._continuous_mode = False
                    self._running = False
                    break
            except Exception as exc:
                logger.exception("Command execution failed.")
                print(f"[NEXUS] Execution error: {exc}")

            if not self._continuous_mode:
                print("\n[NEXUS] Waiting for wake word...\n")


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST MODE
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    def fake_command_parser(text: str):
        print(f"[COMMAND PARSER] Received: {text!r}")
        if text == "status":
            print("[NEXUS] All systems operational.")
        elif "hello" in text:
            print("[NEXUS] Hello, Senanu.")
        else:
            print(f"[NEXUS] Unknown command: {text}")

    engine = VoiceEngine(command_callback=fake_command_parser)

    try:
        engine.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        engine.stop()