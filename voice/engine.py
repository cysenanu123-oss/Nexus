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
    ):
        self.command_callback = command_callback
        self.wake_phrase = wake_phrase
        self.whisper_model = whisper_model

        self._running = False
        self._continuous_mode = False
        self._thread: Optional[threading.Thread] = None

        # Core systems
        self.listener = MicrophoneListener()

        self.detector = WakeWordDetector(
            wake_phrase=self.wake_phrase,
            model_path="models/wakeword/hey_nexus.onnx",
            listener=self.listener
        )

        self.transcriber = Transcriber(
            model_size=self.whisper_model
        )

    # ─────────────────────────────────────────────────────────
    #  Lifecycle
    # ─────────────────────────────────────────────────────────

    def start(self):
        """Start the voice engine."""
        if self._running:
            return

        logger.info("Starting NEXUS Voice Engine...")

        self.listener.start()
        self.detector.start()

        self._running = True

        self._thread = threading.Thread(
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

        self.detector.stop()
        self.listener.stop()

        if self._thread:
            self._thread.join(timeout=3)

        print("\n[NEXUS] Voice Engine OFFLINE")

    def trigger(self, continuous: bool = True):
        """Manually trigger the voice engine to start listening immediately.
        If continuous=True, it will keep listening until told to stop."""
        if not self._running:
            logger.error("Voice Engine is not running.")
            return
        self._continuous_mode = continuous
        self.detector.trigger()

    # ─────────────────────────────────────────────────────────
    #  Main Runtime Loop
    # ─────────────────────────────────────────────────────────

    def _main_loop(self):
        """Main voice processing loop."""

        while self._running:

            # 1. WAIT FOR WAKE WORD (if not in continuous mode)
            if not self._continuous_mode:
                result = self.detector.wait_for_wake_word()

                if not result:
                    continue

                print("\n[NEXUS] Wake word detected")
            
            print("[NEXUS] Listening...\n")

            # Small pause to avoid clipping
            time.sleep(0.2)

            # PAUSE detector so it stops stealing audio chunks
            self.detector.pause()

            try:
                # 2. RECORD USER SPEECH
                audio = self.listener.capture_phrase(
                    verbose=False,
                )
            finally:
                # RESUME detector
                self.detector.resume()

            if audio is None or len(audio) == 0:
                print("[NEXUS] No speech captured.")
                if self._continuous_mode:
                    print("[NEXUS] Exiting continuous voice mode due to silence.\n")
                    self._continuous_mode = False
                continue

            # 3. TRANSCRIBE
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

            # Handle exit words to drop out of continuous mode
            lower_text = text.lower().strip(".?!,;")
            if lower_text in ["stop", "exit", "quit", "cancel", "nevermind", "stop listening"]:
                print("[NEXUS] Exiting voice mode.\n")
                self._continuous_mode = False
                # If they just said stop/cancel to drop out of voice mode, skip dispatch
                if lower_text not in ["exit", "quit"]:
                    continue

            # 4. EXECUTE COMMAND
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

    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Simple test parser
    def fake_command_parser(text: str):

        print(f"[COMMAND PARSER] Received: {text}")

        if text == "status":
            print("[NEXUS] All systems operational.")

        elif text == "hello":
            print("[NEXUS] Hello, Senanu.")

        else:
            print(f"[NEXUS] Unknown command: {text}")

    engine = VoiceEngine(
        command_callback=fake_command_parser
    )

    try:
        engine.start()

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")
        engine.stop()
