"""
NEXUS — voice/speaker_id.py
Speaker Identification — the voice recognition layer of NEXUS.

Sits in the voice pipeline after wake word detection:

    [MICROPHONE]
         ↓
    listener.py
         ↓
    wakeword.py
         ↓
    speaker_id.py      ← YOU ARE HERE
         ↓
    speech_to_text.py
         ↓
    command parser

Responsibilities:
  - Enroll the owner's voice (record samples, build a profile)
  - Verify every captured phrase against the owner's voice profile
  - Reject commands from unknown speakers
  - Return a confidence score with every verification

Uses SpeechBrain's ECAPA-TDNN speaker embedding model.
Embeddings are stored locally — no cloud, fully offline.

Usage:
    # Enroll (first time setup)
    python voice/speaker_id.py --enroll

    # Verify a WAV file
    python voice/speaker_id.py --verify path/to/audio.wav

    # Test live from mic
    python voice/speaker_id.py --test
"""

from __future__ import annotations

import os
import logging
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger("nexus.voice.speaker_id")

# ── lazy imports ──────────────────────────────────────────────
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from speechbrain.inference.speaker import EncoderClassifier
    _SB_AVAILABLE = True
except ImportError:
    _SB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

SAMPLE_RATE       = 16_000
PROFILE_DIR       = Path("data/speaker_profiles")
OWNER_PROFILE     = PROFILE_DIR / "owner.npy"
ENROLL_SAMPLES    = 10         # number of voice samples to record during enrollment
ENROLL_SEC        = 6.0        # seconds per enrollment sample
VERIFY_THRESHOLD  = 0.40       # cosine similarity — above = accepted
MODEL_SOURCE      = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_SAVE_DIR    = "models/speaker_id"


# ─────────────────────────────────────────────────────────────
#  SPEAKER EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────

class SpeakerEmbedder:
    """
    Wraps SpeechBrain's ECAPA-TDNN model to extract
    192-dimensional speaker embeddings from audio.

    The model is downloaded once (~80MB) and cached locally.
    """

    def __init__(self):
        if not _SB_AVAILABLE:
            raise ImportError(
                "speechbrain not installed.\n"
                "Run: pip install speechbrain"
            )
        if not _TORCH_AVAILABLE:
            raise ImportError("torch not installed.")

        logger.info("Loading SpeechBrain ECAPA-TDNN speaker model...")
        self._model = EncoderClassifier.from_hparams(
            source=MODEL_SOURCE,
            savedir=MODEL_SAVE_DIR,
            run_opts={"device": "cpu"},
        )
        logger.info("Speaker embedding model ready.")

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract a 192-dim speaker embedding from a float32 audio array.

        Parameters
        ----------
        audio : float32 ndarray at 16 kHz

        Returns
        -------
        np.ndarray of shape (192,) — L2-normalized embedding
        """
        import torch

        # Ensure float32, correct shape
        audio = audio.astype(np.float32)
        waveform = torch.FloatTensor(audio).unsqueeze(0)  # (1, samples)

        with torch.no_grad():
            embedding = self._model.encode_batch(waveform)  # (1, 1, 192)

        vec = embedding.squeeze().numpy()   # (192,)
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    return float(np.dot(a, b))


# ─────────────────────────────────────────────────────────────
#  SPEAKER PROFILE
# ─────────────────────────────────────────────────────────────

class SpeakerProfile:
    """
    Stores and manages the owner's voice profile.

    The profile is the mean of N enrollment embeddings — averaged
    to make it robust to slight variations in speech.
    """

    def __init__(self, profile_path: Path = OWNER_PROFILE):
        self._path      = profile_path
        self._embedding: Optional[np.ndarray] = None
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            self._embedding = np.load(str(self._path))
            logger.info("Speaker profile loaded from %s", self._path)
        else:
            logger.info("No speaker profile found at %s — run enrollment.", self._path)

    def save(self, embedding: np.ndarray) -> None:
        """Save a new profile embedding to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self._path), embedding)
        self._embedding = embedding
        logger.info("Speaker profile saved to %s", self._path)

    def is_enrolled(self) -> bool:
        return self._embedding is not None

    def get_embedding(self) -> Optional[np.ndarray]:
        return self._embedding

    def clear(self) -> None:
        if self._path.exists():
            self._path.unlink()
        self._embedding = None
        logger.info("Speaker profile cleared.")


# ─────────────────────────────────────────────────────────────
#  SPEAKER IDENTIFIER — main class
# ─────────────────────────────────────────────────────────────

class SpeakerIdentifier:
    """
    Full speaker identification system for NEXUS.

    Usage:
        sid = SpeakerIdentifier()

        # Check if owner is enrolled
        if not sid.is_enrolled():
            sid.enroll()   # records voice samples interactively

        # Verify a captured audio phrase
        result = sid.verify(audio)
        if result.accepted:
            print(f"Owner verified (score={result.score:.2f})")
        else:
            print(f"Unknown speaker rejected (score={result.score:.2f})")
    """

    def __init__(self, threshold: float = VERIFY_THRESHOLD):
        self.threshold = threshold
        self._embedder = SpeakerEmbedder()
        self._profile  = SpeakerProfile()

    # ── enrollment ────────────────────────────────────────────

    def enroll(
        self,
        audio_samples: Optional[list[np.ndarray]] = None,
    ) -> None:
        """
        Build the owner's voice profile.

        Parameters
        ----------
        audio_samples : list of float32 arrays (optional)
            If provided, use these instead of recording from mic.
            Each array should be at least 2 seconds of speech at 16 kHz.
        """
        if audio_samples is None:
            audio_samples = self._record_enrollment_samples()

        if not audio_samples:
            raise ValueError("No audio samples provided for enrollment.")

        logger.info("Computing speaker embeddings for %d samples...", len(audio_samples))

        embeddings = []
        for i, audio in enumerate(audio_samples):
            emb = self._embedder.embed(audio)
            embeddings.append(emb)
            logger.debug("Sample %d/%d embedded.", i + 1, len(audio_samples))

        # Average embeddings and renormalize
        mean_emb = np.mean(embeddings, axis=0)
        norm     = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb /= norm

        self._profile.save(mean_emb)
        logger.info("Enrollment complete. Speaker profile saved.")

    def _record_enrollment_samples(self) -> list[np.ndarray]:
        """Interactive enrollment — record N samples from the microphone."""
        try:
            from voice.listener import MicrophoneListener
        except ModuleNotFoundError:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from voice.listener import MicrophoneListener

        print(f"\n  NEXUS Speaker Enrollment")
        print(f"  {'─' * 40}")
        print(f"  Recording {ENROLL_SAMPLES} voice samples ({ENROLL_SEC:.0f}s each).")
        print(f"  Speak naturally — say anything for each recording.\n")

        samples = []
        listener = MicrophoneListener(max_phrase_sec=ENROLL_SEC + 1)

        with listener:
            for i in range(ENROLL_SAMPLES):
                input(f"  [{i+1}/{ENROLL_SAMPLES}] Press ENTER then speak for {ENROLL_SEC:.0f}s... ")
                print(f"  \033[91m● REC\033[0m  Speak now...", flush=True)

                audio = listener.capture_phrase(verbose=False)
                if audio is None or len(audio) < SAMPLE_RATE:
                    print(f"  \033[93m⚠ Too short — try again.\033[0m")
                    i -= 1
                    continue

                samples.append(audio)
                duration = len(audio) / SAMPLE_RATE
                print(f"  \033[92m✓ Captured {duration:.1f}s\033[0m")

        print(f"\n  {len(samples)} samples recorded.\n")
        return samples

    # ── verification ──────────────────────────────────────────

    def verify(self, audio: np.ndarray) -> "VerificationResult":
        """
        Verify whether the speaker in `audio` matches the enrolled owner.

        Parameters
        ----------
        audio : float32 ndarray at 16 kHz

        Returns
        -------
        VerificationResult with .accepted (bool) and .score (float)
        """
        if not self._profile.is_enrolled():
            logger.warning("No speaker profile enrolled — accepting all speakers.")
            return VerificationResult(accepted=True, score=1.0, reason="no_profile")

        if audio is None or len(audio) < SAMPLE_RATE // 2:
            logger.debug("Audio too short for speaker verification — skipping.")
            return VerificationResult(accepted=True, score=0.0, reason="too_short")

        try:
            query_emb   = self._embedder.embed(audio)
            owner_emb   = self._profile.get_embedding()
            score       = _cosine_similarity(query_emb, owner_emb)
            accepted    = score >= self.threshold

            logger.info(
                "Speaker verification: score=%.3f  threshold=%.2f  → %s",
                score, self.threshold,
                "\033[92mACCEPTED\033[0m" if accepted else "\033[91mREJECTED\033[0m",
            )

            return VerificationResult(
                accepted=accepted,
                score=score,
                reason="verified" if accepted else "unknown_speaker",
            )

        except Exception as exc:
            logger.error("Speaker verification failed: %s", exc)
            return VerificationResult(accepted=True, score=0.0, reason="error")

    # ── utilities ─────────────────────────────────────────────

    def is_enrolled(self) -> bool:
        return self._profile.is_enrolled()

    def clear_profile(self) -> None:
        self._profile.clear()


# ─────────────────────────────────────────────────────────────
#  VERIFICATION RESULT
# ─────────────────────────────────────────────────────────────

class VerificationResult:
    """
    Result of a speaker verification check.

    Attributes
    ----------
    accepted : bool   — True if speaker matches owner profile
    score    : float  — cosine similarity (0.0 – 1.0)
    reason   : str    — "verified" | "unknown_speaker" | "no_profile" |
                        "too_short" | "error"
    """

    def __init__(self, accepted: bool, score: float, reason: str = ""):
        self.accepted = accepted
        self.score    = score
        self.reason   = reason

    def __repr__(self) -> str:
        status = "ACCEPTED" if self.accepted else "REJECTED"
        return f"VerificationResult({status}, score={self.score:.3f}, reason={self.reason!r})"

    def __bool__(self) -> bool:
        return self.accepted


# ─────────────────────────────────────────────────────────────
#  STANDALONE ENTRY
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n  NEXUS — voice/speaker_id.py")
    print("  " + "─" * 40)

    if not _SB_AVAILABLE:
        print("  ✗ speechbrain not installed. Run: pip install speechbrain")
        sys.exit(1)

    sid = SpeakerIdentifier()

    # ── --enroll ──────────────────────────────────────────────
    if "--enroll" in sys.argv:
        if sid.is_enrolled():
            ans = input("  Profile already exists. Re-enroll? [y/N] ").strip().lower()
            if ans != 'y':
                print("  Enrollment cancelled.")
                sys.exit(0)
            sid.clear_profile()
        sid.enroll()
        print("  ✓ Enrollment complete. Profile saved.")
        sys.exit(0)

    # ── --verify <file> ───────────────────────────────────────
    if "--verify" in sys.argv:
        idx = sys.argv.index("--verify")
        wav_path = sys.argv[idx + 1]

        from scipy.io import wavfile
        from voice.listener import _resample, SAMPLE_RATE as SR

        fs, data = wavfile.read(wav_path)
        audio = data.astype(np.float32) / 32768.0
        if fs != SR:
            audio = _resample(audio, fs, SR)

        result = sid.verify(audio)
        print(f"\n  Result : {result}")
        print(f"  Score  : {result.score:.4f}")
        print(f"  Status : {'✓ ACCEPTED' if result.accepted else '✗ REJECTED'}\n")
        sys.exit(0)

    # ── --test (live mic) ─────────────────────────────────────
    if "--test" in sys.argv:
        if not sid.is_enrolled():
            print("  No profile found. Run: python voice/speaker_id.py --enroll")
            sys.exit(1)

        try:
            from voice.listener import MicrophoneListener
        except ModuleNotFoundError:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from voice.listener import MicrophoneListener

        print(f"\n  Live speaker verification test.")
        print(f"  Threshold : {sid.threshold}")
        print(f"  Speak after each prompt. Ctrl-C to stop.\n")

        listener = MicrophoneListener()
        with listener:
            while True:
                try:
                    input("  Press ENTER to capture and verify... ")
                    print("  \033[91m● REC\033[0m  Speak now...")
                    audio = listener.capture_phrase(verbose=False)
                    if audio is None:
                        print("  No audio captured.")
                        continue
                    result = sid.verify(audio)
                    status = "\033[92m✓ ACCEPTED\033[0m" if result.accepted else "\033[91m✗ REJECTED\033[0m"
                    print(f"  {status}  score={result.score:.4f}\n")
                except (KeyboardInterrupt, EOFError):
                    print("\n  Stopped.")
                    break
        sys.exit(0)

    # ── default: show status ──────────────────────────────────
    print(f"\n  Enrolled : {sid.is_enrolled()}")
    if sid.is_enrolled():
        emb = sid._profile.get_embedding()
        print(f"  Profile  : {OWNER_PROFILE}  (dim={emb.shape[0]})")
    print()
    print("  Usage:")
    print("    python voice/speaker_id.py --enroll           # record your voice")
    print("    python voice/speaker_id.py --verify audio.wav # verify a file")
    print("    python voice/speaker_id.py --test             # live mic test")
    print()