"""
tools/record_wakeword.py
NEXUS Wake Word Recording Tool

Records training samples of "Hey Nexus" for custom wake word training.
Target: 100 positive samples + 100 negative samples

Usage:
    python tools/record_wakeword.py --positive   # record "hey nexus" samples
    python tools/record_wakeword.py --negative   # record background / other speech
    python tools/record_wakeword.py --test        # play back recordings
    python tools/record_wakeword.py --count       # show how many samples collected
"""

from __future__ import annotations

import os
import sys
import time
import wave
import struct
import argparse
import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

SAMPLE_RATE     = 16_000       # Hz — openWakeWord expects 16 kHz
CAPTURE_RATE    = 44_100       # Hz — native hardware rate
CHANNELS        = 1
DTYPE           = "int16"
RECORD_SEC      = 2.0          # seconds per sample
TARGET_POSITIVE = 100          # target number of "hey nexus" samples
TARGET_NEGATIVE = 100          # target number of negative samples

DATA_DIR        = Path("data/wakeword")
POSITIVE_DIR    = DATA_DIR / "hey_nexus"
NEGATIVE_DIR    = DATA_DIR / "negative"

# ─────────────────────────────────────────────────────────────
#  AUDIO HELPERS
# ─────────────────────────────────────────────────────────────

def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate:
        return audio
    g    = gcd(to_rate, from_rate)
    up   = to_rate   // g
    down = from_rate // g
    from scipy.signal import resample_poly
    return resample_poly(audio, up, down).astype(np.int16)


def record_sample(duration: float = RECORD_SEC) -> np.ndarray:
    """Record a single audio sample at native rate, resample to 16 kHz."""
    frames = int(CAPTURE_RATE * duration)
    audio  = sd.rec(frames, samplerate=CAPTURE_RATE, channels=CHANNELS, dtype=DTYPE)
    sd.wait()
    mono = audio[:, 0]
    return _resample(mono, CAPTURE_RATE, SAMPLE_RATE)


def save_wav(audio: np.ndarray, path: Path) -> None:
    """Save int16 audio array as a 16 kHz WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), SAMPLE_RATE, audio)


def play_wav(path: Path) -> None:
    """Play back a WAV file through sounddevice."""
    fs, data = wavfile.read(str(path))
    # Resample to hardware rate for playback
    if fs != CAPTURE_RATE:
        data = _resample(data, fs, CAPTURE_RATE)
    sd.play(data, CAPTURE_RATE)
    sd.wait()


def count_samples(directory: Path) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob("*.wav")))


def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


# ─────────────────────────────────────────────────────────────
#  RMS METER
# ─────────────────────────────────────────────────────────────

def rms_bar(audio: np.ndarray, width: int = 30) -> str:
    rms    = float(np.sqrt(np.mean((audio.astype(np.float32) / 32768.0) ** 2)))
    filled = int(min(rms / 0.1, 1.0) * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}]  RMS={rms:.4f}"


# ─────────────────────────────────────────────────────────────
#  RECORDING SESSIONS
# ─────────────────────────────────────────────────────────────

def record_positive():
    """
    Interactive session to record 'hey nexus' samples.
    Press ENTER to record, 'q' to quit.
    """
    POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
    existing = count_samples(POSITIVE_DIR)

    print(f"\n  \033[92m=== POSITIVE SAMPLES — 'Hey Nexus' ===\033[0m")
    print(f"  Target  : {TARGET_POSITIVE} samples")
    print(f"  Existing: {existing} samples")
    print(f"  Needed  : {max(0, TARGET_POSITIVE - existing)} more")
    print(f"\n  Instructions:")
    print(f"  • Press ENTER, then immediately say '\033[1mHey Nexus\033[0m'")
    print(f"  • Each recording is {RECORD_SEC:.0f} seconds")
    print(f"  • Vary your tone, speed, and distance from mic")
    print(f"  • Say it naturally — not too loud, not too quiet")
    print(f"  • Type 'q' + ENTER to stop\n")

    count = existing
    while count < TARGET_POSITIVE:
        remaining = TARGET_POSITIVE - count
        try:
            key = input(f"  [{count:>3}/{TARGET_POSITIVE}]  Press ENTER to record ({remaining} remaining) or 'q' to quit: ")
        except (EOFError, KeyboardInterrupt):
            break

        if key.strip().lower() == 'q':
            break

        print(f"  \033[93m  Recording in 0.3s... say 'Hey Nexus'\033[0m", flush=True)
        time.sleep(0.3)
        print(f"  \033[91m● REC\033[0m", flush=True)

        try:
            audio = record_sample()
        except Exception as e:
            print(f"  \033[91m✗ Recording failed: {e}\033[0m")
            continue

        # Quick quality check
        rms = float(np.sqrt(np.mean((audio.astype(np.float32) / 32768.0) ** 2)))
        if rms < 0.005:
            print(f"  \033[93m⚠ Too quiet (RMS={rms:.4f}) — discarded. Speak louder.\033[0m")
            continue
        if rms > 0.8:
            print(f"  \033[93m⚠ Too loud (RMS={rms:.4f}) — discarded. Move back from mic.\033[0m")
            continue

        # Save
        fname = POSITIVE_DIR / f"hey_nexus_{timestamp()}.wav"
        save_wav(audio, fname)
        count += 1

        print(f"  \033[92m✓ Saved  {rms_bar(audio)}\033[0m")

    print(f"\n  Session complete. Positive samples: {count_samples(POSITIVE_DIR)}/{TARGET_POSITIVE}")
    if count_samples(POSITIVE_DIR) >= TARGET_POSITIVE:
        print(f"  \033[92m✓ Positive target reached!\033[0m")
    print()


def record_negative():
    """
    Interactive session to record negative samples
    (background noise, other speech, NOT 'hey nexus').
    """
    NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)
    existing = count_samples(NEGATIVE_DIR)

    print(f"\n  \033[93m=== NEGATIVE SAMPLES — background / other speech ===\033[0m")
    print(f"  Target  : {TARGET_NEGATIVE} samples")
    print(f"  Existing: {existing} samples")
    print(f"\n  Instructions:")
    print(f"  • Record anything EXCEPT 'hey nexus'")
    print(f"  • Include: silence, background noise, other sentences,")
    print(f"    music, keyboard sounds, other wake words")
    print(f"  • The more variety the better")
    print(f"  • Type 'q' + ENTER to stop\n")

    count = existing
    while count < TARGET_NEGATIVE:
        remaining = TARGET_NEGATIVE - count
        try:
            key = input(f"  [{count:>3}/{TARGET_NEGATIVE}]  Press ENTER to record ({remaining} remaining) or 'q' to quit: ")
        except (EOFError, KeyboardInterrupt):
            break

        if key.strip().lower() == 'q':
            break

        print(f"  \033[93m  Recording...\033[0m", flush=True)
        time.sleep(0.1)
        print(f"  \033[91m● REC\033[0m", flush=True)

        try:
            audio = record_sample()
        except Exception as e:
            print(f"  \033[91m✗ Recording failed: {e}\033[0m")
            continue

        fname = NEGATIVE_DIR / f"negative_{timestamp()}.wav"
        save_wav(audio, fname)
        count += 1

        rms = float(np.sqrt(np.mean((audio.astype(np.float32) / 32768.0) ** 2)))
        print(f"  \033[92m✓ Saved  {rms_bar(audio)}\033[0m")

    print(f"\n  Session complete. Negative samples: {count_samples(NEGATIVE_DIR)}/{TARGET_NEGATIVE}")
    if count_samples(NEGATIVE_DIR) >= TARGET_NEGATIVE:
        print(f"  \033[92m✓ Negative target reached!\033[0m")
    print()


def show_count():
    pos = count_samples(POSITIVE_DIR)
    neg = count_samples(NEGATIVE_DIR)

    print(f"\n  NEXUS Wake Word Dataset Status")
    print(f"  {'─'*40}")
    print(f"  Positive ('hey nexus') : {pos:>4} / {TARGET_POSITIVE}  {'✓' if pos >= TARGET_POSITIVE else f'{TARGET_POSITIVE - pos} more needed'}")
    print(f"  Negative (other)       : {neg:>4} / {TARGET_NEGATIVE}  {'✓' if neg >= TARGET_NEGATIVE else f'{TARGET_NEGATIVE - neg} more needed'}")
    print(f"  {'─'*40}")

    if pos >= TARGET_POSITIVE and neg >= TARGET_NEGATIVE:
        print(f"  \033[92m✓ Dataset complete — ready to train!\033[0m")
        print(f"\n  Next step: upload data/wakeword/ to Google Colab")
        print(f"  and run the openWakeWord training notebook.")
    else:
        print(f"  Dataset incomplete — keep recording.")
    print()


def test_playback():
    pos_files = sorted(POSITIVE_DIR.glob("*.wav")) if POSITIVE_DIR.exists() else []
    neg_files = sorted(NEGATIVE_DIR.glob("*.wav")) if NEGATIVE_DIR.exists() else []

    if not pos_files and not neg_files:
        print("\n  No recordings found yet.\n")
        return

    print(f"\n  Playing back last 3 positive samples...\n")
    for f in pos_files[-3:]:
        print(f"  ▶  {f.name}")
        try:
            play_wav(f)
            time.sleep(0.3)
        except Exception as e:
            print(f"     Error: {e}")
    print()


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NEXUS Wake Word Recording Tool"
    )
    parser.add_argument("--positive", action="store_true", help="Record 'hey nexus' samples")
    parser.add_argument("--negative", action="store_true", help="Record negative/background samples")
    parser.add_argument("--count",    action="store_true", help="Show sample counts")
    parser.add_argument("--test",     action="store_true", help="Play back last recordings")
    args = parser.parse_args()

    print("\n  \033[92mNEXUS — Wake Word Recording Tool\033[0m")
    print(f"  Save directory: {DATA_DIR.resolve()}\n")

    if args.positive:
        record_positive()
    elif args.negative:
        record_negative()
    elif args.count:
        show_count()
    elif args.test:
        test_playback()
    else:
        show_count()
        print("  Usage:")
        print("    python tools/record_wakeword.py --positive   # record 'hey nexus'")
        print("    python tools/record_wakeword.py --negative   # record background noise")
        print("    python tools/record_wakeword.py --count      # check progress")
        print("    python tools/record_wakeword.py --test       # play back samples\n")


if __name__ == "__main__":
    main()