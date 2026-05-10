"""
tools/train_wakeword.py
NEXUS — Wake Word Model Trainer

Trains a small CNN on the recorded WAV samples and exports a
fully self-contained hey_nexus.onnx (no external .data file).

Usage:
    python tools/train_wakeword.py

Output:
    models/wakeword/hey_nexus.onnx   ← drop-in for voice/wakeword.py
"""

from __future__ import annotations

import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd
import torchaudio.transforms as T

# ─────────────────────────────────────────────────────────────
#  PATHS & CONFIG
# ─────────────────────────────────────────────────────────────

ROOT         = Path(__file__).resolve().parent.parent
POSITIVE_DIR = ROOT / "data" / "wakeword" / "hey_nexus"
NEGATIVE_DIR = ROOT / "data" / "wakeword" / "negative"
OUTPUT_PATH  = ROOT / "models" / "wakeword" / "hey_nexus.onnx"

SAMPLE_RATE   = 16_000
N_MELS        = 32
HOP_LENGTH    = 160
WIN_LENGTH    = 400
N_FFT         = 512
TARGET_FRAMES = 101     # ~1 second of audio at these settings
CLIP_SAMPLES  = SAMPLE_RATE * 2   # 2 seconds max per clip

EPOCHS        = 40
BATCH_SIZE    = 16
LR            = 1e-3
VAL_SPLIT     = 0.15    # 15% held out for validation
SEED          = 42

# ─────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────

_mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    power=2.0,
)
_to_db = T.AmplitudeToDB(top_db=80)


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV file using scipy, returns (mono float32 array, sample_rate)."""
    sr, data = wavfile.read(str(path))
    # Convert to float32 in [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    # Mono
    if data.ndim > 1:
        data = data.mean(axis=1)
    # Resample if needed
    if sr != SAMPLE_RATE:
        g    = gcd(SAMPLE_RATE, sr)
        data = resample_poly(data, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return data, SAMPLE_RATE


def wav_to_features(path: Path) -> torch.Tensor:
    """
    Load a WAV → resample to 16 kHz → mel-spectrogram → dB → normalise.
    Returns shape (1, 32, 101)  — channel-first for the CNN.
    """
    audio, sr = _load_wav(path)
    waveform  = torch.from_numpy(audio).unsqueeze(0)   # (1, samples)

    # Pad / trim to CLIP_SAMPLES
    n = waveform.shape[1]
    if n < CLIP_SAMPLES:
        waveform = F.pad(waveform, (0, CLIP_SAMPLES - n))
    else:
        waveform = waveform[:, :CLIP_SAMPLES]

    mel    = _mel_transform(waveform)          # (1, 32, frames)
    mel_db = _to_db(mel)

    # Trim / pad frames axis to TARGET_FRAMES
    f = mel_db.shape[2]
    if f < TARGET_FRAMES:
        mel_db = F.pad(mel_db, (0, TARGET_FRAMES - f))
    else:
        mel_db = mel_db[:, :, :TARGET_FRAMES]

    # Per-sample normalisation
    mean = mel_db.mean()
    std  = mel_db.std() + 1e-8
    mel_db = (mel_db - mean) / std

    return mel_db   # (1, 32, 101)


# ─────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────

class WakeWordDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        features = wav_to_features(path)
        return features, torch.tensor(label, dtype=torch.float32)


def load_dataset() -> tuple[WakeWordDataset, WakeWordDataset]:
    """Load all WAVs, split into train/val, return two datasets."""
    pos_files = sorted(POSITIVE_DIR.glob("*.wav"))
    neg_files = sorted(NEGATIVE_DIR.glob("*.wav"))

    if not pos_files:
        sys.exit(f"[ERROR] No positive samples found in {POSITIVE_DIR}")
    if not neg_files:
        sys.exit(f"[ERROR] No negative samples found in {NEGATIVE_DIR}")

    print(f"  Positive samples : {len(pos_files)}")
    print(f"  Negative samples : {len(neg_files)}")

    all_samples = (
        [(p, 1) for p in pos_files] +
        [(p, 0) for p in neg_files]
    )

    random.seed(SEED)
    random.shuffle(all_samples)

    split = int(len(all_samples) * (1 - VAL_SPLIT))
    train_samples = all_samples[:split]
    val_samples   = all_samples[split:]

    print(f"  Train : {len(train_samples)}  |  Val : {len(val_samples)}")
    return WakeWordDataset(train_samples), WakeWordDataset(val_samples)


# ─────────────────────────────────────────────────────────────
#  MODEL — small CNN binary classifier
# ─────────────────────────────────────────────────────────────

class WakeWordCNN(nn.Module):
    """
    Tiny CNN — designed to run in real-time on CPU.
    Input : (batch, 1, 32, 101)   — mel-spectrogram
    Output: (batch, 1)            — sigmoid probability of wake word
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # → (16, 16, 50)

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # → (32, 8, 25)

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), # → (64, 4, 4) = 1024
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return torch.sigmoid(x).squeeze(1)


# ─────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def train():
    print("\n" + "─" * 50)
    print("  NEXUS — Wake Word Model Trainer")
    print("─" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")
    print(f"  Epochs : {EPOCHS}  |  Batch : {BATCH_SIZE}  |  LR : {LR}")
    print()

    # ── Load data ─────────────────────────────────────────────
    print("  Loading dataset...")
    train_ds, val_ds = load_dataset()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────
    model     = WakeWordCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.BCELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters : {total_params:,}")
    print()

    best_val_loss = float("inf")
    best_state    = None

    # ── Epoch loop ────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss  = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * len(y)
            train_correct += ((preds >= 0.5) == y.bool()).sum().item()

        train_loss /= len(train_ds)
        train_acc   = train_correct / len(train_ds) * 100

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                loss  = criterion(preds, y)
                val_loss    += loss.item() * len(y)
                val_correct += ((preds >= 0.5) == y.bool()).sum().item()

        val_loss /= len(val_ds)
        val_acc   = val_correct / len(val_ds) * 100

        scheduler.step(val_loss)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " ← best"
        else:
            marker = ""

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:>3}/{EPOCHS}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%{marker}"
            )

    # ── Export best model ─────────────────────────────────────
    print(f"\n  Best val_loss : {best_val_loss:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    model.cpu()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(1, 1, N_MELS, TARGET_FRAMES)   # (batch, ch, mels, frames)

    torch.onnx.export(
        model,
        dummy,
        str(OUTPUT_PATH),
        export_params=True,           # ← weights baked in — NO external .data file
        opset_version=17,
        do_constant_folding=True,
        input_names=["mel_features"],
        output_names=["wake_probability"],
        dynamic_axes={
            "mel_features":    {0: "batch_size"},
            "wake_probability": {0: "batch_size"},
        },
    )

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\n  ✓ Model exported → {OUTPUT_PATH}")
    print(f"    Size : {size_kb:.1f} KB  (self-contained, no external data)")

    # ── Quick sanity check via ONNX Runtime ──────────────────
    try:
        import onnxruntime as rt
        sess = rt.InferenceSession(str(OUTPUT_PATH), providers=["CPUExecutionProvider"])
        inp  = np.zeros((1, 1, N_MELS, TARGET_FRAMES), dtype=np.float32)
        out  = sess.run(None, {"mel_features": inp})
        print(f"    ONNX sanity check : output={out[0][0]:.4f}  ✓ OK")
    except Exception as e:
        print(f"    ONNX sanity check FAILED: {e}")

    print()
    print("  Training complete. Run:  python voice/wakeword.py")
    print("─" * 50 + "\n")


if __name__ == "__main__":
    train()
