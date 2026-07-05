"""
core/hardware.py
NEXUS — cross-platform device capability detection.

Before Nexus can pick the right brain for a task (or offer to download a bigger
model), it has to know what body it is running on. This module probes the
machine — RAM, CPU, GPU/accelerator memory, disk — on Linux, macOS and Windows,
and turns that into a *capability class*: which local model sizes are realistic.

Design goals:
  * No hard dependencies. Uses psutil/torch if present, but falls back to
    stdlib (/proc, sysctl, ctypes, shutil) so it never crashes on a fresh box.
  * Every probe is wrapped — a failing probe yields "unknown", not an exception.
  * Result is cached; call get_profile() freely.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field

# Standard local-model sizes (in billions of params) we reason about.
_STANDARD_SIZES_B: tuple[float, ...] = (1.5, 3, 7, 8, 13, 34, 70)

# Approx GB of memory needed per billion params, by quantization.
_GB_PER_B = {"q4": 0.6, "q8": 1.1, "f16": 2.1}


def _run(cmd: list[str], timeout: int = 5) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return ""


# ─────────────────────────────────────────────────────────────
#  Individual probes (each returns a best-effort value, never raises)
# ─────────────────────────────────────────────────────────────

def _total_ram_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass

    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return round(int(line.split()[1]) * 1024 / 1e9, 1)
        elif system == "Darwin":
            out = _run(["sysctl", "-n", "hw.memsize"])
            if out:
                return round(int(out) / 1e9, 1)
        elif system == "Windows":
            import ctypes

            class _MEMS(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMS()
            stat.dwLength = ctypes.sizeof(_MEMS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / 1e9, 1)
    except Exception:
        pass

    try:  # POSIX last resort
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except Exception:
        return 0.0


def _disk_free_gb() -> float:
    try:
        return round(shutil.disk_usage(os.path.expanduser("~")).free / 1e9, 1)
    except Exception:
        return 0.0


def _detect_gpu() -> tuple[str, str, float | None, str]:
    """Return (vendor, name, vram_gb_or_None, backend). backend ∈ cuda/mps/rocm/none."""
    # 1) torch — most accurate when installed.
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return ("nvidia", torch.cuda.get_device_name(0),
                    round(props.total_memory / 1e9, 1), "cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return ("apple", "Apple Silicon (Metal)", None, "mps")
    except Exception:
        pass

    # 2) nvidia-smi
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits"])
    if smi:
        try:
            name, mem = [x.strip() for x in smi.splitlines()[0].split(",")]
            return ("nvidia", name, round(float(mem) / 1024, 1), "cuda")
        except Exception:
            pass

    # 3) Apple Silicon by platform (unified memory → vram tracks RAM)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return ("apple", "Apple Silicon", None, "mps")

    # 4) AMD ROCm
    if _run(["rocm-smi", "--showproductname"]):
        return ("amd", "AMD GPU (ROCm)", None, "rocm")

    return ("none", "", 0.0, "none")


# ─────────────────────────────────────────────────────────────
#  Profile
# ─────────────────────────────────────────────────────────────

@dataclass
class HardwareProfile:
    os: str
    arch: str
    cpu_cores: int
    ram_gb: float
    gpu_vendor: str
    gpu_name: str
    vram_gb: float | None          # None = unified/unknown (see effective_accel_gb)
    accelerator: str               # cuda / mps / rocm / none
    disk_free_gb: float
    runnable_sizes_b: list[float] = field(default_factory=list)

    @property
    def has_gpu(self) -> bool:
        return self.accelerator != "none"

    @property
    def effective_accel_gb(self) -> float:
        """Memory usable for a model on the fastest available path."""
        if self.accelerator == "cuda" and self.vram_gb:
            return self.vram_gb
        if self.accelerator in ("mps", "rocm"):
            # Unified / shared memory — assume ~70% of RAM is usable for a model.
            return round(self.ram_gb * 0.7, 1)
        # CPU-only: models run in RAM but slowly; be conservative.
        return round(self.ram_gb * 0.6, 1)

    @property
    def max_model_params_b(self) -> float:
        return max(self.runnable_sizes_b, default=0.0)

    @property
    def tier(self) -> str:
        m = self.max_model_params_b
        if m >= 70:
            return "workstation"   # 70B-class
        if m >= 30:
            return "high"          # 34B-class
        if m >= 13:
            return "mid"           # 13B-class
        if m >= 7:
            return "standard"      # 7-8B-class
        if m >= 3:
            return "light"         # 3B-class
        return "minimal"           # 1.5B or reflex-only

    def can_run(self, params_b: float, quant: str = "q4") -> bool:
        need = params_b * _GB_PER_B.get(quant, 0.6)
        # 0.9 leaves ~10% headroom for KV-cache/context and matches real-world
        # fits (e.g. a 70B q4 ≈ 40 GB runs on a 48 GB card).
        return need <= self.effective_accel_gb * 0.9

    def summary(self) -> str:
        vram = (f"{self.vram_gb} GB VRAM" if self.vram_gb
                else ("unified memory" if self.accelerator in ("mps", "rocm") else "no GPU"))
        sizes = ", ".join(f"{s:g}B" for s in self.runnable_sizes_b) or "reflex-only"
        return (
            "── Device Capability ──────────────────────\n"
            f"  OS/arch   : {self.os} ({self.arch})\n"
            f"  CPU       : {self.cpu_cores} cores\n"
            f"  RAM       : {self.ram_gb} GB\n"
            f"  GPU       : {self.gpu_name or '—'} ({self.accelerator}, {vram})\n"
            f"  Disk free : {self.disk_free_gb} GB\n"
            f"  Tier      : {self.tier}  (can run: {sizes})"
        )


def detect() -> HardwareProfile:
    vendor, name, vram, backend = _detect_gpu()
    profile = HardwareProfile(
        os=platform.system() or "unknown",
        arch=platform.machine() or "unknown",
        cpu_cores=os.cpu_count() or 1,
        ram_gb=_total_ram_gb(),
        gpu_vendor=vendor,
        gpu_name=name,
        vram_gb=vram,
        accelerator=backend,
        disk_free_gb=_disk_free_gb(),
    )
    profile.runnable_sizes_b = [s for s in _STANDARD_SIZES_B if profile.can_run(s)]
    return profile


_cached: HardwareProfile | None = None


def get_profile(refresh: bool = False) -> HardwareProfile:
    global _cached
    if _cached is None or refresh:
        _cached = detect()
    return _cached


if __name__ == "__main__":
    print(get_profile().summary())
