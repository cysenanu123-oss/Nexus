"""
core/model_manager.py
NEXUS — model registry + consent-based, capability-aware model management.

Nexus should grow into the hardware it's given: when a task would benefit from
a stronger brain the device can actually run, it *offers* to download it — and
never downloads anything without the user saying yes.

Responsibilities:
  * Know a catalog of local models (the registry) tagged by size + strength.
  * Filter that catalog by what the current device can run (via core.hardware).
  * Report what's already installed (Ollama primary, GGUF files secondary).
  * `ensure(name, confirm=...)` — install a model on demand, but only after the
    `confirm` callback approves. This is the "ask before downloading" gate.

Backends supported (per project decision): Ollama by default, with a GGUF
download path for advanced/offline use.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from core.hardware import HardwareProfile, get_profile

log = logging.getLogger("nexus.model_manager")

ConfirmFn = Callable[[str], bool]
ProgressFn = Callable[[str], None]


# ─────────────────────────────────────────────────────────────
#  Model catalog
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelSpec:
    name: str                 # canonical id (also the ollama tag when backend=ollama)
    backend: str              # "ollama" | "gguf"
    params_b: float           # billions of parameters
    size_gb: float            # approx download size
    tags: tuple[str, ...]     # capabilities: chat / code / reasoning / vision / reflex
    description: str
    source: str = ""          # ollama tag or gguf URL (defaults to name for ollama)

    @property
    def pull_ref(self) -> str:
        return self.source or self.name


# A small, curated catalog of real models. Extend freely.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "qwen2.5:1.5b": ModelSpec("qwen2.5:1.5b", "ollama", 1.5, 1.0,
        ("chat", "reflex"), "Tiny, instant, offline reflex brain"),
    "qwen2.5:3b": ModelSpec("qwen2.5:3b", "ollama", 3, 2.0,
        ("chat",), "Light general chat"),
    "qwen2.5:7b": ModelSpec("qwen2.5:7b", "ollama", 7, 4.7,
        ("chat", "reasoning"), "Solid general reasoning"),
    "qwen2.5-coder:7b": ModelSpec("qwen2.5-coder:7b", "ollama", 7, 4.7,
        ("code", "reasoning"), "Strong local coding model"),
    "llama3.1:8b": ModelSpec("llama3.1:8b", "ollama", 8, 4.9,
        ("chat", "reasoning"), "Well-rounded 8B reasoning model"),
    "qwen2.5:14b": ModelSpec("qwen2.5:14b", "ollama", 13, 9.0,
        ("chat", "reasoning"), "Stronger reasoning, needs a real GPU"),
    "qwen2.5:32b": ModelSpec("qwen2.5:32b", "ollama", 34, 20.0,
        ("reasoning",), "High-end local reasoning"),
    "llama3.1:70b": ModelSpec("llama3.1:70b", "ollama", 70, 40.0,
        ("reasoning",), "Workstation-class reasoning"),
    "moondream": ModelSpec("moondream", "ollama", 1.8, 1.7,
        ("vision",), "Tiny vision-language model — describes camera frames"),
    "llava:7b": ModelSpec("llava:7b", "ollama", 7, 4.7,
        ("vision", "chat"), "Vision-language model for scene understanding"),
    "llama3.2-vision:11b": ModelSpec("llama3.2-vision:11b", "ollama", 11, 7.9,
        ("vision", "reasoning"), "Strong vision-language model (needs a GPU)"),
    "llava:13b": ModelSpec("llava:13b", "ollama", 13, 8.0,
        ("vision", "chat"), "Larger LLaVA for richer scene description"),
    "llava:34b": ModelSpec("llava:34b", "ollama", 34, 20.0,
        ("vision", "reasoning"), "High-end vision-language model"),
}


@dataclass
class EnsureResult:
    ok: bool
    message: str
    model: Optional[str] = None


# ─────────────────────────────────────────────────────────────
#  Manager
# ─────────────────────────────────────────────────────────────

class ModelManager:
    def __init__(
        self,
        profile: Optional[HardwareProfile] = None,
        gguf_dir: Optional[str] = None,
        ollama_bin: str = "ollama",
    ):
        self.profile = profile or get_profile()
        self.ollama_bin = ollama_bin
        self.gguf_dir = Path(gguf_dir or (Path(__file__).parent.parent / "models" / "gguf"))

    # ── discovery ────────────────────────────────────────────────────
    def _ollama_installed_names(self) -> list[str]:
        """Names of models Ollama has locally. [] if Ollama is unavailable."""
        try:
            out = subprocess.run([self.ollama_bin, "list"],
                                 capture_output=True, text=True, timeout=10)
        except Exception as e:
            log.info("Ollama not available for listing: %s", e)
            return []
        if out.returncode != 0:
            return []
        names = []
        for line in out.stdout.splitlines()[1:]:   # skip header
            line = line.strip()
            if line:
                names.append(line.split()[0])       # first column = NAME
        return names

    def _gguf_installed_names(self) -> list[str]:
        if not self.gguf_dir.exists():
            return []
        return [p.name for p in self.gguf_dir.glob("*.gguf")]

    def installed(self) -> list[str]:
        return self._ollama_installed_names() + self._gguf_installed_names()

    def is_installed(self, name: str) -> bool:
        installed = self.installed()
        # Ollama tags may carry a :latest suffix; compare loosely.
        return any(name == i or i.startswith(name + ":") or name.startswith(i)
                   for i in installed)

    # ── capability-aware selection ───────────────────────────────────
    def available_for_device(self) -> list[ModelSpec]:
        """Registry entries the current device can actually run, largest first."""
        fit = [s for s in MODEL_REGISTRY.values() if self.profile.can_run(s.params_b)]
        return sorted(fit, key=lambda s: s.params_b, reverse=True)

    def recommend(self, task: str) -> Optional[ModelSpec]:
        """Best runnable model for a task tag (chat/code/reasoning/vision).
        Prefers the strongest model the device can run."""
        candidates = [s for s in self.available_for_device() if task in s.tags]
        return candidates[0] if candidates else None

    def recommend_upgrade(self, task: str) -> Optional[ModelSpec]:
        """The best runnable model for a task that is NOT yet installed —
        i.e. what Nexus would offer to download to get smarter."""
        rec = self.recommend(task)
        if rec and not self.is_installed(rec.name):
            return rec
        return None

    # ── consent-gated install ────────────────────────────────────────
    def ensure(
        self,
        name: str,
        confirm: Optional[ConfirmFn] = None,
        on_progress: Optional[ProgressFn] = None,
    ) -> EnsureResult:
        """Make `name` available, downloading it only if `confirm` approves."""
        spec = MODEL_REGISTRY.get(name)
        if spec is None:
            return EnsureResult(False, f"Unknown model '{name}'.")

        if self.is_installed(name):
            return EnsureResult(True, f"{name} is already installed.", name)

        if not self.profile.can_run(spec.params_b):
            return EnsureResult(
                False,
                f"Your device ({self.profile.tier} tier, "
                f"~{self.profile.effective_accel_gb} GB usable) can't run {name} "
                f"(needs ~{spec.size_gb} GB). Try a smaller model.",
            )

        prompt = (
            f"Download {name} — {spec.description} "
            f"(~{spec.size_gb} GB, {spec.backend}). "
            f"It fits your {self.profile.tier}-tier device. Proceed?"
        )
        if confirm is None or not confirm(prompt):
            return EnsureResult(False, f"Download of {name} declined.", name)

        if spec.backend == "ollama":
            return self._download_ollama(spec, on_progress)
        return self._download_gguf(spec, on_progress)

    # ── downloads ────────────────────────────────────────────────────
    def _download_ollama(self, spec: ModelSpec, on_progress: Optional[ProgressFn]) -> EnsureResult:
        try:
            proc = subprocess.Popen(
                [self.ollama_bin, "pull", spec.pull_ref],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
        except FileNotFoundError:
            return EnsureResult(False, "Ollama is not installed. See https://ollama.com/download")
        except Exception as e:
            return EnsureResult(False, f"Failed to start download: {e}")

        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line and on_progress:
                on_progress(line)
        code = proc.wait()
        if code == 0:
            return EnsureResult(True, f"{spec.name} downloaded and ready.", spec.name)
        return EnsureResult(False, f"Download failed (exit {code}).", spec.name)

    def _download_gguf(self, spec: ModelSpec, on_progress: Optional[ProgressFn]) -> EnsureResult:
        if not spec.source:
            return EnsureResult(False, f"No GGUF source URL for {spec.name}.")
        try:
            import requests
        except Exception:
            return EnsureResult(False, "The 'requests' package is required for GGUF downloads.")

        self.gguf_dir.mkdir(parents=True, exist_ok=True)
        dest = self.gguf_dir / (spec.name.replace(":", "_") + ".gguf")
        try:
            with requests.get(spec.source, stream=True, timeout=30) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if on_progress and total:
                            on_progress(f"{done / total:.0%} ({done // (1<<20)} MB)")
        except Exception as e:
            if dest.exists():
                dest.unlink(missing_ok=True)
            return EnsureResult(False, f"GGUF download failed: {e}")
        return EnsureResult(True, f"{spec.name} downloaded to {dest}.", spec.name)

    # ── reporting ────────────────────────────────────────────────────
    def report(self) -> str:
        installed = set(self.installed())
        lines = [self.profile.summary(), "", "── Models ─────────────────────────────────"]
        for spec in sorted(MODEL_REGISTRY.values(), key=lambda s: s.params_b):
            runnable = self.profile.can_run(spec.params_b)
            here = any(spec.name == i or i.startswith(spec.name + ":") for i in installed)
            mark = "✓ installed" if here else ("· can download" if runnable else "✗ too big")
            tags = "/".join(spec.tags)
            lines.append(f"  {mark:<14} {spec.name:<18} {spec.size_gb:>4}GB  [{tags}]")
        return "\n".join(lines)


_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager


if __name__ == "__main__":
    print(get_model_manager().report())
