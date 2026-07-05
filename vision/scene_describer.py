"""
vision/scene_describer.py
NEXUS — scene understanding via a vision-language model.

Turns a camera frame (or image file) into a sentence: "a desk with two monitors,
a coffee mug, and a window on the left." It picks the *best VLM the device can
actually run* — moondream on a light laptop, LLaVA-7B/13B or Llama-3.2-Vision on
a real GPU — using the same capability-aware model manager as everything else,
and offers to download it (consent-gated) if it isn't installed yet.

The actual VLM call is injectable (`describe_fn`), so the model-selection and
download-prompt logic is unit-testable without Ollama or a GPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

log = logging.getLogger("nexus.scene_describer")

# Fallback preference order if the model manager can't recommend one.
_VISION_PREFERENCE = ("llava:34b", "llama3.2-vision:11b", "llava:13b",
                      "llava:7b", "moondream")


@dataclass
class SceneResult:
    ok: bool
    text: str
    model: str = ""


class SceneDescriber:
    def __init__(self, llm=None, model_manager=None,
                 describe_fn: Optional[Callable] = None):
        self.llm = llm
        self.model_manager = model_manager
        # describe_fn(image, prompt, model) -> str. Defaults to the LLM's VLM call.
        self._describe_fn = describe_fn

    # ── model selection ──────────────────────────────────────────────
    def pick_model(self) -> Optional[str]:
        """Best runnable vision model for this device (largest that fits)."""
        if self.model_manager is not None:
            rec = self.model_manager.recommend("vision")
            if rec:
                return rec.name
        return None

    def _describe(self, image, prompt: str, model: str) -> str:
        if self._describe_fn is not None:
            return self._describe_fn(image, prompt, model)
        if self.llm is not None:
            return self.llm.describe_image(image, prompt=prompt, model=model)
        return "LLM error: no vision backend"

    # ── main entry ───────────────────────────────────────────────────
    def describe(
        self,
        image,
        prompt: str = "Describe this scene concisely: the place, key objects, and any people.",
        confirm=None,
    ) -> SceneResult:
        """Describe an image. If the chosen VLM isn't installed, offer to
        download it (via the model manager's consent gate)."""
        model = self.pick_model()
        if not model:
            return SceneResult(False,
                "No vision model fits this device. Try `models` to see options.")

        # Ensure the model is present — consent-gated download if not.
        if self.model_manager is not None and not self.model_manager.is_installed(model):
            if confirm is None:
                return SceneResult(False,
                    f"Vision model '{model}' isn't installed. "
                    f"Run: models get {model}", model=model)
            res = self.model_manager.ensure(model, confirm=confirm)
            if not res.ok:
                return SceneResult(False, res.message, model=model)

        text = self._describe(image, prompt, model)
        if not text or text.startswith("LLM error"):
            return SceneResult(False, text or "No description produced.", model=model)
        return SceneResult(True, text.strip(), model=model)
