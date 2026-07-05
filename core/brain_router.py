"""
core/brain_router.py
NEXUS — the tiered brain router.

The idea (see the README "Adaptive Intelligence Roadmap"): route each task to
the *cheapest brain that can handle it*, and only escalate when the cheaper one
is out of its depth.

    REFLEX  — tiny local model. Instant, offline, free.
    LOCAL   — a bigger local model, if the device can run one. Private, offline.
    CLOUD   — a hosted model (Claude / GPT) via API. Smartest, but costs money
              and leaves the device, so it is gated behind explicit consent.

The router starts at the lowest sensible tier, runs it, and escalates only if
the answer looks uncertain/empty AND a higher tier is both available and
allowed. Cloud is never used silently: it requires `llm.allow_cloud` to be on
and (by default) a per-call confirmation — the same consent discipline as the
model downloader and the shell-safety layer.

Backends are injected, so the whole policy is unit-testable without touching a
real model or the network.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Optional

log = logging.getLogger("nexus.brain_router")

ConfirmFn = Callable[[str], bool]


class Tier(IntEnum):
    REFLEX = 1
    LOCAL = 2
    CLOUD = 3


# ─────────────────────────────────────────────────────────────
#  Uncertainty — the escalation trigger
# ─────────────────────────────────────────────────────────────

_UNCERTAIN_PHRASES = (
    "i don't know", "i'm not sure", "i cannot determine", "i can't determine",
    "i don't have enough information", "i'm unable to", "i cannot answer",
    "as an ai", "i don't have access", "i'm not able to", "i do not have",
)


def is_uncertain(text: str) -> bool:
    """True when a response signals the model is out of its depth."""
    if not text or not text.strip():
        return True
    low = text.strip().lower()
    if low.startswith("llm error") or low.startswith("(offline"):
        return True
    if len(low) < 2:
        return True
    return any(p in low for p in _UNCERTAIN_PHRASES)


# ─────────────────────────────────────────────────────────────
#  Difficulty estimate → starting tier
# ─────────────────────────────────────────────────────────────

_HARD_MARKERS = (
    "analyze", "analyse", "design", "architect", "prove", "optimize", "optimise",
    "debug", "why does", "why is", "strategy", "refactor", "compare", "trade-off",
    "tradeoff", "derive", "explain how", "explain why", "walk me through",
    "step by step", "in depth", "pros and cons",
)


def estimate_start_tier(task: str) -> Tier:
    """Pick where to START. We never start at CLOUD (cost/privacy) — hard tasks
    start LOCAL and escalate to cloud only if the local answer is weak."""
    t = task.lower()
    words = len(task.split())
    if words <= 4 and not any(m in t for m in _HARD_MARKERS):
        return Tier.REFLEX
    return Tier.LOCAL


# ─────────────────────────────────────────────────────────────
#  Backends
# ─────────────────────────────────────────────────────────────

class Backend:
    """A single brain. Subclasses implement available() and generate()."""

    name: str = "backend"
    tier: Tier = Tier.LOCAL
    is_local: bool = True
    costs_money: bool = False
    model: str = ""

    def available(self) -> bool:
        raise NotImplementedError

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        raise NotImplementedError


class LocalLLMBackend(Backend):
    """Wraps the existing Ollama-backed core.llm.LLM at a fixed model."""

    def __init__(self, llm, model: str = "", tier: Tier = Tier.LOCAL,
                 task: str = "chat", name: Optional[str] = None):
        self._llm = llm
        self.model = model
        self.tier = tier
        self.task = task
        self.is_local = True
        self.costs_money = False
        self.name = name or f"local:{model or task}"

    def available(self) -> bool:
        return bool(self._llm) and getattr(self._llm, "is_ready", False)

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        return self._llm.chat(prompt, system=system,
                              model=self.model or None, task=self.task)


class CloudBackend(Backend):
    """Hosted model via HTTP API (Anthropic or OpenAI). Requires an API key in
    the environment and network connectivity."""

    def __init__(self, provider: str = "anthropic", model: str = "claude-sonnet-5",
                 max_tokens: int = 1024, api_key: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.tier = Tier.CLOUD
        self.is_local = False
        self.costs_money = True
        self.name = f"cloud:{provider}:{model}"
        self._api_key = api_key or self._key_from_env(provider)

    @staticmethod
    def _key_from_env(provider: str) -> str:
        return os.environ.get(
            "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY", ""
        )

    def available(self) -> bool:
        return bool(self._api_key) and _is_online()

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        try:
            import requests
        except Exception:
            return "LLM error: requests not installed"
        try:
            if self.provider == "anthropic":
                return self._anthropic(requests, prompt, system)
            return self._openai(requests, prompt, system)
        except Exception as e:
            log.warning("Cloud backend %s failed: %s", self.name, e)
            return f"LLM error: {e}"

    def _anthropic(self, requests, prompt: str, system: Optional[str]) -> str:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body, timeout=60,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", [])).strip()

    def _openai(self, requests, prompt: str, system: Optional[str]) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}",
                     "content-type": "application/json"},
            json={"model": self.model, "messages": messages,
                  "max_tokens": self.max_tokens}, timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


# ── connectivity (cached briefly) ────────────────────────────────────────

_online_cache: tuple[float, bool] = (0.0, False)


def _is_online(host: str = "1.1.1.1", port: int = 53, timeout: float = 1.5) -> bool:
    global _online_cache
    ts, val = _online_cache
    if time.time() - ts < 30:
        return val
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        val = True
    except Exception:
        val = False
    _online_cache = (time.time(), val)
    return val


# ─────────────────────────────────────────────────────────────
#  Router
# ─────────────────────────────────────────────────────────────

@dataclass
class RouteResult:
    text: str
    tier_used: Optional[Tier] = None
    backend_name: Optional[str] = None
    escalated: bool = False
    tiers_tried: list[str] = field(default_factory=list)


class BrainRouter:
    def __init__(
        self,
        backends: list[Backend],
        allow_cloud: bool = False,
        cloud_confirm: bool = True,
        uncertain_fn: Callable[[str], bool] = is_uncertain,
    ):
        # Lowest tier first; dedupe identical local models so we don't ask the
        # same model twice on a device where reflex and local resolve equal.
        self.backends = self._dedupe(sorted(backends, key=lambda b: b.tier))
        self.allow_cloud = allow_cloud
        self.cloud_confirm = cloud_confirm
        self.is_uncertain = uncertain_fn

    @staticmethod
    def _dedupe(backends: list[Backend]) -> list[Backend]:
        seen: set[tuple] = set()
        out: list[Backend] = []
        for b in backends:
            key = (b.is_local, b.model or b.name)
            if key in seen:
                continue
            seen.add(key)
            out.append(b)
        return out

    def _cloud_ok(self, confirm: Optional[ConfirmFn], backend: Backend) -> bool:
        if not self.allow_cloud:
            return False
        if self.cloud_confirm:
            msg = (f"Escalate to {backend.name}? This sends your request to a "
                   f"hosted model (leaves the device, may cost money).")
            return confirm is not None and confirm(msg)
        return True

    def route(
        self,
        task: str,
        system: Optional[str] = None,
        min_tier: Optional[Tier] = None,
        confirm: Optional[ConfirmFn] = None,
    ) -> RouteResult:
        start = min_tier or estimate_start_tier(task)
        tried: list[str] = []
        last_text = ""
        last_tier: Optional[Tier] = None
        last_name: Optional[str] = None

        for backend in self.backends:
            if backend.tier < start:
                continue
            if not backend.available():
                continue
            if backend.tier == Tier.CLOUD and not self._cloud_ok(confirm, backend):
                continue

            tried.append(backend.name)
            text = backend.generate(task, system=system)
            last_text, last_tier, last_name = text, backend.tier, backend.name

            if text and not self.is_uncertain(text):
                return RouteResult(
                    text=text, tier_used=backend.tier, backend_name=backend.name,
                    escalated=len(tried) > 1, tiers_tried=tried,
                )

        if tried:
            # Every tier we could reach was uncertain — return the best we got.
            return RouteResult(
                text=last_text or "I couldn't find a confident answer.",
                tier_used=last_tier, backend_name=last_name,
                escalated=len(tried) > 1, tiers_tried=tried,
            )
        return RouteResult(
            text="No brain is available right now (is Ollama running? is a "
                 "model installed? try `models`).",
            tier_used=None, backend_name=None, escalated=False, tiers_tried=[],
        )

    def describe(self) -> str:
        lines = ["── Brain Router ───────────────────────────"]
        for b in self.backends:
            state = "up" if b.available() else "down"
            gate = " (cloud, consent-gated)" if b.tier == Tier.CLOUD else ""
            lines.append(f"  {b.tier.name:<7} {b.name:<28} [{state}]{gate}")
        lines.append(f"  cloud allowed: {self.allow_cloud} · confirm: {self.cloud_confirm}")
        return "\n".join(lines)


def build_default_backends(llm, model_manager=None, cfg=None) -> list[Backend]:
    """Assemble the standard REFLEX/LOCAL/CLOUD backends from what's available."""
    backends: list[Backend] = []
    if llm is not None:
        available = list(getattr(llm, "_available_models", []) or [])
        # Reflex = smallest available model (fallback to task-based pick).
        reflex_model = min(available, key=len) if available else ""
        backends.append(LocalLLMBackend(llm, model=reflex_model, tier=Tier.REFLEX,
                                        task="fast", name="local-reflex"))
        # Local reasoning = a stronger installed model if the manager knows one.
        local_model = ""
        if model_manager is not None:
            rec = model_manager.recommend("reasoning")
            if rec and model_manager.is_installed(rec.name):
                local_model = rec.name
        backends.append(LocalLLMBackend(llm, model=local_model, tier=Tier.LOCAL,
                                        task="chat", name="local-reasoning"))

    provider = model = None
    allow_cloud = False
    confirm = True
    if cfg is not None:
        try:
            allow_cloud = bool(cfg.get("llm.allow_cloud", False))
            confirm = bool(cfg.get("llm.cloud_confirm", True))
            provider = cfg.get("llm.cloud_provider", "anthropic")
            model = cfg.get("llm.cloud_model", "claude-sonnet-5")
        except Exception:
            pass
    if provider:
        backends.append(CloudBackend(provider=provider,
                                     model=model or "claude-sonnet-5"))
    return backends


if __name__ == "__main__":
    # Demo with a couple of fake backends.
    class Fake(Backend):
        def __init__(self, name, tier, reply, up=True):
            self.name, self.tier, self._reply, self._up = name, tier, reply, up
            self.is_local = tier != Tier.CLOUD
            self.costs_money = tier == Tier.CLOUD
            self.model = name
        def available(self): return self._up
        def generate(self, prompt, system=None): return self._reply

    r = BrainRouter(
        [Fake("reflex", Tier.REFLEX, "I don't know"),
         Fake("local", Tier.LOCAL, "I'm not sure"),
         Fake("cloud", Tier.CLOUD, "The answer is 42.")],
        allow_cloud=True, cloud_confirm=False,
    )
    res = r.route("explain why the sky is blue")
    print(res.tiers_tried, "->", res.backend_name, ":", res.text)
