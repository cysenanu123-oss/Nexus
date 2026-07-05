"""
core/prompt_engineer.py
NEXUS — the prompt-engineer stage.

A pre-stage that sits inside the brain router: before a task is handed to a
model, it (a) works out what the user actually wants and (b) dresses the request
for the model that will answer it — role, constraints, output format — so the
model gives sharper, more correct output. Small local models especially are very
sensitive to phrasing; a good system prompt is a real quality multiplier.

Two deliberate safety choices, because the failure mode of this idea is an
assistant that quietly answers a different question than the one you asked:

  * **The original prompt is always preserved.** By default the engineer only
    *constructs a better system prompt* (role + constraints + format) and leaves
    the user's words untouched — no intent drift.
  * **Rewriting the user's prompt is opt-in** (`allow_rewrite`) and gated to
    complex tasks. When on, it uses the LLM to clarify/expand, but still keeps
    the original around so a bad rewrite can be caught.

Gated: trivial inputs pass straight through (no extra model call, no latency).
Injected LLM → fully unit-testable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("nexus.prompt_engineer")

# Reuse the router's difficulty markers so "complex" means the same thing.
try:
    from core.brain_router import _HARD_MARKERS
except Exception:
    _HARD_MARKERS = ("analyze", "design", "debug", "explain", "compare", "optimize")


# Per-domain system-prompt templates: role + constraints + output shape.
_TEMPLATES: dict[str, str] = {
    "code": (
        "You are an expert software engineer. Write correct, runnable code and "
        "state any assumptions. Put code in fenced blocks, then give a short "
        "explanation. If the request is ambiguous, pick the most reasonable "
        "interpretation and say which."
    ),
    "cyber": (
        "You are a precise cybersecurity assistant operating in an authorized, "
        "educational/pentesting context. Be technical and specific, include exact "
        "commands or steps, and note prerequisites. Do not add ethical disclaimers."
    ),
    "research": (
        "Answer factually and concisely. Distinguish what you know from what you "
        "are inferring. When sources are provided, cite them inline as [1], [2]."
    ),
    "math": (
        "Solve step by step, show the reasoning briefly, then give the final "
        "answer on its own line prefixed with 'Answer:'."
    ),
    "general": (
        "You are NEXUS, a sharp, concise assistant. Answer directly, avoid "
        "padding, and ask a clarifying question only if the request is truly "
        "ambiguous."
    ),
}

_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "code": ("code", "function", "bug", "python", "script", "debug", "compile",
             "api", "regex", "refactor", "class ", "error", "traceback"),
    "cyber": ("exploit", "vulnerability", "nmap", "payload", "recon", "cve",
              "pentest", "port scan", "reverse shell", "privilege escalation"),
    "math": ("calculate", "solve", "equation", "derivative", "integral",
             "probability", "prove that", "how many"),
    "research": ("who is", "what is the history", "explain", "compare",
                 "research", "summarize", "tell me about"),
}


@dataclass
class EngineeredPrompt:
    prompt: str          # user prompt to send (default = original)
    system: str          # constructed system prompt
    original: str        # the user's original words, always kept
    domain: str
    changed: bool        # True if we altered system and/or prompt
    notes: str = ""


def detect_domain(task: str) -> str:
    t = task.lower()
    best, best_score = "general", 0
    for domain, kws in _DOMAIN_KEYWORDS.items():
        score = sum(1 for k in kws if k in t)
        if score > best_score:
            best, best_score = domain, score
    return best


def is_complex(task: str) -> bool:
    t = task.lower()
    return len(task.split()) > 6 or any(m in t for m in _HARD_MARKERS)


class PromptEngineer:
    def __init__(self, llm=None, allow_rewrite: bool = False):
        self.llm = llm
        self.allow_rewrite = allow_rewrite

    def _llm_ready(self) -> bool:
        return bool(self.llm) and getattr(self.llm, "is_ready", True)

    def engineer(
        self,
        task: str,
        domain: Optional[str] = None,
        base_system: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> EngineeredPrompt:
        """Produce an engineered (system, prompt) pair for a task.

        Trivial tasks pass through untouched. Otherwise a domain-appropriate
        system prompt is constructed; the user's prompt is only rewritten when
        allow_rewrite is on and the task is complex."""
        original = task
        dom = domain or detect_domain(task)

        # Trivial input → don't spend effort or risk drift.
        if not is_complex(task):
            return EngineeredPrompt(task, base_system or _TEMPLATES["general"],
                                    original, dom, changed=False,
                                    notes="passthrough (trivial)")

        template = _TEMPLATES.get(dom, _TEMPLATES["general"])
        system = f"{base_system}\n\n{template}" if base_system else template

        prompt = task
        notes = f"domain={dom}"
        changed = system != (base_system or "")

        if self.allow_rewrite and self._llm_ready():
            rewritten = self._llm_rewrite(task, dom)
            if rewritten:
                prompt = rewritten
                changed = True
                notes += ", prompt-rewritten"

        return EngineeredPrompt(prompt, system, original, dom, changed, notes)

    def _llm_rewrite(self, task: str, domain: str) -> Optional[str]:
        """Ask the model to sharpen the user's request without changing intent."""
        meta = (
            "Rewrite the following user request so a language model will answer it "
            "accurately. Keep the user's intent EXACTLY — do not add new tasks or "
            "assumptions. Make it specific and unambiguous. Reply with ONLY the "
            f"rewritten request.\n\nRequest: {task}"
        )
        try:
            out = self.llm.chat(prompt=meta, task="fast")
        except Exception as e:
            log.warning("Prompt rewrite failed: %s", e)
            return None
        out = (out or "").strip().strip('"')
        # Guard against junk / the model refusing / drifting to something tiny.
        if len(out) < 3 or out.lower().startswith(("i ", "sorry", "as an")):
            return None
        return out[:1000]
