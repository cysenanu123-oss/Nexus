"""
core/world_state.py
NEXUS — the live "world model".

The fusion loop keeps this object up to date with NEXUS's best current picture
of the situation: where it is, what it's looking at, who's around, what's on
screen, what the user seems to be doing. Everything the proactive layer decides
is derived from here — it's the difference between "a set of capabilities" and
"an assistant that knows the current context".

WorldState is deliberately simple and snapshot-friendly so it can be read from
another thread without ceremony.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class WorldState:
    location: Optional[str] = None        # recognized place, e.g. "office"
    scene: Optional[str] = None           # one-line VLM description
    people: list[str] = field(default_factory=list)   # recognized speakers/faces
    on_screen: Optional[str] = None       # summary of screen content
    activity: Optional[str] = None         # inferred activity, e.g. "coding"
    updated_at: float = 0.0

    # Fields a sensor is allowed to write (guards against typos in observations).
    _WRITABLE = ("location", "scene", "people", "on_screen", "activity")

    def apply(self, observation: dict, now: Optional[float] = None) -> bool:
        """Merge an observation dict into the state. Returns True if anything
        actually changed."""
        changed = False
        for key, value in observation.items():
            if key not in self._WRITABLE:
                continue
            if value is None:
                continue
            if getattr(self, key) != value:
                setattr(self, key, value)
                changed = True
        if changed:
            self.updated_at = now if now is not None else time.time()
        return changed

    def snapshot(self) -> dict:
        d = asdict(self)
        d.pop("updated_at", None)
        return d

    def summary(self) -> str:
        parts = []
        if self.location:
            parts.append(f"location: {self.location}")
        if self.people:
            parts.append(f"people: {', '.join(self.people)}")
        if self.activity:
            parts.append(f"activity: {self.activity}")
        if self.on_screen:
            parts.append(f"screen: {self.on_screen[:60]}")
        if self.scene:
            parts.append(f"scene: {self.scene[:60]}")
        if not parts:
            return "── World State ──\n  (nothing observed yet)"
        age = f"{int(time.time() - self.updated_at)}s ago" if self.updated_at else "—"
        body = "\n".join(f"  {p}" for p in parts)
        return f"── World State (updated {age}) ──\n{body}"
