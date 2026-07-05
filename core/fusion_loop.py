"""
core/fusion_loop.py
NEXUS — the always-on fusion loop + proactivity engine.

This is the nervous system: one background service that polls the sensors
(place recognition, scene description, screen, speaker) on their own cadences,
fuses their observations into a single live WorldState, and lets a set of
triggers decide — rarely and with taste — whether to say something proactively.

The design keeps `tick()` as a pure, manually-callable step (poll due sensors →
update state → evaluate triggers → return messages). `start()/stop()` just wrap
`tick()` in a background thread. Because sensors, triggers and the clock are all
injected, the whole thing is unit-testable without threads, a camera, or a mic.

Proactivity discipline: a global minimum gap between messages and de-duplication
of the last message, because the fastest way to make an assistant hated is to
have it talk too much.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from core.world_state import WorldState

log = logging.getLogger("nexus.fusion")

# A trigger inspects (old_state, new_state) and optionally returns a message.
Trigger = Callable[[WorldState, WorldState], Optional[str]]


@dataclass
class Sensor:
    """A source of observations. `read` returns a dict merged into WorldState
    (or None for 'nothing new'). It is polled at most once per `interval` sec."""
    name: str
    interval: float
    read: Callable[[], Optional[dict]]


# ─────────────────────────────────────────────────────────────
#  Default triggers (proactive, but restrained)
# ─────────────────────────────────────────────────────────────

def location_change_trigger(old: WorldState, new: WorldState) -> Optional[str]:
    if new.location and new.location != old.location:
        if old.location:
            return f"You've moved from the {old.location} to the {new.location}."
        return f"You're now in the {new.location}."
    return None


def person_arrived_trigger(old: WorldState, new: WorldState) -> Optional[str]:
    newcomers = [p for p in new.people if p not in old.people]
    if newcomers:
        return f"{', '.join(newcomers)} just arrived."
    return None


DEFAULT_TRIGGERS: tuple[Trigger, ...] = (
    location_change_trigger,
    person_arrived_trigger,
)


# ─────────────────────────────────────────────────────────────
#  Fusion loop
# ─────────────────────────────────────────────────────────────

class FusionLoop:
    def __init__(
        self,
        sensors: Optional[list[Sensor]] = None,
        triggers: Optional[list[Trigger]] = None,
        min_proactive_gap: float = 60.0,
        on_message: Optional[Callable[[str], None]] = None,
        clock: Callable[[], float] = time.time,
    ):
        self.state = WorldState()
        self.sensors = sensors or []
        self.triggers = list(triggers) if triggers is not None else list(DEFAULT_TRIGGERS)
        self.min_proactive_gap = min_proactive_gap
        self.on_message = on_message
        self.clock = clock

        self._last_run: dict[str, float] = {}
        self._last_msg_at: float = 0.0
        self._last_msg: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ── the pure step ────────────────────────────────────────────────
    def tick(self) -> list[str]:
        """Poll due sensors, update the world state, evaluate triggers. Returns
        any proactive messages emitted this tick."""
        now = self.clock()
        old = copy.deepcopy(self.state)

        changed = False
        for sensor in self.sensors:
            if now - self._last_run.get(sensor.name, -1e9) < sensor.interval:
                continue
            self._last_run[sensor.name] = now
            try:
                obs = sensor.read()
            except Exception as e:
                log.warning("Sensor %s failed: %s", sensor.name, e)
                continue
            if obs and self.state.apply(obs, now=now):
                changed = True

        if not changed:
            return []

        messages: list[str] = []
        for trig in self.triggers:
            try:
                msg = trig(old, self.state)
            except Exception as e:
                log.warning("Trigger failed: %s", e)
                continue
            if not msg:
                continue
            if now - self._last_msg_at < self.min_proactive_gap:
                continue                       # rate-limited
            if msg == self._last_msg:
                continue                       # de-dupe
            messages.append(msg)
            self._last_msg_at = now
            self._last_msg = msg
            if self.on_message:
                self.on_message(msg)
        return messages

    # ── background thread wrapper ────────────────────────────────────
    def start(self, poll_interval: float = 2.0) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def _run():
            log.info("Fusion loop started.")
            while not self._stop.is_set():
                try:
                    self.tick()
                except Exception as e:
                    log.warning("Fusion tick error: %s", e)
                self._stop.wait(poll_interval)
            log.info("Fusion loop stopped.")

        self._thread = threading.Thread(target=_run, daemon=True, name="nexus-fusion")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())


def build_default_sensors(brain) -> list[Sensor]:
    """Wire sensors to a Brain's perception/vision capabilities. Each sensor
    returns None when its hardware/dep is unavailable, so the loop stays safe."""
    sensors: list[Sensor] = []

    def read_place():
        try:
            res = brain.where_am_i()          # captures a frame internally
            if res and getattr(res, "known", False):
                return {"location": res.place}
        except Exception:
            return None
        return None

    def read_scene():
        try:
            frame = brain.capture_camera_frame()
            if frame is None:
                return None
            res = brain.describe_scene(frame)
            if res and getattr(res, "ok", False):
                return {"scene": res.text}
        except Exception:
            return None
        return None

    if getattr(brain, "place_recognizer", None):
        sensors.append(Sensor("place", interval=5.0, read=read_place))
    if getattr(brain, "scene_describer", None):
        sensors.append(Sensor("scene", interval=20.0, read=read_scene))
    return sensors
