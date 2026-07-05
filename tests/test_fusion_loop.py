"""Tests for core/world_state.py and core/fusion_loop.py — fusion + proactivity."""

from core.world_state import WorldState
from core.fusion_loop import FusionLoop, Sensor, location_change_trigger


# ── clock helper ─────────────────────────────────────────────────────

class Clock:
    def __init__(self, t=1000.0):
        self.t = t
    def __call__(self):
        return self.t
    def advance(self, dt):
        self.t += dt


def sensor(name, script, interval=0.0):
    """A sensor whose read() returns successive items from `script` (a list)."""
    it = iter(script)
    def read():
        try:
            return next(it)
        except StopIteration:
            return None
    return Sensor(name, interval=interval, read=read)


# ── WorldState ───────────────────────────────────────────────────────

def test_worldstate_apply_detects_change():
    w = WorldState()
    assert w.apply({"location": "office"}) is True
    assert w.apply({"location": "office"}) is False   # same value → no change
    assert w.apply({"location": "kitchen"}) is True


def test_worldstate_ignores_unknown_and_none_fields():
    w = WorldState()
    assert w.apply({"bogus": "x"}) is False
    assert w.apply({"location": None}) is False
    assert w.location is None


# ── fusion loop ──────────────────────────────────────────────────────

def test_tick_updates_state_from_sensors():
    clk = Clock()
    loop = FusionLoop(sensors=[sensor("place", [{"location": "office"}])],
                      clock=clk, min_proactive_gap=0)
    msgs = loop.tick()
    assert loop.state.location == "office"
    assert any("office" in m for m in msgs)


def test_no_change_no_messages():
    clk = Clock()
    loop = FusionLoop(sensors=[sensor("place", [{"location": "office"},
                                                {"location": "office"}])],
                      clock=clk, min_proactive_gap=0)
    assert loop.tick()                 # first: change → message
    assert loop.tick() == []           # second: same → nothing


def test_location_change_message():
    clk = Clock()
    scr = [{"location": "office"}, {"location": "kitchen"}]
    loop = FusionLoop(sensors=[sensor("place", scr)], clock=clk, min_proactive_gap=0)
    assert loop.tick() == ["You're now in the office."]
    assert loop.tick() == ["You've moved from the office to the kitchen."]


def test_rate_limiting_suppresses_rapid_messages():
    clk = Clock()
    scr = [{"location": "office"}, {"location": "kitchen"}, {"location": "garage"}]
    loop = FusionLoop(sensors=[sensor("place", scr)], clock=clk, min_proactive_gap=60)

    assert loop.tick()                 # office → message
    assert loop.tick() == []           # kitchen within 60s → suppressed
    clk.advance(61)
    assert loop.tick()                 # garage after gap → allowed again
    assert loop.state.location == "garage"


def test_sensor_interval_respected():
    clk = Clock()
    calls = {"n": 0}
    def read():
        calls["n"] += 1
        return {"location": f"place{calls['n']}"}
    loop = FusionLoop(sensors=[Sensor("place", interval=10.0, read=read)],
                      clock=clk, min_proactive_gap=0)
    loop.tick()                        # polled (n=1)
    loop.tick()                        # too soon → not polled
    assert calls["n"] == 1
    clk.advance(10)
    loop.tick()                        # now polled again (n=2)
    assert calls["n"] == 2


def test_sensor_exception_does_not_crash_tick():
    clk = Clock()
    def boom():
        raise RuntimeError("camera unplugged")
    loop = FusionLoop(
        sensors=[Sensor("bad", 0.0, boom),
                 sensor("place", [{"location": "office"}])],
        clock=clk, min_proactive_gap=0)
    msgs = loop.tick()                 # bad sensor logged, good one still works
    assert loop.state.location == "office"
    assert msgs


def test_on_message_callback_fires():
    clk = Clock()
    got = []
    loop = FusionLoop(sensors=[sensor("place", [{"location": "office"}])],
                      on_message=got.append, clock=clk, min_proactive_gap=0)
    loop.tick()
    assert got == ["You're now in the office."]


def test_person_arrival_trigger():
    clk = Clock()
    scr = [{"people": ["Cyril"]}, {"people": ["Cyril", "Ama"]}]
    loop = FusionLoop(sensors=[sensor("faces", scr)], clock=clk, min_proactive_gap=0)
    loop.tick()
    assert loop.tick() == ["Ama just arrived."]
