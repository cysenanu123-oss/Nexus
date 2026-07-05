"""
Tests for core/brain_router.py — the tiered brain router.
All backends are fakes; no real model or network is touched.
"""

import pytest

from core.brain_router import (
    BrainRouter, Backend, Tier, is_uncertain, estimate_start_tier,
)


class Fake(Backend):
    def __init__(self, name, tier, reply="ok", up=True):
        self.name = name
        self.tier = tier
        self._reply = reply
        self._up = up
        self.is_local = tier != Tier.CLOUD
        self.costs_money = tier == Tier.CLOUD
        self.model = name
        self.calls = 0

    def available(self):
        return self._up

    def generate(self, prompt, system=None):
        self.calls += 1
        return self._reply


# ── uncertainty ──────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "", "   ", "x", "I don't know", "I'm not sure about that",
    "LLM error: boom", "As an AI, I cannot help",
])
def test_uncertain_true(text):
    assert is_uncertain(text)


@pytest.mark.parametrize("text", [
    "The capital of France is Paris.",
    "Here is a working solution: ...",
])
def test_uncertain_false(text):
    assert not is_uncertain(text)


# ── difficulty ───────────────────────────────────────────────────────

def test_trivial_starts_reflex():
    assert estimate_start_tier("hi there") == Tier.REFLEX


def test_hard_starts_local():
    assert estimate_start_tier("analyze this architecture for me") == Tier.LOCAL
    assert estimate_start_tier("explain why the sky is blue in detail") == Tier.LOCAL


# ── routing / escalation ─────────────────────────────────────────────

def test_stops_at_first_confident_tier():
    reflex = Fake("reflex", Tier.REFLEX, "Paris.")
    local = Fake("local", Tier.LOCAL, "should not run")
    r = BrainRouter([reflex, local])
    res = r.route("capital of france")   # trivial → starts reflex
    assert res.backend_name == "reflex"
    assert not res.escalated
    assert local.calls == 0


def test_escalates_when_lower_tier_uncertain():
    local = Fake("local", Tier.LOCAL, "I don't know")
    cloud = Fake("cloud", Tier.CLOUD, "The answer is 42.")
    r = BrainRouter([local, cloud], allow_cloud=True, cloud_confirm=False)
    res = r.route("analyze the meaning of life")
    assert res.backend_name == "cloud"
    assert res.escalated
    assert res.tiers_tried == ["local", "cloud"]


def test_cloud_never_used_when_disallowed():
    local = Fake("local", Tier.LOCAL, "I'm not sure")
    cloud = Fake("cloud", Tier.CLOUD, "confident cloud answer")
    r = BrainRouter([local, cloud], allow_cloud=False)
    res = r.route("analyze something hard")
    assert cloud.calls == 0
    assert res.backend_name == "local"          # falls back to best local attempt


def test_cloud_requires_confirmation_by_default():
    local = Fake("local", Tier.LOCAL, "I don't know")
    cloud = Fake("cloud", Tier.CLOUD, "cloud answer")
    r = BrainRouter([local, cloud], allow_cloud=True, cloud_confirm=True)

    # No confirm callback → cloud is skipped.
    assert r.route("analyze x").backend_name != "cloud"
    assert cloud.calls == 0

    # Declining → skipped.
    assert r.route("analyze x", confirm=lambda m: False).backend_name != "cloud"
    assert cloud.calls == 0

    # Approving → used.
    res = r.route("analyze x", confirm=lambda m: True)
    assert res.backend_name == "cloud"
    assert cloud.calls == 1


def test_unavailable_backends_are_skipped():
    down = Fake("local", Tier.LOCAL, "great answer", up=False)
    cloud = Fake("cloud", Tier.CLOUD, "cloud answer")
    r = BrainRouter([down, cloud], allow_cloud=True, cloud_confirm=False)
    res = r.route("analyze x")
    assert res.backend_name == "cloud"
    assert down.calls == 0


def test_no_backend_available_returns_helpful_message():
    r = BrainRouter([Fake("local", Tier.LOCAL, "x", up=False)])
    res = r.route("anything")
    assert res.tier_used is None
    assert "available" in res.text.lower()


def test_all_uncertain_returns_best_effort():
    reflex = Fake("reflex", Tier.REFLEX, "I don't know")
    local = Fake("local", Tier.LOCAL, "I'm not sure either")
    r = BrainRouter([reflex, local])   # no cloud
    res = r.route("hi")                 # trivial → starts reflex, both uncertain
    assert res.text == "I'm not sure either"
    assert res.backend_name == "local"


def test_min_tier_overrides_estimate():
    reflex = Fake("reflex", Tier.REFLEX, "reflex answer")
    local = Fake("local", Tier.LOCAL, "local answer")
    r = BrainRouter([reflex, local])
    res = r.route("hi", min_tier=Tier.LOCAL)   # force skip reflex
    assert res.backend_name == "local"
    assert reflex.calls == 0


def test_duplicate_local_models_deduped():
    a = Fake("local-reflex", Tier.REFLEX, "ans")
    b = Fake("local-reflex", Tier.LOCAL, "ans")   # same model id
    a.model = b.model = "qwen2.5:1.5b"
    r = BrainRouter([a, b])
    assert len(r.backends) == 1
