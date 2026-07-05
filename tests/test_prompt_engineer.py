"""Tests for core/prompt_engineer.py and its integration with the router."""

from core.prompt_engineer import PromptEngineer, detect_domain, is_complex
from core.brain_router import BrainRouter, Backend, Tier


# ── domain + complexity ──────────────────────────────────────────────

def test_detect_domain():
    assert detect_domain("write a python function to sort a list") == "code"
    assert detect_domain("run an nmap port scan and find the exploit") == "cyber"
    assert detect_domain("calculate the derivative of x^2") == "math"
    assert detect_domain("hello there") == "general"


def test_is_complex_gate():
    assert not is_complex("hi")
    assert not is_complex("what time is it")
    assert is_complex("explain how tcp handshakes work")
    assert is_complex("design a database schema for a shop with many tables")


# ── engineering ──────────────────────────────────────────────────────

def test_trivial_task_passes_through_untouched():
    pe = PromptEngineer()
    ep = pe.engineer("hi")
    assert not ep.changed
    assert ep.prompt == "hi"
    assert ep.original == "hi"


def test_complex_task_gets_domain_system_prompt_but_keeps_prompt():
    pe = PromptEngineer()               # allow_rewrite off by default
    ep = pe.engineer("write a python function to parse a large CSV file")
    assert ep.domain == "code"
    assert "software engineer" in ep.system.lower()
    assert ep.prompt == "write a python function to parse a large CSV file"  # unchanged
    assert ep.original == ep.prompt
    assert ep.changed


def test_base_system_is_preserved_and_extended():
    pe = PromptEngineer()
    ep = pe.engineer("check this host for a privilege escalation vulnerability",
                     base_system="You are NEXUS.")
    assert ep.domain == "cyber"
    assert ep.system.startswith("You are NEXUS.")
    assert "cybersecurity" in ep.system.lower()


class FakeLLM:
    is_ready = True
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []
    def chat(self, prompt, system=None, task="chat"):
        self.prompts.append(prompt)
        return self.reply


def test_rewrite_is_opt_in_and_preserves_original():
    llm = FakeLLM("Parse a 1GB CSV in Python using streaming; return rows as dicts.")
    pe = PromptEngineer(llm=llm, allow_rewrite=True)
    ep = pe.engineer("write a python thing to read a big csv")
    assert ep.changed
    assert ep.prompt == "Parse a 1GB CSV in Python using streaming; return rows as dicts."
    assert ep.original == "write a python thing to read a big csv"   # kept
    assert "rewrite" in llm.prompts[0].lower()


def test_rewrite_off_by_default_does_not_call_llm():
    llm = FakeLLM("should not be used")
    pe = PromptEngineer(llm=llm)         # allow_rewrite False
    ep = pe.engineer("explain how dns resolution works end to end")
    assert ep.prompt == "explain how dns resolution works end to end"
    assert llm.prompts == []


def test_rewrite_rejects_junk_and_keeps_original():
    llm = FakeLLM("As an AI, I cannot ...")   # refusal-ish → rejected
    pe = PromptEngineer(llm=llm, allow_rewrite=True)
    ep = pe.engineer("design a resilient microservice architecture")
    assert ep.prompt == "design a resilient microservice architecture"


def test_rewrite_survives_llm_exception():
    class Boom:
        is_ready = True
        def chat(self, prompt, system=None, task="chat"):
            raise RuntimeError("model down")
    pe = PromptEngineer(llm=Boom(), allow_rewrite=True)
    ep = pe.engineer("optimize this slow database query for a huge table")
    assert ep.prompt == "optimize this slow database query for a huge table"


# ── router integration ───────────────────────────────────────────────

class RecordingBackend(Backend):
    def __init__(self):
        self.name, self.tier, self.model = "local", Tier.LOCAL, "m"
        self.is_local, self.costs_money = True, False
        self.seen_system = None
        self.seen_prompt = None
    def available(self):
        return True
    def generate(self, prompt, system=None):
        self.seen_prompt, self.seen_system = prompt, system
        return "a confident answer"


def test_router_applies_prompt_engineer():
    backend = RecordingBackend()
    pe = PromptEngineer()
    router = BrainRouter([backend], prompt_engineer=pe)
    router.route("design a fault-tolerant queue system", min_tier=Tier.LOCAL)
    # The backend received the engineered system prompt, not a bare one.
    assert backend.seen_system is not None
    assert "engineer" in backend.seen_system.lower() or "nexus" in backend.seen_system.lower()


def test_router_without_prompt_engineer_passes_raw():
    backend = RecordingBackend()
    router = BrainRouter([backend])     # no prompt engineer
    router.route("hello", system="BASE", min_tier=Tier.LOCAL)
    assert backend.seen_system == "BASE"
    assert backend.seen_prompt == "hello"
