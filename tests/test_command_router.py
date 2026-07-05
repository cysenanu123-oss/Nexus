"""
Tests for core/command_router.py — the table-driven replacement for the
Brain._route if/elif chain.
"""

from core.command_router import CommandRouter, normalize


def test_normalize_strips_case_and_punctuation():
    assert normalize("  Trends!  ") == "trends"
    assert normalize("What Do You Know?") == "what do you know"


def test_exact_match_first_match_wins():
    hits = []
    r = (CommandRouter()
         .exact("trends", to=lambda t: "A", label="a")
         .exact("trends", to=lambda t: "B", label="b"))
    assert r.dispatch("trends") == "A"


def test_exact_is_forgiving_of_punctuation_and_case():
    r = CommandRouter().exact("end session", to=lambda t: "ended")
    assert r.dispatch("End Session") == "ended"
    assert r.dispatch("end session!") == "ended"


def test_no_match_returns_none():
    r = CommandRouter().exact("trends", to=lambda t: "x")
    assert r.dispatch("something else") is None
    assert r.find("something else") is None


def test_prefix_match_and_argument_passthrough():
    r = CommandRouter().prefix("session ", to=lambda t: t.split()[1])
    assert r.dispatch("session 3") == "3"
    assert r.dispatch("sessions") is None  # no trailing space → not a prefix hit


def test_contains_match():
    r = CommandRouter().contains("acquire skill", to=lambda t: "acq")
    assert r.dispatch("please acquire skill from github") == "acq"
    assert r.dispatch("nothing here") is None


def test_predicate_match():
    r = CommandRouter().predicate(lambda t: t.isdigit(), to=lambda t: int(t) * 2)
    assert r.dispatch("21") == 42
    assert r.dispatch("abc") is None


def test_registration_order_is_respected():
    # exact registered before prefix: exact wins for an exact hit.
    r = (CommandRouter()
         .exact("focus", to=lambda t: "exact")
         .prefix("foc", to=lambda t: "prefix"))
    assert r.dispatch("focus") == "exact"
    assert r.dispatch("focing") == "prefix"


def test_len_reflects_route_count():
    r = CommandRouter().exact("a", to=lambda t: 1).prefix("b", to=lambda t: 2)
    assert len(r) == 2
