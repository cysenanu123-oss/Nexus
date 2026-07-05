"""Tests for core/web_agent.py — the autonomous research loop, fully faked."""

from core.web_agent import WebAgent


# ── fakes ────────────────────────────────────────────────────────────

class FakeSearchResponse:
    def __init__(self, urls):
        self._urls = urls
    @property
    def success(self):
        return bool(self._urls)
    @property
    def urls(self):
        return list(self._urls)


class FakeSearcher:
    """Returns a different URL set per query so we can test refinement."""
    def __init__(self, by_query):
        self.by_query = by_query
        self.queries = []
    def search(self, query, max_results=10):
        self.queries.append(query)
        return FakeSearchResponse(self.by_query.get(query, []))


class FakePage:
    def __init__(self, url, ok=True, title="", text="content"):
        self.url, self._ok, self.title, self.text = url, ok, title, text
    @property
    def success(self):
        return self._ok


class FakeFetcher:
    def __init__(self, fail_urls=()):
        self.fetched = []
        self.fail = set(fail_urls)
    def fetch(self, url):
        self.fetched.append(url)
        return FakePage(url, ok=url not in self.fail, title=f"Title of {url}")


class FakeSummary:
    def __init__(self, url, title, text="a useful summary"):
        self.url, self.title, self.text = url, title, text
    @property
    def success(self):
        return bool(self.text)


class FakeSummarizer:
    def summarize(self, page, topic=""):
        return FakeSummary(page.url, page.title)


class FakeLLM:
    is_ready = True
    def __init__(self, refine_to="second query"):
        self.refine_to = refine_to
    def chat(self, prompt, system=None, task="chat"):
        if "improved search query" in prompt:
            return self.refine_to
        if prompt.startswith("Answer the question"):
            return "Synthesized answer citing [1]."
        return "ok"


# ── tests ────────────────────────────────────────────────────────────

def test_happy_path_returns_answer_and_sources():
    searcher = FakeSearcher({"what is x": ["https://a.com", "https://b.com"]})
    agent = WebAgent(searcher, FakeFetcher(), FakeSummarizer(),
                     llm=FakeLLM(), min_sources=2, pages_per_iter=3)
    res = agent.research("what is x")
    assert res.success
    assert res.answer == "Synthesized answer citing [1]."
    assert {s.url for s in res.sources} == {"https://a.com", "https://b.com"}
    assert res.iterations == 1


def test_ssrf_urls_are_filtered_before_fetch():
    searcher = FakeSearcher({"q": ["http://169.254.169.254/meta",
                                   "http://localhost/admin",
                                   "https://public.com"]})
    fetcher = FakeFetcher()
    agent = WebAgent(searcher, fetcher, FakeSummarizer(), llm=FakeLLM(),
                     min_sources=1)
    agent.research("q")
    # Only the public URL should have been fetched.
    assert fetcher.fetched == ["https://public.com"]


def test_no_results_reports_failure_gracefully():
    agent = WebAgent(FakeSearcher({}), FakeFetcher(), FakeSummarizer(), llm=None)
    res = agent.research("obscure question")
    assert not res.success
    assert "couldn't find" in res.answer.lower()


def test_refines_query_and_iterates_when_insufficient():
    searcher = FakeSearcher({
        "start": ["https://one.com"],            # round 1: only 1 source
        "second query": ["https://two.com"],     # round 2 after refine: 1 more
    })
    agent = WebAgent(searcher, FakeFetcher(), FakeSummarizer(),
                     llm=FakeLLM(refine_to="second query"),
                     min_sources=2, max_iterations=3, pages_per_iter=2)
    res = agent.research("start")
    assert searcher.queries == ["start", "second query"]
    assert res.iterations == 2
    assert len(res.sources) == 2


def test_respects_max_iterations():
    # Every round finds only 1 source; min_sources=5 is never met.
    keys = ["start query", "refined one", "refined two", "refined three"]
    searcher = FakeSearcher({q: [f"https://{q.replace(' ', '')}.com"] for q in keys})
    llm = FakeLLM()
    # Make refine cycle through queries.
    seq = iter(["refined one", "refined two", "refined three", "refined four"])
    llm.chat = lambda prompt, system=None, task="chat": (
        next(seq) if "improved search query" in prompt else "answer")
    agent = WebAgent(searcher, FakeFetcher(), FakeSummarizer(),
                     llm=llm, min_sources=5, max_iterations=2)
    res = agent.research("start query")
    assert res.iterations == 2       # stopped at the cap, not chasing forever


def test_without_llm_falls_back_to_stitched_summaries():
    searcher = FakeSearcher({"q": ["https://a.com", "https://b.com"]})
    agent = WebAgent(searcher, FakeFetcher(), FakeSummarizer(), llm=None,
                     min_sources=2)
    res = agent.research("q")
    assert res.success
    assert "what i found" in res.answer.lower()


def test_sources_deduped_across_iterations():
    searcher = FakeSearcher({
        "start": ["https://dup.com"],
        "second query": ["https://dup.com", "https://new.com"],
    })
    agent = WebAgent(searcher, FakeFetcher(), FakeSummarizer(),
                     llm=FakeLLM(refine_to="second query"),
                     min_sources=3, max_iterations=2, pages_per_iter=3)
    res = agent.research("start")
    urls = [s.url for s in res.sources]
    assert urls.count("https://dup.com") == 1
