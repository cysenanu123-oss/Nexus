"""
core/llm.py
NEXUS Local LLM Interface — wraps Ollama for all AI responses.

Single entry point for everything that needs a language model:
    - Conversation (mistral)
    - Code help (codellama)
    - Research summarization (mistral)
    - Intent parsing fallback (mistral)

Usage:
    from core.llm import LLM
    llm = LLM()
    response = llm.chat("explain what a buffer overflow is")
    code_help = llm.code("fix this python function", code="def foo():\n  pass")
"""

import logging
import time
from typing import Optional, Iterator

log = logging.getLogger("nexus.llm")

# ─────────────────────────────────────────────
# Available models
# ─────────────────────────────────────────────

MODELS = {
    "chat":     "qwen2.5:1.5b",       # general conversation
    "code":     "qwen2.5-coder:1.5b", # coding assistant
    "fast":     "qwen2.5:1.5b",       # fast responses
    "research": "qwen2.5:1.5b",       # summarization
}

OLLAMA_HOST = "http://localhost:11434"


# ─────────────────────────────────────────────
# LLM Class
# ─────────────────────────────────────────────

class LLM:
    """
    Unified interface to local Ollama models.

    Automatically selects the right model for each task.
    Falls back gracefully if Ollama isn't running.
    """

    def __init__(self, host: str = OLLAMA_HOST):
        self.host    = host
        self._client = None
        self._available_models: list[str] = []
        self._ready  = False
        self._connect()

    def _connect(self):
        try:
            import ollama
            self._client = ollama.Client(host=self.host)
            # Test connection
            models = self._client.list()
            self._available_models = [m.model for m in models.models]
            self._ready = True
            log.info(f"Ollama connected — models: {self._available_models}")
        except ImportError:
            log.warning("ollama package not installed: pip install ollama")
        except Exception as e:
            log.warning(f"Ollama not reachable at {self.host}: {e}")
            log.warning("Start Ollama with: ollama serve")

    @property
    def is_ready(self) -> bool:
        return self._ready

    def _pick_model(self, task: str) -> str:
        """Pick the best available model for a task."""
        preferred = MODELS.get(task, "mistral")

        # Check if preferred is available
        for m in self._available_models:
            if preferred in m:
                return m

        # Fallback to any available model
        if self._available_models:
            return self._available_models[0]

        return preferred  # try anyway

    # ─────────────────────────────────────────────
    # Core chat method
    # ─────────────────────────────────────────────

    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        history: Optional[list[dict]] = None,
        model: Optional[str] = None,
        task: str = "chat",
        stream: bool = False,
    ) -> str:
        """
        Send a message and get a response.

        Parameters
        ----------
        prompt  : user message
        system  : system prompt (sets personality/context)
        history : list of {"role": "user"/"assistant", "content": "..."}
        model   : override model selection
        task    : "chat" | "code" | "research" | "fast"
        stream  : if True, prints tokens as they arrive

        Returns
        -------
        str — full response text
        """
        if not self._ready:
            return self._offline_fallback(prompt)

        selected_model = model or self._pick_model(task)
        messages = []

        # System prompt
        if system:
            messages.append({"role": "system", "content": system})
        else:
            messages.append({
                "role": "system",
                "content": (
                    "You are NEXUS, a sharp, concise AI assistant running locally "
                    "on a Linux machine owned by Cyril — a Telecom Engineering student "
                    "and developer in Ghana who works on cybersecurity, "
                    "full-stack development, and AI projects. "
                    "Be direct and helpful. Never pad responses unnecessarily."
                )
            })

        # Conversation history
        if history:
            messages.extend(history)

        # Current message
        messages.append({"role": "user", "content": prompt})

        try:
            t0 = time.time()
            log.info(f"LLM request → {selected_model} ({task})")

            if stream:
                return self._stream_response(selected_model, messages)

            response = self._client.chat(
                model=selected_model,
                messages=messages,
            )
            text    = response.message.content.strip()
            elapsed = time.time() - t0
            log.info(f"LLM response in {elapsed:.2f}s ({len(text)} chars)")
            return text

        except Exception as e:
            log.error(f"LLM chat failed: {e}")
            return f"LLM error: {e}"

    def _stream_response(self, model: str, messages: list[dict]) -> str:
        """Stream tokens to terminal, return full text."""
        full = []
        try:
            for chunk in self._client.chat(
                model=model,
                messages=messages,
                stream=True,
            ):
                token = chunk.message.content
                print(token, end="", flush=True)
                full.append(token)
            print()  # newline after stream
        except Exception as e:
            log.error(f"Stream failed: {e}")
        return "".join(full)

    # ─────────────────────────────────────────────
    # Specialized methods
    # ─────────────────────────────────────────────

    def code(
        self,
        instruction: str,
        code: Optional[str] = None,
        language: str = "python",
    ) -> str:
        """
        Ask codellama for coding help.

        Parameters
        ----------
        instruction : what you want (explain, fix, improve, write)
        code        : the code to work on (optional)
        language    : programming language hint
        """
        system = (
            f"You are an expert {language} programmer. "
            "Give concise, working code. "
            "When fixing bugs, explain what was wrong in one line. "
            "When explaining, be brief and technical. "
            "Output only what's needed — no padding."
        )

        if code:
            prompt = f"{instruction}\n\n```{language}\n{code}\n```"
        else:
            prompt = instruction

        return self.chat(prompt, system=system, task="code")

    def summarize(self, text: str, topic: str = "") -> str:
        """
        Summarize a block of text (for research module).

        Parameters
        ----------
        text  : content to summarize
        topic : what we were researching (for context)
        """
        system = (
            "You are a research assistant. "
            "Summarize the given content concisely. "
            "Extract key facts, concepts, and actionable information. "
            "Use bullet points. Be thorough but avoid fluff."
        )

        topic_ctx = f"Topic: {topic}\n\n" if topic else ""
        prompt    = f"{topic_ctx}Summarize this:\n\n{text[:4000]}"

        return self.chat(prompt, system=system, task="research")

    def explain(self, concept: str, level: str = "technical") -> str:
        """
        Explain a concept — used by the cybersecurity and coding modules.

        Parameters
        ----------
        concept : what to explain
        level   : "simple" | "technical" | "expert"
        """
        system = (
            f"Explain at a {level} level. "
            "Be accurate and direct. "
            "Use examples where helpful. "
            "Keep it under 200 words unless complexity demands more."
        )
        return self.chat(f"Explain: {concept}", system=system)

    def ask(self, question: str, context: str = "") -> str:
        """
        General question answering with optional context.
        Used when screen content or documents are involved.
        """
        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {question}"
        else:
            prompt = question
        return self.chat(prompt)

    def classify_intent(self, text: str) -> dict:
        """
        Use LLM as intent classifier fallback.
        Returns dict with intent, target, confidence.
        """
        import json

        system = (
            "You are an intent classifier for an AI assistant. "
            "Parse the user input and return ONLY a JSON object with keys: "
            "intent (snake_case string), target (string or null), "
            "action (string or null), query (string or null), confidence (0-1 float). "
            "No explanation. No markdown. Just JSON."
        )

        try:
            raw = self.chat(text, system=system, task="fast")
            # Strip markdown fences if present
            raw = raw.strip().strip("```json").strip("```").strip()
            return json.loads(raw)
        except Exception as e:
            log.warning(f"Intent classification failed: {e}")
            return {"intent": "unknown", "confidence": 0.0}

    # ─────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────

    def available_models(self) -> list[str]:
        return list(self._available_models)

    def pull_model(self, model_name: str) -> bool:
        """Download a model if not already available."""
        if not self._ready:
            return False
        try:
            print(f"Pulling {model_name}... (this may take a while)")
            self._client.pull(model_name)
            self._connect()  # refresh model list
            return True
        except Exception as e:
            log.error(f"Pull failed: {e}")
            return False

    def status(self) -> dict:
        """Return current LLM status."""
        return {
            "ready":   self._ready,
            "host":    self.host,
            "models":  self._available_models,
            "chat":    self._pick_model("chat"),
            "code":    self._pick_model("code"),
        }

    def _offline_fallback(self, prompt: str) -> str:
        """Response when Ollama isn't available."""
        return (
            "LLM offline. Start Ollama with: ollama serve\n"
            "Then pull models: ollama pull mistral && ollama pull codellama"
        )


# ─────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────

_llm_instance: Optional[LLM] = None


def get_llm() -> LLM:
    """Return the shared LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLM()
    return _llm_instance


# ─────────────────────────────────────────────
# CLI test — python core/llm.py
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    llm = LLM()

    print("\n─── NEXUS LLM Test ───\n")
    print(f"Status  : {'✓ Online' if llm.is_ready else '✗ Offline'}")
    print(f"Models  : {llm.available_models()}")
    print(f"Chat    : {llm._pick_model('chat')}")
    print(f"Code    : {llm._pick_model('code')}")

    if not llm.is_ready:
        print("\nStart Ollama: ollama serve")
        sys.exit(1)

    if "--chat" in sys.argv:
        idx = sys.argv.index("--chat")
        q   = " ".join(sys.argv[idx + 1:])
        print(f"\nYou: {q}\n")
        print(f"NEXUS: {llm.chat(q)}\n")

    elif "--code" in sys.argv:
        print("\nCode help test:\n")
        result = llm.code(
            "explain what this does",
            code="def fib(n): return n if n<=1 else fib(n-1)+fib(n-2)",
        )
        print(f"NEXUS: {result}\n")

    elif "--summarize" in sys.argv:
        text = "Python is a high-level programming language known for its simple syntax. It supports multiple programming paradigms including procedural, object-oriented, and functional programming."
        print("\nSummarize test:\n")
        print(f"NEXUS: {llm.summarize(text, topic='Python')}\n")

    else:
        print("\nUsage:")
        print("  python core/llm.py --chat what is a buffer overflow")
        print("  python core/llm.py --code")
        print("  python core/llm.py --summarize")
        print()

        # Quick interactive test
        print("Quick test — asking mistral something...\n")
        response = llm.chat("In one sentence, what is NEXUS?")
        print(f"NEXUS: {response}\n")