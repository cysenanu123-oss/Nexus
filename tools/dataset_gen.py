"""
tools/dataset_gen.py
NEXUS Training Dataset Generator

Uses the LLM to synthesize new (instruction, input, output) triples from:
  1. Existing training pairs as seeds
  2. Skill registry descriptions as task templates
  3. Configurable topic seeds

Inspired by LLMs-from-scratch ch07's synthetic data approach.

Usage:
    python tools/dataset_gen.py --count 50
    python tools/dataset_gen.py --count 100 --topics cyber,code,research --out data/synth.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

log = logging.getLogger("nexus.tools.dataset_gen")

_DEFAULT_OUTPUT = Path("data/synth_training.jsonl")
_SEED_PAIRS     = Path("data/training_pairs.jsonl")

_TOPICS = {
    "cyber": [
        "port scanning a target", "enumerating subdomains", "looking up a CVE",
        "searching Exploit-DB", "running nmap scripts", "passive OSINT",
        "analyzing web headers", "brute-forcing a login page", "DNS enumeration",
        "privilege escalation techniques",
    ],
    "code": [
        "writing a Python script", "debugging a function", "refactoring a class",
        "implementing a REST API", "adding error handling", "writing unit tests",
        "using asyncio", "parsing JSON", "working with SQLite", "building a CLI tool",
    ],
    "research": [
        "summarizing a research paper", "explaining a technical concept",
        "comparing two technologies", "looking up how something works",
        "writing a technical overview", "explaining an algorithm",
    ],
    "memory": [
        "storing a fact for later", "recalling something previously saved",
        "looking up a remembered note", "listing recent memories",
    ],
    "assistant": [
        "scheduling a reminder", "setting a calendar event",
        "explaining NEXUS capabilities", "helping plan a project",
        "giving step-by-step instructions",
    ],
}

_GEN_PROMPT = """\
Generate a realistic example of a user question and a helpful NEXUS AI response.

Context: NEXUS is a local-first AI assistant for a cybersecurity and development professional.
Topic area: {topic}
Specific scenario: {scenario}

Write a realistic user message and the ideal NEXUS response.
Respond ONLY with valid JSON in this exact format:
{{"instruction": "<short task description>", "input": "<user message>", "output": "<ideal NEXUS response>"}}

The output should be 2–5 sentences, specific, and technically accurate.
JSON:"""


def _call_llm(prompt: str) -> str:
    """Call Ollama for generation."""
    import urllib.request
    payload = json.dumps({
        "model": "mistral",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 300},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read()).get("response", "")


def _parse_record(raw: str) -> dict | None:
    import re
    m = re.search(r"\{[^}]+\}", raw, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if all(k in obj for k in ("instruction", "input", "output")):
            if obj["output"].strip():
                return obj
    except json.JSONDecodeError:
        pass
    return None


def load_seeds(path: Path, limit: int = 20) -> list[dict]:
    """Load a random sample of existing pairs as style seeds."""
    if not path.exists():
        return []
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                pairs.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
    return random.sample(pairs, min(limit, len(pairs)))


def generate(count: int, topics: list[str], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    failed    = 0

    print(f"Generating {count} synthetic training examples...")
    print(f"Topics: {', '.join(topics)}")

    with open(output_path, "a", encoding="utf-8") as out_f:
        attempts = 0
        while generated < count and attempts < count * 3:
            attempts += 1
            topic    = random.choice(topics)
            scenarios = _TOPICS.get(topic, _TOPICS["assistant"])
            scenario = random.choice(scenarios)

            prompt = _GEN_PROMPT.format(topic=topic, scenario=scenario)

            try:
                raw    = _call_llm(prompt)
                record = _parse_record(raw)
            except Exception as e:
                log.warning("LLM call failed: %s", e)
                failed += 1
                time.sleep(1)
                continue

            if not record:
                failed += 1
                continue

            record["_topic"]  = topic
            record["quality"] = 0.7   # synthetic — mark as medium quality
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            generated += 1

            if generated % 10 == 0:
                print(f"  Generated {generated}/{count}...")

            time.sleep(0.2)

    print(f"\nDone — {generated} examples saved to {output_path}")
    if failed:
        print(f"  (Failed/skipped: {failed})")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s — %(message)s")

    parser = argparse.ArgumentParser(description="Generate synthetic NEXUS training data.")
    parser.add_argument("--count",  type=int, default=50)
    parser.add_argument("--topics", default="cyber,code,research,memory,assistant",
                        help="Comma-separated topic list")
    parser.add_argument("--out",    default=str(_DEFAULT_OUTPUT))
    args = parser.parse_args()

    topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    generate(args.count, topics, Path(args.out))


if __name__ == "__main__":
    main()
