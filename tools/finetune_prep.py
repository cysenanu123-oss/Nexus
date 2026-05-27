"""
tools/finetune_prep.py
NEXUS Fine-Tuning Data Preparer

Converts NEXUS's collected training_pairs.jsonl into Alpaca instruction-format
JSON suitable for fine-tuning with LLMs-from-scratch ch07's pipeline or any
standard SFT trainer (unsloth, axolotl, HuggingFace TRL).

Output format (one JSON object per line):
    {
        "instruction": "<task description>",
        "input": "<optional context, empty string if none>",
        "output": "<expected response>"
    }

Usage:
    python tools/finetune_prep.py
    python tools/finetune_prep.py --input data/training_pairs.jsonl --out data/finetune_dataset.json --min-quality 0.7
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("nexus.tools.finetune_prep")

_DEFAULT_INPUT  = Path("data/training_pairs.jsonl")
_DEFAULT_OUTPUT = Path("data/finetune_dataset.json")
_DEFAULT_MIN_Q  = 0.5    # minimum quality score to include (0.0–1.0)


# ── Instruction templates per intent type ────────────────────────────────────

_INTENT_TEMPLATES = {
    "cyber_scan":      "Perform a cybersecurity scan or reconnaissance task.",
    "cyber_exploit":   "Analyze or exploit a security vulnerability.",
    "code":            "Write or fix code as requested.",
    "research":        "Research the following topic and provide a detailed answer.",
    "memory_store":    "Remember the following fact for future reference.",
    "memory_query":    "Recall information from memory.",
    "calendar":        "Create or manage a calendar event or reminder.",
    "automation":      "Automate the following system task.",
    "conversation":    "Respond to the following message naturally.",
    "autonomous_plan": "Plan and execute the following multi-step task.",
}

_DEFAULT_INSTRUCTION = "Respond to the following user request as NEXUS, a local-first AI assistant."


def _intent_to_instruction(intent: str) -> str:
    for key, template in _INTENT_TEMPLATES.items():
        if key in intent:
            return template
    return _DEFAULT_INSTRUCTION


def _build_alpaca_record(pair: dict) -> dict | None:
    """Convert a raw training pair to Alpaca format."""
    user   = (pair.get("user_input") or pair.get("user") or "").strip()
    nexus  = (pair.get("nexus_response") or pair.get("nexus") or pair.get("response") or "").strip()
    intent = pair.get("intent") or pair.get("category") or ""

    if not user or not nexus:
        return None
    if len(nexus) < 5:
        return None

    instruction = _intent_to_instruction(intent)

    # If user input is short and generic, roll it into the instruction
    if len(user) < 80:
        return {
            "instruction": user,
            "input":       "",
            "output":      nexus,
        }

    return {
        "instruction": instruction,
        "input":       user,
        "output":      nexus,
    }


def load_pairs(path: Path, min_quality: float) -> list[dict]:
    pairs = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("Line %d: JSON error — %s", line_no, e)
                continue

            quality = obj.get("quality", 0.5)
            if quality < min_quality:
                skipped += 1
                continue

            record = _build_alpaca_record(obj)
            if record:
                pairs.append(record)
            else:
                skipped += 1

    log.info("Loaded %d pairs (skipped %d below quality %.1f or empty).",
             len(pairs), skipped, min_quality)
    return pairs


def deduplicate(pairs: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for p in pairs:
        key = (p["instruction"][:100], p["input"][:100])
        if key not in seen:
            seen.add(key)
            result.append(p)
    removed = len(pairs) - len(result)
    if removed:
        log.info("Deduplication removed %d duplicates.", removed)
    return result


def save(pairs: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    log.info("Saved %d training examples to %s", len(pairs), path)


def stats(pairs: list[dict]):
    if not pairs:
        print("No examples to summarize.")
        return
    instruction_lens = [len(p["instruction"]) for p in pairs]
    output_lens      = [len(p["output"]) for p in pairs]
    with_input       = sum(1 for p in pairs if p["input"])

    print(f"\nDataset statistics:")
    print(f"  Total examples : {len(pairs)}")
    print(f"  With context   : {with_input}")
    print(f"  Instruction len: avg={sum(instruction_lens)//len(pairs)}, max={max(instruction_lens)}")
    print(f"  Output len     : avg={sum(output_lens)//len(pairs)}, max={max(output_lens)}")

    # Show a sample
    print(f"\nSample record:")
    sample = pairs[0]
    print(f"  instruction: {sample['instruction'][:80]}")
    print(f"  input:       {sample['input'][:80] or '(empty)'}")
    print(f"  output:      {sample['output'][:120]}")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s — %(message)s")

    parser = argparse.ArgumentParser(description="Prepare NEXUS fine-tuning dataset.")
    parser.add_argument("--input",   default=str(_DEFAULT_INPUT),
                        help="Path to training_pairs.jsonl")
    parser.add_argument("--out",     default=str(_DEFAULT_OUTPUT),
                        help="Output JSON file path")
    parser.add_argument("--min-quality", type=float, default=_DEFAULT_MIN_Q,
                        help="Minimum quality score (0.0–1.0, default 0.5)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Skip deduplication")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.out)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    pairs = load_pairs(input_path, args.min_quality)
    if not args.no_dedup:
        pairs = deduplicate(pairs)

    if not pairs:
        print("No valid training pairs found after filtering.", file=sys.stderr)
        sys.exit(1)

    save(pairs, output_path)
    stats(pairs)
    print(f"\nDataset ready: {output_path}")


if __name__ == "__main__":
    main()
