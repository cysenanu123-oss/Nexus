"""
test_brain.py
Interactive test loop for the NEXUS Brain.
Type commands and get responses. Also lets you mark good/bad responses
to build your training dataset.

Run:
    python test_brain.py
"""

import logging
import os

# Silence lower-level logs during interactive test
logging.basicConfig(level=logging.WARNING)

from core.brain import Brain

HELP = """
Commands:
  <anything>     → NEXUS thinks and responds
  !good          → mark last response as good (training data)
  !bad <answer>  → mark last response as bad, optionally provide correct answer
  !history       → show conversation so far
  !memory        → show stored long-term memory
  !clear         → clear session context
  !help          → show this message
  exit / quit    → exit
"""

def main():
    os.makedirs("data", exist_ok=True)

    print("\n" + "─"*50)
    print("  NEXUS Brain Test Interface")
    print("  Training mode active — use !good / !bad")
    print("─"*50)
    print(HELP)

    brain = Brain(user_name="Cyril", use_llm=True)

    while True:
        try:
            user_input = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  NEXUS offline.\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("\n  NEXUS: Later, Cyril.\n")
            break

        # ── Meta commands ────────────────────────────────────────────────
        if user_input == "!good":
            print(f"  NEXUS: {brain.mark_good()}")
            continue

        if user_input.startswith("!bad"):
            correction = user_input[4:].strip()
            print(f"  NEXUS: {brain.mark_bad(correction)}")
            continue

        if user_input == "!history":
            print("\n  " + brain.context.summary().replace("\n", "\n  "))
            continue

        if user_input == "!memory":
            all_mem = brain.memory.recall_by_category("user_note")
            if all_mem:
                for m in all_mem:
                    print(f"  • {m['key']}: {m['value']}")
            else:
                print("  (memory empty)")
            continue

        if user_input == "!clear":
            brain.context.clear()
            print("  Context cleared.")
            continue

        if user_input == "!help":
            print(HELP)
            continue

        # ── Normal input ─────────────────────────────────────────────────
        response = brain.think(user_input)
        print(f"\n  NEXUS: {response}")

if __name__ == "__main__":
    main()
