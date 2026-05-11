"""
core/coding_assistant.py
NEXUS Coding Assistant — screen-aware code help powered by codellama.

Reads code from screen via OCR, sends to local LLM, returns help.

Usage:
    from core.coding_assistant import CodingAssistant
    ca = CodingAssistant()
    ca.explain_screen()   # explain code currently on screen
    ca.fix_screen()       # fix bugs in code on screen
    ca.ask("how do I reverse a string in python")
"""

import logging
import re
from typing import Optional

log = logging.getLogger("nexus.coding")


class CodingAssistant:
    """
    Screen-aware coding assistant.

    Pipeline:
        User asks → Vision reads screen → OCR extracts code
        → LLM (codellama) analyzes → Response returned
    """

    def __init__(self):
        from core.llm import get_llm
        self._llm    = get_llm()
        self._vision = None
        log.info("CodingAssistant ready.")

    def _get_vision(self):
        if self._vision is None:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from vision.vision import Vision
            self._vision = Vision()
        return self._vision

    # ─────────────────────────────────────────────
    # Screen-based operations
    # ─────────────────────────────────────────────

    def read_code_from_screen(self) -> str:
        """Extract code from the current screen."""
        try:
            vision = self._get_vision()
            text   = vision.read_screen()
            return self._extract_code_blocks(text)
        except Exception as e:
            log.error(f"Screen read failed: {e}")
            return ""

    def _extract_code_blocks(self, text: str) -> str:
        """
        Try to extract code from OCR text.
        Looks for indentation patterns, function defs, imports, etc.
        """
        lines = text.splitlines()
        code_lines = []
        in_code = False

        for line in lines:
            stripped = line.strip()

            # Code indicators
            is_code = (
                stripped.startswith(("def ", "class ", "import ", "from ",
                                     "if ", "for ", "while ", "return ",
                                     "try:", "except", "#", "//", "/*",
                                     "public ", "private ", "function "))
                or (len(line) > 0 and line[0] in " \t" and stripped)
                or re.match(r"^\s*[\w.]+\s*[=({\[<]", line)
            )

            if is_code:
                in_code = True

            if in_code or is_code:
                code_lines.append(line)

        return "\n".join(code_lines) if code_lines else text

    def explain_screen(self) -> str:
        """Explain the code currently visible on screen."""
        code = self.read_code_from_screen()
        if not code:
            return "No code detected on screen."

        log.info("Explaining code from screen...")
        return self._llm.code(
            "Explain what this code does, step by step.",
            code=code,
        )

    def fix_screen(self) -> str:
        """Find and fix bugs in code currently on screen."""
        code = self.read_code_from_screen()
        if not code:
            return "No code detected on screen."

        log.info("Looking for bugs in screen code...")
        return self._llm.code(
            "Find bugs or issues in this code and provide the fixed version.",
            code=code,
        )

    def improve_screen(self) -> str:
        """Suggest improvements for code on screen."""
        code = self.read_code_from_screen()
        if not code:
            return "No code detected on screen."

        return self._llm.code(
            "Suggest improvements for this code — performance, readability, or best practices.",
            code=code,
        )

    def document_screen(self) -> str:
        """Generate docstrings/comments for code on screen."""
        code = self.read_code_from_screen()
        if not code:
            return "No code detected on screen."

        return self._llm.code(
            "Add proper docstrings and inline comments to this code.",
            code=code,
        )

    # ─────────────────────────────────────────────
    # Direct questions
    # ─────────────────────────────────────────────

    def ask(self, question: str, language: str = "python") -> str:
        """Ask a coding question without screen context."""
        return self._llm.code(question, language=language)

    def write(self, description: str, language: str = "python") -> str:
        """Write code from a description."""
        return self._llm.code(
            f"Write {language} code to: {description}",
            language=language,
        )

    def explain_error(self, error: str) -> str:
        """Explain an error message and how to fix it."""
        code_on_screen = self.read_code_from_screen()
        context = f"Code:\n{code_on_screen}\n\n" if code_on_screen else ""
        return self._llm.code(
            f"{context}Explain this error and how to fix it:\n{error}"
        )

    def convert(self, code: str, from_lang: str, to_lang: str) -> str:
        """Convert code from one language to another."""
        return self._llm.code(
            f"Convert this {from_lang} code to {to_lang}:",
            code=code,
            language=to_lang,
        )

    def review(self, code: str) -> str:
        """Full code review — security, performance, style."""
        return self._llm.code(
            "Do a full code review. Check for: security issues, "
            "performance problems, style issues, and potential bugs.",
            code=code,
        )


# ─────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Fix module path for standalone execution
    root_dir = str(Path(__file__).resolve().parent.parent)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    logging.basicConfig(level=logging.INFO)

    ca = CodingAssistant()
    print("\n─── NEXUS Coding Assistant ───\n")

    if "--screen" in sys.argv:
        print("Reading code from screen...\n")
        print(ca.explain_screen())

    elif "--ask" in sys.argv:
        idx = sys.argv.index("--ask")
        q   = " ".join(sys.argv[idx + 1:])
        print(f"Q: {q}\n")
        print(ca.ask(q))

    elif "--write" in sys.argv:
        idx = sys.argv.index("--write")
        d   = " ".join(sys.argv[idx + 1:])
        print(f"Writing: {d}\n")
        print(ca.write(d))

    elif "--error" in sys.argv:
        idx = sys.argv.index("--error")
        err = " ".join(sys.argv[idx + 1:])
        print(f"Error: {err}\n")
        print(ca.explain_error(err))

    else:
        print("Usage:")
        print("  python core/coding_assistant.py --screen        # explain code on screen")
        print("  python core/coding_assistant.py --ask <q>       # ask coding question")
        print("  python core/coding_assistant.py --write <desc>  # write code")
        print("  python core/coding_assistant.py --error <msg>   # explain error")
        print()
        print("Quick test:\n")
        result = ca.ask("write a python one-liner to reverse a string")
        print(f"NEXUS: {result}\n")