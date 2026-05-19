"""
core/skill_acquirer.py
NEXUS Skill Acquirer — learns new skills from GitHub repos and web pages.

Workflow for a GitHub repo:
  1. git clone --depth=1 <url>  (or git pull if already cloned)
  2. Read README for high-level context
  3. AST-parse all non-test Python files → extract public functions with docstrings
  4. Ask LLM to group those functions into high-level skills
  5. Register each skill in the SkillRegistry
  6. Source files live in  data/skills/acquired/<repo-name>/

Workflow for a single .py file:
  Fetch → save → parse → register (same end result, shorter path)

Workflow for a generic URL (docs, README, tutorial):
  Fetch with researcher → LLM extracts skill descriptions → register
"""

from __future__ import annotations

import ast
import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("nexus.skill_acquirer")

_ACQUIRED_DIR = Path(__file__).parent.parent / "data" / "skills" / "acquired"

# Files we skip when scanning a repo
_SKIP_PATTERNS = {
    "test_", "_test.", "setup.py", "conftest", "migration",
    "__pycache__", ".git", "venv", "env/", "node_modules",
}


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

@dataclass
class FunctionInfo:
    name:      str
    docstring: str
    args:      list[str]
    file_path: str
    line:      int


# ─────────────────────────────────────────────────────────────
#  ACQUIRER
# ─────────────────────────────────────────────────────────────

class SkillAcquirer:
    """Learns skills from external sources and registers them."""

    def __init__(self, llm=None, researcher=None):
        self._llm        = llm
        self._researcher = researcher

        try:
            from core.stream_output import get_output
            self._out = get_output()
        except ImportError:
            self._out = None

        _ACQUIRED_DIR.mkdir(parents=True, exist_ok=True)

    def _say(self, method: str, *args, **kwargs):
        if self._out:
            getattr(self._out, method, self._out.thinking)(*args, **kwargs)

    # ── Public API ────────────────────────────────────────────

    def acquire(self, url: str) -> list:
        """
        Auto-detect URL type and acquire skills from it.
        Returns list of registered Skill objects.
        """
        self._say("thinking", f"Acquiring skills from: {url}")

        url = url.strip().rstrip("/")
        if "github.com" in url and not url.endswith(".py"):
            return self._from_github_repo(url)
        if url.endswith(".py") or "raw.githubusercontent.com" in url:
            return self._from_python_file(url)
        return self._from_webpage(url)

    # ── GitHub repo ───────────────────────────────────────────

    def _from_github_repo(self, repo_url: str) -> list:
        repo_url  = re.sub(r"\.git$", "", repo_url)
        repo_name = repo_url.rstrip("/").split("/")[-1]
        dest      = _ACQUIRED_DIR / repo_name

        if dest.exists():
            self._say("running", f"Updating existing clone of {repo_name}")
            subprocess.run(
                ["git", "-C", str(dest), "pull", "--quiet"],
                capture_output=True, timeout=60,
            )
        else:
            self._say("running", f"Cloning {repo_name} ...")
            result = subprocess.run(
                ["git", "clone", "--depth=1", "--quiet", repo_url, str(dest)],
                capture_output=True, text=True, timeout=180,
            )
            if result.returncode != 0:
                self._say("fail", f"Clone failed: {result.stderr.strip()[:200]}")
                return []

        self._say("reading", f"Scanning Python files in {repo_name}")

        readme    = self._read_readme(dest)
        py_files  = [
            f for f in dest.rglob("*.py")
            if not any(skip in str(f) for skip in _SKIP_PATTERNS)
        ][:40]

        self._say("thinking", f"Analyzing {len(py_files)} Python files...")
        functions: list[FunctionInfo] = []
        for pf in py_files:
            functions.extend(self._extract_functions(pf))

        self._say("thinking", f"Extracted {len(functions)} functions — summarizing into skills")
        skills = self._summarize_skills(functions, readme, repo_name, repo_url, dest)

        from core.skill_registry import get_registry
        reg = get_registry()
        for s in skills:
            reg.register(s)

        self._say("learned", f"Acquired {len(skills)} skill(s) from {repo_name}")
        return skills

    # ── Single Python file ────────────────────────────────────

    def _from_python_file(self, url: str) -> list:
        import urllib.request
        try:
            self._say("searching", f"Fetching {url}")
            with urllib.request.urlopen(url, timeout=30) as resp:
                code = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            self._say("fail", f"Fetch failed: {exc}")
            return []

        filename = url.split("/")[-1].split("?")[0]
        dest = _ACQUIRED_DIR / "individual" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(code, encoding="utf-8")

        functions = self._extract_functions(dest)
        skills    = self._summarize_skills(functions, "", filename, url, dest.parent)

        from core.skill_registry import get_registry
        reg = get_registry()
        for s in skills:
            reg.register(s)

        self._say("learned", f"Acquired {len(skills)} skill(s) from {filename}")
        return skills

    # ── Webpage / docs ────────────────────────────────────────

    def _from_webpage(self, url: str) -> list:
        if not self._researcher:
            self._say("fail", "Researcher not available — can't fetch webpage")
            return []

        self._say("searching", f"Reading page: {url}")
        try:
            content = self._researcher.fetch(url)
        except Exception as exc:
            self._say("fail", f"Could not fetch {url}: {exc}")
            return []

        if not content or len(content) < 100:
            return []

        if not (self._llm and self._llm.is_ready):
            self._say("fail", "LLM not available — can't parse webpage into skills")
            return []

        self._say("thinking", "Extracting skill knowledge from page content...")

        prompt = (
            f"Analyze this page and extract usable skills for an AI assistant.\n"
            f"Page content:\n{content[:3000]}\n\n"
            "Return a JSON array of up to 6 skills:\n"
            '[{"name":"snake_case","description":"one sentence","category":'
            '"cyber|research|code|system|communication|utility","tags":["tag"],'
            '"usage_example":"natural language example"}]'
        )
        try:
            raw  = self._llm.chat(prompt, task="fast")
            raw  = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            items = json.loads(raw)
        except Exception as exc:
            self._say("fail", f"LLM extraction failed: {exc}")
            return []

        from core.skill_registry import Skill, get_registry
        reg    = get_registry()
        skills = []
        for item in items:
            name = item.get("name", "unnamed").lower().replace(" ", "_")[:50]
            s = Skill(
                name          = name,
                description   = item.get("description", "")[:300],
                category      = item.get("category", "utility"),
                source        = f"learned:{url}",
                tags          = item.get("tags", []),
                usage_example = item.get("usage_example", ""),
            )
            reg.register(s)
            skills.append(s)

        self._say("learned", f"Extracted {len(skills)} skill(s) from webpage")
        return skills

    # ── AST extraction ────────────────────────────────────────

    def _extract_functions(self, file_path: Path) -> list[FunctionInfo]:
        """Parse a Python file and return public functions that have docstrings."""
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            tree   = ast.parse(source)
        except Exception:
            return []

        funcs = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_") or len(node.name) < 4:
                continue
            docstring = ast.get_docstring(node) or ""
            if len(docstring) < 15:
                continue
            args = [a.arg for a in node.args.args if a.arg not in ("self", "cls")]
            funcs.append(FunctionInfo(
                name      = node.name,
                docstring = docstring[:400],
                args      = args,
                file_path = str(file_path),
                line      = node.lineno,
            ))
        return funcs

    # ── LLM summarization ─────────────────────────────────────

    def _summarize_skills(
        self,
        funcs:     list[FunctionInfo],
        readme:    str,
        repo_name: str,
        source_url: str,
        repo_path: Path,
    ) -> list:
        """Convert raw functions into high-level Skill objects, using LLM if available."""
        from core.skill_registry import Skill

        if not funcs:
            return []

        # Try LLM grouping first
        if self._llm and self._llm.is_ready:
            skills = self._llm_group(funcs, readme, repo_name, source_url, repo_path)
            if skills:
                return skills

        # Fallback: direct 1:1 from docstrings (no LLM)
        skills = []
        for fn in funcs[:20]:
            try:
                rel = str(Path(fn.file_path).relative_to(repo_path.parent if repo_path.is_dir() else repo_path))
            except ValueError:
                rel = fn.file_path
            s = Skill(
                name          = f"{repo_name}_{fn.name}"[:60].lower().replace("-", "_"),
                description   = fn.docstring.split("\n")[0][:150],
                category      = "utility",
                source        = f"github:{source_url}",
                tags          = [repo_name],
                usage_example = f"{fn.name}({', '.join(fn.args)})",
                code_path     = rel,
                invoke_fn     = fn.name,
            )
            skills.append(s)
        return skills

    def _llm_group(
        self,
        funcs:     list[FunctionInfo],
        readme:    str,
        repo_name: str,
        source_url: str,
        repo_path: Path,
    ) -> list:
        """Ask LLM to cluster functions into meaningful, high-level skills."""
        from core.skill_registry import Skill

        fn_block = "\n".join(
            f"{i+1}. {fn.name}({', '.join(fn.args)}): {fn.docstring[:180]}"
            for i, fn in enumerate(funcs[:50])
        )
        readme_snip = readme[:800] if readme else "No README."

        prompt = (
            f"Repository: {repo_name}\n"
            f"README excerpt: {readme_snip}\n\n"
            f"Functions:\n{fn_block}\n\n"
            "Group these into at most 10 high-level skills an AI assistant can use.\n"
            "Each skill should represent a meaningful capability, not just a function.\n"
            "Return ONLY a JSON array:\n"
            '[{"name":"snake_case_name","description":"one clear sentence",'
            '"category":"cyber|research|code|system|communication|utility",'
            '"tags":["tag1","tag2"],"usage_example":"natural language usage",'
            '"invoke_fn":"primary_function_name"}]'
        )

        try:
            raw   = self._llm.chat(prompt, task="fast")
            raw   = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            items = json.loads(raw)
        except Exception as exc:
            log.warning("LLM skill grouping failed: %s", exc)
            return []

        skills = []
        for item in items:
            invoke_fn = item.get("invoke_fn", "")
            code_path = ""
            for fn in funcs:
                if fn.name == invoke_fn:
                    try:
                        code_path = str(Path(fn.file_path).relative_to(repo_path.parent))
                    except ValueError:
                        code_path = fn.file_path
                    break

            slug = f"{repo_name}_{item.get('name','skill')}"[:60].lower().replace("-","_")
            s = Skill(
                name          = slug,
                description   = item.get("description","")[:300],
                category      = item.get("category","utility"),
                source        = f"github:{source_url}",
                tags          = item.get("tags", [repo_name]),
                usage_example = item.get("usage_example",""),
                code_path     = code_path,
                invoke_fn     = invoke_fn,
                invoke_module = repo_name.replace("-","_"),
            )
            skills.append(s)
        return skills

    # ── Helpers ───────────────────────────────────────────────

    def _read_readme(self, repo_path: Path) -> str:
        for name in ["README.md", "README.rst", "README.txt", "README"]:
            p = repo_path / name
            if p.exists():
                return p.read_text(encoding="utf-8", errors="replace")[:2000]
        return ""


# ── Singleton ──────────────────────────────────────────────────

_acquirer: Optional[SkillAcquirer] = None


def get_acquirer(llm=None, researcher=None) -> SkillAcquirer:
    global _acquirer
    if _acquirer is None:
        _acquirer = SkillAcquirer(llm=llm, researcher=researcher)
    return _acquirer
