"""
core/skill_manager.py
NEXUS Skill Manager — learn, store, and reuse skills across sessions.

A "skill" is anything NEXUS can learn and reuse:
    - A Python function or script
    - A bash command sequence
    - A workflow (do X then Y then Z)
    - A prompt template for a specific task
    - A cybersecurity technique

Skills are stored locally in data/skills/ as JSON files.
NEXUS can call them by name, list them, and apply them.

Usage:
    from core.skill_manager import SkillManager
    sm = SkillManager()

    # Save a skill
    sm.save_skill("port_scan", code="nmap -sV {target}", language="bash",
                  description="Quick port scan with version detection")

    # Use a skill
    result = sm.run_skill("port_scan", target="192.168.1.1")

    # List skills
    sm.list_skills()

    # Extract a skill from text/code the user pastes
    sm.extract_and_save(raw_text, name="my_script")
"""

from __future__ import annotations

import json
import time
import logging
import subprocess
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any

log = logging.getLogger("nexus.skills")

SKILLS_DIR = Path("data/skills")
SKILLS_INDEX = SKILLS_DIR / "index.json"


# ─────────────────────────────────────────────────────────────
#  SKILL DATA CLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class Skill:
    """
    A stored, reusable skill.

    Attributes
    ----------
    name        : unique identifier (e.g. "port_scan")
    description : what it does
    language    : "python" | "bash" | "prompt" | "workflow"
    code        : the actual code, command, or prompt template
    parameters  : list of {name, description, required} dicts
    tags        : list of category tags (e.g. ["cyber", "network"])
    author      : "nexus" or "user"
    created_at  : timestamp
    use_count   : how many times it's been called
    last_used   : timestamp of last use
    """
    name:        str
    description: str
    language:    str
    code:        str
    parameters:  list[dict] = field(default_factory=list)
    tags:        list[str]  = field(default_factory=list)
    author:      str        = "user"
    created_at:  float      = field(default_factory=time.time)
    use_count:   int        = 0
    last_used:   float      = 0.0
    skill_id:    str        = ""

    def __post_init__(self):
        if not self.skill_id:
            self.skill_id = hashlib.md5(
                f"{self.name}{self.created_at}".encode()
            ).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "skill_id":    self.skill_id,
            "name":        self.name,
            "description": self.description,
            "language":    self.language,
            "code":        self.code,
            "parameters":  self.parameters,
            "tags":        self.tags,
            "author":      self.author,
            "created_at":  self.created_at,
            "use_count":   self.use_count,
            "last_used":   self.last_used,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Skill":
        return cls(
            skill_id    = d.get("skill_id", ""),
            name        = d["name"],
            description = d.get("description", ""),
            language    = d.get("language", "python"),
            code        = d.get("code", ""),
            parameters  = d.get("parameters", []),
            tags        = d.get("tags", []),
            author      = d.get("author", "user"),
            created_at  = d.get("created_at", time.time()),
            use_count   = d.get("use_count", 0),
            last_used   = d.get("last_used", 0.0),
        )

    def __str__(self) -> str:
        return (
            f"Skill({self.name!r}, lang={self.language}, "
            f"tags={self.tags}, used={self.use_count}x)"
        )


# ─────────────────────────────────────────────────────────────
#  SKILL RUN RESULT
# ─────────────────────────────────────────────────────────────

@dataclass
class SkillResult:
    """Result of running a skill."""
    skill_name: str
    success:    bool
    output:     str
    error:      str   = ""
    elapsed:    float = 0.0

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"[{status}] {self.skill_name}: {self.output[:200]}"


# ─────────────────────────────────────────────────────────────
#  SKILL MANAGER
# ─────────────────────────────────────────────────────────────

class SkillManager:
    """
    Manages the NEXUS skill library.

    A skill is learned once and reused forever.
    Skills can be Python code, bash commands, or prompt templates.

    Usage:
        sm = SkillManager()

        # Add a skill manually
        sm.save_skill(
            name="port_scan",
            description="Scan a target for open ports",
            language="bash",
            code="nmap -sV -T4 {target}",
            parameters=[{"name": "target", "description": "IP or hostname", "required": True}],
            tags=["cyber", "network"],
        )

        # Run a skill
        result = sm.run_skill("port_scan", target="192.168.1.1")
        print(result.output)

        # Extract skill from pasted code
        sm.extract_and_save("def reverse_string(s): return s[::-1]", name="reverse_string")

        # List all skills
        for skill in sm.list_skills():
            print(skill)
    """

    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._skills: dict[str, Skill] = {}
        self._load_all()
        log.info("SkillManager ready — %d skills loaded.", len(self._skills))

    # ── CRUD ──────────────────────────────────────────────────

    def save_skill(
        self,
        name:        str,
        code:        str,
        description: str       = "",
        language:    str       = "python",
        parameters:  list      = None,
        tags:        list[str] = None,
        author:      str       = "user",
        overwrite:   bool      = False,
    ) -> Skill:
        """
        Save a new skill to the library.

        Parameters
        ----------
        name        : unique skill name (snake_case recommended)
        code        : the code, command, or prompt
        description : what this skill does
        language    : "python" | "bash" | "prompt" | "workflow"
        parameters  : list of parameter dicts
        tags        : category tags
        author      : "user" or "nexus"
        overwrite   : replace existing skill with same name

        Returns
        -------
        Skill object
        """
        name = name.lower().strip().replace(" ", "_")

        if name in self._skills and not overwrite:
            log.warning("Skill %r already exists. Use overwrite=True to replace.", name)
            return self._skills[name]

        skill = Skill(
            name        = name,
            description = description or f"Skill: {name}",
            language    = language,
            code        = code,
            parameters  = parameters or [],
            tags        = tags or [],
            author      = author,
        )

        self._skills[name] = skill
        self._save_skill_file(skill)
        self._save_index()

        log.info("Skill saved: %s (%s)", name, language)
        return skill

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self._skills.get(name.lower().strip())

    def delete_skill(self, name: str) -> bool:
        """Delete a skill."""
        name = name.lower().strip()
        if name not in self._skills:
            return False

        skill = self._skills.pop(name)
        skill_file = self.skills_dir / f"{name}.json"
        if skill_file.exists():
            skill_file.unlink()

        self._save_index()
        log.info("Skill deleted: %s", name)
        return True

    def list_skills(
        self,
        tag: Optional[str] = None,
        language: Optional[str] = None,
    ) -> list[Skill]:
        """
        List all skills, optionally filtered.

        Parameters
        ----------
        tag      : filter by tag (e.g. "cyber")
        language : filter by language (e.g. "bash")
        """
        skills = list(self._skills.values())

        if tag:
            skills = [s for s in skills if tag.lower() in s.tags]
        if language:
            skills = [s for s in skills if s.language == language.lower()]

        return sorted(skills, key=lambda s: s.use_count, reverse=True)

    def search_skills(self, query: str) -> list[Skill]:
        """Search skills by name, description, or tags."""
        query = query.lower()
        results = []
        for skill in self._skills.values():
            if (query in skill.name or
                query in skill.description.lower() or
                any(query in tag for tag in skill.tags)):
                results.append(skill)
        return sorted(results, key=lambda s: s.use_count, reverse=True)

    # ── Execution ──────────────────────────────────────────────

    def run_skill(self, name: str, **kwargs) -> SkillResult:
        """
        Run a skill by name.

        Parameters
        ----------
        name   : skill name
        **kwargs : parameter values (e.g. target="192.168.1.1")

        Returns
        -------
        SkillResult
        """
        skill = self.get_skill(name)
        if not skill:
            return SkillResult(
                skill_name=name, success=False,
                output="", error=f"Skill '{name}' not found.",
            )

        t0 = time.time()

        try:
            if skill.language == "bash":
                result = self._run_bash(skill, **kwargs)
            elif skill.language == "python":
                result = self._run_python(skill, **kwargs)
            elif skill.language == "prompt":
                result = self._run_prompt(skill, **kwargs)
            else:
                result = SkillResult(
                    skill_name=name, success=False,
                    output="", error=f"Unknown language: {skill.language}",
                )

            result.elapsed = time.time() - t0

            # Update usage stats
            skill.use_count += 1
            skill.last_used  = time.time()
            self._save_skill_file(skill)

            log.info(
                "Ran skill %r in %.2fs — success=%s",
                name, result.elapsed, result.success,
            )
            return result

        except Exception as e:
            log.error("Skill %r failed: %s", name, e)
            return SkillResult(
                skill_name=name, success=False,
                output="", error=str(e),
                elapsed=time.time() - t0,
            )

    def _run_bash(self, skill: Skill, **kwargs) -> SkillResult:
        """Execute a bash skill."""
        cmd = skill.code
        for key, val in kwargs.items():
            cmd = cmd.replace(f"{{{key}}}", str(val))

        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        output = (result.stdout or result.stderr or "").strip()
        return SkillResult(
            skill_name=skill.name,
            success=result.returncode == 0,
            output=output or "(no output)",
            error="" if result.returncode == 0 else result.stderr,
        )

    def _run_python(self, skill: Skill, **kwargs) -> SkillResult:
        """Execute a Python skill in a subprocess."""
        code = skill.code
        for key, val in kwargs.items():
            code = code.replace(f"{{{key}}}", str(val))

        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        output = (result.stdout or "").strip()
        return SkillResult(
            skill_name=skill.name,
            success=result.returncode == 0,
            output=output or "(no output)",
            error=result.stderr.strip() if result.returncode != 0 else "",
        )

    def _run_prompt(self, skill: Skill, **kwargs) -> SkillResult:
        """Run a prompt-based skill through the LLM."""
        prompt = skill.code
        for key, val in kwargs.items():
            prompt = prompt.replace(f"{{{key}}}", str(val))

        try:
            from core.llm import get_llm
            llm    = get_llm()
            output = llm.chat(prompt)
            return SkillResult(
                skill_name=skill.name, success=True, output=output
            )
        except Exception as e:
            return SkillResult(
                skill_name=skill.name, success=False,
                output="", error=str(e),
            )

    # ── Skill Extraction ───────────────────────────────────────

    def extract_and_save(
        self,
        raw_text: str,
        name: str = "",
        tags: list[str] = None,
    ) -> Optional[Skill]:
        """
        Extract a skill from raw text (code the user pastes).
        Tries to detect language and parameters automatically.

        Parameters
        ----------
        raw_text : pasted code or command
        name     : optional name (auto-generated if not given)
        tags     : optional tags

        Returns
        -------
        Skill if extraction succeeded, None otherwise
        """
        raw_text = raw_text.strip()
        if not raw_text:
            return None

        # Detect language
        language = self._detect_language(raw_text)

        # Auto-generate name from first line if not provided
        if not name:
            first_line = raw_text.splitlines()[0].strip()
            if "def " in first_line:
                name = first_line.split("def ")[1].split("(")[0].strip()
            elif first_line.startswith("#"):
                name = first_line.lstrip("#").strip().lower().replace(" ", "_")[:30]
            else:
                name = f"skill_{int(time.time())}"

        # Extract parameters (look for {placeholder} patterns)
        import re
        params_raw  = re.findall(r"\{(\w+)\}", raw_text)
        parameters  = [
            {"name": p, "description": f"Parameter: {p}", "required": True}
            for p in set(params_raw)
        ]

        # Auto-generate description using LLM
        description = self._auto_describe(raw_text, language)

        return self.save_skill(
            name        = name,
            code        = raw_text,
            description = description,
            language    = language,
            parameters  = parameters,
            tags        = tags or [language],
        )

    def _detect_language(self, code: str) -> str:
        """Detect programming language from code."""
        code_lower = code.lower()
        if code_lower.startswith("#!/bin/bash") or re.search(r"\bnmap\b|\bsudo\b|\bapt\b|\becho\b", code):
            return "bash"
        if "def " in code or "import " in code or "print(" in code:
            return "python"
        if "function " in code or "const " in code or "var " in code:
            return "javascript"
        if code_lower.startswith("#!"):
            return "bash"
        return "python"

    def _auto_describe(self, code: str, language: str) -> str:
        """Use LLM to generate a one-line description of the code."""
        try:
            from core.llm import get_llm
            llm = get_llm()
            if not llm.is_ready:
                return f"A {language} skill."
            prompt = f"In one sentence, what does this {language} code do?\n\n{code[:500]}"
            return llm.chat(prompt, task="fast")
        except Exception:
            return f"A {language} skill."

    # ── Persistence ───────────────────────────────────────────

    def _save_skill_file(self, skill: Skill) -> None:
        """Save one skill to its JSON file."""
        path = self.skills_dir / f"{skill.name}.json"
        path.write_text(
            json.dumps(skill.to_dict(), indent=2, ensure_ascii=False)
        )

    def _load_all(self) -> None:
        """Load all skill files from disk."""
        for path in self.skills_dir.glob("*.json"):
            if path.name == "index.json":
                continue
            try:
                data  = json.loads(path.read_text())
                skill = Skill.from_dict(data)
                self._skills[skill.name] = skill
            except Exception as e:
                log.warning("Failed to load skill %s: %s", path.name, e)

    def _save_index(self) -> None:
        """Save the skills index for quick listing."""
        index = {
            name: {
                "description": s.description,
                "language":    s.language,
                "tags":        s.tags,
                "use_count":   s.use_count,
            }
            for name, s in self._skills.items()
        }
        SKILLS_INDEX.write_text(json.dumps(index, indent=2))

    def print_skills(self) -> None:
        """Print all skills to terminal."""
        skills = self.list_skills()
        if not skills:
            print("\n  No skills saved yet.\n")
            return

        print(f"\n  {'NAME':<25} {'LANG':<10} {'USED':<6} {'DESCRIPTION'}")
        print("  " + "─" * 70)
        for s in skills:
            print(
                f"  {s.name:<25} {s.language:<10} {s.use_count:<6} "
                f"{s.description[:40]}"
            )
        print()


# ─────────────────────────────────────────────────────────────
#  SINGLETON
# ─────────────────────────────────────────────────────────────

import re

_skill_manager: Optional[SkillManager] = None


def get_skill_manager() -> SkillManager:
    global _skill_manager
    if _skill_manager is None:
        _skill_manager = SkillManager()
    return _skill_manager


# ─────────────────────────────────────────────────────────────
#  CLI TEST — python core/skill_manager.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    sm = SkillManager()

    if "--list" in sys.argv:
        sm.print_skills()

    elif "--add" in sys.argv:
        sm.save_skill(
            name="port_scan",
            description="Scan a target for open ports with version detection",
            language="bash",
            code="nmap -sV -T4 {target}",
            parameters=[{"name": "target", "description": "IP or hostname", "required": True}],
            tags=["cyber", "network"],
        )
        sm.save_skill(
            name="reverse_string",
            description="Reverse a string in Python",
            language="python",
            code="s = '{text}'\nprint(s[::-1])",
            parameters=[{"name": "text", "description": "String to reverse", "required": True}],
            tags=["coding", "python"],
        )
        print("  Sample skills added.\n")
        sm.print_skills()

    elif "--run" in sys.argv:
        idx  = sys.argv.index("--run")
        name = sys.argv[idx + 1]
        kwargs = {}
        for arg in sys.argv[idx + 2:]:
            if "=" in arg:
                k, v = arg.split("=", 1)
                kwargs[k] = v
        print(f"\n  Running skill: {name!r} with {kwargs}\n")
        result = sm.run_skill(name, **kwargs)
        print(f"  Success : {result.success}")
        print(f"  Output  :\n{result.output}\n")

    else:
        print("\n  NEXUS Skill Manager")
        print("  Usage:")
        print("    python core/skill_manager.py --list")
        print("    python core/skill_manager.py --add")
        print("    python core/skill_manager.py --run port_scan target=192.168.1.1")
        print()
        sm.print_skills()
