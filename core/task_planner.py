"""
core/task_planner.py
NEXUS Task Planner — logical planning engine backed by skills, memory, and research.

When NEXUS gets an arbitrary task it doesn't have a hardcoded handler for,
TaskPlanner takes over:

  1. Search the skill registry → what capabilities do we already have?
  2. Search memory → have we done something like this before?
  3. If knowledge is thin, go online and research "how to <task>"
  4. Ask the LLM to compose a concrete step plan using available skills
  5. If a required skill doesn't exist, request creation on the spot
  6. Execute step by step, narrating each action
  7. Store the learned procedure so next time skips steps 1-4

Example
-------
  Task: "send an email to john@example.com about our meeting"

  → Registry search: no "email" skill yet
  → Memory search: nothing stored
  → Web research: "how to send email python gmail smtp"
  → LLM builds 4-step plan:
      1. Research Gmail SMTP credentials needed
      2. Create skill "send_email" (LLM writes Python function)
      3. Execute: compose + send via SMTP
      4. Store procedure in memory
  → Executes, skill file saved to data/skills/created/send_email.py
  → Procedure logged → next time goes straight to step 3
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("nexus.task_planner")

_PROCEDURE_LOG = Path(__file__).parent.parent / "data" / "task_procedures.jsonl"
_CREATED_DIR   = Path(__file__).parent.parent / "data" / "skills" / "created"


# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class TaskStep:
    n:           int
    description: str
    skill:       str           # action name: shell | memory | research | notify |
                               # llm_respond | acquire_skill | create_skill | cyber | calendar
    params:      dict          = field(default_factory=dict)
    result:      str           = ""
    success:     bool          = False
    skippable:   bool          = False


@dataclass
class TaskPlan:
    task:        str
    steps:       list[TaskStep] = field(default_factory=list)
    skills_used: list[str]      = field(default_factory=list)
    context:     dict           = field(default_factory=dict)
    procedure:   str            = ""    # human-readable summary of how to do this task
    error:       str            = ""

    @property
    def success(self) -> bool:
        return not self.error and all(s.success or s.skippable for s in self.steps)


# ─────────────────────────────────────────────────────────────
#  TASK PLANNER
# ─────────────────────────────────────────────────────────────

class TaskPlanner:
    """
    Logical planning engine for arbitrary tasks.

    Builds a concrete step-by-step plan using skill registry + memory +
    online research + LLM reasoning, then executes it and learns from the outcome.
    """

    def __init__(self, memory=None, llm=None, researcher=None):
        self._memory     = memory
        self._llm        = llm
        self._researcher = researcher

        try:
            from core.stream_output import get_output
            self._out = get_output()
        except ImportError:
            self._out = None

        try:
            from core.skill_registry import get_registry
            self._registry = get_registry()
        except Exception:
            self._registry = None

        _PROCEDURE_LOG.parent.mkdir(parents=True, exist_ok=True)
        _CREATED_DIR.mkdir(parents=True, exist_ok=True)

    def _say(self, method: str, *args, **kwargs):
        if self._out:
            getattr(self._out, method, self._out.thinking)(*args, **kwargs)

    # ── Public API ────────────────────────────────────────────

    def plan_and_execute(self, task: str, context: dict = None) -> str:
        """Build a plan, execute it, learn from it. Returns summary string."""
        ctx = dict(context or {})
        self._say("thinking", f"Planning task: {task!r}")

        plan = self._build_plan(task, ctx)
        if plan.error:
            return f"Could not plan: {plan.error}"

        self._show_plan(plan)
        self._execute_plan(plan)
        self._learn(task, plan)
        return self._format_result(plan)

    def plan_only(self, task: str, context: dict = None) -> TaskPlan:
        """Build without executing (useful for inspection / tests)."""
        return self._build_plan(task, dict(context or {}))

    # ── Plan building ─────────────────────────────────────────

    def _build_plan(self, task: str, ctx: dict) -> TaskPlan:
        """
        Build the best possible plan by:
        1. Recalling a stored procedure from memory
        2. Finding matching skills in the registry
        3. Researching online if needed
        4. Asking LLM to compose the final plan
        5. Heuristic fallback if LLM unavailable
        """
        # 1. Memory recall
        procedure = self._recall_procedure(task)
        if procedure:
            self._say("thinking", "Found stored procedure — adapting it ...")
            plan = self._plan_from_procedure(task, procedure, ctx)
            if plan and not plan.error:
                return plan

        # 2. Skill search
        skills = self._registry.search(task, limit=6) if self._registry else []
        if skills:
            self._say("thinking",
                      f"Found {len(skills)} matching skill(s): "
                      f"{', '.join(s.name for s in skills[:3])}")

        # 3. Online research if we lack sufficient skill coverage
        research = ""
        if not self._has_enough_skills(task, skills):
            research = self._research_how_to(task)

        # 4. LLM plan
        if self._llm and self._llm.is_ready:
            plan = self._llm_plan(task, skills, research, ctx)
            if plan and not plan.error:
                return plan

        # 5. Heuristic fallback
        return self._fallback_plan(task, skills, ctx)

    # ── Recall ────────────────────────────────────────────────

    def _recall_procedure(self, task: str) -> Optional[str]:
        """Try to find a stored procedure for a similar task."""
        # Check NEXUS memory
        if self._memory:
            try:
                results = self._memory.search_memory(task)
                for r in results:
                    if r.get("category") == "procedure":
                        return r.get("value", "")
            except Exception:
                pass

        # Check procedure log (recent 50 entries)
        if not _PROCEDURE_LOG.exists():
            return None
        task_words = set(task.lower().split())
        try:
            lines = _PROCEDURE_LOG.read_text(encoding="utf-8").strip().split("\n")
            for line in reversed(lines[-50:]):
                try:
                    entry = json.loads(line)
                    entry_words = set(entry.get("task", "").lower().split())
                    overlap = len(task_words & entry_words) / max(len(task_words), 1)
                    if overlap >= 0.55 and entry.get("procedure"):
                        return entry["procedure"]
                except Exception:
                    continue
        except Exception:
            pass
        return None

    # ── Skill relevance ───────────────────────────────────────

    def _has_enough_skills(self, task: str, skills: list) -> bool:
        if not skills:
            return False
        task_words = set(task.lower().split())
        for skill in skills[:2]:
            haystack = set(
                f"{skill.name} {skill.description} {' '.join(skill.tags)}".lower().split()
            )
            overlap = len(task_words & haystack) / max(len(task_words), 1)
            if overlap >= 0.35:
                return True
        return False

    # ── Research ──────────────────────────────────────────────

    def _research_how_to(self, task: str) -> str:
        """Search the web for how to accomplish the task."""
        if not self._researcher:
            return ""
        query = f"how to {task} programmatically step by step"
        self._say("searching", f"Researching: {query[:80]}")
        try:
            results = self._researcher.search(query, max_results=3)
            if not results:
                return ""
            # Persist findings in knowledge base
            try:
                from core.knowledge import get_knowledge_base
                kb = get_knowledge_base()
                for r in results[:2]:
                    kb.store(
                        topic   = f"how_to_{task[:40].replace(' ','_')}",
                        content = r.get("snippet", "")[:500],
                        source  = r.get("url", "web"),
                        tags    = task.split()[:5],
                    )
            except Exception:
                pass
            return "\n".join(
                r.get("snippet", "") for r in results if r.get("snippet")
            )[:1500]
        except Exception as exc:
            log.debug("Research failed: %s", exc)
            return ""

    # ── LLM plan ─────────────────────────────────────────────

    def _llm_plan(self, task: str, skills: list, research: str, ctx: dict) -> Optional[TaskPlan]:
        skills_text = "\n".join(
            f"- {s.name}: {s.description}  (e.g. \"{s.usage_example}\")"
            for s in skills
        ) or "No specific skills found — use generic actions."

        system = (
            "You are NEXUS's task planner. Build a concrete, minimal step plan.\n\n"
            "Available action types for the 'skill' field:\n"
            "  shell        — run a bash/terminal command\n"
            "  memory       — store or search memory\n"
            "  research     — web search for information\n"
            "  notify       — send a desktop notification\n"
            "  llm_respond  — use LLM to compose a text response\n"
            "  acquire_skill — clone a GitHub repo to get a skill\n"
            "  create_skill — ask LLM to write a new Python skill function\n"
            "  cyber        — cybersecurity action (scan, recon, etc.)\n"
            "  calendar     — scheduling / calendar event\n"
            "  <skill_name> — use a registered skill by its exact name\n\n"
            "Rules:\n"
            "- Keep it to ≤6 steps\n"
            "- Each step uses exactly ONE action\n"
            "- If a skill needs to be created first, add a create_skill step before using it\n"
            "- Return ONLY JSON:\n"
            '{"steps":[{"n":1,"skill":"action","description":"what this does",'
            '"params":{},"skippable":false}],'
            '"procedure":"one paragraph: how to do this type of task in general"}'
        )

        prompt = (
            f"Task: {task}\n"
            f"Context: {json.dumps(ctx)}\n"
            f"Available skills:\n{skills_text}\n"
        )
        if research:
            prompt += f"\nResearch findings:\n{research[:800]}"

        try:
            raw  = self._llm.chat(prompt, system=system, task="fast")
            raw  = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(raw)
            steps = [
                TaskStep(
                    n           = s["n"],
                    skill       = s.get("skill", "llm_respond"),
                    description = s.get("description", ""),
                    params      = s.get("params", {}),
                    skippable   = s.get("skippable", False),
                )
                for s in data.get("steps", [])
            ]
            return TaskPlan(
                task        = task,
                steps       = steps,
                context     = ctx,
                skills_used = [s.skill for s in steps],
                procedure   = data.get("procedure", ""),
            )
        except Exception as exc:
            log.warning("LLM plan failed: %s", exc)
            return None

    def _plan_from_procedure(self, task: str, procedure: str, ctx: dict) -> Optional[TaskPlan]:
        """Re-hydrate a stored procedure into executable steps via LLM."""
        if not (self._llm and self._llm.is_ready):
            return None
        prompt = (
            f"Stored procedure for similar task:\n{procedure}\n\n"
            f"Current task: {task}\n\n"
            "Adapt this procedure into ≤5 executable steps.\n"
            'Return ONLY JSON: {"steps":[{"n":1,"skill":"shell|memory|research|llm_respond",'
            '"description":"...","params":{}}]}'
        )
        try:
            raw  = self._llm.chat(prompt, task="fast")
            raw  = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(raw)
            steps = [
                TaskStep(n=s["n"], skill=s.get("skill","llm_respond"),
                         description=s.get("description",""), params=s.get("params",{}))
                for s in data.get("steps", [])
            ]
            return TaskPlan(task=task, steps=steps, context=ctx, procedure=procedure)
        except Exception:
            return None

    def _fallback_plan(self, task: str, skills: list, ctx: dict) -> TaskPlan:
        """Minimal heuristic plan when LLM isn't available."""
        steps = [
            TaskStep(n=1, skill="memory",
                     description=f"Search memory: {task[:60]}",
                     params={"action": "search", "query": task}),
        ]
        if skills:
            top = skills[0]
            steps.append(TaskStep(n=2, skill=top.name,
                                  description=f"Use skill: {top.description}",
                                  params={"skill": top.name}))
        else:
            steps.append(TaskStep(n=2, skill="research",
                                  description=f"Research: {task[:60]}",
                                  params={"query": task}, skippable=True))
            steps.append(TaskStep(n=3, skill="llm_respond",
                                  description="Compose response from gathered context",
                                  params={"task": task}))
        return TaskPlan(task=task, steps=steps, context=ctx,
                        skills_used=[s.skill for s in steps])

    # ── Display ───────────────────────────────────────────────

    def _show_plan(self, plan: TaskPlan):
        self._say("blank")
        self._say("planning", f"Task: {plan.task}")
        self._say("blank")
        total = len(plan.steps)
        for step in plan.steps:
            opt = " (optional)" if step.skippable else ""
            self._say("step", step.n, total, f"[{step.skill}] {step.description}{opt}")
        self._say("blank")

    # ── Execution ─────────────────────────────────────────────

    def _execute_plan(self, plan: TaskPlan):
        total = len(plan.steps)
        for step in plan.steps:
            try:
                self._say("thinking", f"Step {step.n}/{total}: {step.description}")
                step.result  = self._run_step(step, plan.context)
                step.success = True
                self._say("step", step.n, total, step.description, done=True)
            except Exception as exc:
                log.warning("Step %d failed: %s", step.n, exc)
                step.result  = str(exc)
                step.success = False
                if not step.skippable:
                    self._say("fail", f"Step {step.n}: {exc}")

    def _run_step(self, step: TaskStep, ctx: dict) -> str:
        skill = step.skill
        p     = step.params

        # ── Generic actions ───────────────────────────────────

        if skill == "shell":
            cmd = p.get("cmd", "echo 'no command specified'")
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            return proc.stdout.strip() or proc.stderr.strip() or "(no output)"

        if skill == "memory":
            if not self._memory:
                return "Memory not available"
            action = p.get("action", "search")
            if action == "search":
                results = self._memory.search_memory(p.get("query", step.description))
                ctx["memory_results"] = results
                return f"Found {len(results)} memory entries"
            if action == "store":
                self._memory.remember(p.get("key","note"), p.get("value",""), category="plan")
                return "Stored in memory"
            return f"Unknown memory action: {action}"

        if skill == "research":
            if not self._researcher:
                return "Researcher not available"
            results = self._researcher.search(p.get("query", step.description))
            snippets = [r.get("snippet","") for r in (results or [])[:3] if r.get("snippet")]
            ctx["research_results"] = snippets
            return "\n".join(snippets) or "No results"

        if skill == "notify":
            from core.autonomous_planner import _system_notify
            ok = _system_notify(p.get("title","NEXUS"), p.get("body",""))
            return "Notification sent" if ok else "notify-send unavailable"

        if skill == "llm_respond":
            if not (self._llm and self._llm.is_ready):
                # Fall back to raw memory/research results
                parts = []
                if ctx.get("memory_results"):
                    parts.append("Memory: " + "; ".join(
                        f"{r.get('key')}: {r.get('value')}" for r in ctx["memory_results"][:3]
                    ))
                if ctx.get("research_results"):
                    parts.append("Research: " + " ".join(ctx["research_results"][:2]))
                return "\n".join(parts) or "No information found."

            mem      = ctx.get("memory_results", [])
            research = ctx.get("research_results", [])
            ctx_text = ""
            if mem:
                ctx_text += "Memory:\n" + "\n".join(
                    f"  • {r.get('key')}: {r.get('value')}" for r in mem[:3]
                )
            if research:
                ctx_text += "\n\nResearch:\n" + " ".join(research[:2])

            prompt = f"Task: {p.get('task', step.description)}\n{ctx_text}"
            result = self._llm.chat(prompt, task="conversational")
            ctx["final_response"] = result
            return result[:300]

        if skill == "acquire_skill":
            url = p.get("url", "")
            if not url:
                return "No URL provided for skill acquisition"
            from core.skill_acquirer import SkillAcquirer
            acq    = SkillAcquirer(llm=self._llm, researcher=self._researcher)
            skills = acq.acquire(url)
            return f"Acquired {len(skills)} skill(s) from {url}"

        if skill == "create_skill":
            return self._create_skill(p, ctx)

        if skill == "cyber":
            try:
                from cyber.cyber import CyberBrain
                cb     = CyberBrain(verbose=False)
                result = cb.run(p.get("cmd",""))
                ctx["cyber_result"] = result[:500]
                return result[:250]
            except Exception as exc:
                return f"Cyber action failed: {exc}"

        if skill == "calendar":
            from core.autonomous_planner import AutonomousPlanner, AStep
            ap = AutonomousPlanner(self._memory, self._llm, self._researcher)
            fake = AStep(n=step.n, action="calendar",
                         description=step.description, params=p)
            return ap._run_step(fake, ctx)

        # ── Try registry skill by name ────────────────────────

        if self._registry:
            reg_skill = self._registry.get(skill)
            if reg_skill:
                self._registry.update_usage(skill)
                # If the skill has a code_path we can try to import + call it
                if reg_skill.invoke_module and reg_skill.invoke_fn:
                    try:
                        import importlib
                        mod = importlib.import_module(reg_skill.invoke_module)
                        fn  = getattr(mod, reg_skill.invoke_fn)
                        result = fn(**p)
                        return str(result)[:300]
                    except Exception as exc:
                        return f"Skill '{skill}' invocation failed: {exc}"
                return f"Skill '{skill}' noted — {reg_skill.description}"

        return f"No handler for: {skill}"

    # ── Skill creation ────────────────────────────────────────

    def _create_skill(self, params: dict, ctx: dict) -> str:
        """Use LLM to write a new Python skill function and register it."""
        if not (self._llm and self._llm.is_ready):
            return "LLM not available to create skill"

        name        = params.get("name", "new_skill").lower().replace(" ", "_")
        description = params.get("description", "")
        context_str = params.get("context", "")

        self._say("thinking", f"Creating new skill: {name}")

        prompt = (
            f"Write a self-contained Python function named `{name}` that: {description}\n"
            f"Additional context: {context_str}\n\n"
            "Requirements:\n"
            "- Accept clear keyword arguments\n"
            "- Return a string describing what was done\n"
            "- Import everything it needs inside the function\n"
            "- Handle exceptions gracefully (return error message, don't raise)\n"
            "- Include a short docstring\n"
            "Return ONLY the Python function code — no explanation, no markdown."
        )

        try:
            code = self._llm.chat(prompt, task="code")
            code = code.strip().removeprefix("```python").removeprefix("```").removesuffix("```").strip()

            skill_path = _CREATED_DIR / f"{name}.py"
            skill_path.write_text(code, encoding="utf-8")

            from core.skill_registry import Skill, get_registry
            skill = Skill(
                name          = name,
                description   = description or f"LLM-created skill",
                category      = params.get("category", "utility"),
                source        = "created",
                tags          = params.get("tags", [name]),
                code_path     = str(skill_path),
                invoke_fn     = name,
            )
            get_registry().register(skill)

            self._say("learned", f"Created skill: {name}")
            ctx[f"created_skill_{name}"] = str(skill_path)
            return f"Skill '{name}' created → {skill_path}"
        except Exception as exc:
            return f"Skill creation failed: {exc}"

    # ── Learning ──────────────────────────────────────────────

    def _learn(self, task: str, plan: TaskPlan):
        """Persist the procedure so future tasks can skip planning."""
        procedure = plan.procedure or " → ".join(
            s.description for s in plan.steps if s.success
        )
        if not procedure:
            return

        entry = {
            "task":      task,
            "procedure": procedure,
            "steps":     len(plan.steps),
            "success":   plan.success,
            "skills":    plan.skills_used,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            with open(_PROCEDURE_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

        if self._memory and plan.success:
            try:
                self._memory.remember(
                    f"procedure_{task[:40].replace(' ','_')}",
                    procedure,
                    category="procedure",
                )
            except Exception:
                pass

    # ── Summary ───────────────────────────────────────────────

    def _format_result(self, plan: TaskPlan) -> str:
        # If LLM composed a final answer, return it directly
        if plan.context.get("final_response"):
            return plan.context["final_response"]

        succeeded = sum(1 for s in plan.steps if s.success)
        failed    = sum(1 for s in plan.steps if not s.success and not s.skippable)
        lines     = [f"Done: {succeeded}/{len(plan.steps)} steps completed."]

        if failed:
            lines.append(f"  ⚠  {failed} step(s) failed.")

        for step in plan.steps:
            if step.success and step.result and len(step.result) > 30:
                lines.append(f"  • {step.description}:\n    {step.result[:200]}")

        return "\n".join(lines)


# ── Singleton ──────────────────────────────────────────────────

_planner: Optional[TaskPlanner] = None


def get_task_planner(memory=None, llm=None, researcher=None) -> TaskPlanner:
    global _planner
    if _planner is None:
        _planner = TaskPlanner(memory=memory, llm=llm, researcher=researcher)
    return _planner
