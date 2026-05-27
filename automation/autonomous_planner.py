"""
automation/autonomous_planner.py
NEXUS Autonomous Planner — AI planning + full physical execution.

This is the single unified planner. It handles BOTH:
  - High-level AI goals (calendar events, reminders, memory, cyber tasks, code)
  - Physical computer control (mouse clicks, keyboard typing, shell commands,
    window management, URL navigation, screenshots)

Execution agents used:
  AI steps    → handled directly (calendar, memory, notify, reminder, cyber)
  GUI steps   → GUIAgent (pyautogui / xdotool)
  Shell steps → ShellAgent (subprocess, file ops, app launch)
  Full tasks  → Automation.run() (full planner→executor pipeline)

Flow:
  User says anything complex
       ↓
  EntryModel classifies (CALENDAR / PLAN / CODE / CYBER / ...)
       ↓
  AutonomousPlanner.execute()
       ↓
  _build_plan() — picks the right plan builder:
    • schedule_meeting / set_reminder / add_calendar → AI calendar steps
    • pentest                                        → cyber pipeline
    • code_plan                                      → CodeEngine
    • physical task (open app, click, type, etc.)    → Automation.run()
    • generic                                        → TaskPlanner (skill+research+LLM)
       ↓
  _execute_plan() — runs each AStep via _run_step()
       ↓
  _format_summary() → response string

Example (calendar):
  "remid me i have a metting tomorow on zoom at 3pm"
  → Normalized → 7-step AI plan → .ics saved, reminder set, memory stored

Example (physical):
  "open firefox, go to github.com, search for NEXUS AI"
  → Detected as physical → Automation.run() → 5-step GUI plan → executed

Example (generic):
  "send an email to john about the meeting"
  → TaskPlanner → skill search → research → LLM plan → optional skill creation
"""

from __future__ import annotations

import json
import logging
import subprocess
import re
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("nexus.autonomous_planner")

_PLAN_LOG = Path(__file__).parent.parent / "data" / "plan_history.jsonl"

# Physical task trigger words — route to Automation.run() instead of AI planning
_PHYSICAL_TRIGGERS = re.compile(
    r"\b(?:open|launch|start|click|double.click|right.click|type|press|"
    r"scroll|drag|drop|navigate to|go to|browse to|visit|search (?:for|on)|"
    r"maximize|minimize|close window|focus|screenshot|take a screenshot|"
    r"volume up|volume down|brightness|save (?:as|the file)|copy to clipboard|"
    r"ssh into|git clone|pip install|apt install|run command|execute command)\b",
    re.I,
)

# Don't route to physical if these AI-handled categories appear
_AI_OVERRIDE = re.compile(
    r"\b(?:remind me|reminder|calendar|meeting|schedule|remember|store|"
    r"cyber|scan|recon|exploit|pentest|code|plan how|implement|analyze)\b",
    re.I,
)


# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class AStep:
    """One step in an autonomous plan."""
    n:           int
    description: str
    action:      str          # resolve_date | resolve_time | shell | gui | automation |
                              # calendar | notify | reminder | memory | cyber | code_engine | respond
    params:      dict         = field(default_factory=dict)
    result:      str          = ""
    success:     bool         = False
    skippable:   bool         = False

    def __str__(self) -> str:
        status = "✓" if self.success else ("→" if not self.result else "✗")
        return f"  {status} Step {self.n}: {self.description}"


@dataclass
class AutonomousPlan:
    """A complete autonomous execution plan."""
    goal:    str
    steps:   list[AStep]  = field(default_factory=list)
    context: dict         = field(default_factory=dict)
    summary: str          = ""
    error:   str          = ""

    @property
    def success(self) -> bool:
        return not self.error and all(
            s.success or s.skippable for s in self.steps
        )


# ─────────────────────────────────────────────────────────────
#  CALENDAR HELPERS
# ─────────────────────────────────────────────────────────────

def _ics_event(title: str, dt_str: str, duration_min: int = 60) -> str:
    """Generate a minimal iCalendar (.ics) event string."""
    try:
        dt = datetime.fromisoformat(dt_str)
    except Exception:
        dt = datetime.now()
    end   = dt + timedelta(minutes=duration_min)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return (
        "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//NEXUS AI//EN\n"
        "BEGIN:VEVENT\n"
        f"UID:{stamp}@nexus.ai\nDTSTAMP:{stamp}\n"
        f"DTSTART:{dt.strftime('%Y%m%dT%H%M%S')}\n"
        f"DTEND:{end.strftime('%Y%m%dT%H%M%S')}\n"
        f"SUMMARY:{title}\n"
        "END:VEVENT\nEND:VCALENDAR\n"
    )


def _system_notify(title: str, body: str, urgency: str = "normal") -> bool:
    """Send a desktop notification — cross-platform."""
    try:
        from core.platform_utils import notify
        return notify(title, body, urgency=urgency)
    except Exception:
        return False


def _schedule_at_command(cmd: str, when: str) -> bool:
    """Schedule a shell command with `at`. when = 'HH:MM YYYY-MM-DD'"""
    try:
        proc = subprocess.Popen(
            ["at", when],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        proc.communicate(input=cmd, timeout=10)
        return proc.returncode == 0
    except Exception as exc:
        log.debug("at command failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────
#  AUTONOMOUS PLANNER
# ─────────────────────────────────────────────────────────────

class AutonomousPlanner:
    """
    The unified self-thinking execution system.

    Handles complex goals with both AI reasoning (calendar, memory, cyber)
    and physical computer control (GUI, shell, keyboard, mouse).
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

        # Lazy-loaded physical agents
        self._gui_agent   = None
        self._shell_agent = None
        self._automation  = None

        _PLAN_LOG.parent.mkdir(parents=True, exist_ok=True)
        log.info("AutonomousPlanner ready (unified AI+physical).")

    def _say(self, method: str, *args, **kwargs):
        if self._out:
            getattr(self._out, method, self._out.thinking)(*args, **kwargs)

    # ── Physical agent lazy loaders ───────────────────────────

    def _get_gui_agent(self):
        if self._gui_agent is None:
            try:
                from automation.gui_agent import GUIAgent
                self._gui_agent = GUIAgent()
            except Exception as exc:
                log.warning("GUIAgent unavailable: %s", exc)
        return self._gui_agent

    def _get_shell_agent(self):
        if self._shell_agent is None:
            try:
                from automation.shell_agent import ShellAgent
                self._shell_agent = ShellAgent()
            except Exception as exc:
                log.warning("ShellAgent unavailable: %s", exc)
        return self._shell_agent

    def _get_automation(self):
        if self._automation is None:
            try:
                from automation.automation import Automation
                self._automation = Automation(verbose=True, confirm_high_risk=False)
            except Exception as exc:
                log.warning("Automation unavailable: %s", exc)
        return self._automation

    # ── Public API ────────────────────────────────────────────

    def execute(self, goal: str, entry_result=None, normalized_input=None) -> str:
        """
        Build and execute an autonomous plan for the given goal.
        Returns human-readable summary.
        """
        self._say("blank")
        self._say("thinking", f"Breaking down goal: {goal!r}")

        # Pull context from entry result / normalizer
        entities: dict = {}
        if entry_result:
            entities.update(entry_result.entities or {})
        if normalized_input:
            entities.update(normalized_input.entities or {})
            if normalized_input.date:
                entities["date"] = normalized_input.date
            if normalized_input.time:
                entities["time"] = normalized_input.time

        hint = entry_result.intent_hint if entry_result else None

        # ── Physical task fast-path ───────────────────────────
        # If task smells like GUI/shell automation and has no AI override, use Automation
        if self._is_physical_task(goal, hint):
            return self._run_physical(goal)

        # Build AI plan
        plan = self._build_plan(goal, hint, entities)

        if plan.error:
            return f"Could not plan: {plan.error}"

        self._show_plan(plan)
        self._execute_plan(plan)
        self._store_plan(goal, plan)
        return self._format_summary(plan)

    # ── Physical task detection ───────────────────────────────

    def _is_physical_task(self, goal: str, hint: Optional[str]) -> bool:
        """True if task involves physical computer control rather than AI reasoning."""
        if hint in ("schedule_meeting", "set_reminder", "add_calendar", "show_calendar",
                    "pentest", "code_plan", "remember", "recall"):
            return False
        has_physical = bool(_PHYSICAL_TRIGGERS.search(goal))
        has_ai       = bool(_AI_OVERRIDE.search(goal))
        return has_physical and not has_ai

    def _run_physical(self, goal: str) -> str:
        """Delegate physical task to Automation.run() and narrate the result."""
        self._say("thinking", f"Physical task detected — using automation engine")
        auto = self._get_automation()
        if not auto:
            return f"Automation engine unavailable. Could not perform: {goal}"
        try:
            result = auto.run(goal)
            if result.success:
                self._say("done", f"{result.steps_passed}/{result.steps_run} steps completed")
            else:
                self._say("fail", f"Automation failed at step {result.steps_run}")
            return result.voice_summary or result.detail or "Done."
        except Exception as exc:
            log.warning("Automation.run() failed: %s", exc)
            return f"Could not automate: {exc}"

    # ── Plan builders ─────────────────────────────────────────

    def _build_plan(self, goal: str, hint: Optional[str], entities: dict) -> AutonomousPlan:
        builders = {
            "schedule_meeting": self._plan_schedule_meeting,
            "set_reminder":     self._plan_set_reminder,
            "add_calendar":     self._plan_add_calendar,
            "show_calendar":    self._plan_show_calendar,
            "pentest":          self._plan_pentest,
            "code_plan":        self._plan_code_work,
        }
        builder = builders.get(hint or "")
        if builder:
            return builder(goal, entities)
        return self._plan_generic(goal, entities)

    # ── Calendar / scheduling plans ───────────────────────────

    def _plan_schedule_meeting(self, goal: str, entities: dict) -> AutonomousPlan:
        steps = []
        ctx: dict[str, Any] = {}

        # Resolve date
        date_val = entities.get("date")
        if not date_val:
            from core.normalizer import _resolve_relative_date
            date_val = _resolve_relative_date(goal)
        ctx["date"] = date_val or date.today().isoformat()
        steps.append(AStep(n=1, action="resolve_date",
                           description=f"Resolve date → {ctx['date']}",
                           params={"date": ctx["date"]}))

        # Resolve time
        time_val = entities.get("time")
        if not time_val:
            from core.normalizer import _resolve_time
            time_val = _resolve_time(goal)
        ctx["time"] = time_val or "09:00"
        steps.append(AStep(n=2, action="resolve_time",
                           description=f"Resolve time → {ctx['time']}",
                           params={"time": ctx["time"]}))

        # Meeting title
        app   = entities.get("app", "")
        title = self._extract_meeting_title(goal, app)
        ctx["title"] = title
        steps.append(AStep(n=3, action="resolve_title",
                           description=f"Meeting title → {title!r}",
                           params={"title": title}))

        # ICS file
        dt_str = f"{ctx['date']}T{ctx['time']}:00"
        ctx["datetime"] = dt_str
        desktop = Path.home() / "Desktop"
        cal_dir = desktop if desktop.exists() else Path(__file__).parent.parent / "data" / "calendar"
        cal_dir.mkdir(parents=True, exist_ok=True)
        ics_path = cal_dir / f"meeting_{ctx['date'].replace('-','')}.ics"
        steps.append(AStep(n=4, action="calendar",
                           description=f"Create calendar event: {title} @ {ctx['date']} {ctx['time']}",
                           params={"title": title, "dt_str": dt_str, "ics_path": str(ics_path)}))

        # Desktop notification
        steps.append(AStep(n=5, action="notify",
                           description="Desktop notification: meeting added",
                           params={"title": "NEXUS — Meeting Scheduled",
                                   "body": f"{title}\n{ctx['date']} at {ctx['time']}"},
                           skippable=True))

        # Reminder (30 min before)
        reminder_offset = entities.get("reminder_offset", "30 minute")
        try:
            offset_mins = int(re.search(r"\d+", reminder_offset).group())
        except Exception:
            offset_mins = 30
        meeting_dt  = datetime.fromisoformat(dt_str)
        reminder_dt = meeting_dt - timedelta(minutes=offset_mins)
        ctx["reminder_time"] = reminder_dt.strftime("%H:%M")
        ctx["reminder_date"] = reminder_dt.strftime("%Y-%m-%d")
        steps.append(AStep(n=6, action="reminder",
                           description=f"Set reminder {offset_mins}min before → {ctx['reminder_time']}",
                           params={"title": f"NEXUS REMINDER: {title} in {offset_mins} min",
                                   "reminder_time": ctx["reminder_time"],
                                   "reminder_date": ctx["reminder_date"],
                                   "body": f"Your meeting starts at {ctx['time']}"},
                           skippable=True))

        # Memory store
        steps.append(AStep(n=7, action="memory",
                           description="Store in NEXUS memory",
                           params={"key": f"meeting_{ctx['date']}",
                                   "value": f"{title} at {ctx['time']} on {ctx['date']}"}))

        return AutonomousPlan(goal=goal, steps=steps, context=ctx)

    def _plan_set_reminder(self, goal: str, entities: dict) -> AutonomousPlan:
        steps = []
        ctx: dict = {}
        date_val = entities.get("date") or date.today().isoformat()
        time_val = entities.get("time") or "09:00"
        ctx["date"] = date_val
        ctx["time"] = time_val
        subject = re.sub(r"remind me (to|about|i have|that)?", "", goal, flags=re.I).strip()
        ctx["subject"] = subject or "reminder"

        steps.append(AStep(n=1, action="resolve_date",
                           description=f"Date → {date_val}", params={"date": date_val}))
        steps.append(AStep(n=2, action="resolve_time",
                           description=f"Time → {time_val}", params={"time": time_val}))
        steps.append(AStep(n=3, action="notify",
                           description=f"Schedule desktop reminder: {subject[:40]}",
                           params={"title": "NEXUS REMINDER", "body": subject[:200]},
                           skippable=True))
        steps.append(AStep(n=4, action="reminder",
                           description=f"System reminder at {time_val}",
                           params={"title": f"NEXUS: {subject[:60]}",
                                   "reminder_time": time_val,
                                   "reminder_date": date_val,
                                   "body": subject},
                           skippable=True))
        steps.append(AStep(n=5, action="memory",
                           description="Store reminder in memory",
                           params={"key": f"reminder_{date_val}", "value": f"{subject} at {time_val}"}))

        return AutonomousPlan(goal=goal, steps=steps, context=ctx)

    def _plan_add_calendar(self, goal: str, entities: dict) -> AutonomousPlan:
        return self._plan_schedule_meeting(goal, entities)

    def _plan_show_calendar(self, goal: str, entities: dict) -> AutonomousPlan:
        desktop = Path.home() / "Desktop"
        cal_dir = desktop if desktop.exists() else Path(__file__).parent.parent / "data" / "calendar"
        steps = [
            AStep(n=1, action="shell",
                  description="List stored calendar events",
                  params={"cmd": f"ls {cal_dir}/*.ics 2>/dev/null || echo 'No .ics files found'"}),
            AStep(n=2, action="memory",
                  description="Recall meetings from memory",
                  params={"key": f"meeting_{date.today().isoformat()}"}),
        ]
        return AutonomousPlan(goal=goal, steps=steps, context={})

    def _plan_pentest(self, goal: str, entities: dict) -> AutonomousPlan:
        target = entities.get("url") or entities.get("host", "")
        steps = [
            AStep(n=1, action="authorize_check",
                  description=f"Verify {target!r} is in authorized scope",
                  params={"target": target}),
            AStep(n=2, action="cyber",
                  description="Port scan — discover open services",
                  params={"cmd": f"scan ports on {target}"}),
            AStep(n=3, action="cyber",
                  description="Subdomain enumeration",
                  params={"cmd": f"find subdomains {target}"}),
            AStep(n=4, action="cyber",
                  description="HTTP header fingerprint",
                  params={"cmd": f"http headers {target}"}),
            AStep(n=5, action="cyber",
                  description="Vulnerability scan",
                  params={"cmd": f"vuln scan {target}"}),
            AStep(n=6, action="memory",
                  description="Save findings to memory",
                  params={"key": f"pentest_{target}", "value": "See recon reports"}),
        ]
        return AutonomousPlan(goal=goal, steps=steps, context={"target": target})

    def _plan_code_work(self, goal: str, entities: dict) -> AutonomousPlan:
        steps = [
            AStep(n=1, action="code_engine",
                  description=f"Run code engine: {goal[:60]}",
                  params={"instruction": goal}),
        ]
        return AutonomousPlan(goal=goal, steps=steps, context={})

    def _plan_generic(self, goal: str, entities: dict) -> AutonomousPlan:
        """
        Generic plan — check TaskPlanner first (skill + research + LLM),
        then LLM-only, then minimal fallback.
        """
        try:
            from core.task_planner import get_task_planner
            tp     = get_task_planner(memory=self._memory, llm=self._llm, researcher=self._researcher)
            result = tp.plan_and_execute(goal, context=entities)
            steps  = [AStep(n=1, action="respond",
                            description="Task completed via TaskPlanner",
                            params={"goal": goal}, result=result, success=True)]
            plan   = AutonomousPlan(goal=goal, steps=steps, context=entities)
            plan.summary = result
            return plan
        except Exception as exc:
            log.warning("TaskPlanner delegation failed: %s", exc)

        if self._llm and self._llm.is_ready:
            return self._llm_plan(goal, entities)

        steps = [
            AStep(n=1, action="memory_search",
                  description=f"Search memory: {goal[:50]}", params={"query": goal}),
            AStep(n=2, action="respond",
                  description="Compose response from context", params={"goal": goal}),
        ]
        return AutonomousPlan(goal=goal, steps=steps, context=entities)

    def _llm_plan(self, goal: str, entities: dict) -> AutonomousPlan:
        """Ask LLM to generate a plan."""
        self._say("thinking", "Asking LLM to generate plan...")
        system = (
            "You are NEXUS's autonomous planner. Break the user's goal into concrete steps.\n"
            "Return ONLY JSON:\n"
            '{"steps":[{"n":1,"action":"shell|notify|memory|calendar|reminder|gui|automation|'
            'search|respond","description":"what this does","params":{},"skippable":false}],'
            '"summary":"one-line goal"}'
        )
        try:
            raw  = self._llm.chat(f"Goal: {goal}\nEntities: {json.dumps(entities)}",
                                  system=system, task="fast")
            raw  = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(raw)
            steps = [
                AStep(n=s["n"], action=s.get("action","respond"),
                      description=s.get("description",""), params=s.get("params",{}),
                      skippable=s.get("skippable",False))
                for s in data.get("steps", [])
            ]
            return AutonomousPlan(goal=goal, steps=steps, context=entities,
                                  summary=data.get("summary",""))
        except Exception as exc:
            log.warning("LLM plan failed: %s", exc)
            return AutonomousPlan(goal=goal, error=str(exc))

    # ── Plan display ──────────────────────────────────────────

    def _show_plan(self, plan: AutonomousPlan):
        self._say("blank")
        self._say("planning", f"Goal: {plan.goal}")
        self._say("blank")
        total = len(plan.steps)
        for step in plan.steps:
            skip = " (optional)" if step.skippable else ""
            self._say("step", step.n, total, step.description + skip)
        self._say("blank")

    # ── Plan execution ────────────────────────────────────────

    def _execute_plan(self, plan: AutonomousPlan):
        """Execute each step, record results."""
        total = len(plan.steps)
        for step in plan.steps:
            try:
                self._say("thinking", f"Executing step {step.n}/{total}: {step.description}")
                result       = self._run_step(step, plan.context)
                step.result  = result
                step.success = True
                self._say("step", step.n, total, step.description, done=True)
            except Exception as exc:
                log.warning("Step %d failed: %s", step.n, exc)
                step.result  = str(exc)
                step.success = False
                if not step.skippable:
                    self._say("fail", f"Step {step.n} failed: {exc}")

    def _run_step(self, step: AStep, ctx: dict) -> str:
        """Dispatch a single step to its executor — AI or physical."""
        a = step.action
        p = step.params

        # ── AI steps ──────────────────────────────────────────

        if a == "resolve_date":
            ctx["date_resolved"] = p.get("date", "")
            return f"Date resolved: {p.get('date')}"

        if a == "resolve_time":
            ctx["time_resolved"] = p.get("time", "")
            return f"Time resolved: {p.get('time')}"

        if a == "resolve_title":
            ctx["title_resolved"] = p.get("title", "")
            return f"Title resolved: {p.get('title')}"

        if a == "calendar":
            ics  = _ics_event(p["title"], p["dt_str"])
            path = Path(p["ics_path"])
            path.write_text(ics, encoding="utf-8")
            ctx["ics_path"] = str(path)
            return f"Calendar event saved to {path}"

        if a == "notify":
            ok = _system_notify(p.get("title", "NEXUS"), p.get("body", ""))
            return "Notification sent" if ok else "Notification failed (notify-send unavailable)"

        if a == "reminder":
            remind_time = p.get("reminder_time", "09:00")
            remind_date = p.get("reminder_date", date.today().isoformat())
            title       = p.get("title", "NEXUS Reminder")
            body        = p.get("body", "")
            from core.platform_utils import IS_WINDOWS
            if IS_WINDOWS:
                import urllib.parse
                ps_body = body.replace("'", " ")
                cmd = (f"powershell -Command \"Add-Type -AssemblyName System.Windows.Forms; "
                       f"[System.Windows.Forms.MessageBox]::Show('{ps_body}', '{title}')\"")
            else:
                cmd = f'notify-send -u critical "{title}" "{body}"'
            ok          = _schedule_at_command(cmd, f"{remind_time} {remind_date}")
            return (f"Reminder scheduled at {remind_time} on {remind_date}"
                    if ok else f"at command unavailable — reminder not scheduled")

        if a == "memory":
            if self._memory:
                key = p.get("key", "note")
                val = p.get("value", "")
                if val:
                    self._memory.remember(key, val, category="plan")
                    return "Stored in memory"
                # Recall mode
                results = self._memory.search_memory(key)
                if results:
                    return "; ".join(f"{r['key']}: {r['value']}" for r in results[:3])
                return "Nothing found in memory"
            return "Memory not available"

        if a == "memory_search":
            if self._memory:
                results = self._memory.search_memory(p.get("query", ""))
                ctx["memory_results"] = results
                return f"Found {len(results)} memory entries"
            return "Memory not available"

        if a == "respond":
            result = p.get("result") or ctx.get("memory_results") or p.get("goal","")
            if isinstance(result, list):
                return "\n".join(f"• {r.get('key')}: {r.get('value')}" for r in result[:3])
            return str(result)[:300] if result else "No result to report."

        if a == "authorize_check":
            try:
                from cyber.recon import ReconEngine
                recon = ReconEngine()
                if not recon._is_authorized(p.get("target", "")):
                    return f"NOT authorized. Run: authorize {p.get('target')}"
                return "Target authorized ✓"
            except Exception:
                return "Authorization check failed"

        if a == "cyber":
            try:
                from cyber.cyber import CyberBrain
                cb     = CyberBrain(verbose=False)
                result = cb.run(p.get("cmd", ""))
                ctx[f"cyber_{step.n}"] = result[:500]
                return result[:200]
            except Exception as exc:
                return f"Cyber action failed: {exc}"

        if a == "code_engine":
            try:
                from core.code_engine import get_engine
                engine = get_engine(llm=self._llm, researcher=self._researcher)
                return engine.work(p.get("instruction", step.description))
            except Exception as exc:
                return f"Code engine: {exc}"

        # ── Physical steps (shell) ────────────────────────────

        if a == "shell":
            cmd = p.get("cmd", "")
            if not cmd:
                return "No command specified"
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            return proc.stdout.strip() or proc.stderr.strip() or "(no output)"

        if a == "shell_cmd":
            # Route through ShellAgent for structured execution
            shell = self._get_shell_agent()
            if not shell:
                # Fallback: direct subprocess
                cmd = p.get("cmd", p.get("target", ""))
                if cmd:
                    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                    return proc.stdout.strip() or proc.stderr.strip() or "(no output)"
                return "ShellAgent unavailable and no command specified"

            class _FakeStep:
                def __init__(self, action, target, params, timeout_sec=30.0):
                    self.action, self.target, self.params = action, target, params
                    self.timeout_sec = timeout_sec

            fake_action = p.get("shell_action", "run_command")
            fake_target = p.get("cmd") or p.get("target", "")
            fake_step   = _FakeStep(fake_action, fake_target, p)
            ok, output  = shell.run(fake_step)
            return output if ok else f"Shell failed: {output}"

        # ── Physical steps (GUI) ──────────────────────────────

        if a == "gui":
            gui = self._get_gui_agent()
            if not gui:
                return "GUIAgent unavailable — install pyautogui and xdotool"

            class _FakeGUIStep:
                def __init__(self, action, target, params, timeout_sec=30.0):
                    self.action, self.target, self.params = action, target, params
                    self.timeout_sec = timeout_sec

            gui_action = p.get("gui_action", "click")
            gui_target = p.get("target", "")
            fake       = _FakeGUIStep(gui_action, gui_target, p)
            ok, output = gui.run(fake)
            return output if ok else f"GUI action failed: {output}"

        # ── Full physical automation delegation ───────────────

        if a == "automation":
            instruction = p.get("instruction", step.description)
            return self._run_physical(instruction)

        if a == "launch_app":
            shell = self._get_shell_agent()
            if shell:
                class _FS:
                    def __init__(self, target):
                        self.action, self.target, self.params = "launch_app", target, {}
                        self.timeout_sec = 10.0
                ok, out = shell.run(_FS(p.get("app", step.description)))
                return out
            return f"Launching {p.get('app', 'app')} via subprocess"

        if a == "click":
            gui = self._get_gui_agent()
            if not gui:
                return "GUIAgent unavailable"
            class _CS:
                def __init__(self, target, params):
                    self.action, self.target, self.params = "click", target, params
                    self.timeout_sec = 10.0
            ok, out = gui.run(_CS(p.get("target",""), p))
            return out if ok else f"Click failed: {out}"

        if a == "type_text":
            gui = self._get_gui_agent()
            if not gui:
                return "GUIAgent unavailable"
            class _TS:
                def __init__(self, text):
                    self.action, self.target, self.params = "type_text", "", {"text": text}
                    self.timeout_sec = 30.0
            ok, out = gui.run(_TS(p.get("text", "")))
            return out if ok else f"Type failed: {out}"

        if a == "press_key":
            gui = self._get_gui_agent()
            if not gui:
                return "GUIAgent unavailable"
            class _KS:
                def __init__(self, key):
                    self.action, self.target, self.params = "press_key", key, {}
                    self.timeout_sec = 5.0
            ok, out = gui.run(_KS(p.get("key", "return")))
            return out if ok else f"Key press failed: {out}"

        if a == "hotkey":
            gui = self._get_gui_agent()
            if not gui:
                return "GUIAgent unavailable"
            keys = p.get("keys", [])
            combo = "+".join(keys) if keys else p.get("combo", "")
            class _HS:
                def __init__(self, combo, keys):
                    self.action = "hotkey"
                    self.target = combo
                    self.params = {"keys": keys}
                    self.timeout_sec = 5.0
            ok, out = gui.run(_HS(combo, keys))
            return out if ok else f"Hotkey failed: {out}"

        if a == "scroll":
            gui = self._get_gui_agent()
            if not gui:
                return "GUIAgent unavailable"
            class _SS:
                def __init__(self, direction, amount):
                    self.action = "scroll"
                    self.target = ""
                    self.params = {"direction": direction, "amount": amount}
                    self.timeout_sec = 5.0
            ok, out = gui.run(_SS(p.get("direction","down"), p.get("amount", 3)))
            return out if ok else f"Scroll failed: {out}"

        if a == "navigate_url":
            gui = self._get_gui_agent()
            if gui:
                class _NS:
                    def __init__(self, url):
                        self.action, self.target, self.params = "navigate_url", url, {}
                        self.timeout_sec = 15.0
                ok, out = gui.run(_NS(p.get("url", step.description)))
                return out if ok else f"Navigate failed: {out}"
            # Fallback: cross-platform URL open
            url = p.get("url", "")
            if url:
                from core.platform_utils import open_url
                open_url(url)
                return f"Opened {url} in browser"
            return "No URL specified"

        if a == "screenshot":
            gui = self._get_gui_agent()
            if gui:
                class _PCS:
                    def __init__(self):
                        self.action = "screenshot"
                        self.target = "data/screenshots"
                        self.params = {}
                        self.timeout_sec = 15.0
                ok, out = gui.run(_PCS())
                return out
            return "GUIAgent unavailable for screenshot"

        return f"Unknown action: {a}"

    # ── Plan storage ──────────────────────────────────────────

    def _store_plan(self, goal: str, plan: AutonomousPlan):
        summary = plan.summary or " → ".join(
            s.description for s in plan.steps if s.success
        )
        entry = {
            "goal":      goal,
            "summary":   summary,
            "steps":     len(plan.steps),
            "success":   plan.success,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            with open(_PLAN_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    # ── Summary ───────────────────────────────────────────────

    def _format_summary(self, plan: AutonomousPlan) -> str:
        if plan.summary:
            return plan.summary

        lines = []
        if plan.context.get("date_resolved") or plan.context.get("date"):
            dt    = plan.context.get("date_resolved") or plan.context.get("date")
            tm    = plan.context.get("time_resolved") or plan.context.get("time", "")
            title = plan.context.get("title_resolved") or plan.context.get("title", plan.goal)
            lines.append(f"Scheduled: {title}")
            lines.append(f"  Date : {dt}   Time: {tm}")
            if plan.context.get("ics_path"):
                lines.append(f"  File : {plan.context['ics_path']}")
            if plan.context.get("reminder_time"):
                lines.append(f"  Reminder: {plan.context['reminder_time']} on {plan.context.get('reminder_date','')}")

        succeeded = sum(1 for s in plan.steps if s.success)
        failed    = sum(1 for s in plan.steps if not s.success and not s.skippable)

        if not lines:
            lines.append(f"Plan executed: {succeeded}/{len(plan.steps)} steps succeeded.")
        if failed:
            lines.append(f"⚠  {failed} step(s) failed (check logs).")
        for step in plan.steps:
            if step.result and not step.success:
                lines.append(f"  Step {step.n}: {step.result}")

        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _extract_meeting_title(goal: str, app: str) -> str:
        cleaned = re.sub(
            r"remind me (i have |about |that )?(a |an )?|schedule (a |an )?|"
            r"add (a |an )?|create (a |an )?|set (a |an )?|i have (a |an )?",
            "", goal, flags=re.I
        ).strip()
        cleaned = re.sub(
            r"\b(?:tomorrow|today|next\s+\w+|on\s+\w+day|at\s+\d+[:\d]*\s*(?:am|pm)?|"
            r"\d+\s*(?:am|pm))\b", "", cleaned, flags=re.I
        ).strip()
        if app:
            cleaned = re.sub(rf"\b(?:on\s+)?{re.escape(app)}\b", "", cleaned, flags=re.I).strip()
        # Strip trailing prepositions left by removals
        cleaned = re.sub(r"\b(?:on|at|in|via|through|using)\s*$", "", cleaned, flags=re.I).strip()
        # Capitalise platform name
        platform = app.capitalize() if app else ""
        if platform and platform.lower() not in cleaned.lower():
            cleaned = f"{cleaned} ({platform})".strip()
        if not cleaned or len(cleaned) < 3:
            cleaned = f"Meeting{' on ('+platform+')' if platform else ''}"
        return cleaned[:80]
