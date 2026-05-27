"""
automation/planner.py
NEXUS Task Planner — converts one natural language instruction into
an ordered sequence of atomic, executable steps.

Each step has:
  - a type   (shell | gui | web | api | wait | check | llm)
  - a target (what to act on)
  - an action (what to do)
  - params   (arguments for the executor)
  - depends_on (list of step indices that must succeed first)

Example:

  Instruction: "Open Chrome, go to YouTube, search for NEXUS AI"

  Plan:
    Step 1: type=gui,   action=open_app,     target=google-chrome
    Step 2: type=gui,   action=wait_window,  target=Chrome,         depends_on=[1]
    Step 3: type=gui,   action=navigate_url, target=youtube.com,    depends_on=[2]
    Step 4: type=gui,   action=type_text,    target=search_box,     params={"text": "NEXUS AI"}, depends_on=[3]
    Step 5: type=gui,   action=press_key,    target=Return,         depends_on=[4]

Pipeline:
    User instruction (text)
           ↓
    planner.plan(instruction) → ExecutionPlan
           ↓
    executor.run(plan) → results
"""

from __future__ import annotations

import re
import logging
import shutil
from dataclasses import dataclass, field
from typing import Optional, Any

log = logging.getLogger("nexus.automation.planner")


# ─────────────────────────────────────────────────────────────
#  STEP TYPES
# ─────────────────────────────────────────────────────────────

STEP_TYPES = {
    "shell":  "Run a shell command (subprocess)",
    "gui":    "GUI automation (pyautogui — mouse, keyboard, windows)",
    "web":    "Search the web or open a URL in a browser",
    "wait":   "Wait for a condition or time",
    "check":  "Check a condition (screen text, file existence, etc.)",
    "api":    "Make an API call (WhatsApp, email, notifications)",
    "llm":    "Ask the LLM for a sub-result (generate text, code, etc.)",
}


# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class Step:
    """
    One atomic, executable step in a plan.

    Attributes
    ----------
    index       : step number (1-indexed)
    type        : "shell" | "gui" | "web" | "wait" | "check" | "api" | "llm"
    action      : what to do (e.g. "open_app", "click", "type_text", "run_cmd")
    target      : what to act on (app name, URL, file path, selector, etc.)
    params      : extra parameters for the executor
    depends_on  : indices of steps that must succeed before this one runs
    description : human-readable description of what this step does
    optional    : if True, failure won't abort the plan
    retry       : number of retry attempts on failure (0 = no retry)
    timeout_sec : max seconds to wait for this step to complete
    """
    index:       int
    type:        str
    action:      str
    target:      str             = ""
    params:      dict            = field(default_factory=dict)
    depends_on:  list[int]       = field(default_factory=list)
    description: str             = ""
    optional:    bool            = False
    retry:       int             = 0
    timeout_sec: float           = 30.0

    def __str__(self) -> str:
        dep = f" (after {self.depends_on})" if self.depends_on else ""
        return f"Step {self.index}: [{self.type}] {self.action} → {self.target!r}{dep}"


@dataclass
class ExecutionPlan:
    """
    Complete plan for executing an instruction.

    Attributes
    ----------
    instruction : original user instruction
    steps       : ordered list of Step objects
    goal        : what success looks like
    estimated_duration_sec : rough estimate
    risk_level  : "low" | "medium" | "high" (used for confirmation)
    requires_confirmation : whether to ask user before running
    """
    instruction: str
    steps:       list[Step]     = field(default_factory=list)
    goal:        str            = ""
    estimated_duration_sec: float = 0.0
    risk_level:  str            = "low"
    requires_confirmation: bool = False
    error:       str            = ""

    @property
    def success(self) -> bool:
        return bool(self.steps) and not self.error

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def summary(self) -> str:
        lines = [
            f"Plan: {self.instruction!r}",
            f"Goal: {self.goal}",
            f"Steps: {self.step_count}",
            f"Risk: {self.risk_level}",
            f"Est. duration: {self.estimated_duration_sec:.1f}s",
        ]
        for step in self.steps:
            opt = " [optional]" if step.optional else ""
            lines.append(f"  {step}{opt}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return (
            f"ExecutionPlan({self.step_count} steps, "
            f"risk={self.risk_level}, "
            f"instruction={self.instruction!r})"
        )


# ─────────────────────────────────────────────────────────────
#  RULE-BASED PATTERNS
#  Fast path — no LLM needed for common tasks
# ─────────────────────────────────────────────────────────────

# Each rule: (pattern, plan_builder_function)
# Plan builders return list[dict] — each dict becomes a Step

def _plan_open_app(match, instruction: str) -> list[dict]:
    app = match.group("app").strip()
    return [
        {"type": "shell", "action": "launch_app", "target": app,
         "description": f"Launch {app}"},
        {"type": "gui",   "action": "wait_window", "target": app,
         "description": f"Wait for {app} window to appear",
         "depends_on": [1], "optional": True, "timeout_sec": 10.0},
    ]


def _plan_open_url(match, instruction: str) -> list[dict]:
    url = match.group("url").strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return [
        {"type": "shell", "action": "launch_app",   "target": "firefox",
         "description": "Open browser"},
        {"type": "gui",   "action": "wait_window",  "target": "firefox",
         "description": "Wait for browser to load", "depends_on": [1],
         "optional": True, "timeout_sec": 8.0},
        {"type": "gui",   "action": "navigate_url", "target": url,
         "description": f"Navigate to {url}", "depends_on": [1]},
    ]


def _plan_type_text(match, instruction: str) -> list[dict]:
    # Handle all three named groups (quoted single, quoted double, unquoted)
    text = (match.group("text") or
            match.groupdict().get("text2") or
            match.groupdict().get("text3") or "").strip()
    return [
        {"type": "gui", "action": "focus_active_window", "target": "",
         "description": "Focus current window"},
        {"type": "gui", "action": "type_text", "target": "",
         "params": {"text": text},
         "description": f"Type: {text!r}", "depends_on": [1]},
    ]


def _plan_close_app(match, instruction: str) -> list[dict]:
    app = match.group("app").strip()
    return [
        {"type": "shell", "action": "kill_app", "target": app,
         "description": f"Close {app}"},
    ]


def _plan_create_file(match, instruction: str) -> list[dict]:
    path = match.group("path").strip()
    return [
        {"type": "shell", "action": "create_file", "target": path,
         "params": {"content": ""},
         "description": f"Create file: {path}"},
    ]


def _plan_create_dir(match, instruction: str) -> list[dict]:
    path = match.group("path").strip()
    return [
        {"type": "shell", "action": "run_command",
         "target": f"mkdir -p {path}",
         "description": f"Create directory: {path}"},
    ]


def _plan_delete_file(match, instruction: str) -> list[dict]:
    path = match.group("path").strip()
    return [
        {"type": "check", "action": "check_file_exists", "target": path,
         "description": f"Verify {path} exists before deletion"},
        {"type": "shell", "action": "delete_file", "target": path,
         "description": f"Delete: {path}", "depends_on": [1]},
    ]


def _plan_run_command(match, instruction: str) -> list[dict]:
    cmd = match.group("cmd").strip()
    return [
        {"type": "shell", "action": "run_command", "target": cmd,
         "description": f"Run: {cmd}", "timeout_sec": 60.0},
    ]


def _plan_take_screenshot(match, instruction: str) -> list[dict]:
    return [
        {"type": "shell", "action": "screenshot", "target": "data/screenshots/",
         "description": "Take a screenshot"},
    ]


def _plan_search_web(match, instruction: str) -> list[dict]:
    query = match.group("query").strip()
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    return [
        {"type": "shell", "action": "launch_app",   "target": "firefox",
         "description": "Open browser"},
        {"type": "gui",   "action": "wait_window",  "target": "firefox",
         "description": "Wait for browser", "depends_on": [1],
         "optional": True, "timeout_sec": 8.0},
        {"type": "gui",   "action": "navigate_url", "target": url,
         "description": f"Search for: {query}", "depends_on": [1]},
    ]


def _plan_install_pip(match, instruction: str) -> list[dict]:
    pkg = match.group("pkg").strip()
    return [
        {"type": "shell", "action": "run_command",
         "target": f"pip install {pkg}",
         "description": f"Install Python package: {pkg}",
         "params": {"timeout": 120}},
        {"type": "check", "action": "check_python_import",
         "target": pkg.split("[")[0],   # strip extras like [all]
         "description": f"Verify {pkg} installed",
         "depends_on": [1], "optional": True},
    ]


def _plan_install_apt(match, instruction: str) -> list[dict]:
    pkg = match.group("pkg").strip()
    return [
        {"type": "shell", "action": "run_command",
         "target": f"sudo apt-get install -y {pkg}",
         "description": f"Install system package: {pkg}",
         "params": {"timeout": 180}},
    ]


def _plan_git_clone(match, instruction: str) -> list[dict]:
    url = match.group("url").strip()
    return [
        {"type": "shell", "action": "run_command",
         "target": f"git clone {url}",
         "description": f"Clone repository: {url}",
         "params": {"timeout": 120}},
    ]


def _plan_git_commit(match, instruction: str) -> list[dict]:
    msg = match.group("msg").strip()
    return [
        {"type": "shell", "action": "run_command",
         "target": "git add -A",
         "description": "Stage all changes"},
        {"type": "shell", "action": "run_command",
         "target": f'git commit -m "{msg}"',
         "description": f"Commit with message: {msg}",
         "depends_on": [1]},
    ]


def _plan_copy_file(match, instruction: str) -> list[dict]:
    src = match.group("src").strip()
    dst = match.group("dst").strip()
    return [
        {"type": "check", "action": "check_file_exists", "target": src,
         "description": f"Verify source exists: {src}"},
        {"type": "shell", "action": "run_command",
         "target": f"cp -r {src} {dst}",
         "description": f"Copy {src} → {dst}",
         "depends_on": [1]},
    ]


def _plan_move_file(match, instruction: str) -> list[dict]:
    src = match.group("src").strip()
    dst = match.group("dst").strip()
    return [
        {"type": "check", "action": "check_file_exists", "target": src,
         "description": f"Verify source exists: {src}"},
        {"type": "shell", "action": "run_command",
         "target": f"mv {src} {dst}",
         "description": f"Move {src} → {dst}",
         "depends_on": [1]},
    ]


def _plan_rename_file(match, instruction: str) -> list[dict]:
    src = match.group("src").strip()
    dst = match.group("dst").strip()
    return [
        {"type": "shell", "action": "run_command",
         "target": f"mv {src} {dst}",
         "description": f"Rename {src} → {dst}"},
    ]


def _plan_click(match, instruction: str) -> list[dict]:
    target = match.group("target").strip() if "target" in match.groupdict() else ""
    return [
        {"type": "gui", "action": "click",
         "target": target,
         "description": f"Click: {target!r}"},
    ]


def _plan_double_click(match, instruction: str) -> list[dict]:
    target = match.group("target").strip() if "target" in match.groupdict() else ""
    return [
        {"type": "gui", "action": "double_click",
         "target": target,
         "description": f"Double-click: {target!r}"},
    ]


def _plan_right_click(match, instruction: str) -> list[dict]:
    target = match.group("target").strip() if "target" in match.groupdict() else ""
    return [
        {"type": "gui", "action": "right_click",
         "target": target,
         "description": f"Right-click: {target!r}"},
    ]


def _plan_press_key(match, instruction: str) -> list[dict]:
    key = match.group("key").strip()
    return [
        {"type": "gui", "action": "press_key",
         "target": key,
         "description": f"Press key: {key}"},
    ]


def _plan_hotkey(match, instruction: str) -> list[dict]:
    combo = match.group("combo").strip()
    return [
        {"type": "gui", "action": "hotkey",
         "target": combo,
         "params": {"keys": combo.split("+")},
         "description": f"Press hotkey: {combo}"},
    ]


def _plan_scroll(match, instruction: str) -> list[dict]:
    direction = match.group("direction").strip()
    amount_str = match.groupdict().get("amount", "3") or "3"
    try:
        amount = int(amount_str)
    except (ValueError, TypeError):
        amount = 3
    return [
        {"type": "gui", "action": "scroll",
         "target": "",
         "params": {"direction": direction, "amount": amount},
         "description": f"Scroll {direction} by {amount}"},
    ]


def _plan_maximize_window(match, instruction: str) -> list[dict]:
    return [
        {"type": "gui", "action": "maximize_window", "target": "",
         "description": "Maximize current window"},
    ]


def _plan_minimize_window(match, instruction: str) -> list[dict]:
    return [
        {"type": "gui", "action": "minimize_window", "target": "",
         "description": "Minimize current window"},
    ]


def _plan_close_window(match, instruction: str) -> list[dict]:
    return [
        {"type": "gui", "action": "close_window", "target": "",
         "description": "Close current window"},
    ]


def _plan_focus_window(match, instruction: str) -> list[dict]:
    target = match.group("target").strip() if "target" in match.groupdict() else ""
    return [
        {"type": "gui", "action": "focus_window", "target": target,
         "description": f"Focus window: {target!r}"},
    ]


def _plan_wait(match, instruction: str) -> list[dict]:
    seconds = float(match.group("seconds").strip())
    return [
        {"type": "wait", "action": "sleep",
         "target": "",
         "params": {"seconds": seconds},
         "description": f"Wait {seconds}s"},
    ]


def _plan_wait_for_text(match, instruction: str) -> list[dict]:
    text = match.group("text").strip()
    return [
        {"type": "wait", "action": "wait_for_text",
         "target": text,
         "params": {"timeout": 30},
         "description": f"Wait until screen shows: {text!r}"},
    ]


def _plan_check_file(match, instruction: str) -> list[dict]:
    path = match.group("path").strip()
    return [
        {"type": "check", "action": "check_file_exists",
         "target": path,
         "description": f"Check if file exists: {path}"},
    ]


def _plan_read_file(match, instruction: str) -> list[dict]:
    path = match.group("path").strip()
    return [
        {"type": "shell", "action": "run_command",
         "target": f"cat {path}",
         "description": f"Read file contents: {path}"},
    ]


def _plan_write_file(match, instruction: str) -> list[dict]:
    path = match.group("path").strip()
    content = match.groupdict().get("content", "").strip()
    return [
        {"type": "shell", "action": "write_file",
         "target": path,
         "params": {"content": content},
         "description": f"Write to file: {path}"},
    ]


def _plan_list_files(match, instruction: str) -> list[dict]:
    path = (match.group("path").strip()
            if "path" in match.groupdict() and match.group("path") else ".")
    return [
        {"type": "shell", "action": "run_command",
         "target": f"ls -la {path}",
         "description": f"List files in: {path}"},
    ]


def _plan_find_files(match, instruction: str) -> list[dict]:
    pattern = match.group("pattern").strip()
    return [
        {"type": "shell", "action": "run_command",
         "target": f"find . -name '{pattern}'",
         "description": f"Find files matching: {pattern}"},
    ]


def _plan_grep(match, instruction: str) -> list[dict]:
    pattern = match.group("pattern").strip()
    path    = (match.groupdict().get("path") or ".").strip()
    return [
        {"type": "shell", "action": "run_command",
         "target": f"grep -r '{pattern}' {path}",
         "description": f"Search for '{pattern}' in {path}"},
    ]


def _plan_zip(match, instruction: str) -> list[dict]:
    src = match.group("src").strip()
    dst = match.group("dst").strip() if match.groupdict().get("dst") else src + ".zip"
    return [
        {"type": "shell", "action": "run_command",
         "target": f"zip -r {dst} {src}",
         "description": f"Compress {src} → {dst}"},
    ]


def _plan_unzip(match, instruction: str) -> list[dict]:
    src = match.group("src").strip()
    dst = (match.groupdict().get("dst") or ".").strip()
    return [
        {"type": "shell", "action": "run_command",
         "target": f"unzip {src} -d {dst}",
         "description": f"Extract {src} → {dst}"},
    ]


def _plan_chmod(match, instruction: str) -> list[dict]:
    mode = match.group("mode").strip()
    path = match.group("path").strip()
    return [
        {"type": "shell", "action": "run_command",
         "target": f"chmod {mode} {path}",
         "description": f"Set permissions {mode} on {path}"},
    ]


def _plan_python_run(match, instruction: str) -> list[dict]:
    script = match.group("script").strip()
    return [
        {"type": "check", "action": "check_file_exists", "target": script,
         "description": f"Verify script exists: {script}"},
        {"type": "shell", "action": "run_command",
         "target": f"python3 {script}",
         "description": f"Run Python script: {script}",
         "depends_on": [1], "timeout_sec": 120.0},
    ]


def _plan_volume(match, instruction: str) -> list[dict]:
    action = match.group("action").lower().strip()
    level  = match.groupdict().get("level", "").strip()
    cmd_map = {
        "up":     "amixer -D pulse sset Master 5%+",
        "down":   "amixer -D pulse sset Master 5%-",
        "mute":   "amixer -D pulse sset Master mute",
        "unmute": "amixer -D pulse sset Master unmute",
        "max":    "amixer -D pulse sset Master 100%",
    }
    if level and level.isdigit():
        cmd = f"amixer -D pulse sset Master {level}%"
        desc = f"Set volume to {level}%"
    else:
        cmd  = cmd_map.get(action, "amixer -D pulse sset Master 5%+")
        desc = f"Volume {action}"
    return [{"type": "shell", "action": "run_command",
             "target": cmd, "description": desc}]


def _plan_brightness(match, instruction: str) -> list[dict]:
    from core.platform_utils import brightness_cmd
    action = match.group("action").lower().strip()
    level  = match.groupdict().get("level", "").strip()
    cmd = brightness_cmd(action, level if level and level.isdigit() else None)
    return [{"type": "shell", "action": "run_command",
             "target": cmd, "description": f"Brightness {action}"}]


def _plan_notify(match, instruction: str) -> list[dict]:
    from core.platform_utils import IS_WINDOWS
    msg = match.group("msg").strip()
    if IS_WINDOWS:
        # Use PowerShell toast — no extra dependency
        ps = (f"Add-Type -AssemblyName System.Windows.Forms; "
              f"[System.Windows.Forms.MessageBox]::Show('{msg}', 'NEXUS')")
        cmd = f"powershell -WindowStyle Hidden -Command \"{ps}\""
    else:
        cmd = f'notify-send "NEXUS" "{msg}"'
    return [
        {"type": "shell", "action": "run_command",
         "target": cmd, "description": f"Send notification: {msg}"},
    ]


def _plan_clipboard_copy(match, instruction: str) -> list[dict]:
    from core.platform_utils import IS_WINDOWS, IS_MAC
    text = match.group("text").strip()
    if IS_WINDOWS:
        cmd = f"echo {text} | clip"
    elif IS_MAC:
        cmd = f"echo -n '{text}' | pbcopy"
    else:
        if shutil.which("xclip"):
            cmd = f"echo -n '{text}' | xclip -selection clipboard"
        elif shutil.which("wl-copy"):
            cmd = f"echo -n '{text}' | wl-copy"
        else:
            cmd = f"echo -n '{text}' | xsel --clipboard --input"
    return [
        {"type": "shell", "action": "run_command",
         "target": cmd, "description": f"Copy to clipboard: {text!r}"},
    ]


def _plan_system_info(match, instruction: str) -> list[dict]:
    from core.platform_utils import system_info_cmd
    return [
        {"type": "shell", "action": "run_command",
         "target": system_info_cmd(),
         "description": "Get system information"},
    ]


def _plan_kill_process(match, instruction: str) -> list[dict]:
    from core.platform_utils import kill_process_cmd
    proc = match.group("proc").strip()
    return [
        {"type": "shell", "action": "run_command",
         "target": kill_process_cmd(proc),
         "description": f"Kill process: {proc}"},
    ]


def _plan_network_info(match, instruction: str) -> list[dict]:
    from core.platform_utils import network_info_cmd
    return [
        {"type": "shell", "action": "run_command",
         "target": network_info_cmd(),
         "description": "Show network information"},
    ]


def _plan_ssh(match, instruction: str) -> list[dict]:
    host = match.group("host").strip()
    user = (match.groupdict().get("user") or "").strip()
    target = f"{user}@{host}" if user else host
    return [
        {"type": "shell", "action": "run_command",
         "target": f"ssh {target}",
         "description": f"SSH into {target}",
         "params": {"interactive": True}, "timeout_sec": 300.0},
    ]


def _plan_save(match, instruction: str) -> list[dict]:
    return [
        {"type": "gui", "action": "hotkey",
         "target": "ctrl+s",
         "params": {"keys": ["ctrl", "s"]},
         "description": "Save current file (Ctrl+S)"},
    ]


def _plan_save_as(match, instruction: str) -> list[dict]:
    filename = (match.groupdict().get("name") or "").strip()
    steps = [
        {"type": "gui", "action": "hotkey",
         "target": "ctrl+shift+s",
         "params": {"keys": ["ctrl", "shift", "s"]},
         "description": "Open Save As dialog (Ctrl+Shift+S)"},
        {"type": "wait", "action": "sleep",
         "target": "", "params": {"seconds": 0.5},
         "description": "Wait for dialog to open",
         "depends_on": [1]},
    ]
    if filename:
        steps += [
            {"type": "gui", "action": "type_text",
             "target": "", "params": {"text": filename},
             "description": f"Type filename: {filename!r}",
             "depends_on": [2]},
            {"type": "gui", "action": "press_key",
             "target": "return",
             "description": "Confirm save",
             "depends_on": [3]},
        ]
    return steps


# ─────────────────────────────────────────────────────────────
#  RULE LIST: (pattern, builder, risk_level)
# ─────────────────────────────────────────────────────────────

RULES: list[tuple[re.Pattern, callable, str]] = [

    # ── App management ────────────────────────────────────────
    (re.compile(r"^(?:open|launch|start)\s+(?P<app>[\w\s\-\.]+?)(?:\s+and|\s+then|$)", re.I),
     _plan_open_app, "low"),
    (re.compile(r"^(?:close|quit|kill)\s+(?P<app>[\w\s\-\.]+?)(?:\s+and|\s+then|$)", re.I),
     _plan_close_app, "low"),

    # ── File / directory operations ───────────────────────────
    (re.compile(r"^(?:create|make|touch)\s+(?:file\s+)?(?P<path>[\w./~\-\s]+\.[\w]+)", re.I),
     _plan_create_file, "low"),
    (re.compile(r"^(?:create|make|mkdir)\s+(?:dir(?:ectory)?\s+)?(?P<path>[\w./~\-\s]+)$", re.I),
     _plan_create_dir, "low"),
    (re.compile(r"^(?:delete|remove|rm)\s+(?:file\s+)?(?P<path>[\w./~\-\s]+)", re.I),
     _plan_delete_file, "medium"),
    (re.compile(r"^(?:read|cat|show contents of)\s+(?:file\s+)?(?P<path>[\w./~\-\s]+)", re.I),
     _plan_read_file, "low"),
    (re.compile(r"^write\s+(?:\"(?P<content>[^\"]+)\"|\s*)?\s*to\s+(?P<path>[\w./~\-]+)", re.I),
     _plan_write_file, "low"),
    (re.compile(r"^list\s+(?:files\s+(?:in\s+)?)?(?P<path>[\w./~\-]*)$", re.I),
     _plan_list_files, "low"),
    (re.compile(r"^find\s+(?:files?\s+)?(?:named\s+)?(?P<pattern>[\w.*?\-]+)", re.I),
     _plan_find_files, "low"),
    (re.compile(r"^(?:grep|search)\s+(?:for\s+)?(?P<pattern>[\w\s]+)(?:\s+in\s+(?P<path>[\w./~\-]+))?$", re.I),
     _plan_grep, "low"),
    (re.compile(r"^copy\s+(?P<src>[\w./~\-]+)\s+(?:to\s+)?(?P<dst>[\w./~\-]+)", re.I),
     _plan_copy_file, "low"),
    (re.compile(r"^move\s+(?P<src>[\w./~\-]+)\s+(?:to\s+)?(?P<dst>[\w./~\-]+)", re.I),
     _plan_move_file, "medium"),
    (re.compile(r"^rename\s+(?P<src>[\w./~\-]+)\s+(?:to\s+)?(?P<dst>[\w./~\-]+)", re.I),
     _plan_rename_file, "low"),
    (re.compile(r"^(?:zip|compress|archive)\s+(?P<src>[\w./~\-]+)(?:\s+(?:to|as)\s+(?P<dst>[\w./~\-]+))?", re.I),
     _plan_zip, "low"),
    (re.compile(r"^(?:unzip|extract)\s+(?P<src>[\w./~\-]+)(?:\s+(?:to|into)\s+(?P<dst>[\w./~\-]+))?", re.I),
     _plan_unzip, "low"),
    (re.compile(r"^chmod\s+(?P<mode>[0-7]{3,4})\s+(?P<path>[\w./~\-]+)", re.I),
     _plan_chmod, "medium"),
    (re.compile(r"^(?:check|does|if)\s+(?:file\s+)?(?P<path>[\w./~\-]+)\s+exist", re.I),
     _plan_check_file, "low"),

    # ── Browser / URL ─────────────────────────────────────────
    (re.compile(r"^(?:go to|open|navigate to|visit)\s+(?P<url>https?://\S+|www\.\S+|\w+\.\w{2,})", re.I),
     _plan_open_url, "low"),
    (re.compile(r"^(?:search|google|look up)\s+(?:for\s+)?(?P<query>.+)$", re.I),
     _plan_search_web, "low"),

    # ── Package install ───────────────────────────────────────
    (re.compile(r"^(?:pip install|install pip)\s+(?P<pkg>[\w\-\[\],]+)", re.I),
     _plan_install_pip, "medium"),
    (re.compile(r"^(?:apt install|apt-get install|sudo apt install)\s+(?P<pkg>[\w\-]+)", re.I),
     _plan_install_apt, "medium"),

    # ── Git ───────────────────────────────────────────────────
    (re.compile(r"^git clone\s+(?P<url>\S+)", re.I),
     _plan_git_clone, "low"),
    (re.compile(r"^git commit\s+(?:\-m\s+)?[\"']?(?P<msg>[^\"']+)[\"']?$", re.I),
     _plan_git_commit, "medium"),

    # ── Python ────────────────────────────────────────────────
    (re.compile(r"^(?:run|execute)\s+(?P<script>[\w./~\-]+\.py)(?:\s+.*)?$", re.I),
     _plan_python_run, "medium"),

    # ── GUI: text / keyboard ──────────────────────────────────
    (re.compile(r"^type\s+(?:\"(?P<text>[^\"]+)\"|'(?P<text2>[^']+)'|(?P<text3>.+)$)", re.I),
     _plan_type_text, "low"),
    (re.compile(r"^press\s+hotkey\s+(?P<combo>[\w\+]+)$", re.I),
     _plan_hotkey, "low"),
    (re.compile(r"^press\s+(?P<key>[\w\s\+]+)$", re.I),
     _plan_press_key, "low"),

    # ── GUI: mouse ────────────────────────────────────────────
    (re.compile(r"^double.click\s+(?P<target>.+)$", re.I),
     _plan_double_click, "low"),
    (re.compile(r"^right.click\s+(?P<target>.+)$", re.I),
     _plan_right_click, "low"),
    (re.compile(r"^click\s+(?P<target>.+)$", re.I),
     _plan_click, "low"),

    # ── GUI: scroll / window ──────────────────────────────────
    (re.compile(r"^scroll\s+(?P<direction>up|down|left|right)(?:\s+(?P<amount>\d+))?", re.I),
     _plan_scroll, "low"),
    (re.compile(r"^maximize(?:\s+window)?", re.I),
     _plan_maximize_window, "low"),
    (re.compile(r"^minimize(?:\s+window)?", re.I),
     _plan_minimize_window, "low"),
    (re.compile(r"^close\s+window", re.I),
     _plan_close_window, "low"),
    (re.compile(r"^(?:focus|switch to)\s+(?P<target>[\w\s\-]+)\s+window", re.I),
     _plan_focus_window, "low"),

    # ── Wait ──────────────────────────────────────────────────
    (re.compile(r"^wait\s+(?:for\s+)?(?P<seconds>\d+(?:\.\d+)?)\s+seconds?", re.I),
     _plan_wait, "low"),
    (re.compile(r"^wait\s+(?:until|for)\s+[\"']?(?P<text>[^\"']+)[\"']?\s+(?:appears|is visible|on screen)", re.I),
     _plan_wait_for_text, "low"),

    # ── System ────────────────────────────────────────────────
    (re.compile(r"^(?:run|execute)\s+(?:command\s+)?(?P<cmd>.+)$", re.I),
     _plan_run_command, "medium"),
    (re.compile(r"^(?:kill|stop)\s+(?:process\s+)?(?P<proc>[\w\-\.]+)", re.I),
     _plan_kill_process, "medium"),
    (re.compile(r"^(?:take|capture)\s+(?:a\s+)?screenshot", re.I),
     _plan_take_screenshot, "low"),
    (re.compile(r"^(?:system info|sysinfo|show resources)", re.I),
     _plan_system_info, "low"),
    (re.compile(r"^(?:network info|show network|ip address)", re.I),
     _plan_network_info, "low"),
    (re.compile(r"^(?:volume|set volume)\s+(?P<action>up|down|mute|unmute|max|to)\s*(?P<level>\d+)?", re.I),
     _plan_volume, "low"),
    (re.compile(r"^brightness\s+(?P<action>up|down|to)\s*(?P<level>\d+)?", re.I),
     _plan_brightness, "low"),
    (re.compile(r"^(?:notify|notify-send|notification)\s+(?P<msg>.+)$", re.I),
     _plan_notify, "low"),
    (re.compile(r"^copy\s+[\"']?(?P<text>[^\"']+)[\"']?\s+to\s+clipboard", re.I),
     _plan_clipboard_copy, "low"),
    (re.compile(r"^ssh\s+(?:(?P<user>[\w\-]+)@)?(?P<host>[\w\-\.]+)", re.I),
     _plan_ssh, "medium"),

    # ── File save ─────────────────────────────────────────────
    (re.compile(r"^save\s+as\s+(?P<name>[\w./~\-\.]+)", re.I),
     _plan_save_as, "low"),
    (re.compile(r"^save\s+(?:the\s+)?(?:file|document)?$", re.I),
     _plan_save, "low"),
]


# High-risk patterns that always require user confirmation
HIGH_RISK_PATTERNS = re.compile(
    r"(delete|remove|rm|format|wipe|destroy|drop\s+table|"
    r"sudo\s+rm\s+-rf|shutdown|reboot|kill\s+process|"
    r"truncate|overwrite|purge)",
    re.I,
)


# ─────────────────────────────────────────────────────────────
#  TASK PLANNER
# ─────────────────────────────────────────────────────────────

class TaskPlanner:
    """
    Converts a natural language instruction into an ExecutionPlan.

    Stage 1: Rule-based fast path (handles common patterns instantly)
    Stage 2: LLM-based planning (for complex / multi-step instructions)
    Stage 3: Fallback to single shell step if nothing matches

    Usage:
        planner = TaskPlanner()
        plan = planner.plan("open firefox and go to youtube and search for NEXUS AI")
        print(plan.summary())
        for step in plan.steps:
            print(step)
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self._llm    = None
        log.info("TaskPlanner ready — llm=%s", use_llm)

    # ── Public API ────────────────────────────────────────────

    def plan(self, instruction: str) -> ExecutionPlan:
        """
        Build an execution plan for the given instruction.

        Parameters
        ----------
        instruction : natural language task description

        Returns
        -------
        ExecutionPlan
        """
        instruction = instruction.strip()
        if not instruction:
            return ExecutionPlan(instruction="", error="Empty instruction.")

        log.info("Planning: %r", instruction)

        # Split compound instructions at conjunctions
        segments = self._split_instruction(instruction)

        if len(segments) > 1:
            return self._plan_compound(instruction, segments)

        # Single instruction — try rules first
        plan = self._rule_plan(instruction)
        if plan.success:
            return plan

        # LLM fallback
        if self.use_llm:
            plan = self._llm_plan(instruction)
            if plan.success:
                return plan

        # Last resort: treat as raw shell command
        return self._fallback_plan(instruction)

    def explain(self, instruction: str) -> str:
        """Return a human-readable explanation of what would be done."""
        plan = self.plan(instruction)
        if not plan.success:
            return f"Could not plan: {plan.error}"
        return plan.summary()

    # ── Compound instruction splitting ────────────────────────

    def _split_instruction(self, text: str) -> list[str]:
        """Split 'do X and then do Y' or 'do X, do Y' into ['do X', 'do Y']."""
        splitters = re.compile(
            r"\s*(?:and then|then|after that|and also|;|,\s*(?:then\s*)?|\s+then\s+)\s*",
            re.I,
        )
        parts = splitters.split(text)
        return [p.strip() for p in parts if p.strip()]

    def _plan_compound(self, instruction: str, segments: list[str]) -> ExecutionPlan:
        """Plan each segment and concatenate, adjusting step indices."""
        all_steps: list[Step] = []
        offset = 0

        for segment in segments:
            sub_plan = self.plan(segment)
            if not sub_plan.steps:
                continue
            for step in sub_plan.steps:
                step.index      += offset
                step.depends_on  = [d + offset for d in step.depends_on]
                all_steps.append(step)
            offset += len(sub_plan.steps)

        if not all_steps:
            return ExecutionPlan(
                instruction=instruction,
                error="No steps could be planned for any segment.",
            )

        risk = "high" if HIGH_RISK_PATTERNS.search(instruction) else "medium"
        est  = sum(s.timeout_sec * 0.5 for s in all_steps)

        return ExecutionPlan(
            instruction = instruction,
            steps       = all_steps,
            goal        = f"Execute: {instruction}",
            risk_level  = risk,
            requires_confirmation = risk == "high",
            estimated_duration_sec = est,
        )

    # ── Rule-based planning ───────────────────────────────────

    def _rule_plan(self, instruction: str) -> ExecutionPlan:
        """Try all rules and return the first match."""
        for pattern, builder, risk in RULES:
            match = pattern.match(instruction.strip())
            if not match:
                continue

            try:
                raw_steps = builder(match, instruction)
            except Exception as e:
                log.debug("Rule builder failed for %r: %s", instruction, e)
                continue

            if not raw_steps:
                continue

            steps = []
            for i, s in enumerate(raw_steps, 1):
                steps.append(Step(
                    index       = i,
                    type        = s["type"],
                    action      = s["action"],
                    target      = s.get("target", ""),
                    params      = s.get("params", {}),
                    depends_on  = s.get("depends_on", []),
                    description = s.get("description", f"Step {i}"),
                    optional    = s.get("optional", False),
                    retry       = s.get("retry", 0),
                    timeout_sec = s.get("timeout_sec", 30.0),
                ))

            is_risky = bool(HIGH_RISK_PATTERNS.search(instruction))
            final_risk = "high" if is_risky else risk
            est = sum(s.timeout_sec * 0.4 for s in steps)

            plan = ExecutionPlan(
                instruction = instruction,
                steps       = steps,
                goal        = f"Execute: {instruction}",
                risk_level  = final_risk,
                requires_confirmation = is_risky,
                estimated_duration_sec = est,
            )
            log.info("Rule-based plan: %d steps, risk=%s", len(steps), plan.risk_level)
            return plan

        return ExecutionPlan(instruction=instruction, error="No rule matched.")

    # ── LLM-based planning ────────────────────────────────────

    def _llm_plan(self, instruction: str) -> ExecutionPlan:
        """Use the NEXUS LLM to generate a structured plan."""
        try:
            from core.llm import get_llm
            llm = get_llm()
            if not llm.is_ready:
                return ExecutionPlan(instruction=instruction, error="LLM offline.")

            system = """You are an automation planner for a Linux AI assistant called NEXUS.
Convert the user instruction into a JSON execution plan.

STEP TYPES:
- shell: run shell commands, launch apps, file operations
- gui: mouse clicks, keyboard input, window management (via pyautogui)
- web: open URLs, search the web
- wait: pause for N seconds or wait for a condition
- check: verify a condition before continuing

Return ONLY this JSON, nothing else:
{
  "goal": "what success looks like",
  "risk": "low|medium|high",
  "steps": [
    {
      "type": "shell|gui|web|wait|check",
      "action": "specific action name",
      "target": "what to act on",
      "params": {},
      "depends_on": [],
      "description": "human description",
      "optional": false,
      "retry": 0,
      "timeout_sec": 30
    }
  ]
}

Action names by type:
- shell: run_command, launch_app, kill_app, create_file, delete_file, write_file
- gui: click, double_click, right_click, type_text, press_key, hotkey, scroll,
       navigate_url, wait_window, focus_window, maximize_window, minimize_window,
       close_window, screenshot, drag_drop
- web: open_url, search_web
- wait: sleep, wait_for_text, wait_for_window
- check: check_screen_text, check_file_exists, check_process_running, check_python_import"""

            raw = llm.chat(instruction, system=system, task="fast")

            import json
            raw = raw.strip().strip("```json").strip("```").strip()
            data = json.loads(raw)

            steps = []
            for i, s in enumerate(data.get("steps", []), 1):
                steps.append(Step(
                    index       = i,
                    type        = s.get("type", "shell"),
                    action      = s.get("action", "run_command"),
                    target      = s.get("target", ""),
                    params      = s.get("params", {}),
                    depends_on  = s.get("depends_on", []),
                    description = s.get("description", f"Step {i}"),
                    optional    = s.get("optional", False),
                    retry       = s.get("retry", 0),
                    timeout_sec = float(s.get("timeout_sec", 30)),
                ))

            risk = data.get("risk", "medium")
            plan = ExecutionPlan(
                instruction = instruction,
                steps       = steps,
                goal        = data.get("goal", instruction),
                risk_level  = risk,
                requires_confirmation = risk == "high",
                estimated_duration_sec = len(steps) * 2.0,
            )
            log.info("LLM plan: %d steps, risk=%s", len(steps), risk)
            return plan

        except Exception as e:
            log.warning("LLM planning failed: %s", e)
            return ExecutionPlan(instruction=instruction, error=str(e))

    # ── Fallback ──────────────────────────────────────────────

    def _fallback_plan(self, instruction: str) -> ExecutionPlan:
        """Treat the instruction as a raw shell command."""
        step = Step(
            index       = 1,
            type        = "shell",
            action      = "run_command",
            target      = instruction,
            description = f"Run: {instruction}",
        )
        return ExecutionPlan(
            instruction = instruction,
            steps       = [step],
            goal        = f"Execute: {instruction}",
            risk_level  = "medium" if HIGH_RISK_PATTERNS.search(instruction) else "low",
        )


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    # Allow running as: python automation/planner.py ...
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    logging.basicConfig(level=logging.INFO)

    planner = TaskPlanner(use_llm=False)

    tests = [
        "open firefox",
        "go to github.com",
        "search for NEXUS AI python",
        "take a screenshot",
        "wait 3 seconds",
        "create file /tmp/test.txt",
        "run command ls -la",
        "copy /tmp/a.txt /tmp/b.txt",
        "zip my_project output.zip",
        "pip install requests",
        "press hotkey ctrl+c",
        "scroll down 5",
        "open mousepad",
        "open firefox and go to youtube.com and search for python tutorials",
        "open mousepad, type hello world, save as test.txt",
    ]

    if len(sys.argv) > 1:
        tests = [" ".join(sys.argv[1:])]

    for instruction in tests:
        print(f"\n  Instruction: {instruction!r}")
        plan = planner.plan(instruction)
        if plan.success:
            print(f"  Steps: {plan.step_count} | Risk: {plan.risk_level} | "
                  f"Est: {plan.estimated_duration_sec:.1f}s")
            for step in plan.steps:
                print(f"    {step}")
        else:
            print(f"  Error: {plan.error}")