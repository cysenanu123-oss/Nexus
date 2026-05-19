"""
automation/__init__.py
NEXUS Automation Module — autonomous task execution engine.

The automation pipeline converts a natural language instruction into
an ordered sequence of steps and executes them reliably.

Pipeline:
    User instruction (text)
           ↓
    TaskPlanner   → ExecutionPlan (ordered Steps)
           ↓
    Executor      → runs each Step via the right agent
           ↓
    ShellAgent    → system commands, file ops, installs
    GUIAgent      → mouse, keyboard, window management
    WebAgent      → searches web for unknown commands
    AppReader     → reads running apps via accessibility API
           ↓
    Reporter      → real-time progress display

Quick usage:
    from automation import Automation
    auto = Automation()
    result = auto.run("open firefox and go to github.com")
    print(result.summary())
"""

from automation.automation          import Automation
from automation.planner             import TaskPlanner, ExecutionPlan, Step
from automation.executor            import Executor, ExecutionResult
from automation.reporter            import Reporter
from automation.autonomous_planner  import AutonomousPlanner, AutonomousPlan, AStep

__all__ = [
    "Automation",
    "TaskPlanner",
    "ExecutionPlan",
    "Step",
    "Executor",
    "ExecutionResult",
    "Reporter",
    "AutonomousPlanner",
    "AutonomousPlan",
    "AStep",
]