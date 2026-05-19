"""
core/autonomous_planner.py
Backward-compatibility shim — the real implementation lives in
automation/autonomous_planner.py (merged with physical execution).

All imports that use  `from core.autonomous_planner import ...`
continue to work unchanged.
"""

from automation.autonomous_planner import (  # noqa: F401
    AutonomousPlanner,
    AutonomousPlan,
    AStep,
    _ics_event,
    _system_notify,
    _schedule_at_command,
)

__all__ = [
    "AutonomousPlanner",
    "AutonomousPlan",
    "AStep",
    "_ics_event",
    "_system_notify",
    "_schedule_at_command",
]
