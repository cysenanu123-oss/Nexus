"""
core/retry_system.py

NEXUS Robust Retry System — Advanced command execution with 5-attempt retry logic
using multiple strategies.

Features:
  - 5 configurable retry attempts with escalating strategies
  - Multiple retry strategies per attempt (exponential backoff, jitter, different timeouts)
  - Graceful degradation for different failure modes
  - Command environment modification between attempts
  - Detailed logging of all attempts and failures
  - Support for different command types (subprocess, shell, api calls, etc.)

Retry Strategies:
  1. Quick retry (immediate) - for transient network issues
  2. Standard retry (2s delay) - for temporary resource issues
  3. Extended retry (5s delay + higher timeout) - for slow responses
  4. Alternative approach (different command/params) - for method failures
  5. Last resort (15s delay + maximum timeout) - for severe issues

Usage:
    from core.retry_system import RetryExecutor, RetryConfig

    executor = RetryExecutor()
    result = executor.execute_with_retry(
        command="ping google.com",
        config=RetryConfig(max_attempts=5)
    )
"""

import os
import sys
import time
import random
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Optional, Any, Union, Dict, List
from enum import Enum

from core.shell_safety import check_command

log = logging.getLogger("nexus.retry_system")


class FailureType(Enum):
    """Types of command failures that trigger different retry strategies."""
    TIMEOUT = "timeout"
    NETWORK = "network"
    PERMISSION = "permission"
    RESOURCE = "resource"
    COMMAND_NOT_FOUND = "command_not_found"
    INVALID_ARGS = "invalid_args"
    SYSTEM_ERROR = "system_error"
    UNKNOWN = "unknown"


class RetryStrategy(Enum):
    """Available retry strategies for different attempts."""
    IMMEDIATE = "immediate"          # No delay, same command
    STANDARD = "standard"            # 2s delay, same command
    EXTENDED = "extended"            # 5s delay, higher timeout
    ALTERNATIVE = "alternative"      # Different command/approach
    LAST_RESORT = "last_resort"      # Maximum delay, all options


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 5
    base_timeout: float = 30.0
    enable_jitter: bool = True
    enable_alternative_commands: bool = True
    log_all_attempts: bool = True

    # Strategy-specific configurations
    immediate_delay: float = 0.0
    standard_delay: float = 2.0
    extended_delay: float = 5.0
    last_resort_delay: float = 15.0

    # Timeout multipliers per attempt
    timeout_multipliers: List[float] = field(default_factory=lambda: [1.0, 1.0, 2.0, 2.5, 3.0])

    # Alternative commands to try for different failure types
    alternative_commands: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class RetryAttempt:
    """Details of a single retry attempt."""
    attempt_number: int
    strategy: RetryStrategy
    command: Union[str, List[str]]
    timeout: float
    delay_before: float
    timestamp: float
    success: bool = False
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    error: Optional[Exception] = None
    failure_type: FailureType = FailureType.UNKNOWN
    elapsed_time: float = 0.0


@dataclass
class RetryResult:
    """Result of executing a command with retry logic."""
    success: bool = False
    final_stdout: str = ""
    final_stderr: str = ""
    final_exit_code: Optional[int] = None
    attempts: List[RetryAttempt] = field(default_factory=list)
    total_elapsed: float = 0.0
    failure_reason: str = ""

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def successful_attempt(self) -> Optional[RetryAttempt]:
        """Return the attempt that succeeded, if any."""
        for attempt in self.attempts:
            if attempt.success:
                return attempt
        return None

    def summary(self) -> str:
        """Human-readable summary of the retry result."""
        if self.success:
            successful = self.successful_attempt
            return (f"✓ SUCCESS after {self.attempt_count} attempt(s) "
                   f"({self.total_elapsed:.2f}s total) - "
                   f"Strategy: {successful.strategy.value}")
        else:
            return (f"✗ FAILED after {self.attempt_count} attempt(s) "
                   f"({self.total_elapsed:.2f}s total) - "
                   f"Reason: {self.failure_reason}")


class RetryExecutor:
    """
    Robust command executor with intelligent retry strategies.

    Handles subprocess execution, shell commands, and can be extended
    for other types of operations.
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self._setup_default_alternatives()

    def _setup_default_alternatives(self):
        """Setup default alternative commands for common cases."""
        if not self.config.alternative_commands:
            self.config.alternative_commands = {
                "ping": ["ping -c 1", "ping -c 3", "nping", "fping"],
                "curl": ["curl -f", "wget -O-", "python3 -m urllib.request"],
                "git": ["git --no-pager", "GIT_TERMINAL_PROMPT=0 git"],
                "ssh": ["ssh -o ConnectTimeout=10", "ssh -o BatchMode=yes"],
                "wget": ["wget --timeout=30", "curl -O"],
                "nmap": ["nmap -T4", "nmap -T3", "masscan"],
            }

    def execute_with_retry(
        self,
        command: Union[str, List[str], Callable],
        config: Optional[RetryConfig] = None,
        **kwargs
    ) -> RetryResult:
        """
        Execute a command with intelligent retry logic.

        Args:
            command: Command to execute (string, list, or callable)
            config: Override default retry configuration
            **kwargs: Additional arguments passed to execution method

        Returns:
            RetryResult with success status and attempt details
        """
        cfg = config or self.config
        result = RetryResult()
        start_time = time.time()

        log.info("Starting retry execution: %s (max_attempts=%d)",
                str(command)[:100], cfg.max_attempts)

        for attempt_num in range(1, cfg.max_attempts + 1):
            strategy = self._get_strategy_for_attempt(attempt_num)

            # Determine command and timeout for this attempt
            current_cmd = self._prepare_command(command, attempt_num, strategy, result)
            timeout = self._calculate_timeout(cfg.base_timeout, attempt_num, cfg.timeout_multipliers)
            delay = self._calculate_delay(strategy, attempt_num, cfg)

            # Apply delay before attempt (except first attempt)
            if attempt_num > 1:
                if cfg.enable_jitter:
                    delay += random.uniform(0, delay * 0.2)  # Add 0-20% jitter
                log.info(f"Waiting {delay:.1f}s before attempt {attempt_num}")
                time.sleep(delay)

            # Create attempt record
            attempt = RetryAttempt(
                attempt_number=attempt_num,
                strategy=strategy,
                command=current_cmd,
                timeout=timeout,
                delay_before=delay if attempt_num > 1 else 0.0,
                timestamp=time.time()
            )

            # Execute the attempt
            self._execute_attempt(attempt, **kwargs)
            result.attempts.append(attempt)

            if cfg.log_all_attempts or not attempt.success:
                log.info(f"Attempt {attempt_num}/{cfg.max_attempts}: {self._format_attempt_result(attempt)}")

            # Check if successful
            if attempt.success:
                result.success = True
                result.final_stdout = attempt.stdout
                result.final_stderr = attempt.stderr
                result.final_exit_code = attempt.exit_code
                break
            else:
                # Analyze failure for next attempt strategy
                attempt.failure_type = self._analyze_failure(attempt)
                if not result.failure_reason:
                    result.failure_reason = self._format_failure_reason(attempt)

        result.total_elapsed = time.time() - start_time

        # Log final result
        if result.success:
            log.info(result.summary())
        else:
            log.warning(result.summary())
            log.warning("All retry attempts failed. Final error: %s", result.failure_reason)

        return result

    def _get_strategy_for_attempt(self, attempt_num: int) -> RetryStrategy:
        """Determine which retry strategy to use for this attempt number."""
        strategies = [
            RetryStrategy.IMMEDIATE,    # Attempt 1
            RetryStrategy.STANDARD,     # Attempt 2
            RetryStrategy.EXTENDED,     # Attempt 3
            RetryStrategy.ALTERNATIVE,  # Attempt 4
            RetryStrategy.LAST_RESORT   # Attempt 5
        ]
        return strategies[min(attempt_num - 1, len(strategies) - 1)]

    def _prepare_command(
        self,
        original_cmd: Union[str, List[str]],
        attempt_num: int,
        strategy: RetryStrategy,
        previous_result: RetryResult
    ) -> Union[str, List[str]]:
        """Prepare the command for this specific attempt, potentially modifying it."""

        if strategy != RetryStrategy.ALTERNATIVE or attempt_num < 4:
            return original_cmd

        # For alternative strategy, try different command variations
        if isinstance(original_cmd, str):
            base_cmd = original_cmd.split()[0]

            # Check if we have alternative commands for this base command
            alternatives = self.config.alternative_commands.get(base_cmd, [])
            if alternatives and attempt_num <= len(alternatives) + 3:
                alt_idx = attempt_num - 4  # Start alternatives from attempt 4
                if alt_idx < len(alternatives):
                    # Replace base command with alternative
                    parts = original_cmd.split()
                    alt_cmd = alternatives[alt_idx]
                    if " " in alt_cmd:
                        # Alternative has arguments, replace whole command start
                        alt_parts = alt_cmd.split()
                        return " ".join(alt_parts + parts[1:])
                    else:
                        # Simple replacement of base command
                        parts[0] = alt_cmd
                        return " ".join(parts)

        return original_cmd

    def _calculate_timeout(self, base_timeout: float, attempt_num: int, multipliers: List[float]) -> float:
        """Calculate timeout for this attempt."""
        multiplier = multipliers[min(attempt_num - 1, len(multipliers) - 1)]
        return base_timeout * multiplier

    def _calculate_delay(self, strategy: RetryStrategy, attempt_num: int, config: RetryConfig) -> float:
        """Calculate delay before this attempt."""
        delays = {
            RetryStrategy.IMMEDIATE: config.immediate_delay,
            RetryStrategy.STANDARD: config.standard_delay,
            RetryStrategy.EXTENDED: config.extended_delay,
            RetryStrategy.ALTERNATIVE: config.extended_delay,
            RetryStrategy.LAST_RESORT: config.last_resort_delay
        }
        return delays.get(strategy, config.standard_delay)

    def _execute_attempt(self, attempt: RetryAttempt, **kwargs) -> None:
        """Execute a single attempt and populate the attempt object with results."""
        start_time = time.time()

        try:
            if isinstance(attempt.command, str):
                # Shell command execution — deny-list guard (pipes/redirects
                # allowed here, but destructive commands are refused).
                verdict = check_command(attempt.command, allow_shell_operators=True)
                if not verdict.ok:
                    attempt.exit_code = 126
                    attempt.stdout = ""
                    attempt.stderr = f"refused by safety layer: {verdict.reason}"
                    attempt.success = False
                    attempt.failure_type = FailureType.PERMISSION
                    attempt.elapsed_time = time.time() - start_time
                    return
                result = subprocess.run(
                    attempt.command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=attempt.timeout,
                    **kwargs
                )
                attempt.exit_code = result.returncode
                attempt.stdout = result.stdout
                attempt.stderr = result.stderr
                attempt.success = (result.returncode == 0)

            elif isinstance(attempt.command, list):
                # List command execution
                result = subprocess.run(
                    attempt.command,
                    capture_output=True,
                    text=True,
                    timeout=attempt.timeout,
                    **kwargs
                )
                attempt.exit_code = result.returncode
                attempt.stdout = result.stdout
                attempt.stderr = result.stderr
                attempt.success = (result.returncode == 0)

            elif callable(attempt.command):
                # Callable execution
                try:
                    result = attempt.command(timeout=attempt.timeout, **kwargs)
                    if isinstance(result, tuple):
                        attempt.success, attempt.stdout = result
                    else:
                        attempt.success = bool(result)
                        attempt.stdout = str(result) if result else ""
                except Exception as e:
                    attempt.success = False
                    attempt.error = e
                    attempt.stderr = str(e)
            else:
                attempt.success = False
                attempt.stderr = f"Unsupported command type: {type(attempt.command)}"

        except subprocess.TimeoutExpired as e:
            attempt.success = False
            attempt.error = e
            attempt.stderr = f"Command timed out after {attempt.timeout}s"
            attempt.failure_type = FailureType.TIMEOUT

        except subprocess.CalledProcessError as e:
            attempt.success = False
            attempt.error = e
            attempt.exit_code = e.returncode
            attempt.stdout = e.stdout or ""
            attempt.stderr = e.stderr or ""

        except FileNotFoundError as e:
            attempt.success = False
            attempt.error = e
            attempt.stderr = f"Command not found: {e}"
            attempt.failure_type = FailureType.COMMAND_NOT_FOUND

        except PermissionError as e:
            attempt.success = False
            attempt.error = e
            attempt.stderr = f"Permission denied: {e}"
            attempt.failure_type = FailureType.PERMISSION

        except Exception as e:
            attempt.success = False
            attempt.error = e
            attempt.stderr = f"Unexpected error: {e}"
            attempt.failure_type = FailureType.SYSTEM_ERROR

        attempt.elapsed_time = time.time() - start_time

    def _analyze_failure(self, attempt: RetryAttempt) -> FailureType:
        """Analyze the type of failure to inform next retry strategy."""
        if attempt.failure_type != FailureType.UNKNOWN:
            return attempt.failure_type

        stderr = attempt.stderr.lower()

        if "timeout" in stderr or "timed out" in stderr:
            return FailureType.TIMEOUT
        elif any(net_err in stderr for net_err in ["network", "connection", "unreachable", "host not found"]):
            return FailureType.NETWORK
        elif any(perm_err in stderr for perm_err in ["permission", "access denied", "forbidden"]):
            return FailureType.PERMISSION
        elif "command not found" in stderr or "not recognized" in stderr:
            return FailureType.COMMAND_NOT_FOUND
        elif any(res_err in stderr for res_err in ["resource", "busy", "unavailable", "locked"]):
            return FailureType.RESOURCE
        elif attempt.exit_code == 1:
            return FailureType.INVALID_ARGS
        else:
            return FailureType.SYSTEM_ERROR

    def _format_attempt_result(self, attempt: RetryAttempt) -> str:
        """Format attempt result for logging."""
        status = "✓" if attempt.success else "✗"
        strategy_info = f"[{attempt.strategy.value}]"
        timing = f"({attempt.elapsed_time:.2f}s)"

        if attempt.success:
            return f"{status} {strategy_info} {timing}"
        else:
            error_info = attempt.stderr[:100] if attempt.stderr else str(attempt.error)[:100]
            return f"{status} {strategy_info} {timing} - {error_info}"

    def _format_failure_reason(self, attempt: RetryAttempt) -> str:
        """Format the failure reason for final result."""
        if attempt.stderr:
            return attempt.stderr[:200]
        elif attempt.error:
            return str(attempt.error)[:200]
        elif attempt.exit_code:
            return f"Command failed with exit code {attempt.exit_code}"
        else:
            return "Unknown failure"


# ─────────────────────────────────────────────────────────────
# Convenience functions for common use cases
# ─────────────────────────────────────────────────────────────

def execute_shell_with_retry(
    command: str,
    max_attempts: int = 5,
    timeout: float = 30.0,
    **kwargs
) -> RetryResult:
    """Convenience function for executing shell commands with retry."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_timeout=timeout
    )
    executor = RetryExecutor(config)
    return executor.execute_with_retry(command, **kwargs)


def execute_subprocess_with_retry(
    command: List[str],
    max_attempts: int = 5,
    timeout: float = 30.0,
    **kwargs
) -> RetryResult:
    """Convenience function for executing subprocess commands with retry."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_timeout=timeout
    )
    executor = RetryExecutor(config)
    return executor.execute_with_retry(command, **kwargs)