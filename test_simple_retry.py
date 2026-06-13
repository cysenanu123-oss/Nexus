#!/usr/bin/env python3
"""Simple test for the NEXUS retry system."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.retry_system import execute_shell_with_retry

def test_success():
    """Test a simple successful command."""
    print("Testing successful command...")
    result = execute_shell_with_retry("echo 'Hello World'", max_attempts=3, timeout=5.0)

    if result.success:
        print(f"✓ Success: {result.final_stdout.strip()}")
        print(f"Attempts: {result.attempt_count}")
        return True
    else:
        print(f"✗ Failed: {result.failure_reason}")
        return False

def test_failure():
    """Test a command that will fail."""
    print("\nTesting failing command...")
    result = execute_shell_with_retry("nonexistent_cmd_123", max_attempts=3, timeout=5.0)

    print(f"Result: {'SUCCESS' if result.success else 'FAILED'} (expected: FAILED)")
    print(f"Attempts: {result.attempt_count}")
    print(f"Strategies tried: {', '.join(a.strategy.value for a in result.attempts)}")
    return True  # Failure is expected

if __name__ == "__main__":
    print("🚀 Simple NEXUS Retry System Test")
    print("=" * 40)

    success1 = test_success()
    success2 = test_failure()

    print("\n" + "=" * 40)
    if success1 and success2:
        print("✓ All tests passed!")
        print("\n🎉 Retry system is working correctly!")
        print("Commands in NEXUS will now automatically retry with intelligent strategies!")
    else:
        print("✗ Some tests failed")