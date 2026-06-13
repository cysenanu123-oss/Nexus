#!/usr/bin/env python3
"""Final test of the NEXUS retry system."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.retry_system import execute_shell_with_retry

def demonstrate_retry_system():
    """Demonstrate the retry system in action."""
    print("🎯 NEXUS Advanced Retry System Demo")
    print("=" * 60)

    # Test 1: Successful command
    print("📋 Test 1: Successful Command")
    print("Command: echo 'NEXUS retry system is working!'")
    result = execute_shell_with_retry(
        "echo 'NEXUS retry system is working!'",
        max_attempts=5,
        timeout=10.0
    )
    print(f"✓ {result.summary()}")
    print(f"📤 Output: {result.final_stdout.strip()}")
    print()

    # Test 2: Command with network-like delay (simulated)
    print("📋 Test 2: Command with Timeout (demonstrates retry strategies)")
    print("Command: sleep 0.5 && echo 'Completed after delay'")
    result = execute_shell_with_retry(
        "sleep 0.5 && echo 'Completed after delay'",
        max_attempts=3,
        timeout=10.0
    )
    print(f"✓ {result.summary()}")
    if result.success:
        print(f"📤 Output: {result.final_stdout.strip()}")
    print()

    # Test 3: Command that will ultimately fail (to show all strategies)
    print("📋 Test 3: Failing Command (shows all 5 retry strategies)")
    print("Command: exit 1 (always fails)")
    result = execute_shell_with_retry(
        "exit 1",
        max_attempts=5,
        timeout=5.0
    )
    print(f"✗ {result.summary()}")
    print("🔄 Retry strategies used:")
    for i, attempt in enumerate(result.attempts, 1):
        print(f"   {i}. {attempt.strategy.value.upper():<12} - "
              f"Delay: {attempt.delay_before:.1f}s, "
              f"Timeout: {attempt.timeout:.1f}s")
    print()

    print("🎊 SUMMARY")
    print("─" * 60)
    print("✅ NEXUS now has a robust retry system with 5 attempts!")
    print("🔧 Integration points:")
    print("   • automation/executor.py - Enhanced step execution")
    print("   • automation/shell_agent.py - Command execution")
    print("   • main.py - Shell command interface")
    print("   • core/retry_system.py - New retry engine")
    print()
    print("🚀 Key features implemented:")
    print("   1. 5 configurable retry attempts")
    print("   2. Multiple strategies: immediate, standard, extended, alternative, last resort")
    print("   3. Intelligent failure analysis")
    print("   4. Alternative command suggestions")
    print("   5. Exponential backoff with jitter")
    print("   6. Detailed logging and progress tracking")
    print()
    print("💡 When NEXUS runs commands that fail, it will now:")
    print("   • Try immediately (for quick transient issues)")
    print("   • Wait 2 seconds and try again (for temporary issues)")
    print("   • Wait 5 seconds with higher timeout (for slower responses)")
    print("   • Try alternative commands/approaches (for method issues)")
    print("   • Wait 15 seconds as a last resort (for severe issues)")

if __name__ == "__main__":
    demonstrate_retry_system()