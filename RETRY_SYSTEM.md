# NEXUS Advanced Retry System

## Overview

NEXUS now includes a robust retry system that automatically retries failed commands up to 5 times using intelligent strategies. No more manual retries when commands fail due to network issues, temporary resource problems, or transient errors!

## Features

### 🔄 **5-Attempt Retry Logic**
- **Attempt 1**: Immediate retry (for quick transient failures)
- **Attempt 2**: Standard retry (2s delay)
- **Attempt 3**: Extended retry (5s delay + higher timeout)
- **Attempt 4**: Alternative approach (different command/parameters)
- **Attempt 5**: Last resort (15s delay + maximum settings)

### 🧠 **Intelligent Failure Analysis**
The system automatically detects failure types:
- **Network failures**: Connection timeouts, unreachable hosts
- **Permission errors**: Access denied, forbidden operations
- **Resource issues**: Busy resources, locked files
- **Command errors**: Command not found, invalid arguments
- **System errors**: Unexpected failures

### 🛠 **Alternative Command Support**
For common commands, the system tries alternatives when the original fails:
- `ping` → `ping -c 1` → `ping -c 3` → `nping` → `fping`
- `curl` → `curl -f` → `wget -O-` → `python3 -m urllib.request`
- `git` → `git --no-pager` → `GIT_TERMINAL_PROMPT=0 git`
- `ssh` → `ssh -o ConnectTimeout=10` → `ssh -o BatchMode=yes`

### ⚡ **Exponential Backoff with Jitter**
- Smart delays between retries prevent overwhelming failing services
- Random jitter (0-20%) prevents thundering herd problems
- Timeout increases with each attempt for slow operations

## Integration Points

### Automatic Integration
The retry system is automatically used in:

1. **Automation System** (`automation/executor.py`)
   - All automation steps now use intelligent retry
   - Configurable per-step retry settings

2. **Shell Commands** (`automation/shell_agent.py`)
   - All shell command executions are automatically retried
   - Detailed logging of retry attempts

3. **Main Shell Interface** (`main.py`)
   - The `shell` command in NEXUS CLI uses retry system
   - Shows retry progress to users

### Manual Usage
You can also use the retry system directly in your code:

```python
from core.retry_system import execute_shell_with_retry, RetryConfig, RetryExecutor

# Simple shell command with retry
result = execute_shell_with_retry(
    "ping google.com",
    max_attempts=5,
    timeout=10.0
)

# Custom configuration
config = RetryConfig(
    max_attempts=3,
    base_timeout=30.0,
    enable_jitter=True,
    alternative_commands={
        "mycommand": ["mycommand --fast", "mycommand --slow", "fallback_command"]
    }
)

executor = RetryExecutor(config)
result = executor.execute_with_retry("mycommand --option")

# Using with custom callable
def my_operation(timeout=30.0):
    # Your custom logic here
    return True, "Operation succeeded"

result = executor.execute_with_retry(my_operation)
```

## Configuration Options

### RetryConfig Parameters
- `max_attempts`: Maximum retry attempts (default: 5)
- `base_timeout`: Base timeout in seconds (default: 30.0)
- `enable_jitter`: Add randomness to delays (default: True)
- `enable_alternative_commands`: Try alternative commands (default: True)
- `log_all_attempts`: Log every attempt (default: True)
- `timeout_multipliers`: Timeout scaling per attempt (default: [1.0, 1.0, 2.0, 2.5, 3.0])

### Strategy-Specific Delays
- `immediate_delay`: 0.0s
- `standard_delay`: 2.0s
- `extended_delay`: 5.0s
- `last_resort_delay`: 15.0s

## Result Information

The retry system returns detailed `RetryResult` objects containing:

```python
result = execute_shell_with_retry("some_command")

print(f"Success: {result.success}")
print(f"Attempts: {result.attempt_count}")
print(f"Total time: {result.total_elapsed:.2f}s")
print(f"Output: {result.final_stdout}")
print(f"Summary: {result.summary()}")

# Access individual attempts
for attempt in result.attempts:
    print(f"Attempt {attempt.attempt_number}: {attempt.strategy.value}")
    print(f"  Success: {attempt.success}")
    print(f"  Time: {attempt.elapsed_time:.2f}s")
    print(f"  Error: {attempt.stderr}")
```

## Logging

The retry system provides comprehensive logging:

```
INFO: Starting retry execution: ping google.com (max_attempts=5)
INFO: Attempt 1/5: ✓ [immediate] (0.05s)
INFO: ✓ SUCCESS after 1 attempt(s) (0.05s total) - Strategy: immediate
```

For failures:
```
WARNING: Attempt 1/5: ✗ [immediate] (3.00s) - Command timed out after 3.0s
INFO: Waiting 2.1s before attempt 2
WARNING: Attempt 2/5: ✗ [standard] (3.00s) - Command timed out after 3.0s
WARNING: All retry attempts failed. Final error: Command timed out after 3.0s
```

## Benefits

### For Users
- **Increased Reliability**: Commands succeed more often
- **Better Experience**: Less manual intervention required
- **Transparency**: Clear feedback on retry progress
- **Configurability**: Adjust retry behavior per needs

### For Developers
- **Robust Operations**: Network and system operations more reliable
- **Error Handling**: Automatic recovery from transient failures
- **Debugging**: Detailed logs for troubleshooting
- **Extensibility**: Easy to add new retry strategies

## Examples

### Network Operations
```python
# Robust ping with retries
result = execute_shell_with_retry("ping -c 1 google.com")

# HTTP requests with fallbacks
result = execute_shell_with_retry("curl https://api.example.com/data")
```

### System Operations
```python
# File operations that might fail due to locks
result = execute_shell_with_retry("cp large_file.dat backup/")

# Package installations
result = execute_shell_with_retry("pip install some-package")
```

### Custom Operations
```python
config = RetryConfig(
    max_attempts=3,
    alternative_commands={
        "deploy": ["deploy.sh", "deploy.sh --force", "deploy_fallback.sh"]
    }
)

executor = RetryExecutor(config)
result = executor.execute_with_retry("deploy.sh --env production")
```

## Testing

Run the test suite to verify the retry system:

```bash
python3 test_simple_retry.py      # Basic functionality test
python3 test_final_retry.py       # Comprehensive demo
```

---

🎉 **The retry system makes NEXUS significantly more robust and reliable for real-world usage!**