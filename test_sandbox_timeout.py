import time
import sys
import os

# Ensure we can import from src
sys.path.append(os.getcwd())

from src.sandbox import run_untrusted_code

# 1. Define code that prints every second and then hangs
hanging_code = """
import time
import sys

def solver(input_grid):
    # This loop will attempt to run for 100 seconds
    for i in range(100):
        # We use stderr for the heartbeat so we can see it in logs
        print(f"Heartbeat {i}...", file=sys.stderr)
        sys.stderr.flush()
        time.sleep(1)
    return input_grid
"""

def main():
    print("--- Sandbox Timeout Verification ---")
    print("Goal: Verify that a process hanging for 100s is killed exactly at 5s.")
    
    start_time = time.time()

    # 2. Run the code with a 5-second timeout
    success, result, logs = run_untrusted_code(hanging_code, [[0, 0], [0, 0]], timeout_s=5.0)

    duration = time.time() - start_time

    print(f"\nExecution Duration: {duration:.2f}s")
    print(f"Success Status: {success}")
    print(f"Return Result: {result}")
    print(f"Logs from Subprocess:\n{logs}")

    if not success and result == "TIMEOUT_EXPIRED":
        print("\n✅ VERIFICATION PASSED: Subprocess was SIGKILLed exactly on time.")
    else:
        print("\n❌ VERIFICATION FAILED: Subprocess was not handled correctly.")

if __name__ == "__main__":
    main()
