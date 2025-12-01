import logging
import sys
import json
import datetime
import traceback
import threading
from pathlib import Path

# Global lock to ensure atomic writes to the failures file
_failures_lock = threading.Lock()

def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configures the root logger.
    
    - INFO level goes to stdout (standard application output).
    - ERROR/WARNING goes to stderr.
    - DEBUG level goes to stderr if verbose is True.
    """
    logger = logging.getLogger("arc_agi")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Handler for standard info/results (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    # Simple format for CLI output, or just message for cleanliness if we are printing tables manually
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    
    # Handler for debug/errors (stderr)
    # We filter this so it only takes ERROR (always) and DEBUG (if verbose)
    # But standard StreamHandler(sys.stderr) takes everything >= level.
    
    # If verbose, we want detailed logs to stderr
    if verbose:
        debug_handler = logging.StreamHandler(sys.stderr)
        debug_handler.setLevel(logging.DEBUG)
        debug_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        debug_handler.setFormatter(debug_formatter)
        logger.addHandler(debug_handler)
    
    # We generally don't attach the console_handler to the root logger if we are 
    # manually printing tables with print(). 
    # However, for this refactor, we will use logger.info() for standard messages.
    # But carefully: print_table_header/row in reporting.py uses print() for strict formatting.
    # We should probably leave "presentation" prints as prints, and usage logs as logs.
    
    return logger

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"arc_agi.{name}")

def log_failure(
    run_timestamp: str,
    task_id: str,
    run_id: str,
    error: Exception,
    model: str = None,
    step: str = None,
    test_index: int = None
):
    """
    Logs a structured failure record to a JSONL file.
    Designed to be a 'System of Record' for terminal failures.
    """
    try:
        # Construct the failures file path: logs/{timestamp}_failures.jsonl
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        failures_path = log_dir / f"{run_timestamp}_failures.jsonl"

        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "task_id": task_id,
            "test_index": test_index,
            "step": step,
            "model": model,
            "run_id": run_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "stack_trace": traceback.format_exc()
        }

        # Serialize safely (handle non-serializable types)
        json_entry = json.dumps(entry, default=str)

        with _failures_lock:
            with open(failures_path, "a", encoding="utf-8") as f:
                f.write(json_entry + "\n")
                f.flush() # Ensure it hits the disk
                
    except Exception as e:
        # Fallback: Don't let logging crash the application
        print(f"CRITICAL: Failed to log failure to DLQ: {e}", file=sys.stderr)
