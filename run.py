import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime

from src.solver_engine import run_solver_mode
from src.default_engine import run_default_mode
from src.tasks import load_task
from src.logging import PrefixedStdout
from src.run_utils import find_task_path

def execute_task(args, task_path: Path, test_index: int, run_timestamp: str):
    task_id = task_path.stem
    prefix = f"{task_id}:{test_index} "
    
    # Update args for default mode compatibility
    # default_mode uses args.task and args.test
    args.task = str(task_path)
    args.test = test_index
    
    with PrefixedStdout(prefix):
        try:
            if args.solver or args.solver_testing:
                run_solver_mode(task_id, test_index, args.verbose, is_testing=args.solver_testing, run_timestamp=run_timestamp)
            else:
                run_default_mode(args)
        except Exception as e:
            # We catch exception here to ensure one task crashing doesn't stop the entire directory run
            # But we rely on engines to handle their own logging/crashing mostly.
            # run_solver_mode raises exception on critical error.
            # run_default_mode might exit sys.exit(1).
            # If run_default_mode calls sys.exit(1), it kills the process.
            # We should check run_default_mode.
            raise e

def main():
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser(description="Run ARC task test cases with multiple models in parallel.")
    
    # Task selection group
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", help="Task ID (e.g., 38007db0) or path to JSON file")
    task_group.add_argument("--task-directory", help="Directory containing task JSON files to run in batch")
    
    parser.add_argument("--test", type=int, default=1, help="Test case index (1-based, default: 1). Ignored if --task-directory is used.")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers (default: 10)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--models", type=str, help="Comma-separated list of models to run")
    parser.add_argument("--hint", type=str, default=None, help="Optional hint to provide to the model")
    parser.add_argument("--image", action="store_true", help="Generate an image for the task and include it in the prompt.")
    parser.add_argument("--trigger-deep-thinking", action="store_true", help="Append a deep thinking procedure to the prompt.")
    parser.add_argument("--generate-hint", action="store_true", help="Generate a hint for the task using a separate model call.")
    parser.add_argument("--generate-hint-model", type=str, default="gpt-5.1-high", help="Model to use for generating hints.")
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--solver", action="store_true", help="Enable solver mode.")
    mode_group.add_argument("--solver-testing", action="store_true", help="Enable solver testing mode with a smaller set of models.")
    
    args = parser.parse_args()

    # Suppress Pydantic warnings from the Anthropic library
    warnings.filterwarnings("ignore", message=r"Pydantic serializer warnings:", category=UserWarning)

    if args.task_directory:
        if args.test != 1:
             print(f"Warning: --test argument ({args.test}) is ignored when using --task-directory. Running all tests for all tasks.", file=sys.stderr)

        directory = Path(args.task_directory)
        if not directory.exists() or not directory.is_dir():
            print(f"Error: Directory '{directory}' does not exist.", file=sys.stderr)
            sys.exit(1)
            
        task_files = list(directory.glob("*.json"))
        if not task_files:
            print(f"No JSON files found in '{directory}'.", file=sys.stderr)
            sys.exit(0)
            
        print(f"Found {len(task_files)} task files in {directory}. Starting batch execution...")
        
        for task_file in sorted(task_files):
            try:
                task = load_task(task_file)
                num_tests = len(task.test)
                for i in range(num_tests):
                    test_index = i + 1
                    # We wrap execution in try/except to prevent one task from killing the loop 
                    # IF the underlying engine raises exceptions.
                    # BUT default_engine uses sys.exit(1). This is a problem for batch mode.
                    try:
                        execute_task(args, task_file, test_index, run_timestamp)
                    except SystemExit as se:
                        if se.code != 0:
                            print(f"Task {task_file.name} test {test_index} failed with exit code {se.code}", file=sys.stderr)
                        # If code is 0, it's success, continue.
                    except Exception as e:
                        print(f"Task {task_file.name} test {test_index} raised exception: {e}", file=sys.stderr)

            except Exception as e:
                print(f"Error loading task {task_file}: {e}", file=sys.stderr)
                
    else:
        # Single task mode
        try:
            task_path = find_task_path(args.task)
            execute_task(args, task_path, args.test, run_timestamp)
        except FileNotFoundError as e:
             print(f"Error: {e}", file=sys.stderr)
             sys.exit(1)
        except SystemExit as se:
            sys.exit(se.code)

if __name__ == "__main__":
    main()