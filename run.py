import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime

from src.tasks import load_task
from src.run_utils import find_task_path
from src.execution import execute_task
from src.submission import generate_submission
from src.batch_processing import run_batch_execution

# Conditional import for rich
try:
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

def main():
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser(description="Run ARC task test cases with multiple models in parallel.")
    
    # Task selection group
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", help="Task ID (e.g., 38007db0) or path to JSON file")
    task_group.add_argument("--task-directory", help="Directory containing task JSON files to run in batch")
    
    parser.add_argument("--test", type=int, default=1, help="Test case index (1-based, default: 1). Ignored if --task-directory is used.")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers (default: 10) PER TASK.")
    parser.add_argument("--task-workers", type=int, default=20, help="Number of tasks to run in parallel (default: 20). CAUTION: Divides global rate limits by this factor.")
    parser.add_argument("--objects", action="store_true", help="Run a 3-step pipeline: Extraction -> Transformation -> Solution.")
    parser.add_argument("--step-5-only", action="store_true", help="Run only Step 5 (Full Search) of the solver.")
    parser.add_argument("--objects-only", action="store_true", help="Run only the Objects Pipeline sub-strategy in Step 5.")
    parser.add_argument("--force-step-5", action="store_true", help="Force execution of Step 5 even if a solution is found in earlier steps.")
    parser.add_argument("--force-step-2", action="store_true", help="Force the pipeline to stop after Step 2 (First Check), proceeding to Step Finish regardless of outcome.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--models", type=str, help="Comma-separated list of models to run")
    parser.add_argument("--hint", type=str, default=None, help="Optional hint to provide to the model")
    parser.add_argument("--image", action="store_true", help="Generate an image for the task and include it in the prompt.")
    parser.add_argument("--trigger-deep-thinking", action="store_true", help="Append a deep thinking procedure to the prompt.")
    parser.add_argument("--generate-hint", action="store_true", help="Generate a hint for the task using a separate model call.")
    parser.add_argument("--generate-hint-model", type=str, default="gpt-5.1-high", help="Model to use for generating hints.")
    parser.add_argument("--judge-model", type=str, default=None, help="Model to use for the auditing judges (Logic & Consistency).")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable the rich dashboard in batch mode.")
    parser.add_argument("--submissions-directory", type=str, default="submissions/", help="Directory to save submission files (default: submissions/).")
    parser.add_argument("--answers-directory", type=str, help="Optional directory containing answer files (with 'output' for test cases).")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--solver", action="store_true", help="Enable solver mode.")
    mode_group.add_argument("--solver-testing", action="store_true", help="Enable solver testing mode with a smaller set of models.")
    
    args = parser.parse_args()

    # Set default judge model if not specified
    if args.judge_model is None:
        if args.solver_testing:
            args.judge_model = "gemini-3-low"
        else:
            args.judge_model = "gemini-3-high"
    
    # If no specific solver mode is chosen, default to --solver
    if not args.solver and not args.solver_testing:
        args.solver = True
    
    warnings.filterwarnings("ignore", message=r"Pydantic serializer warnings:", category=UserWarning)

    answers_dir = Path(args.answers_directory) if args.answers_directory else None
    if answers_dir and not answers_dir.exists():
         print(f"Error: Answers directory '{answers_dir}' does not exist.", file=sys.stderr)
         sys.exit(1)

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
            
        # Prepare tasks
        tasks_to_run = []
        for task_file in sorted(task_files):
            try:
                # Load task briefly just to count tests, we load again in worker
                # We don't need answer path here strictly, but good to know
                task = load_task(task_file)
                num_tests = len(task.test)
                for i in range(num_tests):
                    tasks_to_run.append((task_file, i + 1))
            except Exception as e:
                print(f"Error loading task {task_file}: {e}", file=sys.stderr)

        total_tasks = len(tasks_to_run)
        print(f"Found {len(task_files)} task files. Total test cases: {total_tasks}")
        print(f"Starting batch execution with {args.task_workers} parallel task workers...")
        
        rate_limit_scale = 1.0 / max(1, args.task_workers)
        
        # Determine if we should use the dashboard
        use_dashboard = RICH_AVAILABLE and sys.stdout.isatty() and not args.no_dashboard and (args.solver or args.solver_testing)

        final_results = run_batch_execution(args, tasks_to_run, run_timestamp, rate_limit_scale, use_dashboard, answers_dir)
                        
        # Generate Submission File
        generate_submission(final_results, args.submissions_directory, run_timestamp)

    else:
        # Single task mode
        try:
            task_path = find_task_path(args.task)
            answer_path = None
            if answers_dir:
                 answer_path = answers_dir / task_path.name
            
            execute_task(args, task_path, args.test, run_timestamp, answer_path=answer_path)
        except FileNotFoundError as e:
             print(f"Error: {e}", file=sys.stderr)
             sys.exit(1)
        except SystemExit as se:
            sys.exit(se.code)

if __name__ == "__main__":
    main()