import sys
import warnings
import os
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime

from src.tasks import load_task
from src.run_utils import find_task_path
from src.execution import execute_task
from src.submission import generate_submission
from src.batch_processing import run_batch_execution



def run_app(
    task=None,
    task_directory=None,
    test=1,
    task_workers=20,
    startup_delay=20.0,
    task_limit=None,
    objects=False,
    step_5_only=False,
    objects_only=False,
    force_step_5=False,
    force_step_2=False,
    verbose=0,
    models=None,
    hint=None,
    image=False,
    trigger_deep_thinking=False,
    generate_hint=False,
    generate_hint_model="gpt-5.1-high",
    judge_model=None,
    old_pick_solution=False,
    submissions_directory="submissions/",
    answers_directory=None,
    solver=False,
    solver_testing=False,
):
    # Construct args namespace to pass around internally as many legacy functions expect it
    args = SimpleNamespace(
        task=task,
        task_directory=task_directory,
        test=test,
        task_workers=task_workers,
        task_limit=task_limit,
        objects=objects,
        step_5_only=step_5_only,
        objects_only=objects_only,
        force_step_5=force_step_5,
        force_step_2=force_step_2,
        verbose=verbose,
        models=models,
        hint=hint,
        image=image,
        trigger_deep_thinking=trigger_deep_thinking,
        generate_hint=generate_hint,
        generate_hint_model=generate_hint_model,
        judge_model=judge_model,
        old_pick_solution=old_pick_solution,
        submissions_directory=submissions_directory,
        answers_directory=answers_directory,
        solver=solver,
        solver_testing=solver_testing
    )

    if os.getenv("ARC_AGI_INSECURE_SSL", "").lower() == "true":
        print("WARNING: SSL verification disabled (ARC_AGI_INSECURE_SSL=true)")
        
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Set default judge model if not specified
    if args.judge_model is None:
        if args.solver_testing:
            args.judge_model = "gemini-3-low"
        else:
            args.judge_model = "gemini-3-high"
    
    # If no specific solver mode is chosen, default to --solver
    if not args.solver and not args.solver_testing:
        args.solver = True
    
    if args.solver_testing:
        print("Solver testing mode activated.")
        print(f"Judge model: {args.judge_model}")
        print()
    else:
        print("Solver mode activated.")
        print(f"Judge model: {args.judge_model}")
        print()
    
    warnings.filterwarnings("ignore", message=r"Pydantic serializer warnings:", category=UserWarning)

    # Ensure logs directory exists
    logs_dir = Path("logs")
    if not logs_dir.exists():
        logs_dir.mkdir(parents=True, exist_ok=True)

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
            
        task_files = sorted(task_files)
        if args.task_limit:
            print(f"Limiting execution to first {args.task_limit} tasks.")
            task_files = task_files[:args.task_limit]

        # Prepare tasks
        tasks_to_run = []
        for task_file in task_files:
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
        print()
        
        rate_limit_scale = 1.0 / max(1, args.task_workers)
        
        final_results = run_batch_execution(args, tasks_to_run, run_timestamp, rate_limit_scale, answers_dir, startup_delay=startup_delay)
                        
        # Generate Submission File
        generate_submission(final_results, args.submissions_directory, run_timestamp)

    else:
        # Single task mode
        try:
            task_path = find_task_path(args.task)
            answer_path = None
            if answers_dir:
                 answer_path = answers_dir / task_path.name
            
            # Print Table Header
            print("Legend: ⚡ Running   ⏳ Queued   ✅ Done")
            print()
            print("| Status        | Task:Test  | Step  | Phase           | Time   | Message")
            print("|---------------|------------|-------|-----------------|--------|------------------------------")

            execute_task(args, task_path, args.test, run_timestamp, answer_path=answer_path)
        except FileNotFoundError as e:
             print(f"Error: {e}", file=sys.stderr)
             sys.exit(1)
        except SystemExit as se:
            sys.exit(se.code)