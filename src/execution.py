import sys
import time
from pathlib import Path
from src.solver_engine import run_solver_mode
from src.default_engine import run_default_mode
from src.logging import PrefixedStdout
from src.parallel import set_rate_limit_scaling

class DummyFile:
    def write(self, x): pass
    def flush(self): pass
    @property
    def encoding(self): return 'utf-8'

def execute_task(args, task_path: Path, test_index: int, run_timestamp: str, rate_limit_scale: float = 1.0, progress_queue=None, quiet=False, answer_path: Path = None):
    # Apply rate limit scaling (only affects this process)
    if rate_limit_scale != 1.0:
        set_rate_limit_scaling(rate_limit_scale)
        
    task_id = task_path.stem
    
    # Update args for default mode compatibility
    args.task = str(task_path)
    args.test = test_index
    
    predictions = None

    if quiet:
        # Redirect stdout/stderr to dummy file to prevent interrupting the UI
        devnull = DummyFile()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            if args.solver or args.solver_testing:
                predictions = run_solver_mode(
                    task_id, test_index, args.verbose, 
                    is_testing=args.solver_testing, 
                    run_timestamp=run_timestamp, 
                    task_path=task_path, 
                    progress_queue=progress_queue, 
                    answer_path=answer_path,
                    step_5_only=args.step_5_only,
                    objects_only=args.objects_only,
                    force_step_5=args.force_step_5,
                    force_step_2=args.force_step_2
                )
            else:
                # default mode doesn't support progress_queue yet, so it will just run silently if quiet=True
                predictions = run_default_mode(args, answer_path=answer_path)
        except Exception as e:
             # If we have a queue, try to report the crash
            if progress_queue:
                error_msg = str(e)
                if len(error_msg) > 50:
                    error_msg = error_msg[:47] + "..."
                progress_queue.put({
                    "task_id": task_id,
                    "test_index": test_index,
                    "status": "ERROR",
                    "step": f"Error: {error_msg}",
                    "outcome": "FAIL",
                    "event": "FINISH",
                    "timestamp": time.time()
                })
            raise e
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    else:
        # Standard verbose mode
        prefix = f"{task_id}:{test_index} "
        with PrefixedStdout(prefix):
            try:
                if args.solver or args.solver_testing:
                    predictions = run_solver_mode(
                        task_id, test_index, args.verbose, 
                        is_testing=args.solver_testing, 
                        run_timestamp=run_timestamp, 
                        task_path=task_path, 
                        progress_queue=progress_queue, 
                        answer_path=answer_path,
                        step_5_only=args.step_5_only,
                        objects_only=args.objects_only,
                        force_step_5=args.force_step_5,
                        force_step_2=args.force_step_2
                    )
                else:
                    predictions = run_default_mode(args, answer_path=answer_path)
            except Exception as e:
                raise e
    
    return task_id, test_index, predictions
