import sys
import time
from pathlib import Path
from src.solver_engine import run_solver_mode
from src.logging import PrefixedStdout
from src.parallel import set_rate_limit_scaling



def execute_task(args, task_path: Path, test_index: int, run_timestamp: str, rate_limit_scale: float = 1.0, answer_path: Path = None, status_counters=None):
    if status_counters:
        running, remaining, finished, lock = status_counters
        with lock:
            running.value += 1
            remaining.value -= 1

    try:
        # Apply rate limit scaling (only affects this process)
        if rate_limit_scale != 1.0:
            set_rate_limit_scaling(rate_limit_scale)
            
        task_id = task_path.stem
        
        predictions = None

        # Standard verbose mode
        task_status = {
            "step": "0",
            "phase": "Init",
            "start_time": time.time()
        }
        
        def get_prefix():
            # Column widths
            w_status = 10
            w_task = 10
            w_step = 5
            w_phase = 15
            w_time = 6
            
            # 1. Status Column
            if status_counters:
                r = status_counters[0].value
                q = status_counters[1].value
                d = status_counters[2].value
                # Format: ⚡3 ⏳1 ✅1
                # We remove brackets to save space and fit in column
                stat_text = f" ⚡{r} ⏳{q} ✅{d}"
            else:
                stat_text = " Single Task"
            
            col_status = stat_text.ljust(w_status)

            # 2. Task Column
            task_str = f"{task_id}:{test_index}"
            col_task = task_str.ljust(w_task)

            # 3. Step Column
            step = task_status.get("step", "0")
            phase = task_status.get("phase", "Init")
            
            if phase == "Finished":
                step_str = "DONE"
            else:
                step_str = f"{step}/5"
            col_step = step_str.center(w_step)

            # 4. Phase Column
            col_phase = phase.ljust(w_phase)

            # 5. Time Column
            elapsed = time.time() - task_status.get("start_time", time.time())
            time_str = f"{elapsed:.1f}s"
            col_time = time_str.rjust(w_time)

            return f"| {col_status} | {col_task} | {col_step} | {col_phase} | {col_time} | "

        with PrefixedStdout(get_prefix, message_width=30):
            try:
                predictions = run_solver_mode(
                    task_id, test_index, args.verbose, 
                    is_testing=args.solver_testing, 
                    run_timestamp=run_timestamp, 
                    task_path=task_path, 
                    answer_path=answer_path,
                    step_5_only=args.step_5_only,
                    objects_only=args.objects_only,
                    force_step_5=args.force_step_5,
                    force_step_2=args.force_step_2,
                    judge_model=args.judge_model,
                    old_pick_solution=args.old_pick_solution,
                    task_status=task_status,
                    openai_background=args.openai_background,
                    enable_step_3_and_4=args.enable_step_3_and_4,
                    judge_consistency_enable=args.judge_consistency_enable,
                    codegen_prompt=args.codegen_prompt
                )
            except Exception as e:
                raise e
        
        return task_id, test_index, predictions
    finally:
        if status_counters:
            running, remaining, finished, lock = status_counters
            with lock:
                running.value -= 1
                finished.value += 1
