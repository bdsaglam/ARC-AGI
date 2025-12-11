import sys
from pathlib import Path

from src.logging import log_failure
from src.solver.state import SolverState
from src.solver.steps import run_step_1, run_step_3, run_step_5, check_is_solved

# Re-export run_solver_mode for backward compatibility if imported elsewhere
def run_solver_mode(task_id: str, test_index: int, verbose: int, is_testing: bool = False, run_timestamp: str = None, task_path: Path = None, answer_path: Path = None, step_5_only: bool = False, objects_only: bool = False, force_step_5: bool = False, force_step_2: bool = False, judge_model: str = "gemini-3-high", old_pick_solution: bool = False, task_status=None):
    
    # Initialize State
    try:
        state = SolverState(task_id, test_index, verbose, is_testing, run_timestamp, task_path, answer_path, judge_model, old_pick_solution=old_pick_solution, task_status=task_status)
    except Exception as e:
        print(f"Error initializing solver state: {e}", file=sys.stderr)
        raise e

    try:
        if is_testing:
            # Models for --solver-testing
            models_step1 = ["claude-sonnet-4.5-no-thinking", "gpt-5.1-none", "claude-sonnet-4.5-no-thinking", "gpt-5.1-none", "claude-opus-4.5-no-thinking"]
            models_step3 = ["claude-sonnet-4.5-no-thinking", "gpt-5.1-none"]
            models_step5 = ["claude-sonnet-4.5-no-thinking", "gpt-5.1-none"]
            hint_generation_model = "gpt-5.1-none"
        else:
            # Models for --solver
            models_step1 = ["claude-opus-4.5-thinking-60000", "claude-opus-4.5-thinking-60000", "gemini-3-high", "gemini-3-high", "gpt-5.1-high", "gpt-5.1-high"]
            models_step3 = ["claude-opus-4.5-thinking-60000", "claude-opus-4.5-thinking-60000", "gemini-3-high", "gemini-3-high", "gpt-5.1-high", "gpt-5.1-high"]
            models_step5 = ["claude-opus-4.5-thinking-60000", "claude-opus-4.5-thinking-60000", "gemini-3-high", "gemini-3-high", "gpt-5.1-high", "gpt-5.1-high"]
            hint_generation_model = "gemini-3-high"

        # Skip logic
        should_run_early_steps = not (step_5_only or objects_only)

        if should_run_early_steps:
            # STEP 1
            run_step_1(state, models_step1)

            # STEP 2
            state.set_status(step=2)
            finish, _ = check_is_solved(state, "step_2", force_finish=force_step_2, continue_if_solved=force_step_5)
            if finish:
                return state.finalize("step_finish")

            # STEP 3
            run_step_3(state, models_step3)

            # STEP 4
            state.set_status(step=4)
            finish, _ = check_is_solved(state, "step_4", continue_if_solved=force_step_5)
            if finish:
                return state.finalize("step_finish")
        else:
             print("\nSkipping Steps 1-4 (Deep Search Only Mode)")

        # STEP 5
        run_step_5(state, models_step5, hint_generation_model, objects_only=objects_only)

        # STEP FINISH
        return state.finalize("step_finish")
        
    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 50:
            error_msg = error_msg[:47] + "..."
        log_failure(
            run_timestamp=run_timestamp if run_timestamp else "unknown_timestamp",
            task_id=task_id,
            run_id="SOLVER_ENGINE_MAIN_LOOP",
            error=e,
            model="SYSTEM",
            step="MAIN",
            test_index=test_index
        )
        print(f"CRITICAL ERROR in run_solver_mode: {e}", file=sys.stderr)
        raise e
