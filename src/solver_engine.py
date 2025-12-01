import sys
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from openai import OpenAI
from anthropic import Anthropic
from google import genai


# Import from existing project modules
from src.config import get_api_keys
from src.tasks import load_task, build_prompt
from src.image_generation import generate_and_save_image
from src.hint_generation import generate_hint
from src.logging import setup_logging, log_failure
from src.parallel import run_single_model
from src.run_utils import find_task_path, pick_solution, is_solved

class ProgressReporter:
    def __init__(self, queue, task_id, test_index):
        self.queue = queue
        self.task_id = task_id
        self.test_index = test_index

    def emit(self, status, step, outcome=None, event=None, predictions=None):
        if self.queue is None:
            return
        self.queue.put({
            "task_id": self.task_id,
            "test_index": self.test_index,
            "status": status,
            "step": step,
            "outcome": outcome,
            "event": event,
            "predictions": predictions,
            "timestamp": time.time(),
        })

def run_models_in_parallel(models_to_run, run_id_counts, step_name, prompt, test_example, openai_client, anthropic_client, google_keys, verbose, image_path=None, run_timestamp=None):
    all_results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        
        # Generate unique run IDs
        run_list = []
        for model_name in models_to_run:
            count = run_id_counts.get(model_name, 0) + 1
            run_id_counts[model_name] = count
            run_id = f"{model_name}_{count}_{step_name}"
            run_list.append({"name": model_name, "run_id": run_id})

        future_to_run_id = {
            executor.submit(run_single_model, run["name"], run["run_id"], prompt, test_example, openai_client, anthropic_client, google_keys, verbose, image_path, run_timestamp): run["run_id"]
            for run in run_list
        }

        for future in as_completed(future_to_run_id):
            res = future.result()
            if res:
                all_results.append(res)
    return all_results

def run_solver_mode(task_id: str, test_index: int, verbose: bool, is_testing: bool = False, run_timestamp: str = None, task_path: Path = None, progress_queue=None, answer_path: Path = None):
    reporter = ProgressReporter(progress_queue, task_id, test_index)
    reporter.emit("RUNNING", "Initializing", event="START")
    
    try:
        if is_testing:
            print("Solver testing mode activated.")
            # Models for --solver-testing
            models_step1 = ["claude-sonnet-4.5-no-thinking", "gpt-5.1-none", "gemini-3-low"]
            models_step3 = ["claude-sonnet-4.5-no-thinking", "gpt-5.1-none"]
            models_step5 = ["claude-sonnet-4.5-no-thinking", "gpt-5.1-none"]
            hint_generation_model = "gpt-5.1-none"
        else:
            print("Solver mode activated.")
            # Models for --solver
            models_step1 = ["claude-sonnet-4.5-thinking-60000", "claude-sonnet-4.5-thinking-60000", "claude-opus-4.5-thinking-60000", "claude-opus-4.5-thinking-60000", "gpt-5.1-high", "gpt-5.1-high", "gemini-3-high", "gemini-3-high"]
            models_step3 = ["claude-opus-4.5-thinking-60000", "claude-opus-4.5-thinking-60000", "gpt-5.1-high", "gpt-5.1-high", "gemini-3-high", "gemini-3-high"]
            models_step5 = ["claude-opus-4.5-thinking-60000", "claude-opus-4.5-thinking-60000", "gpt-5.1-high", "gpt-5.1-high", "gemini-3-high", "gemini-3-high"]
            hint_generation_model = "gpt-5.1-high"
        
        setup_logging(verbose)

        start_time = time.time()
        total_cost = 0.0

        def print_summary():
            duration = time.time() - start_time
            print(f"\nTotal Execution Time: {duration:.2f}s")
            print(f"Total Cost: ${total_cost:.4f}")

        def write_step_log(step_name: str, data: dict, timestamp: str):
            log_path = Path("logs") / f"{timestamp}_{task_id}_{test_index}_{step_name}.json"
            with open(log_path, "w") as f:
                json.dump(data, f, indent=4, default=lambda o: '<not serializable>')
            if verbose:
                print(f"Saved log for {step_name} to {log_path}")

        if task_path is None:
            try:
                task_path = find_task_path(task_id)
            except FileNotFoundError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        openai_key, claude_key, google_keys = get_api_keys()
        http_client = httpx.Client(timeout=3600.0)
        openai_client = OpenAI(api_key=openai_key, http_client=http_client) if openai_key else None
        anthropic_client = Anthropic(api_key=claude_key, http_client=http_client) if claude_key else None
        # google_client instantiation removed as we now pass keys directly

        try:
            task = load_task(task_path, answer_path=answer_path)
        except Exception as e:
            print(f"Error loading task: {e}", file=sys.stderr)
            http_client.close()
            sys.exit(1)

        test_idx = test_index - 1
        if test_idx < 0 or test_idx >= len(task.test):
            print(f"Error: Test index {test_index} is out of range.", file=sys.stderr)
            http_client.close()
            sys.exit(1)
        test_example = task.test[test_idx]
        
        run_id_counts = {}
        candidates_object = {}

        def process_results(results, step_log):
            nonlocal candidates_object
            nonlocal total_cost
            initial_solutions = len(candidates_object)
            for res in results:
                if res:
                    total_cost += res.get("cost", 0)
                    run_key = f"{res['run_id']}_{time.time()}"
                    step_log[run_key] = {
                        "duration_seconds": round(res.get("duration", 0), 2),
                        "total_cost": res.get("cost", 0),
                        "input_tokens": res.get("input_tokens", 0),
                        "output_tokens": res.get("output_tokens", 0),
                        "cached_tokens": res.get("cached_tokens", 0),
                        "Full raw LLM call": res["prompt"],
                        "Full raw LLM response": res["full_response"],
                        "Extracted grid": res["grid"],
                    }
                    if res["grid"] is not None:
                        grid_tuple = tuple(tuple(row) for row in res["grid"])
                        if grid_tuple not in candidates_object:
                            candidates_object[grid_tuple] = {"grid": res["grid"], "count": 0, "models": [], "is_correct": res["is_correct"]}
                        candidates_object[grid_tuple]["count"] += 1
                        candidates_object[grid_tuple]["models"].append(res["run_id"])
            new_solutions = len(candidates_object) - initial_solutions
            print(f"Found {new_solutions} new unique solutions.")

        def finalize_result(candidates_object, step_log_name):
             # Check if we have ground truth
            has_ground_truth = test_example.output is not None
            
            picked_solutions, result = pick_solution(candidates_object)
            
            # Determine outcome string
            if not has_ground_truth:
                outcome = "SUBMITTED"
            else:
                outcome = "PASS" if result else "FAIL"
                
            finish_log = {
                "candidates_object": {str(k): v for k, v in candidates_object.items()},
                "picked_solutions": picked_solutions,
                "result": outcome,
                "correct_solution": test_example.output
            }
            write_step_log("step_finish", finish_log, run_timestamp)
            print_summary()
            http_client.close()
            reporter.emit("COMPLETED", "Finished", outcome=outcome, event="FINISH", predictions=picked_solutions)
            return picked_solutions

        # STEP 1
        print("\n--- STEP 1: Initial model run ---")
        reporter.emit("RUNNING", "Step 1 (Shallow search)", event="STEP_CHANGE")
        step_1_log = {}
        print(f"Running {len(models_step1)} models...")
        prompt_step1 = build_prompt(task.train, test_example)
        results_step1 = run_models_in_parallel(models_step1, run_id_counts, "step_1", prompt_step1, test_example, openai_client, anthropic_client, google_keys, verbose, run_timestamp=run_timestamp)
        process_results(results_step1, step_1_log)
        write_step_log("step_1", step_1_log, run_timestamp)

        # STEP 2
        print("\n--- STEP 2: First check ---")
        reporter.emit("RUNNING", "Step 2 (Evaluation)", event="STEP_CHANGE")
        solved = is_solved(candidates_object)
        step_2_log = {"candidates_object": {str(k): v for k, v in candidates_object.items()}, "is_solved": solved}
        write_step_log("step_2", step_2_log, run_timestamp)
        if solved:
            print("is_solved() is TRUE, moving to STEP FINISH.")
            return finalize_result(candidates_object, "step_finish")

        # STEP 3
        print("\n--- STEP 3: Second model run ---")
        reporter.emit("RUNNING", "Step 3 (Extended search)", event="STEP_CHANGE")
        step_3_log = {}
        print(f"Running {len(models_step3)} models...")
        prompt_step3 = build_prompt(task.train, test_example)
        results_step3 = run_models_in_parallel(models_step3, run_id_counts, "step_3", prompt_step3, test_example, openai_client, anthropic_client, google_keys, verbose, run_timestamp=run_timestamp)
        process_results(results_step3, step_3_log)
        write_step_log("step_3", step_3_log, run_timestamp)

        # STEP 4
        print("\n--- STEP 4: Second check ---")
        reporter.emit("RUNNING", "Step 4 (Evaluation)", event="STEP_CHANGE")
        solved = is_solved(candidates_object)
        step_4_log = {"candidates_object": {str(k): v for k, v in candidates_object.items()}, "is_solved": solved}
        write_step_log("step_4", step_4_log, run_timestamp)
        if solved:
            print("is_solved() is TRUE, moving to STEP FINISH.")
            return finalize_result(candidates_object, "step_finish")

        # STEP 5
        print("\n--- STEP 5: Final model runs (in parallel) ---")
        reporter.emit("RUNNING", "Step 5 (Full search)", event="STEP_CHANGE")
        step_5_log = {"trigger-deep-thinking": {}, "image": {}, "generate-hint": {}}

        # Generate image once for both visual and hint steps to avoid matplotlib race conditions
        common_image_path = f"logs/{run_timestamp}_{task_id}_{test_index}_step5_common.png"
        generate_and_save_image(task, common_image_path)

        def run_deep_thinking_step():
            print(f"Running {len(models_step5)} models with deep thinking...")
            prompt_deep = build_prompt(task.train, test_example, trigger_deep_thinking=True)
            results_deep = run_models_in_parallel(models_step5, run_id_counts, "step_5_deep_thinking", prompt_deep, test_example, openai_client, anthropic_client, google_keys, verbose, run_timestamp=run_timestamp)
            return "trigger-deep-thinking", results_deep

        def run_image_step(img_path):
            print(f"Running {len(models_step5)} models with image...")
            # Image is already generated
            prompt_image = build_prompt(task.train, test_example, image_path=img_path)
            results_image = run_models_in_parallel(models_step5, run_id_counts, "step_5_image", prompt_image, test_example, openai_client, anthropic_client, google_keys, verbose, image_path=img_path, run_timestamp=run_timestamp)
            return "image", results_image

        def run_hint_step(img_path):
            print(f"Running {len(models_step5)} models with generated hint...")
            # Image is already generated
            hint_data = generate_hint(task, img_path, hint_generation_model, verbose)
            if hint_data and hint_data["hint"]:
                step_5_log["generate-hint"]["hint_generation"] = {
                    "Full raw LLM call": hint_data["prompt"],
                    "Full raw LLM response": hint_data["full_response"],
                    "Extracted hint": hint_data["hint"],
                }
                prompt_hint = build_prompt(task.train, test_example, strategy=hint_data["hint"])
                results_hint = run_models_in_parallel(models_step5, run_id_counts, "step_5_generate_hint", prompt_hint, test_example, openai_client, anthropic_client, google_keys, verbose, run_timestamp=run_timestamp)
                return "generate-hint", results_hint
            return "generate-hint", []

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(run_deep_thinking_step),
                executor.submit(run_image_step, common_image_path),
                executor.submit(run_hint_step, common_image_path)
            ]
            for future in as_completed(futures):
                step_name, results = future.result()
                process_results(results, step_5_log[step_name])

        write_step_log("step_5", step_5_log, run_timestamp)

        # STEP FINISH
        print("\n--- STEP FINISH: Pick and print solution ---")
        return finalize_result(candidates_object, "step_finish")
        
    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 50:
            error_msg = error_msg[:47] + "..."
        reporter.emit("ERROR", f"Error: {error_msg}", outcome="FAIL", event="FINISH")
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