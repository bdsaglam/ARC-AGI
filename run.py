import argparse
import sys
import os
import time
import warnings
import re
from pathlib import Path
import httpx
from anthropic import Anthropic
from openai import OpenAI
from google import genai
import certifi
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils import format_grid

# Import from existing project modules
from src.config import get_api_keys
from src.tasks import load_task, build_prompt
from src.models import call_model, calculate_cost, parse_model_arg
from src.utils import parse_grid_from_text, verify_prediction
from src.logging import setup_logging
from src.image_generation import generate_and_save_image
from src.hint_generation import generate_hint

# Constants
DEFAULT_MODELS = [
    "claude-sonnet-4.5-thinking-1024",
    "claude-opus-4.5-thinking-1024",
    "gpt-5.1-low"
]

MODEL_WEIGHTS = {
    "claude-opus-4.5-thinking-60000": 16,
    "gemini-3-high": 15,
    "claude-opus-4.5-thinking-16000": 14,
    "gpt-5.1-high": 13,
    "claude-sonnet-4.5-thinking-60000": 12,
    "claude-opus-4.5-thinking-4000": 11,
    "claude-opus-4.5-thinking-1024": 10,
    "claude-opus-4.5-no-thinking": 9,
    "claude-sonnet-4.5-thinking-16000": 8,
    "gemini-3-low": 7,
    "gpt-5.1-medium": 6,
    "claude-sonnet-4.5-thinking-4000": 5,
    "claude-sonnet-4.5-thinking-1024": 4,
    "claude-sonnet-4.5-no-thinking": 3,
    "gpt-5.1-low": 2,
    "gpt-5.1-none": 1
}

def find_task_path(task_id: str) -> Path:
    if task_id.endswith(".json"):
        p = Path(task_id)
        if p.exists():
            return p
        task_id = p.stem
    candidate = Path("data/arc-agi-2-evaluation") / f"{task_id}.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Task file for '{task_id}' not found in data/arc-agi-2-evaluation/.")

def run_single_model(model_name, run_id, prompt, test_example, openai_client, anthropic_client, google_client, verbose, image_path=None):
    prefix = f"[{run_id}]"
    if verbose:
        print(f"{prefix} Initiating call...")
        if image_path:
            print(f"{prefix} Including image: {image_path}")

    cost = 0.0
    duration = 0.0
    full_response = ""
    try:
        start_ts = time.perf_counter()
        response = call_model(
            openai_client=openai_client,
            anthropic_client=anthropic_client,
            google_client=google_client,
            prompt=prompt,
            model_arg=model_name,
            image_path=image_path,
            return_strategy=False,
            verbose=verbose
        )
        duration = time.perf_counter() - start_ts
        full_response = response.text
        
        try:
            model_config = parse_model_arg(model_name)
            cost = calculate_cost(model_config, response)
        except Exception:
            pass

        if verbose:
            print(f"{prefix} Response received.")

        grid_text = response.text
        
        try:
            predicted_grid = parse_grid_from_text(grid_text)
            is_correct = verify_prediction(predicted_grid, test_example.output)
            
            if verbose:
                if is_correct:
                    print(f"{prefix} Result: PASS")
                else:
                    print(f"{prefix} Result: FAIL")
                    print(f"\n{prefix} Predicted Grid:")
                    print(grid_text)
            
            return {"model": model_name, "run_id": run_id, "grid": predicted_grid, "is_correct": is_correct, "cost": cost, "duration": duration, "prompt": prompt, "full_response": full_response}
                    
        except ValueError as e:
            if verbose:
                print(f"{prefix} Result: FAIL (Parse Error: {e})")
                print(f"\n{prefix} Raw Output:\n{grid_text}")
            return {"model": model_name, "run_id": run_id, "grid": None, "is_correct": False, "cost": cost, "duration": duration, "prompt": prompt, "full_response": full_response}

    except Exception as e:
        print(f"{prefix} Error during execution: {e}", file=sys.stderr)
        return {"model": model_name, "run_id": run_id, "grid": None, "is_correct": False, "cost": cost, "duration": duration, "prompt": prompt, "full_response": str(e)}

def get_group_sort_key(group):
    count = group['count']
    max_weight = 0
    for run_id in group['models']:
        # Strip the suffix like _1_step_1 to get the base model name
        base_model = re.sub(r'_\d+_.*$', '', run_id)
        weight = MODEL_WEIGHTS.get(base_model, 0)
        if weight > max_weight:
            max_weight = weight
    return (count, max_weight)

def is_solved(candidates_object) -> float:
    return 0.8

def pick_solution(candidates_object):
    sorted_groups = sorted(candidates_object.values(), key=get_group_sort_key, reverse=True)
    
    print("\n" + "="*40)
    print("FINAL OUTCOME")
    print("="*40)
    
    is_solved_flag = False
    top_groups = sorted_groups[:2]
    
    if len(top_groups) > 0 and top_groups[0]["is_correct"]:
        is_solved_flag = True
    elif len(top_groups) > 1 and top_groups[1]["is_correct"]:
        is_solved_flag = True
        
    if is_solved_flag:
        print("Outcome: SOLVED")
    else:
        print("Outcome: FAILED")

    print("\n--- Debug Info ---")
    if not top_groups:
        print("No solutions generated.")
    else:
        for i, group in enumerate(top_groups):
            print(f"Group {i+1}: Count={group['count']}, Correct={group['is_correct']}")
            print(f"  Models: {', '.join(group['models'])}")
            
    return top_groups, is_solved_flag

def run_models_in_parallel(models_to_run, run_id_counts, step_name, prompt, test_example, openai_client, anthropic_client, google_client, verbose, image_path=None):
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
            executor.submit(run_single_model, run["name"], run["run_id"], prompt, test_example, openai_client, anthropic_client, google_client, verbose, image_path): run["run_id"] 
            for run in run_list
        }

        for future in as_completed(future_to_run_id):
            res = future.result()
            if res:
                all_results.append(res)
    return all_results

def run_solver_mode(task_id: str, test_index: int, verbose: bool):
    print("Solver mode activated.")
    setup_logging(verbose)

    def write_step_log(step_name: str, data: dict):
        log_path = Path("logs") / f"{task_id}_{test_index}_{step_name}.json"
        with open(log_path, "w") as f:
            json.dump(data, f, indent=4, default=lambda o: '<not serializable>')
        if verbose:
            print(f"Saved log for {step_name} to {log_path}")

    try:
        task_path = find_task_path(task_id)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    openai_key, claude_key, google_key = get_api_keys()
    http_client = httpx.Client(timeout=3600.0)
    openai_client = OpenAI(api_key=openai_key, http_client=http_client) if openai_key else None
    anthropic_client = Anthropic(api_key=claude_key, http_client=http_client) if claude_key else None
    google_client = genai.Client(api_key=google_key) if google_key else None

    try:
        task = load_task(task_path)
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
        initial_solutions = len(candidates_object)
        for res in results:
            if res:
                run_key = f"{res['run_id']}_{time.time()}"
                step_log[run_key] = {
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

    # STEP 1
    print("\n--- STEP 1: Initial model run ---")
    step_1_log = {}
    models_step1 = ["claude-sonnet-4.5-no-thinking", "gpt-5.1-none"]
    print(f"Running {len(models_step1)} models...")
    prompt_step1 = build_prompt(task.train, test_example)
    results_step1 = run_models_in_parallel(models_step1, run_id_counts, "step_1", prompt_step1, test_example, openai_client, anthropic_client, google_client, verbose)
    process_results(results_step1, step_1_log)
    write_step_log("step_1", step_1_log)

    # STEP 2
    print("\n--- STEP 2: First check ---")
    solved_prob = is_solved(candidates_object)
    step_2_log = {"candidates_object": {str(k): v for k, v in candidates_object.items()}, "is_solved_prob": solved_prob}
    write_step_log("step_2", step_2_log)
    if solved_prob > 0.9:
        print("is_solved() > 0.9, moving to STEP FINISH.")
        picked_solutions, result = pick_solution(candidates_object)
        finish_log = {"candidates_object": {str(k): v for k, v in candidates_object.items()}, "picked_solutions": picked_solutions, "result": "PASS" if result else "FAIL", "correct_solution": test_example.output}
        write_step_log("step_finish", finish_log)
        http_client.close()
        return

    # STEP 3
    print("\n--- STEP 3: Second model run ---")
    step_3_log = {}
    models_step3 = ["claude-sonnet-4.5-no-thinking", "gpt-5.1-none"]
    print(f"Running {len(models_step3)} models...")
    prompt_step3 = build_prompt(task.train, test_example)
    results_step3 = run_models_in_parallel(models_step3, run_id_counts, "step_3", prompt_step3, test_example, openai_client, anthropic_client, google_client, verbose)
    process_results(results_step3, step_3_log)
    write_step_log("step_3", step_3_log)

    # STEP 4
    print("\n--- STEP 4: Second check ---")
    solved_prob = is_solved(candidates_object)
    step_4_log = {"candidates_object": {str(k): v for k, v in candidates_object.items()}, "is_solved_prob": solved_prob}
    write_step_log("step_4", step_4_log)
    if solved_prob > 0.9:
        print("is_solved() > 0.9, moving to STEP FINISH.")
        picked_solutions, result = pick_solution(candidates_object)
        finish_log = {"candidates_object": {str(k): v for k, v in candidates_object.items()}, "picked_solutions": picked_solutions, "result": "PASS" if result else "FAIL", "correct_solution": test_example.output}
        write_step_log("step_finish", finish_log)
        http_client.close()
        return

    # STEP 5
    print("\n--- STEP 5: Final model runs ---")
    step_5_log = {"trigger-deep-thinking": {}, "image": {}, "generate-hint": {}}
    models_step5 = ["claude-sonnet-4.5-no-thinking", "gpt-5.1-none"]
    
    print(f"Running {len(models_step5)} models with deep thinking...")
    prompt_deep = build_prompt(task.train, test_example, trigger_deep_thinking=True)
    results_deep = run_models_in_parallel(models_step5, run_id_counts, "step_5_deep_thinking", prompt_deep, test_example, openai_client, anthropic_client, google_client, verbose)
    process_results(results_deep, step_5_log["trigger-deep-thinking"])
    
    print(f"Running {len(models_step5)} models with image...")
    image_path = f"logs/{task_id}_{test_index}_step5_image.png"
    generate_and_save_image(task, image_path)
    prompt_image = build_prompt(task.train, test_example, image_path=image_path)
    results_image = run_models_in_parallel(models_step5, run_id_counts, "step_5_image", prompt_image, test_example, openai_client, anthropic_client, google_client, verbose, image_path=image_path)
    process_results(results_image, step_5_log["image"])

    print(f"Running {len(models_step5)} models with generated hint...")
    hint_image_path = f"logs/{task_id}_{test_index}_step5_generate_hint.png"
    hint_data = generate_hint(task, hint_image_path, "gpt-5.1-none", verbose)
    if hint_data and hint_data["hint"]:
        step_5_log["generate-hint"]["hint_generation"] = {
            "Full raw LLM call": hint_data["prompt"],
            "Full raw LLM response": hint_data["full_response"],
            "Extracted hint": hint_data["hint"],
        }
        prompt_hint = build_prompt(task.train, test_example, strategy=hint_data["hint"])
        results_hint = run_models_in_parallel(models_step5, run_id_counts, "step_5_generate_hint", prompt_hint, test_example, openai_client, anthropic_client, google_client, verbose)
        process_results(results_hint, step_5_log["generate-hint"])
    
    write_step_log("step_5", step_5_log)

    # STEP FINISH
    print("\n--- STEP FINISH: Pick and print solution ---")
    picked_solutions, result = pick_solution(candidates_object)
    finish_log = {"candidates_object": {str(k): v for k, v in candidates_object.items()}, "picked_solutions": picked_solutions, "result": "PASS" if result else "FAIL", "correct_solution": test_example.output}
    write_step_log("step_finish", finish_log)
        
    http_client.close()

def main():
    parser = argparse.ArgumentParser(description="Run a single ARC task test case with multiple models in parallel.")
    parser.add_argument("--task", required=True, help="Task ID (e.g., 38007db0)")
    parser.add_argument("--test", type=int, default=1, help="Test case index (1-based, default: 1)")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers (default: 10)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--models", type=str, help="Comma-separated list of models to run")
    parser.add_argument("--hint", type=str, default=None, help="Optional hint to provide to the model")
    parser.add_argument("--image", action="store_true", help="Generate an image for the task and include it in the prompt.")
    parser.add_argument("--trigger-deep-thinking", action="store_true", help="Append a deep thinking procedure to the prompt.")
    parser.add_argument("--generate-hint", action="store_true", help="Generate a hint for the task using a separate model call.")
    parser.add_argument("--generate-hint-model", type=str, default="gpt-5.1-high", help="Model to use for generating hints.")
    parser.add_argument("--solver", action="store_true", help="Enable solver mode.")
    
    args = parser.parse_args()

    if args.solver:
        run_solver_mode(args.task, args.test, args.verbose)
        sys.exit(0)

    warnings.filterwarnings("ignore", message=r"Pydantic serializer warnings:", category=UserWarning)

    if args.models:
        models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models_to_run = DEFAULT_MODELS

    setup_logging(args.verbose)

    try:
        task_path = find_task_path(args.task)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    openai_key, claude_key, google_key = get_api_keys()
    
    if not claude_key:
        print("Error: Anthropic API key not found.", file=sys.stderr)
        sys.exit(1)
    if not openai_key:
        print("Warning: OpenAI API key not found. GPT-5.1 models will fail.", file=sys.stderr)
    if not google_key:
        print("Warning: Google API key not found. Gemini models will fail.", file=sys.stderr)

    os.environ["SSL_CERT_FILE"] = certifi.where()
    http_client = httpx.Client(timeout=3600.0, transport=httpx.HTTPTransport(retries=3, verify=False), limits=httpx.Limits(keepalive_expiry=3600), verify=False)
    anthropic_client = Anthropic(api_key=claude_key, http_client=http_client)
    openai_client = OpenAI(api_key=openai_key, http_client=http_client) if openai_key else None
    google_client = genai.Client(api_key=google_key) if google_key else None

    try:
        task = load_task(task_path)
    except Exception as e:
        print(f"Error loading task: {e}", file=sys.stderr)
        http_client.close()
        sys.exit(1)

    test_idx = args.test - 1
    if test_idx < 0 or test_idx >= len(task.test):
        print(f"Error: Test index {args.test} is out of range. Task has {len(task.test)} test cases.", file=sys.stderr)
        http_client.close()
        sys.exit(1)
    test_example = task.test[test_idx]

    hint = args.hint
    if args.generate_hint:
        print("Generating hint...")
        hint_data = generate_hint(task, f"logs/{args.task}_{args.test}_generate_hint.png", args.generate_hint_model, args.verbose)
        if hint_data and hint_data["hint"]:
            hint = hint_data["hint"]
            print(f"Generated hint: {hint}")
        else:
            print("Warning: Failed to generate hint.")

    image_path = None
    if args.image:
        image_path = generate_and_save_image(task, f"logs/{args.task}_{args.test}_image.png")

    prompt = build_prompt(task.train, test_example, strategy=hint, image_path=image_path, trigger_deep_thinking=args.trigger_deep_thinking)

    total_calls = len(models_to_run)
    print(f"Starting parallel execution for {total_calls} models...")
    
    start_time = time.perf_counter()

    all_results = []
    future_to_run_id = {}
    
    # Generate unique run IDs using enumerate
    run_list = [
        {"name": model_name, "run_id": f"{model_name}_{i+1}_main"}
        for i, model_name in enumerate(models_to_run)
    ]

    pending_run_ids = [run["run_id"] for run in run_list]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for run in run_list:
            future = executor.submit(run_single_model, run["name"], run["run_id"], prompt, test_example, openai_client, anthropic_client, google_client, args.verbose, image_path)
            future_to_run_id[future] = run["run_id"]
        
        completed_count = 0
        for future in as_completed(future_to_run_id):
            completed_count += 1
            run_id = future_to_run_id[future]
            
            if run_id in pending_run_ids:
                pending_run_ids.remove(run_id)
            
            elapsed = time.perf_counter() - start_time
            
            waiting_str = ", ".join(pending_run_ids)
            if len(waiting_str) > 50:
                waiting_str = waiting_str[:47] + "..."
            if not waiting_str:
                waiting_str = "(none)"

            status_str = "(Error)"
            res = None
            try:
                res = future.result()
                if res:
                    if res["grid"] is not None:
                        status_str = "(Grid Received)"
                    else:
                        status_str = "(No Grid)"
            except Exception as exc:
                status_str = "(Error)"
                print(f"\nThread generated an exception: {exc}", file=sys.stderr)

            print(f"{elapsed:.1f}s {completed_count}/{total_calls} done {run_id} {status_str} Wait: {waiting_str}")

            if res:
                all_results.append(res)

    http_client.close()
    total_duration = time.perf_counter() - start_time

    grouped_solutions = {}
    total_cost = 0.0
    
    for res in all_results:
        total_cost += res.get("cost", 0.0)
        
        if res["grid"] is None:
            continue
            
        grid_tuple = tuple(tuple(row) for row in res["grid"])
        
        if grid_tuple not in grouped_solutions:
            grouped_solutions[grid_tuple] = {"grid": res["grid"], "count": 0, "models": [], "is_correct": res["is_correct"]}
        
        grouped_solutions[grid_tuple]["count"] += 1
        grouped_solutions[grid_tuple]["models"].append(res["run_id"])

    sorted_groups = sorted(grouped_solutions.values(), key=get_group_sort_key, reverse=True)
    
    print("\n" + "="*40)
    print("FINAL OUTCOME")
    print("="*40)
    
    is_solved = False
    top_groups = sorted_groups[:2]
    
    if len(top_groups) > 0 and top_groups[0]["is_correct"]:
        is_solved = True
    elif len(top_groups) > 1 and top_groups[1]["is_correct"]:
        is_solved = True
        
    if is_solved:
        print("Outcome: SOLVED")
    else:
        print("Outcome: FAILED")
    
    print(f"Total Duration: {total_duration:.2f}s")
    print(f"Total Cost: ${total_cost:.4f}")
    
    print("Breakdown:")
    for res in all_results:
        cost_val = res.get('cost', 0.0)
        dur_val = res.get('duration', 0.0)
        run_id = res.get('run_id', res.get('model'))
        print(f" - {run_id}: ${cost_val:.4f} ({dur_val:.2f}s)")
        
    print("\n--- Debug Info ---")
    if not top_groups:
        print("No solutions generated.")
    else:
        for i, group in enumerate(top_groups):
            print(f"Group {i+1}: Count={group['count']}, Correct={group['is_correct']}")
            print(f"  Models: {', '.join(group['models'])}")
            
        for i in range(2, len(sorted_groups)):
            group = sorted_groups[i]
            if group['is_correct']:
                print(f"Group {i+1}: Count={group['count']}, Correct={group['is_correct']}")
                print(f"  Models: {', '.join(group['models'])}")

if __name__ == "__main__":
    main()
