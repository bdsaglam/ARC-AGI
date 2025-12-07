import sys
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from openai import OpenAI
from anthropic import Anthropic
from google import genai
import certifi


# Import from existing project modules
from src.config import get_api_keys
from src.tasks import load_task, build_prompt, build_objects_extraction_prompt
from src.image_generation import generate_and_save_image
from src.hint_generation import generate_hint
from src.logging import setup_logging
from src.parallel import run_single_model
from src.run_utils import find_task_path

DEFAULT_MODELS = [
    "claude-sonnet-4.5-thinking-1024",
    "claude-opus-4.5-thinking-1024",
    "gpt-5.1-low"
]

def run_default_mode(args, answer_path: Path = None):
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

    openai_key, claude_key, google_keys = get_api_keys()
    
    if not claude_key:
        print("Error: Anthropic API key not found.", file=sys.stderr)
        sys.exit(1)
    if not openai_key:
        print("Warning: OpenAI API key not found. GPT-5.1 models will fail.", file=sys.stderr)
    if not google_keys:
        print("Warning: Google API keys not found. Gemini models will fail.", file=sys.stderr)

    os.environ["SSL_CERT_FILE"] = certifi.where()
    http_client = httpx.Client(timeout=3600.0, transport=httpx.HTTPTransport(retries=3, verify=False), limits=httpx.Limits(keepalive_expiry=3600), verify=False)
    anthropic_client = Anthropic(api_key=claude_key, http_client=http_client)
    openai_client = OpenAI(api_key=openai_key, http_client=http_client) if openai_key else None
    # google_client removed

    try:
        task = load_task(task_path, answer_path=answer_path)
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
            future = executor.submit(run_single_model, run["name"], run["run_id"], prompt, test_example, openai_client, anthropic_client, google_keys, args.verbose, image_path)
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

    sorted_groups = sorted(grouped_solutions.values(), key=lambda g: g['count'], reverse=True)
    
    print("\n" + "="*40)
    print("FINAL OUTCOME")
    print("="*40)
    
    is_solved = False
    unknown_status = False
    top_groups = sorted_groups[:2]
    
    if len(top_groups) > 0:
         if top_groups[0]["is_correct"] is None:
             unknown_status = True
         elif top_groups[0]["is_correct"]:
             is_solved = True
         elif len(top_groups) > 1 and top_groups[1]["is_correct"]:
             is_solved = True
        
    if unknown_status:
        print("Outcome: SUBMITTED (No Ground Truth)")
    elif is_solved:
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
            c_str = str(group['is_correct']) if group['is_correct'] is not None else "Unknown"
            print(f"Group {i+1}: Count={group['count']}, Correct={c_str}")
            print(f"  Models: {', '.join(group['models'])}")
            
        for i in range(2, len(sorted_groups)):
            group = sorted_groups[i]
            if group['is_correct'] is True:
                print(f"Group {i+1}: Count={group['count']}, Correct={group['is_correct']}")
                print(f"  Models: {', '.join(group['models'])}")
    
    return top_groups