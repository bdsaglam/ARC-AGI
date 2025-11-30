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
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils import format_grid

# Import from existing project modules
from src.config import get_api_keys
from src.tasks import load_task, build_prompt
from src.models import call_model, calculate_cost, parse_model_arg
from src.utils import parse_grid_from_text, verify_prediction
from src.logging import setup_logging
from src.image_generation import generate_and_save_image

# Constants
DEFAULT_MODELS = [
    "claude-sonnet-4.5-thinking-1024", 
    "claude-sonnet-4.5-thinking-1024",
    "claude-opus-4.5-thinking-1024", 
    "claude-opus-4.5-thinking-1024",
    "gpt-5.1-low",
    "gpt-5.1-low"
]

# Model hierarchy for sorting (Higher value = Higher priority)
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
    """Searches for the task JSON file in data/arc-agi-2-evaluation/. """
    # Handle case where user provides full path or filename
    if task_id.endswith(".json"):
        p = Path(task_id)
        if p.exists():
            return p
        # If it's just a filename, assume it's in the eval dir
        task_id = p.stem

    # Look in arc-agi-2-evaluation
    candidate = Path("data/arc-agi-2-evaluation") / f"{task_id}.json"

    if candidate.exists():
        return candidate
    
    raise FileNotFoundError(f"Task file for '{task_id}' not found in data/arc-agi-2-evaluation/.")

def run_single_model(model_name, prompt, test_example, openai_client, anthropic_client, google_client, verbose, image_path=None):
    """
    Helper function to run a single model and return result data.
    Verbose logs are printed only if verbose=True.
    Errors are always printed to stderr.
    """
    prefix = f"[{model_name}]"
    if verbose:
        print(f"{prefix} Initiating call...")
        if image_path:
            print(f"{prefix} Including image: {image_path}")

    cost = 0.0
    duration = 0.0
    try:
        start_ts = time.perf_counter()
        # Call Model
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
        
        # Calculate cost
        try:
            model_config = parse_model_arg(model_name)
            cost = calculate_cost(model_config, response)
        except Exception:
            pass # Ignore cost calculation errors to ensure flow continues

        if verbose:
            print(f"{prefix} Response received.")

        # Process Result
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
            
            return {
                "model": model_name,
                "grid": predicted_grid,
                "is_correct": is_correct,
                "cost": cost,
                "duration": duration
            }
                    
        except ValueError as e:
            if verbose:
                print(f"{prefix} Result: FAIL (Parse Error: {e})")
                print(f"\n{prefix} Raw Output:\n{grid_text}")
            return {
                "model": model_name,
                "grid": None, # Failed parse
                "is_correct": False,
                "cost": cost,
                "duration": duration
            }

    except Exception as e:
        # ALWAYS print errors to stderr
        print(f"{prefix} Error during execution: {e}", file=sys.stderr)
        return None

def get_group_sort_key(group):
    """
    Returns a tuple for sorting: (count, max_model_weight).
    """
    count = group['count']
    
    # Find the max weight among all models in this group
    max_weight = 0
    for model in group['models']:
        weight = MODEL_WEIGHTS.get(model, 0) # Default to 0 if unknown
        if weight > max_weight:
            max_weight = weight
            
    return (count, max_weight)

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
    
    args = parser.parse_args()

    # Suppress Pydantic serialization warnings from google-genai SDK
    # These are known harmless false positives due to SDK internal type mismatches
    warnings.filterwarnings(
        "ignore", 
        message=r"Pydantic serializer warnings:", 
        category=UserWarning
    )

    # Determine models to run
    if args.models:
        models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models_to_run = DEFAULT_MODELS

    # Setup basic logging
    setup_logging(args.verbose)

    try:
        task_path = find_task_path(args.task)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Setup API keys
    openai_key, claude_key, google_key = get_api_keys()
    
    if not claude_key:
        print("Error: Anthropic API key not found.", file=sys.stderr)
        sys.exit(1)
    
    if not openai_key:
        print("Warning: OpenAI API key not found. GPT-5.1 models will fail.", file=sys.stderr)

    if not google_key:
        print("Warning: Google API key not found. Gemini models will fail.", file=sys.stderr)

    # Setup HTTP client (mirroring solver.py for robustness)
    os.environ["SSL_CERT_FILE"] = certifi.where()
    http_client = httpx.Client(
        timeout=3600.0,
        transport=httpx.HTTPTransport(retries=3, verify=False),
        limits=httpx.Limits(keepalive_expiry=3600),
        verify=False
    )

    anthropic_client = Anthropic(api_key=claude_key, http_client=http_client)
    openai_client = OpenAI(api_key=openai_key, http_client=http_client) if openai_key else None
    google_client = genai.Client(api_key=google_key) if google_key else None

    try:
        task = load_task(task_path)
    except Exception as e:
        print(f"Error loading task: {e}", file=sys.stderr)
        http_client.close()
        sys.exit(1)

    # Validate test index
    test_idx = args.test - 1
    if test_idx < 0 or test_idx >= len(task.test):
        print(f"Error: Test index {args.test} is out of range. Task has {len(task.test)} test cases.", file=sys.stderr)
        http_client.close()
        sys.exit(1)

    test_example = task.test[test_idx]

    # Generate image if requested
    image_path = None
    if args.image:
        image_path = generate_and_save_image(task, args.task, "logs")

    # Build Prompt
    prompt = build_prompt(task.train, test_example, strategy=args.hint, image_path=image_path, trigger_deep_thinking=args.trigger_deep_thinking)

    total_calls = len(models_to_run)
    print(f"Starting parallel execution for {total_calls} models...")
    
    start_time = time.perf_counter()

    all_results = []
    # Track which model corresponds to which future
    future_to_model = {}
    # Track pending models (using list to handle duplicates correctly by index if needed, but for display names are fine)
    # Actually, simple counting is robust. For specific names, we can map future -> name.
    pending_models_names = list(models_to_run) 

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for model_name in models_to_run:
            future = executor.submit(
                run_single_model,
                model_name,
                prompt,
                test_example,
                openai_client,
                anthropic_client,
                google_client,
                args.verbose,
                image_path,
            )
            future_to_model[future] = model_name
        
        completed_count = 0
        for future in as_completed(future_to_model):
            completed_count += 1
            model_name = future_to_model[future]
            
            # Remove one instance of this model name from pending list
            if model_name in pending_models_names:
                pending_models_names.remove(model_name)
            
            # Calculate elapsed time
            elapsed = time.perf_counter() - start_time
            
            # Construct waiting list string (summarized if too long)
            waiting_str = ", ".join(pending_models_names)
            if len(waiting_str) > 50:
                waiting_str = waiting_str[:47] + "..."
            if not waiting_str:
                waiting_str = "(none)"

            # Determine status
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
                # We print the exception detail to stderr but keep the main output clean
                print(f"\nThread generated an exception: {exc}", file=sys.stderr)

            # Progress format: Xs X/Y done ModelName (Status) Wait: ModelA, ModelB...
            print(f"{elapsed:.1f}s {completed_count}/{total_calls} done {model_name} {status_str} Wait: {waiting_str}")

            if res:
                all_results.append(res)

    http_client.close()
    total_duration = time.perf_counter() - start_time

    # Group results and calc total cost
    grouped_solutions = {}
    total_cost = 0.0
    
    for res in all_results:
        total_cost += res.get("cost", 0.0)
        
        if res["grid"] is None:
            continue # Skip failed parses for grouping, but we counted cost
            
        # Make grid hashable
        grid_tuple = tuple(tuple(row) for row in res["grid"])
        
        if grid_tuple not in grouped_solutions:
            grouped_solutions[grid_tuple] = {
                "grid": res["grid"],
                "count": 0,
                "models": [],
                "is_correct": res["is_correct"]
            }
        
        grouped_solutions[grid_tuple]["count"] += 1
        grouped_solutions[grid_tuple]["models"].append(res["model"])

    # Sort groups by count (descending) then by max model weight (descending)
    sorted_groups = sorted(grouped_solutions.values(), key=get_group_sort_key, reverse=True)
    
    print("\n" + "="*40)
    print("FINAL OUTCOME")
    print("="*40)
    
    is_solved = False
    
    # Check top 2 groups (if they exist)
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
    
    # Breakdown cost and time
    print("Breakdown:")
    for res in all_results:
        cost_val = res.get('cost', 0.0)
        dur_val = res.get('duration', 0.0)
        print(f" - {res['model']}: ${cost_val:.4f} ({dur_val:.2f}s)")
        
    print("\n--- Debug Info ---")
    if not top_groups:
        print("No solutions generated.")
    else:
        # Print top 2 groups
        for i, group in enumerate(top_groups):
            print(f"Group {i+1}: Count={group['count']}, Correct={group['is_correct']}")
            print(f"  Models: {', '.join(group['models'])}")
            
        # Print any other correct groups
        for i in range(2, len(sorted_groups)):
            group = sorted_groups[i]
            if group['is_correct']:
                print(f"Group {i+1}: Count={group['count']}, Correct={group['is_correct']}")
                print(f"  Models: {', '.join(group['models'])}")

if __name__ == "__main__":
    main()