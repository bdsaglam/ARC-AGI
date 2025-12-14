import time
import re
import sys
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from openai import OpenAI
from anthropic import Anthropic
from google import genai

from src.models import call_model, parse_model_arg, calculate_cost
from src.grid import parse_grid_from_text, verify_prediction
from src.rate_limiter import RateLimiter
from src.config import PROVIDER_RATE_LIMITS
from src.logging import log_failure

# Initialize global rate limiters per provider
LIMITERS = {
    name: RateLimiter(**config)
    for name, config in PROVIDER_RATE_LIMITS.items()
}

_SCALED = False

def set_rate_limit_scaling(factor: float):
    """
    Scales the rate limits for all providers by a factor.
    Used when running multiple worker processes to divide the global rate limit.
    """
    global _SCALED
    if _SCALED:
        return
    _SCALED = True

    if factor == 1.0:
        return

    for name, limiter in LIMITERS.items():
        original_rate = limiter.rate
        new_rate = original_rate * factor
        # Apply minimum floor of 1 request per minute to avoid divide by zero or effective hang
        if new_rate < 1.0:
            new_rate = 1.0
        
        limiter.rate = new_rate
        limiter.per_seconds = 60.0 / new_rate

def extract_tag_content(text: str, tag_name: str) -> str | None:
    """Extracts content between <tag>...</tag>."""
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def run_single_model(model_name, run_id, prompt, test_example, openai_client, anthropic_client, google_keys, verbose, image_path=None, run_timestamp=None, task_id=None, test_index=None, step_name=None, use_background=False):
    original_model_name = model_name
    prefix = f"[{run_id}]"
    if verbose:
        print(f"{prefix} Initiating call...")
        if image_path:
            print(f"{prefix} Including image: {image_path}")

    cost = 0.0
    duration = 0.0
    full_response = ""
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    timings = []
    try:
        # Acquire rate limit token
        try:
            model_config = parse_model_arg(model_name)
            provider = model_config.provider
            if provider == "gemini": # Map internal name to config key
                provider = "google"
            
            if provider in LIMITERS:
                if verbose:
                    print(f"{prefix} Waiting for rate limit token ({provider})...")
                LIMITERS[provider].acquire()
        except Exception as e:
            print(f"{prefix} Warning: Failed to acquire rate limit token: {e}", file=sys.stderr)

        start_ts = time.perf_counter()
        response = call_model(
            openai_client=openai_client,
            anthropic_client=anthropic_client,
            google_keys=google_keys,
            prompt=prompt,
            model_arg=model_name,
            image_path=image_path,
            return_strategy=False,
            verbose=verbose,
            task_id=task_id,
            test_index=test_index,
            step_name=step_name,
            use_background=use_background,
            run_timestamp=run_timestamp,
            timing_tracker=timings
        )
        duration = time.perf_counter() - start_ts
        full_response = response.text
        input_tokens = response.prompt_tokens
        output_tokens = response.completion_tokens
        cached_tokens = response.cached_tokens
        
        # Handle model fallback (e.g., OpenAI -> Opus)
        if response.model_name and response.model_name != model_name:
            if verbose:
                print(f"{prefix} Model fallback occurred: {model_name} -> {response.model_name}")
            
            # Update run_id to reflect the new model
            run_id = run_id.replace(model_name, response.model_name, 1)
            model_name = response.model_name
        
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
                elif is_correct is False:
                    print(f"{prefix} Result: FAIL")
                else:
                    print(f"{prefix} Result: UNKNOWN (No Ground Truth)")
                    print(grid_text)
            
            return {"model": model_name, "requested_model": original_model_name, "run_id": run_id, "grid": predicted_grid, "is_correct": is_correct, "cost": cost, "duration": duration, "prompt": prompt, "full_response": full_response, "input_tokens": input_tokens, "output_tokens": output_tokens, "cached_tokens": cached_tokens, "timing_breakdown": timings}
                    
        except ValueError as e:
            if verbose:
                print(f"{prefix} Result: FAIL (Parse Error: {e})")
                print(f"\n{prefix} Raw Output:\n{grid_text}")
            return {"model": model_name, "requested_model": original_model_name, "run_id": run_id, "grid": None, "is_correct": False, "cost": cost, "duration": duration, "prompt": prompt, "full_response": full_response, "input_tokens": input_tokens, "output_tokens": output_tokens, "cached_tokens": cached_tokens, "timing_breakdown": timings}

    except Exception as e:
        # Loud error reporting to bypass buffering
        error_msg = f"\n!!! CRITICAL ERROR in {model_name} ({run_id}) !!!\n{str(e)}\n{traceback.format_exc()}\n"
        try:
            os.write(2, error_msg.encode('utf-8', errors='replace'))
        except OSError:
            pass # Fallback if stderr is closed

        print(f"{prefix} Error during execution: {e}", file=sys.stderr)
        
        if run_timestamp:
             log_failure(
                run_timestamp=run_timestamp,
                task_id=task_id if task_id else "UNKNOWN",
                run_id=run_id,
                error=e,
                model=model_name,
                test_index=test_index
            )
            
        return {"model": model_name, "requested_model": original_model_name, "run_id": run_id, "grid": None, "is_correct": False, "cost": cost, "duration": duration, "prompt": prompt, "full_response": str(e), "input_tokens": input_tokens, "output_tokens": output_tokens, "cached_tokens": cached_tokens, "timing_breakdown": timings}

def run_models_in_parallel(models_to_run, run_id_counts, step_name, prompt, test_example, openai_client, anthropic_client, google_keys, verbose, image_path=None, run_timestamp=None, task_id=None, test_index=None, completion_message: str = None, on_task_complete=None, use_background=False):
    all_results = []
    
    # Wrapper for debugging queue times
    def debug_run_single_model(queue_time, *args, **kwargs):
        start_wait = time.time() - queue_time
        if start_wait > 0.1:  # Only print if waiting more than 100ms
            run_id = args[1] if len(args) > 1 else kwargs.get('run_id', 'unknown')
            print(f"DEBUG: Task {run_id} waited in queue for {start_wait:.2f}s")
        return run_single_model(*args, **kwargs)

    with ThreadPoolExecutor(max_workers=20) as executor:
        
        # Generate unique run IDs
        run_list = []
        for model_name in models_to_run:
            count = run_id_counts.get(model_name, 0) + 1
            run_id_counts[model_name] = count
            run_id = f"{model_name}_{count}_{step_name}"
            run_list.append({"name": model_name, "run_id": run_id})

        future_to_run_id = {
            executor.submit(
                debug_run_single_model,
                time.time(), # Capture queue time
                run["name"], run["run_id"], prompt, test_example, openai_client, anthropic_client, google_keys, verbose, image_path, run_timestamp, task_id, test_index, step_name, use_background
            ): run["run_id"]
            for run in run_list
        }

        total_tasks = len(future_to_run_id)
        completed_count = 0

        for future in as_completed(future_to_run_id):
            completed_count += 1
            run_id = future_to_run_id[future]
            try:
                res = future.result()
                if res:
                    all_results.append(res)
                
                # Handle progress updates
                if on_task_complete:
                    on_task_complete()
                elif completion_message:
                    remaining = total_tasks - completed_count
                    print(f"{completion_message}: {remaining} left")
                    
            except Exception as e:
                print(f"Model run {run_id} failed: {e}")
                
    return all_results