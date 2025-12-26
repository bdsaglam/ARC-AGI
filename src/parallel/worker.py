import time
import sys
import os
import traceback
from typing import Optional, List, Dict

from src.models import call_model, parse_model_arg, calculate_cost
from src.grid import parse_grid_from_text, verify_prediction
from src.logging import log_failure
from src.parallel.limiter import LIMITERS
from src.parallel.codegen import extract_and_run_solver

def run_single_model(model_name, run_id, prompt, test_example, openai_client, anthropic_client, google_keys, verbose, image_path=None, run_timestamp=None, task_id=None, test_index=None, step_name=None, use_background=False, execution_mode="grid", train_examples=None):
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
        
        # Handle model fallback
        if response.model_name and response.model_name != model_name:
            if verbose:
                print(f"{prefix} Model fallback occurred: {model_name} -> {response.model_name}")
            
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

        predicted_grid = None
        verification_details = None
        
        if execution_mode == "code":
            # CODE execution path
            try:
                predicted_grid, verification_details = extract_and_run_solver(grid_text, test_example.input, train_examples=train_examples)
            except Exception as e:
                if verbose:
                    print(f"{prefix} Code Execution Failed: {e}")
        else:
            # GRID parsing path (Standard)
            try:
                predicted_grid = parse_grid_from_text(grid_text)
            except ValueError as e:
                if verbose:
                    print(f"{prefix} Result: FAIL (Parse Error: {e})")
                    print(f"\n{prefix} Raw Output:\n{grid_text}")
        
        # Verification
        try:
            is_correct = verify_prediction(predicted_grid, test_example.output)
            
            if verbose:
                if is_correct:
                    print(f"{prefix} Result: PASS")
                elif is_correct is False:
                    print(f"{prefix} Result: FAIL")
                else:
                    print(f"{prefix} Result: UNKNOWN (No Ground Truth)")
            
            return {
                "model": model_name, 
                "requested_model": original_model_name, 
                "run_id": run_id, 
                "grid": predicted_grid, 
                "is_correct": is_correct, 
                "cost": cost, 
                "duration": duration, 
                "prompt": prompt, 
                "full_response": full_response, 
                "input_tokens": input_tokens, 
                "output_tokens": output_tokens, 
                "cached_tokens": cached_tokens, 
                "timing_breakdown": timings,
                "verification_details": verification_details
            }
                    
        except ValueError:
             return {
                 "model": model_name, 
                 "requested_model": original_model_name, 
                 "run_id": run_id, 
                 "grid": None, 
                 "is_correct": False, 
                 "cost": cost, 
                 "duration": duration, 
                 "prompt": prompt, 
                 "full_response": full_response, 
                 "input_tokens": input_tokens, 
                 "output_tokens": output_tokens, 
                 "cached_tokens": cached_tokens, 
                 "timing_breakdown": timings,
                 "verification_details": verification_details
             }

    except Exception as e:
        error_msg = f"\n!!! CRITICAL ERROR in {model_name} ({run_id}) !!!\n{str(e)}\n{traceback.format_exc()}\n"
        try:
            os.write(2, error_msg.encode('utf-8', errors='replace'))
        except OSError:
            pass

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
