import time
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from openai import OpenAI
from anthropic import Anthropic
from google import genai

from src.models import call_model, parse_model_arg, calculate_cost
from src.utils import parse_grid_from_text, verify_prediction
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

def run_single_model(model_name, run_id, prompt, test_example, openai_client, anthropic_client, google_client, verbose, image_path=None, run_timestamp=None):
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
            google_client=google_client,
            prompt=prompt,
            model_arg=model_name,
            image_path=image_path,
            return_strategy=False,
            verbose=verbose
        )
        duration = time.perf_counter() - start_ts
        full_response = response.text
        input_tokens = response.prompt_tokens
        output_tokens = response.completion_tokens
        cached_tokens = response.cached_tokens
        
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
            
            return {"model": model_name, "run_id": run_id, "grid": predicted_grid, "is_correct": is_correct, "cost": cost, "duration": duration, "prompt": prompt, "full_response": full_response, "input_tokens": input_tokens, "output_tokens": output_tokens, "cached_tokens": cached_tokens}
                    
        except ValueError as e:
            if verbose:
                print(f"{prefix} Result: FAIL (Parse Error: {e})")
                print(f"\n{prefix} Raw Output:\n{grid_text}")
            return {"model": model_name, "run_id": run_id, "grid": None, "is_correct": False, "cost": cost, "duration": duration, "prompt": prompt, "full_response": full_response, "input_tokens": input_tokens, "output_tokens": output_tokens, "cached_tokens": cached_tokens}

    except Exception as e:
        print(f"{prefix} Error during execution: {e}", file=sys.stderr)
        
        if run_timestamp:
             log_failure(
                run_timestamp=run_timestamp,
                task_id="UNKNOWN", # Passed context is limited here, improved in next refactor if needed
                run_id=run_id,
                error=e,
                model=model_name
            )
            
        return {"model": model_name, "run_id": run_id, "grid": None, "is_correct": False, "cost": cost, "duration": duration, "prompt": prompt, "full_response": str(e), "input_tokens": input_tokens, "output_tokens": output_tokens, "cached_tokens": cached_tokens}
