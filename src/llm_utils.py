import sys
import time
import random
from typing import Callable, Any, Optional

from src.types import ModelResponse
from src.logging import get_logger
from src.errors import RetryableProviderError, UnknownProviderError

logger = get_logger("llm_utils")

def run_with_retry(
    func: Callable[[], Any],
    max_retries: int = 3,
    task_id: str = None,
    test_index: int = None
) -> Any:
    """
    Generic retry loop helper using RetryableProviderError.
    Retries: 2 times (3 total attempts).
    Sleeps: 60s, 300s.
    """
    # Fixed delays for the 2 retries
    retry_delays = [60, 300]

    for attempt in range(max_retries):
        try:
            start_ts = time.perf_counter()
            return func()
        except RetryableProviderError as e:
            duration = time.perf_counter() - start_ts
            if attempt == max_retries - 1:
                raise e
            
            if attempt < len(retry_delays):
                sleep_time = retry_delays[attempt]
            else:
                sleep_time = retry_delays[-1]

            msg_text = f"Retryable error: {e}. Retrying in {sleep_time}s..."
            
            if isinstance(e, UnknownProviderError):
                logger.error(f"!!! UNKNOWN ERROR (after {duration:.2f}s) - RETRYING (Attempt {attempt + 1}/{max_retries}) !!!")
                logger.error(f"Error details: {e}")
            else:
                logger.warning(f"Retryable error (after {duration:.2f}s): {e}. Retrying in {sleep_time}s (Attempt {attempt + 1}/{max_retries})...")
            
            time.sleep(sleep_time)
            
    # Should not be reached if we raise in the loop
    return None

def orchestrate_two_stage(
    solve_func: Callable[[str], ModelResponse],
    explain_func: Callable[[str, ModelResponse], Optional[ModelResponse]],
    prompt: str,
    return_strategy: bool,
    verbose: bool,
    image_path: str = None,
) -> ModelResponse:
    """
    Orchestrates the Solve -> Explain workflow.
    """
    # Step 1: Solve
    if verbose:
        # We use DEBUG level for prompt dumps, requiring verbose=True in main setup
        logger.debug(f"--- REAL PROMPT STEP 1 (Solve) ---\n{prompt}\n--- END REAL PROMPT STEP 1 ---")
        if image_path:
            logger.debug(f"--- IMAGE PROMPT STEP 1 (Solve) ---\n{image_path}\n--- END IMAGE PROMPT STEP 1 ---")
    
    response1 = solve_func(prompt)
    
    if not return_strategy:
        return response1

    # Step 2: Explain
    step2_input = "Explain the strategy you used in broad terms such that it can be applied on other similar examples and other input data. Do not use any of the example or other actual data in your explanation."
    if verbose:
        logger.debug(f"--- REAL PROMPT STEP 2 (Explain) ---\n{step2_input}\n--- END REAL PROMPT STEP 2 ---")

    response2 = explain_func(step2_input, response1)

    if response2:
        # Combine Usage
        return ModelResponse(
            text=response1.text,
            prompt_tokens=response1.prompt_tokens + response2.prompt_tokens,
            cached_tokens=response1.cached_tokens + response2.cached_tokens,
            completion_tokens=response1.completion_tokens + response2.completion_tokens,
            strategy=response2.text # Step 2 text IS the strategy
        )
    else:
        # Step 2 failed, return Step 1 only
        return response1