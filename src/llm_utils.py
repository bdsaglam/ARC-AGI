import sys
import time
import random
from typing import Callable, Any, Optional

from src.types import ModelResponse
from src.logging import get_logger, log_failure
from src.errors import RetryableProviderError, UnknownProviderError, NonRetryableProviderError

logger = get_logger("llm_utils")

def run_with_retry(
    func: Callable[[], Any],
    max_retries: int = 3,
    task_id: str = None,
    test_index: int = None,
    run_timestamp: str = None,
    model_name: str = None,
    timing_tracker: list[dict] = None
) -> Any:
    """
    Generic retry loop helper using RetryableProviderError.
    Retries: 2 times (3 total attempts).
    Sleeps: 60s, 300s.
    """
    # Fixed delays for the 2 retries
    retry_delays = [60, 300]
    
    # Construct log prefix
    log_prefix = ""
    if task_id:
        log_prefix += f"[Task: {task_id}"
        if test_index is not None:
            log_prefix += f":{test_index}"
        log_prefix += "] "

    for attempt in range(max_retries):
        start_ts = time.perf_counter()
        try:
            result = func()
            duration = time.perf_counter() - start_ts
            if timing_tracker is not None:
                timing_tracker.append({
                    "type": "attempt",
                    "model": model_name,
                    "duration": duration,
                    "status": "success"
                })
            return result
        except NonRetryableProviderError as e:
            duration = time.perf_counter() - start_ts
            if timing_tracker is not None:
                timing_tracker.append({
                    "type": "attempt",
                    "model": model_name,
                    "duration": duration,
                    "status": "failed",
                    "error": str(e)
                })
            # Log terminal errors with context before re-raising
            logger.error(f"{log_prefix}Non-retryable error: {e}")
            raise e
            
        except RetryableProviderError as e:
            duration = time.perf_counter() - start_ts
            
            if timing_tracker is not None:
                timing_tracker.append({
                    "type": "attempt",
                    "model": model_name,
                    "duration": duration,
                    "status": "failed",
                    "error": str(e)
                })

            # Log the retryable failure
            if run_timestamp:
                log_failure(
                    run_timestamp=run_timestamp,
                    task_id=task_id if task_id else "UNKNOWN",
                    run_id="RETRY_LOOP",
                    error=e,
                    model=model_name if model_name else "UNKNOWN",
                    step="RETRY",
                    test_index=test_index,
                    is_retryable=True
                )

            if attempt == max_retries - 1:
                logger.error(f"{log_prefix}Max retries ({max_retries}) exceeded. Final error: {e}")
                raise e
            
            if attempt < len(retry_delays):
                sleep_time = retry_delays[attempt]
            else:
                sleep_time = retry_delays[-1]
            
            if isinstance(e, UnknownProviderError):
                logger.error(f"{log_prefix}!!! UNKNOWN ERROR (after {duration:.2f}s) - RETRYING (Attempt {attempt + 1}/{max_retries}) !!!")
                logger.error(f"{log_prefix}Error details: {e}")
            else:
                logger.warning(f"{log_prefix}Retryable error (after {duration:.2f}s): {e}. Retrying in {sleep_time}s (Attempt {attempt + 1}/{max_retries})...")
            
            if timing_tracker is not None:
                timing_tracker.append({
                    "type": "wait",
                    "duration": sleep_time,
                    "reason": "retry_delay"
                })
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