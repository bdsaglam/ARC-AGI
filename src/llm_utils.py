import sys
import time
from typing import Callable, Any, Optional

from src.types import ModelResponse
from src.logging import get_logger

logger = get_logger("llm_utils")

def run_with_retry(
    func: Callable[[], Any],
    retry_predicate: Callable[[Exception], bool],
    max_retries: int = 7
) -> Any:
    """Generic retry loop helper."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if retry_predicate(e):
                if attempt < max_retries - 1:
                    # Exponential backoff with a cap: 5, 10, 20, 40, 60, 60...
                    delay = min(5 * (2 ** attempt), 60)
                    logger.debug(f"Retryable error encountered: {e}. Retrying in {delay}s (Attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                    continue
            raise e

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