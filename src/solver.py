import sys
import time
from pathlib import Path
from typing import List, Optional

import httpx
from openai import OpenAI
from anthropic import Anthropic
from google import genai

from src.config import get_http_client
from src.models import (
    call_model,
    parse_model_arg,
    calculate_cost,
)
from src.types import TaskResult
from src.tasks import load_task, build_prompt
from src.grid import (
    parse_grid_from_text,
    verify_prediction,
)
from src.logging import get_logger
from src.image_generation import generate_and_save_image

logger = get_logger("solver")

def solve_task(
    openai_key: Optional[str],
    claude_key: Optional[str],
    google_keys: Optional[List[str]],
    task_path: Path,
    model_arg: str,
    strategy: str = None,
    verbose: bool = False,
    return_strategy: bool = False,
    verify: bool = False,
    image: bool = False,
) -> List[TaskResult]:
    # Create a thread-local HTTP client with insecure SSL and long timeouts
    # to prevent connection errors and timeouts in threaded environments.
    http_client = get_http_client(
        timeout=3600.0,
        transport=httpx.HTTPTransport(retries=3),
        limits=httpx.Limits(keepalive_expiry=3600)
    )
    
    openai_client = OpenAI(api_key=openai_key, http_client=http_client) if openai_key else None
    
    anthropic_client = None
    if claude_key:
        anthropic_client = Anthropic(api_key=claude_key, http_client=http_client)
        
    # google_client removed

    try:
        task = load_task(task_path)

        image_path = None
        if image:
            image_path = generate_and_save_image(task, task_path.stem, "logs")

        outcomes: List[TaskResult] = []
        for idx, test_example in enumerate(task.test, start=1):
            prompt = build_prompt(
                task.train,
                test_example,
                strategy=strategy,
                image_path=image_path,
            )
            
            success = False
            duration = 0.0
            cost = 0.0
            strategy_text = None
            verified = None
            
            # If verification is requested, we must extract the strategy to verify it
            should_extract = return_strategy or verify

            try:
                start_time = time.perf_counter()
                
                # Ensure we have the client for the requested model
                if model_arg.startswith("gpt") and not openai_client:
                     raise RuntimeError("OpenAI API key missing.")
                if "claude" in model_arg and not anthropic_client:
                     raise RuntimeError("Anthropic API key missing.")
                if "gemini" in model_arg and not google_keys:
                     raise RuntimeError("Google API keys missing.")

                model_response = call_model(
                    openai_client,
                    anthropic_client,
                    google_keys,
                    prompt,
                    model_arg,
                    image_path=image_path,
                    return_strategy=should_extract,
                    verbose=verbose,
                )
                if verbose:
                    if model_response.strategy:
                        logger.debug(f"--- STRATEGY ---\n{model_response.strategy}\n--- END STRATEGY ---")
                    logger.debug(f"--- OUTPUT ---\n{model_response.text}\n--- END OUTPUT ---")

                duration = time.perf_counter() - start_time

                model_config = parse_model_arg(model_arg)
                cost = calculate_cost(model_config, model_response)

                grid_text = model_response.text
                strategy_text = model_response.strategy
                
                predicted_grid = parse_grid_from_text(grid_text)
                success = verify_prediction(predicted_grid, test_example.output)

                # Verification Logic
                if verify:
                    # Step 1: Double Solve Consistency Check
                    # We already have Run A (model_response). We need Run B.
                    if verbose:
                        logger.info(f"Starting Verification Run B for {task_path.name}...")

                    try:
                        start_time_b = time.perf_counter()
                        model_response_b = call_model(
                            openai_client,
                            anthropic_client,
                            google_keys,
                            prompt,
                            model_arg,
                            image_path=image_path,
                            return_strategy=should_extract,
                            verbose=verbose,
                        )
                        duration += (time.perf_counter() - start_time_b)
                        cost += calculate_cost(model_config, model_response_b)
                        
                        grid_text_b = model_response_b.text
                        predicted_grid_b = parse_grid_from_text(grid_text_b)
                        
                        # Compare Grid A (predicted_grid) vs Grid B (predicted_grid_b)
                        if predicted_grid != predicted_grid_b:
                            verified = False
                            if verbose:
                                logger.info(f"Verification failed: Inconsistent outputs between Run A and Run B.")
                                logger.debug(f"Run A Grid: {predicted_grid}")
                                logger.debug(f"Run B Grid: {predicted_grid_b}")
                        else:
                            # Step 2: Backtesting (LOOCV) using Strategy A
                            if strategy_text:
                                verified = True
                                
                                def _attempt_verification(subset_train, target_ex) -> bool:
                                    """Helper to run a single verification pass."""
                                    nonlocal duration, cost
                                    _v_prompt = build_prompt(
                                        subset_train,
                                        target_ex,
                                        strategy=strategy_text
                                    )
                                    _v_start = time.perf_counter()
                                    try:
                                        _v_resp = call_model(
                                            openai_client,
                                            anthropic_client,
                                            google_keys,
                                            _v_prompt,
                                            model_arg,
                                            return_strategy=False,
                                            verbose=verbose
                                        )
                                        if verbose:
                                            logger.debug(f"--- VERIFICATION OUTPUT (Ex {target_ex.input}) ---\n{_v_resp.text}\n--- END VERIFICATION OUTPUT ---")

                                        duration += (time.perf_counter() - _v_start)
                                        cost += calculate_cost(model_config, _v_resp)
                                        _v_grid = parse_grid_from_text(_v_resp.text)
                                        return verify_prediction(_v_grid, target_ex.output)
                                    except Exception as _e:
                                        logger.error(f"Verification error: {_e}")
                                        return False

                                # Determine repetitions: 2 runs for small datasets (<=2 examples), 1 otherwise
                                num_runs = 2 if len(task.train) <= 2 else 1
                                
                                for i, train_ex in enumerate(task.train):
                                    temp_train = task.train[:i] + task.train[i+1:]
                                    
                                    for run_idx in range(num_runs):
                                        passed = _attempt_verification(temp_train, train_ex)
                                        if not passed:
                                            verified = False
                                            if verbose:
                                                logger.info(f"Verification failed on training example {i+1} (Run {run_idx+1}/{num_runs})")
                                            break
                                    
                                    if not verified:
                                        break
                            else:
                                verified = False # Strategy extraction failed for A
                    except Exception as e:
                        logger.error(f"Verification Run B failed: {e}")
                        verified = False


            except Exception as exc:
                logger.error(f"Task {task_path} test {idx} failed: {type(exc)} {exc}")
            
            outcomes.append(TaskResult(
                task_path=task_path,
                test_index=idx,
                success=success,
                model_arg=model_arg,
                duration=duration,
                cost=cost,
                strategy=strategy_text,
                verified=verified
            ))
        return outcomes
    finally:
        # Clean up the thread-local http client
        http_client.close()