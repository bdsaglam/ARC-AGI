import sys
import time
from pathlib import Path
from typing import List, Optional

import httpx
from openai import OpenAI
from anthropic import Anthropic
from google import genai

from src.models import (
    call_model,
    parse_model_arg,
    calculate_cost,
    TaskResult,
)
from src.tasks import load_task, build_prompt
from src.utils import (
    parse_grid_from_text,
    verify_prediction,
)

def solve_task(
    openai_key: Optional[str],
    claude_key: Optional[str],
    google_key: Optional[str],
    task_path: Path,
    model_arg: str,
    strategy: str = None,
    verbose: bool = False,
    return_strategy: bool = False,
) -> List[TaskResult]:
    # Create a thread-local HTTP client with insecure SSL and long timeouts
    # to prevent connection errors and timeouts in threaded environments.
    http_client = httpx.Client(
        timeout=3600.0,
        transport=httpx.HTTPTransport(retries=3, verify=False),
        limits=httpx.Limits(keepalive_expiry=3600),
        verify=False
    )
    
    openai_client = OpenAI(api_key=openai_key, http_client=http_client) if openai_key else None
    
    anthropic_client = None
    if claude_key:
        anthropic_client = Anthropic(api_key=claude_key, http_client=http_client)
        
    google_client = None
    if google_key:
        google_client = genai.Client(api_key=google_key) # Gemini uses its own transport

    try:
        task = load_task(task_path)
        outcomes: List[TaskResult] = []
        for idx, test_example in enumerate(task.test, start=1):
            prompt = build_prompt(
                task.train,
                test_example,
                strategy=strategy,
            )
            
            success = False
            duration = 0.0
            cost = 0.0
            strategy_text = None
            try:
                start_time = time.perf_counter()
                
                # Ensure we have the client for the requested model
                if model_arg.startswith("gpt") and not openai_client:
                     raise RuntimeError("OpenAI API key missing.")
                if "claude" in model_arg and not anthropic_client:
                     raise RuntimeError("Anthropic API key missing.")
                if "gemini" in model_arg and not google_client:
                     raise RuntimeError("Google API key missing.")

                model_response = call_model(
                    openai_client,
                    anthropic_client,
                    google_client,
                    prompt,
                    model_arg,
                    return_strategy=return_strategy,
                    verbose=verbose,
                )
                if verbose:
                    if model_response.strategy:
                        print(
                            f"--- STRATEGY ---\n{model_response.strategy}\n--- END STRATEGY ---",
                            file=sys.stderr,
                        )
                    print(
                        f"--- OUTPUT ---\n{model_response.text}\n--- END OUTPUT ---",
                        file=sys.stderr,
                    )
                duration = time.perf_counter() - start_time

                model_config = parse_model_arg(model_arg)
                cost = calculate_cost(model_config, model_response)

                grid_text = model_response.text
                strategy_text = model_response.strategy
                
                predicted_grid = parse_grid_from_text(grid_text)
                success = verify_prediction(predicted_grid, test_example.output)
            except Exception as exc:
                print(f"Task {task_path} test {idx} failed: {type(exc)} {exc}", file=sys.stderr)
            
            outcomes.append(TaskResult(
                task_path=task_path,
                test_index=idx,
                success=success,
                model_arg=model_arg,
                duration=duration,
                cost=cost,
                strategy=strategy_text
            ))
        return outcomes
    finally:
        # Clean up the thread-local http client
        http_client.close()
