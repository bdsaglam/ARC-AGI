#!/usr/bin/env python

from __future__ import annotations

import os
import sys
from pathlib import Path
import argparse

import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

from src.config import get_api_keys, get_http_client
from src.tasks import load_task
from src.models import call_model
from src.models import parse_model_arg
from src.logging import setup_logging, get_logger
from src.grid import format_grid
import httpx
from openai import OpenAI
from anthropic import Anthropic
from google import genai

def get_hints(task_path: Path, test_index: int, model_arg: str, verbose: bool = False) -> str:
    """
    Generates hints for a given task and test example.

    Args:
        task_path: Path to the task JSON file.
        test_index: The index of the test example to get hints for. (unused)
        model_arg: The model to use for generating hints.
        verbose: Whether to print verbose output.

    Returns:
        The generated hints as a string.
    """
    logger = get_logger("get_hints")
    openai_key, claude_key, google_keys = get_api_keys()

    http_client = get_http_client(
        timeout=3600.0,
        transport=httpx.HTTPTransport(retries=3),
        limits=httpx.Limits(keepalive_expiry=3600)
    )
    
    openai_client = OpenAI(api_key=openai_key, http_client=http_client) if openai_key else None
    
    anthropic_client = None
    if claude_key:
        anthropic_client = Anthropic(api_key=claude_key, http_client=http_client)
        
    # google_client instantiation removed

    try:
        task = load_task(task_path)
    except ValueError as e:
        logger.error(f"Error loading task: {e}")
        sys.exit(1)

    training_examples = task.train

    prompt = "First, provide a detailed, step-by-step explanation of the transformation that turns the input grids into the output grids. Your explanation should be easy to follow for someone who is new to these kinds of problems. After your explanation, provide a numbered list of 3 single-sentence hints that provide the key non-obvious insights that would ensure a person would not fail on solving this problem.\n"
    
    for i, train_example in enumerate(training_examples):
        prompt += f"--- Example {i+1} ---\n"
        prompt += f"Input:\n{format_grid(train_example.input)}\n"
        prompt += f"Output:\n{format_grid(train_example.output)}\n"

    print(prompt)

    try:
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
            return_strategy=True,
            verbose=verbose,
        )
        if verbose:
            if model_response.strategy:
                logger.debug(f"--- STRATEGY ---\\n{model_response.strategy}\\n--- END STRATEGY ---")
            logger.debug(f"--- OUTPUT ---\\n{model_response.text}\\n--- END OUTPUT ---")
        
        return model_response.strategy
    
    except Exception as exc:
        logger.error(f"Failed to get hints: {exc}")
        sys.exit(1)
    finally:
        http_client.close()


def main():
    """
    Main function for the get_hints script.
    """
    parser = argparse.ArgumentParser(description="Get hints for an ARC task.")
    parser.add_argument("--task", type=str, required=True, help="Path to the task JSON file.")
    parser.add_argument("--test", type=int, required=True, help="Index of the test example.")
    parser.add_argument("--model", type=str, default="gpt-5.1-none", help="The model to use for generating hints.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    task_input = args.task
    if not task_input.endswith(".json"):
        task_path = Path("data/arc-agi-2-evaluation") / f"{task_input}.json"
        if not task_path.exists():
            print(f"Error: Task file not found for ID: {task_input}", file=sys.stderr)
            sys.exit(1)
    else:
        task_path = Path(task_input)
        if not task_path.exists():
            print(f"Error: Task file not found at path: {task_path}", file=sys.stderr)
            sys.exit(1)

    setup_logging(args.verbose)

    hints = get_hints(task_path, args.test, args.model, args.verbose)
    print(hints)


if __name__ == "__main__":
    main()
