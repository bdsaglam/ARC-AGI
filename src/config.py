import argparse
import os
from pathlib import Path
from typing import Optional

from src.models import SUPPORTED_MODELS, ORDERED_MODELS

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send ARC-AGI tasks to OpenAI.")
    parser.add_argument(
        "task_source",
        type=str,
        help="JSON file containing a list of task paths (e.g. data/first_100.json) OR a single task ID (e.g. 15696249).",
    )
    parser.add_argument(
        "--model",
        choices=sorted(SUPPORTED_MODELS),
        default="gpt-5.1-none",
        help="Model to use (e.g. gpt-5.1-none, gpt-5.1-low).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Number of parallel worker threads (default: 20).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Suggested strategy to include in the prompt.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Output full prompt and model answer to stderr.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of times to run the test suite.",
    )
    parser.add_argument(
        "--extract-strategy",
        action="store_true",
        help="Enable two-stage strategy extraction (Solve -> Explain).",
    )
    return parser.parse_args()

def get_api_keys() -> tuple[Optional[str], Optional[str], Optional[str]]:
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    
    claude_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    return openai_key, claude_key, google_key
