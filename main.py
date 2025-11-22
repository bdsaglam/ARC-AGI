from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types

Grid = List[List[int]]
PRICING_PER_1M_TOKENS = {
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "claude-sonnet-4-5-20250929": {
        "input": 3.00,
        "cached_input": 0.30,
        "output": 15.00,
    },
    "gemini-3-pro-preview": {
        "input": 2.00,
        "cached_input": 0.0,
        "output": 12.00,
    },
}

ORDERED_MODELS = [
    "gpt-5.1-none",
    "gpt-5.1-low",
    "gpt-5.1-medium",
    "gpt-5.1-high",
    "claude-sonnet-4.5-no-thinking",
    "claude-sonnet-4.5-thinking-1024",
    "claude-sonnet-4.5-thinking-4000",
    "claude-sonnet-4.5-thinking-16000",
    "claude-sonnet-4.5-thinking-64000",
    "gemini-3-low",
    "gemini-3-high",
]
SUPPORTED_MODELS = set(ORDERED_MODELS)


def get_column_name(model_arg: str) -> str:
    parts = model_arg.split("-")
    formatted_parts = [p.title() if not p[0].isdigit() else p for p in parts]
    name = "-".join(formatted_parts)
    name = name.replace("Gpt", "GPT")
    return name


TABLE_COLUMNS = [get_column_name(m) for m in ORDERED_MODELS]
ResultRecord = Tuple[Path, int, bool, str, float, float]


def parse_model_arg(model_arg: str) -> Tuple[str, str, object]:
    if model_arg not in SUPPORTED_MODELS:
        raise ValueError(f"Model '{model_arg}' not supported. Choose from {SUPPORTED_MODELS}")

    if model_arg.startswith("gpt-5.1-"):
        parts = model_arg.split("-")
        effort = parts[-1]
        base = "-".join(parts[:-1])
        return "openai", base, effort

    if model_arg.startswith("claude-sonnet-4.5-"):
        base = "claude-sonnet-4-5-20250929"
        suffix = model_arg.replace("claude-sonnet-4.5-", "")
        if suffix == "no-thinking":
            return "anthropic", base, 0
        if suffix.startswith("thinking-"):
            try:
                budget = int(suffix.split("-")[1])
                return "anthropic", base, budget
            except (IndexError, ValueError):
                pass

    if model_arg.startswith("gemini-3-"):
        parts = model_arg.split("-")
        effort = parts[-1]
        return "google", "gemini-3-pro-preview", effort

    raise ValueError(f"Unknown model format: {model_arg}")



@dataclass
class Example:
    input: Grid
    output: Grid


@dataclass
class ModelResponse:
    text: str
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int



@dataclass
class Task:
    train: List[Example]
    test: List[Example]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send ARC-AGI tasks to OpenAI.")
    parser.add_argument(
        "task_list",
        type=Path,
        help="JSON file containing a list of task paths (e.g. data/first_100.json).",
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
    return parser.parse_args()


def load_task(json_path: Path) -> Task:
    data = json.loads(json_path.read_text())

    def to_examples(items: List[dict]) -> List[Example]:
        return [Example(input=ex["input"], output=ex["output"]) for ex in items]

    return Task(train=to_examples(data["train"]), test=to_examples(data["test"]))


def load_task_paths(list_path: Path) -> List[Path]:
    payload = json.loads(list_path.read_text())
    if isinstance(payload, dict):
        tasks = payload.get("tasks")
    else:
        tasks = payload
    if not isinstance(tasks, Iterable):
        raise ValueError(f"Task list in {list_path} must be a list.")
    result = []
    for item in tasks:
        if not isinstance(item, str):
            raise ValueError("Each task entry must be a string path.")
        result.append(Path(item))
    return result


def format_grid(grid: Grid) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def build_prompt(train_examples: List[Example], test_example: Example) -> str:
    lines = [
        "You are solving an ARC (Abstraction and Reasoning Corpus) task.",
        "Each grid cell is an integer 0-9 representing a color.",
        "Use the solved examples to infer the transformation and apply it to the test input.",
        "",
        "Solved examples:",
    ]
    for idx, ex in enumerate(train_examples, start=1):
        lines.append(f"Example {idx}:")
        lines.append("input:")
        lines.append(format_grid(ex.input))
        lines.append("output:")
        lines.append(format_grid(ex.output))
        lines.append("")
    lines.append("Test input:")
    lines.append(format_grid(test_example.input))
    lines.append("")
    lines.append("Respond with ONLY the completed output grid, with rows of integers separated by single spaces.")
    return "\n".join(lines)


def call_openai_internal(
    client: OpenAI, prompt: str, model: str, reasoning_effort: str
) -> ModelResponse:
    request_kwargs = {"model": model, "input": prompt}
    if reasoning_effort != "none":
        request_kwargs["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**request_kwargs)
    text_output = None
    for item in response.output or []:
        contents = getattr(item, "content", None)
        if not contents:
            continue
        for content in contents:
            if getattr(content, "type", None) in {"text", "output_text"}:
                text_output = content.text.strip()
                break
        if text_output:
            break

    if not text_output:
        raise RuntimeError("OpenAI response did not contain text output.")

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0

    prompt_details = getattr(usage, "input_tokens_details", None)
    cached_tokens = getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0

    return ModelResponse(
        text=text_output,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
    )


def call_anthropic(
    client: Anthropic, prompt: str, model: str, budget: int
) -> ModelResponse:
    max_tokens = 8192
    if budget and budget > 0:
        max_tokens = max(8192, budget + 4096)

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }

    if budget and budget > 0:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        # Add potential beta headers if required for specific features
        # kwargs["extra_headers"] = {"anthropic-beta": "output-128k-2025-02-19"}

    response = client.messages.create(**kwargs)

    text_parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)

    text = "".join(text_parts).strip()

    p_tokens = response.usage.input_tokens
    c_tokens = response.usage.output_tokens
    # Check for cache read tokens (Anthropic beta/standard feature location varies, assuming usage.cache_read_input_tokens)
    cached = getattr(response.usage, "cache_read_input_tokens", 0) or 0

    return ModelResponse(
        text=text,
        prompt_tokens=p_tokens,
        cached_tokens=cached,
        completion_tokens=c_tokens,
    )


def call_gemini(
    client: genai.Client, prompt: str, model: str, thinking_level: str
) -> ModelResponse:
    # Map thinking_level to thinking_budget since SDK 1.47.0 lacks thinking_level
    budget = 2048
    if thinking_level == "high":
        budget = 16384

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_budget=budget),
        temperature=0.7,
        max_output_tokens=65536,  # Increase max tokens for thinking
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    
    text_parts = []
    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.text:
                text_parts.append(part.text)
    text = "".join(text_parts).strip()

    usage = getattr(response, "usage_metadata", None)
    p_tokens = usage.prompt_token_count if usage else 0
    c_tokens = usage.candidates_token_count if usage else 0
    return ModelResponse(
        text=text,
        prompt_tokens=p_tokens,
        cached_tokens=0,
        completion_tokens=c_tokens,
    )


def call_model(
    openai_client: OpenAI,
    anthropic_client: Anthropic,
    google_client: genai.Client,
    prompt: str,
    model_arg: str,
) -> ModelResponse:
    provider, base_model, config = parse_model_arg(model_arg)

    if provider == "openai":
        return call_openai_internal(openai_client, prompt, base_model, config)
    elif provider == "anthropic":
        if not anthropic_client:
            raise RuntimeError("Anthropic client not initialized.")
        return call_anthropic(anthropic_client, prompt, base_model, config)
    elif provider == "google":
        if not google_client:
            raise RuntimeError("Google client not initialized.")
        return call_gemini(google_client, prompt, base_model, config)

    raise ValueError(f"Unknown provider {provider}")


def parse_grid_from_text(raw_text: str) -> Grid:
    candidate_rows = []
    for line in raw_text.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        if not stripped:
            continue
        tokens = stripped.split()
        if tokens and all(token.isdigit() for token in tokens):
            candidate_rows.append([int(token) for token in tokens])
        else:
            candidate_rows.append(None)

    blocks = []
    current_block = []

    for row in candidate_rows:
        if row is None:
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            if current_block and len(row) != len(current_block[0]):
                blocks.append(current_block)
                current_block = [row]
            else:
                current_block.append(row)

    if current_block:
        blocks.append(current_block)

    if not blocks:
        raise ValueError("Could not parse any numeric rows from OpenAI response.")

    # Return the last block
    return blocks[-1]


def verify_prediction(predicted: Grid, expected: Grid) -> bool:
    return predicted == expected


def solve_task(
    openai_client: OpenAI,
    anthropic_client: Anthropic,
    google_client: genai.Client,
    task_path: Path,
    model_arg: str,
) -> List[ResultRecord]:
    task = load_task(task_path)
    outcomes: List[ResultRecord] = []
    for idx, test_example in enumerate(task.test, start=1):
        prompt = build_prompt(task.train, test_example)
        success = False
        duration = 0.0
        cost = 0.0
        try:
            start_time = time.perf_counter()
            model_response = call_model(
                openai_client, anthropic_client, google_client, prompt, model_arg
            )
            duration = time.perf_counter() - start_time

            _, base_model, _ = parse_model_arg(model_arg)
            pricing = PRICING_PER_1M_TOKENS.get(
                base_model, {"input": 0, "cached_input": 0, "output": 0}
            )

            # Handle Gemini 3 tiered pricing (Long Context)
            if (
                base_model == "gemini-3-pro-preview"
                and model_response.prompt_tokens > 200000
            ):
                pricing = {"input": 4.00, "cached_input": 0.0, "output": 18.00}

            non_cached_input = max(
                0, model_response.prompt_tokens - model_response.cached_tokens
            )

            # NOTE: Cost calculation includes thinking tokens (bundled in completion_tokens for Gemini)
            # NOTE: Time and Cost are printed to STDOUT but NOT saved to Results.md (by user preference).
            cost = (
                (non_cached_input / 1_000_000 * pricing["input"])
                + (
                    model_response.cached_tokens
                    / 1_000_000
                    * pricing.get("cached_input", 0)
                )
                + (model_response.completion_tokens / 1_000_000 * pricing["output"])
            )

            predicted_grid = parse_grid_from_text(model_response.text)
            success = verify_prediction(predicted_grid, test_example.output)
        except Exception as exc:
            print(f"Task {task_path} test {idx} failed: {exc}", file=sys.stderr)
        outcomes.append((task_path, idx, success, model_arg, duration, cost))
    return outcomes


def print_table_header() -> None:
    columns = ["#", "Task", "Test"] + TABLE_COLUMNS + ["Time (s)", "Cost ($)"]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")


def print_result_row(
    row_idx: int,
    task_path: Path,
    test_idx: int,
    success: bool,
    model_arg: str,
    duration: float,
    cost: float,
) -> None:
    column_key = get_column_name(model_arg)

    if column_key not in TABLE_COLUMNS:
        print(
            f"Unsupported reasoning column {column_key}. Update TABLE_COLUMNS to include it.",
            file=sys.stderr,
        )
        return
    values = {column: "-" for column in TABLE_COLUMNS}
    values[column_key] = "PASS" if success else "FAIL"
    row = (
        [str(row_idx), str(task_path), str(test_idx)]
        + [values[col] for col in TABLE_COLUMNS]
        + [f"{duration:.2f}", f"{cost:.2f}"]
    )
    print("| " + " | ".join(row) + " |")


def main() -> None:
    args = parse_args()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    openai_client = OpenAI(api_key=openai_key)

    claude_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    anthropic_client = None
    if claude_key:
        anthropic_client = Anthropic(api_key=claude_key)

    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    google_client = None
    if google_key:
        google_client = genai.Client(api_key=google_key)

    try:
        task_paths = load_task_paths(args.task_list)
    except ValueError as exc:
        raise RuntimeError(f"Failed to read task list {args.task_list}: {exc}") from exc

    print_table_header()
    row_counter = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_path = {
            executor.submit(
                solve_task,
                openai_client,
                anthropic_client,
                google_client,
                path,
                args.model,
            ): path
            for path in task_paths
        }

        for future in as_completed(future_to_path):
            try:
                task_results = future.result()
                for record in task_results:
                    row_counter += 1
                    print_result_row(row_counter, *record)
            except Exception as exc:
                path = future_to_path[future]
                print(
                    f"Task execution for {path} generated an exception: {exc}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
