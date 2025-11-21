from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from openai import OpenAI

Grid = List[List[int]]
MODEL_NAME = "gpt-5.1"
SUPPORTED_REASONING = {"none", "low", "medium", "high"}


@dataclass
class Example:
    input: Grid
    output: Grid


@dataclass
class Task:
    train: List[Example]
    test: List[Example]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send ARC-AGI tasks to OpenAI.")
    parser.add_argument(
        "json_paths",
        type=Path,
        nargs="+",
        help="One or more task JSON files (e.g. data/arc-agi-2-training/00576224.json)",
    )
    parser.add_argument(
        "--reasoning",
        choices=sorted(SUPPORTED_REASONING),
        default="none",
        help="Reasoning effort for OpenAI (supported: none, low, medium, high).",
    )
    return parser.parse_args()


def load_task(json_path: Path) -> Task:
    data = json.loads(json_path.read_text())

    def to_examples(items: List[dict]) -> List[Example]:
        return [Example(input=ex["input"], output=ex["output"]) for ex in items]

    return Task(train=to_examples(data["train"]), test=to_examples(data["test"]))


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


def ensure_reasoning_supported(model: str, reasoning_effort: str) -> None:
    if reasoning_effort not in SUPPORTED_REASONING:
        supported_list = ", ".join(sorted(SUPPORTED_REASONING))
        raise ValueError(
            f"Reasoning effort '{reasoning_effort}' is not supported for {model}. "
            f"Select one of: {supported_list}."
        )


def call_openai(client: OpenAI, prompt: str, reasoning_effort: str) -> str:
    ensure_reasoning_supported(MODEL_NAME, reasoning_effort)
    request_kwargs = {"model": MODEL_NAME, "input": prompt}
    if reasoning_effort != "none":
        request_kwargs["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**request_kwargs)
    for item in response.output or []:
        contents = getattr(item, "content", None)
        if not contents:
            continue
        for content in contents:
            if getattr(content, "type", None) in {"text", "output_text"}:
                return content.text.strip()
    raise RuntimeError("OpenAI response did not contain text output.")


def parse_grid_from_text(raw_text: str) -> Grid:
    rows: Grid = []
    for line in raw_text.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if tokens and all(token.isdigit() for token in tokens):
            rows.append([int(token) for token in tokens])
    if not rows:
        raise ValueError("Could not parse any numeric rows from OpenAI response.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Parsed grid has inconsistent row lengths.")
    return rows


def verify_prediction(predicted: Grid, expected: Grid) -> bool:
    return predicted == expected


def solve_task(client: OpenAI, task_path: Path, reasoning_effort: str) -> None:
    task = load_task(task_path)
    for idx, test_example in enumerate(task.test, start=1):
        print(f"\n[{task_path}] Solving test case {idx} with OpenAI...")
        prompt = build_prompt(task.train, test_example)
        # print("Prompt sent to OpenAI:")
        # print(prompt)
        # print("--- End prompt ---")
        try:
            response_text = call_openai(client, prompt, reasoning_effort)
        except ValueError as exc:
            print(f"Skipping task {task_path}: {exc}")
            return
        print("Model output:")
        print(response_text)
        try:
            predicted_grid = parse_grid_from_text(response_text)
        except ValueError as exc:
            print(f"Failed to parse model output: {exc}")
            continue

        if verify_prediction(predicted_grid, test_example.output):
            print("Result: PASS ✅")
        else:
            print("Result: FAIL ❌")
            print("Predicted grid:")
            print(format_grid(predicted_grid))
            print("Expected grid:")
            print(format_grid(test_example.output))


def main() -> None:
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)

    for path in args.json_paths:
        solve_task(client, path, args.reasoning)


if __name__ == "__main__":
    main()
