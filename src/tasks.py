import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable

from src.utils import Grid, format_grid

@dataclass
class Example:
    input: Grid
    output: Grid

@dataclass
class Task:
    train: List[Example]
    test: List[Example]

def load_task(json_path: Path) -> Task:
    data = json.loads(json_path.read_text())

    def to_examples(items: List[dict]) -> List[Example]:
        return [Example(input=ex["input"], output=ex["output"]) for ex in items]

    return Task(train=to_examples(data["train"]), test=to_examples(data["test"]))

def load_task_paths(list_path: Path) -> List[Path]:
    payload = json.loads(list_path.read_text())
    if not isinstance(payload, dict) or "tasks" not in payload:
        raise ValueError(f"Task list in {list_path} must be a dict with a 'tasks' key.")
    
    tasks = payload["tasks"]
    if not isinstance(tasks, list):
        raise ValueError(f"The 'tasks' value in {list_path} must be a list.")

    result = []
    for item in tasks:
        if not isinstance(item, str):
            raise ValueError("Each task entry must be a string path.")
        result.append(Path(item))
    return result

def build_prompt(
    train_examples: List[Example],
    test_example: Example,
    strategy: str = None,
    suppress_final_instruction: bool = False,
) -> str:
    lines = [
        "You are solving an ARC (Abstraction and Reasoning Corpus) task.",
    ]
    if strategy:
        lines.append(f"Suggested Strategy: {strategy}")

    lines.append("Each grid cell is an integer 0-9 representing a color.")
    lines.append(
        "Use the solved examples to infer the transformation and apply it to the test input."
    )
    lines.append("")
    lines.append("Solved examples:")
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

    if suppress_final_instruction:
        return "\n".join(lines)

    lines.append("Respond with ONLY the completed output grid.")

    return "\n".join(lines)