import datetime
import json
import sys
from pathlib import Path
from typing import List

from src.models import TaskResult, ORDERED_MODELS

# Re-implement get_column_name locally or import if shared. 
# Since it's presentation logic, it fits here.
def get_column_name(model_arg: str) -> str:
    parts = model_arg.split("-")
    formatted_parts = [p.title() if not p[0].isdigit() else p for p in parts]
    name = "-".join(formatted_parts)
    name = name.replace("Gpt", "GPT")
    return name

TABLE_COLUMNS = [get_column_name(m) for m in ORDERED_MODELS]

def print_table_header() -> None:
    # Add Duration and Cost to header
    columns = ["#", "Task", "Test"] + TABLE_COLUMNS + ["Duration", "Cost"]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")

def print_result_row(row_idx: int, result: TaskResult) -> None:
    column_key = get_column_name(result.model_arg)

    # Redundant validation removed. 
    # column_key validity is implicitly guaranteed by SUPPORTED_MODELS check in config parsing
    # and the fact that TABLE_COLUMNS is derived from ORDERED_MODELS.

    values = {column: "-" for column in TABLE_COLUMNS}
    values[column_key] = "PASS" if result.success else "FAIL"
    
    row = (
        [str(row_idx), str(result.task_path), str(result.test_index)]
        + [values[col] for col in TABLE_COLUMNS]
        + [f"{result.duration:.2f}", f"{result.cost:.2f}"]
    )
    print("| " + " | ".join(row) + " |")

def print_summary(all_results: List[TaskResult]) -> None:
    total = len(all_results)
    if total == 0:
        print("\n--- Summary ---")
        print("No results to report.")
        print("---------------")
        return

    passed = sum(1 for r in all_results if r.success)
    percent_pass = (passed / total) * 100 if total > 0 else 0.0

    total_time = sum(r.duration for r in all_results)
    avg_time = total_time / total if total > 0 else 0.0

    total_cost = sum(r.cost for r in all_results)
    avg_cost = total_cost / total if total > 0 else 0.0

    print("\n--- Summary ---")
    print(f"Pass Rate: {percent_pass:.2f}% ({passed}/{total})")
    print(f"Average Time: {avg_time:.2f}s")
    print(f"Average Cost: ${avg_cost:.4f}")
    print(f"Total Cost: ${total_cost:.4f}")
    print("---------------")

def save_json_log(
    all_results: List[TaskResult], 
    model_arg: str, 
    dataset_name: str
) -> None:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{timestamp}_{model_arg}_csv_{dataset_name}.json"
    log_path = log_dir / log_filename

    log_data = []
    for r in all_results:
        log_data.append(
            {
                "model": r.model_arg,
                "task": str(r.task_path),
                "test_index": r.test_index,
                "status": "PASS" if r.success else "FAIL",
                "time": r.duration,
                "cost": r.cost,
                "strategy": r.strategy,
            }
        )

    log_path.write_text(json.dumps(log_data, indent=2))
    print(f"Log saved to: {log_path}")