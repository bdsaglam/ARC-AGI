from __future__ import annotations

import datetime
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

from src.config import parse_args, get_api_keys
from src.tasks import load_task_paths
from src.reporting import (
    print_table_header,
    print_result_row,
    print_summary,
    save_json_log,
)
from src.solver import solve_task

def main() -> None:
    args = parse_args()
    openai_key, claude_key, google_key = get_api_keys()

    task_source = args.task_source
    if task_source.endswith(".json"):
        try:
            task_paths = load_task_paths(Path(task_source))
            dataset_name = Path(task_source).stem
        except ValueError as exc:
            raise RuntimeError(f"Failed to read task list {task_source}: {exc}") from exc
    else:
        if os.path.exists(task_source):
            try:
                task_paths = load_task_paths(Path(task_source))
                dataset_name = Path(task_source).stem
            except:
                pass

        # Check if it is a task ID
        candidate_path = Path("data/arc-agi-2-training") / f"{task_source}.json"
        if candidate_path.exists():
            task_paths = [candidate_path]
            dataset_name = task_source
        else:
            raise RuntimeError(
                f"Input '{task_source}' is not a .json file and not a valid Task ID in data/arc-agi-2-training/"
            )

    print_table_header()
    row_counter = 0
    all_results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_path = {}
        for i in range(args.iterations):
            for path in task_paths:
                future = executor.submit(
                    solve_task,
                    openai_key,
                    claude_key,
                    google_key,
                    path,
                    args.model,
                    args.strategy,
                    args.verbose,
                    return_strategy=args.extract_strategy,
                )
                future_to_path[future] = path

        for future in as_completed(future_to_path):
            try:
                task_results = future.result()
                for result in task_results:
                    row_counter += 1
                    print_result_row(row_counter, result)
                all_results.extend(task_results)
            except Exception as exc:
                path = future_to_path[future]
                print(
                    f"Task execution for {path} generated an exception: {exc}",
                    file=sys.stderr,
                )

    if all_results:
        print_summary(all_results)
        save_json_log(all_results, args.model, dataset_name)

if __name__ == "__main__":
    main()