import concurrent.futures
import multiprocessing
import sys
import time

from src.execution import execute_task

def run_batch_execution(args, tasks_to_run, run_timestamp, rate_limit_scale, answers_directory=None):
    final_results = []

    # Standard logging (interleaved)
    manager = multiprocessing.Manager()
    running = manager.Value('i', 0)
    remaining = manager.Value('i', len(tasks_to_run))
    finished = manager.Value('i', 0)
    lock = manager.Lock()
    status_counters = (running, remaining, finished, lock)

    # Print Table Header
    print("Legend: ⚡ Running   ⏳ Queued   ✅ Done")
    print()
    print("| Status        | Task:Test  | Step  | Phase           | Time   | Message")
    print("|---------------|------------|-------|-----------------|--------|------------------------------")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.task_workers) as executor:
        future_to_task = {}
        for task_path, test_idx in tasks_to_run:
            answer_path = answers_directory / task_path.name if answers_directory else None
            future = executor.submit(execute_task, args, task_path, test_idx, run_timestamp, rate_limit_scale, answer_path, status_counters)
            future_to_task[future] = (task_path, test_idx)
        
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                res = future.result()
                final_results.append(res)
            except Exception as e:
                print(f"Task failed: {e}", file=sys.stderr)

    return final_results
