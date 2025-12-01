import concurrent.futures
import multiprocessing
import queue
import sys
import time
from rich.live import Live

from src.dashboard import render_table, update_task_states
from src.execution import execute_task

def run_batch_execution(args, tasks_to_run, run_timestamp, rate_limit_scale, use_dashboard, answers_directory=None):
    final_results = []

    if use_dashboard:
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        task_states = {}
        
        # Start ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.task_workers) as executor:
            future_to_task = {}
            for task_path, test_idx in tasks_to_run:
                answer_path = answers_directory / task_path.name if answers_directory else None
                future = executor.submit(execute_task, args, task_path, test_idx, run_timestamp, rate_limit_scale, progress_queue, True, answer_path)
                future_to_task[future] = (task_path, test_idx)
            
            remaining_futures = set(future_to_task.keys())
            shutdown_start_time = None
            
            with Live(render_table(task_states), refresh_per_second=4) as live:
                while remaining_futures or not progress_queue.empty():
                    # 1. Drain queue non-blocking
                    while True:
                        try:
                            msg = progress_queue.get_nowait()
                            update_task_states(task_states, msg)
                        except queue.Empty:
                            break
                    
                    # 2. Check futures with small timeout
                    if remaining_futures:
                        done, not_done = concurrent.futures.wait(
                            remaining_futures, timeout=0.1, return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        
                        for fut in done:
                            remaining_futures.remove(fut)
                            task_path, test_idx = future_to_task[fut]
                            try:
                                res = fut.result()
                                final_results.append(res)
                            except SystemExit as se:
                                # Worker exited with sys.exit()
                                key = f"{task_path.stem}:{test_idx}"
                                if key not in task_states or task_states[key].get("status") != "COMPLETED":
                                    status = "COMPLETED" if se.code == 0 else "ERROR"
                                    outcome = "?" if se.code == 0 else "FAIL"
                                    step = "Finished (Silent)" if se.code == 0 else f"Exit Code {se.code}"
                                        
                                    update_task_states(task_states, {
                                        "task_id": task_path.stem,
                                        "test_index": test_idx,
                                        "status": status,
                                        "step": step,
                                        "outcome": outcome,
                                        "event": "FINISH",
                                        "timestamp": time.time()
                                    })
                            except Exception as e:
                                key = f"{task_path.stem}:{test_idx}"
                                if key not in task_states or task_states[key].get("status") != "COMPLETED":
                                        update_task_states(task_states, {
                                        "task_id": task_path.stem,
                                        "test_index": test_idx,
                                        "status": "ERROR",
                                        "step": f"Error: {str(e)}",
                                        "outcome": "FAIL",
                                        "event": "FINISH",
                                        "timestamp": time.time()
                                        })

                        # 3. Re-render
                        live.update(render_table(task_states))
                    else:
                        # All futures done. Now ensure we have received final messages for all tasks.
                        if shutdown_start_time is None:
                            shutdown_start_time = time.time()

                        # If queue has data, keep processing
                        if not progress_queue.empty():
                            time.sleep(0.1)
                            live.update(render_table(task_states))
                            continue
                            
                        # Check if any tasks are still technically "running" according to our state
                        all_terminal = True
                        for state in task_states.values():
                            if state.get("status") not in ("COMPLETED", "ERROR"):
                                all_terminal = False
                                break
                        
                        if all_terminal:
                            live.update(render_table(task_states))
                            break
                            
                        # Timeout safety: Don't wait forever if a message was dropped
                        if time.time() - shutdown_start_time > 5.0:
                            break
                            
                        time.sleep(0.1)
                        live.update(render_table(task_states))

    else:
        # Fallback to plain logging (interleaved)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.task_workers) as executor:
            future_to_task = {}
            for task_path, test_idx in tasks_to_run:
                answer_path = answers_directory / task_path.name if answers_directory else None
                future = executor.submit(execute_task, args, task_path, test_idx, run_timestamp, rate_limit_scale, None, False, answer_path)
                future_to_task[future] = (task_path, test_idx)
            
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    res = future.result()
                    final_results.append(res)
                except Exception as e:
                    print(f"Task failed: {e}", file=sys.stderr)
    
    return final_results
