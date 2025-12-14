from datetime import datetime
import json
import sys
from pathlib import Path

def generate_submission(final_results, submission_dir_path: str, run_timestamp: str):
    submission_dir = Path(submission_dir_path)
    submission_dir.mkdir(parents=True, exist_ok=True)
    submission_file = submission_dir / "submission.json"
    
    submission_data = {}
    
    for task_id, test_idx, preds in final_results:
        if not preds:
            continue
        
        # Save individual task/test result
        # individual_file = submission_dir / f"{run_timestamp}_{task_id}_{test_idx}.json"
        # try:
        #     with open(individual_file, "w") as f:
        #         json.dump(preds, f, indent=2)
        # except Exception as e:
        #     print(f"Error saving individual result for {task_id}:{test_idx}: {e}", file=sys.stderr)
        
    # Re-process to format correctly
    formatted_submission = {}
    
    # Group by task_id
    task_results = {}
    for task_id, test_idx, preds in final_results:
            if task_id not in task_results:
                task_results[task_id] = {}
            task_results[task_id][test_idx] = preds

    for task_id, tests in task_results.items():
        # We need to determine the number of tests. 
        # We can assume the max test_idx found is the number of tests? 
        # Or just use the indices we have.
        # If we are running a subset, we can't produce a valid full submission file anyway.
        # So let's just output what we have, sorted by test_index.
        
        max_idx = max(tests.keys())
        formatted_submission[task_id] = []
        task_aggregated_data = []
        
    for task_id, tests in task_results.items():
        # We need to determine the number of tests. 
        # We can assume the max test_idx found is the number of tests? 
        # Or just use the indices we have.
        # If we are running a subset, we can't produce a valid full submission file anyway.
        # So let's just output what we have, sorted by test_index.
        
        max_idx = max(tests.keys())
        formatted_submission[task_id] = []
        task_aggregated_data = []
        
        for i in range(1, max_idx + 1):
            preds_raw = tests.get(i)
            
            solutions = preds_raw
            usage_stats = None
            
            # Unpack if tuple (solutions, usage_stats)
            if isinstance(preds_raw, tuple) and len(preds_raw) == 2 and isinstance(preds_raw[1], dict):
                solutions, usage_stats = preds_raw
            
            attempt_1 = [[0]] # Default empty/fail
            attempt_2 = [[0]]
            correct_1 = False
            correct_2 = False
            reasoning_1 = None
            reasoning_2 = None
            
            if solutions:
                candidates = []
                # solutions is expected to be a list of candidate dicts from pick_solution_v2
                # e.g. [{"grid": ..., "is_correct": ...}, ...]
                if isinstance(solutions, list):
                    for p in solutions:
                        if isinstance(p, dict) and "grid" in p:
                            candidates.append(p)
                # Fallback legacy checks just in case (though we expect list of dicts now)
                elif isinstance(solutions, tuple) and len(solutions) == 2 and isinstance(solutions[0], list):
                        pass
                        
                if candidates:
                    c1 = candidates[0]
                    attempt_1 = c1["grid"]
                    # explicit check for True to handle None/False safely
                    correct_1 = c1.get("is_correct") is True
                    reasoning_1 = c1.get("reasoning_summary")
                    
                    if len(candidates) > 1:
                        c2 = candidates[1]
                        attempt_2 = c2["grid"]
                        correct_2 = c2.get("is_correct") is True
                        reasoning_2 = c2.get("reasoning_summary")
                    else:
                        # duplicate attempt 1 if no attempt 2
                        attempt_2 = attempt_1
                        correct_2 = correct_1
                        reasoning_2 = reasoning_1
            
            formatted_submission[task_id].append({
                "attempt_1": attempt_1,
                "attempt_2": attempt_2
            })
            
            # Format timestamps to ISO 8601
            try:
                # Parse the run_timestamp (filename safe format) back to datetime object
                start_dt = datetime.strptime(run_timestamp, "%Y-%m-%d_%H-%M-%S")
                # Format as ISO 8601 with +00:00 (assuming UTC for simplicity or adding Z)
                # The example requested +00:00
                start_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            except ValueError:
                # Fallback if run_timestamp isn't in expected format
                start_iso = run_timestamp

            # Generate end timestamp in ISO 8601
            end_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+00:00")
            
            if usage_stats:
                def halve(v, is_int=False):
                    if v is None: return None
                    return v // 2 if is_int else v / 2

                usage_data = {
                    "prompt_tokens": halve(usage_stats.get("prompt_tokens"), True),
                    "completion_tokens": halve(usage_stats.get("completion_tokens"), True),
                    "total_tokens": halve(usage_stats.get("total_tokens"), True),
                    "completion_tokens_details": {
                        "reasoning_tokens": halve(usage_stats.get("reasoning_tokens", 0), True),
                        "accepted_prediction_tokens": halve(usage_stats.get("accepted_prediction_tokens"), True),
                        "rejected_prediction_tokens": halve(usage_stats.get("rejected_prediction_tokens", 0), True)
                    }
                }
                cost_data = {
                    "prompt_cost": halve(usage_stats.get("prompt_cost")),
                    "completion_cost": halve(usage_stats.get("completion_cost")),
                    "reasoning_cost": halve(usage_stats.get("reasoning_cost")),
                    "total_cost": halve(usage_stats.get("total_cost"))
                }
            
            metadata_template_1 = {
                "model": "Johan_Land_Solver_V6",
                "provider": "Johan_Land",
                "start_timestamp": start_iso,
                "end_timestamp": end_iso,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "user",
                            "content": "NA"
                        }
                    },
                    {
                        "index": 1,
                        "message": {
                            "role": "assistant",
                            "content": "NA"
                        }
                    }
                ],
                "reasoning_summary": reasoning_1,
                "kwargs": {
                    "background": "mixed",
                    "stream": "mixed",
                    "reasoning": {
                        "effort": "max with fallbacks for latency, cost and stability"
                    },
                    "max_output_tokens": "max less delta for stability"
                },
                "usage": usage_data,
                "cost": cost_data,
                "task_id": task_id,
                "pair_index": i - 1,
                "test_id": "Johan_Land_Solver_V6_Eval_2_Full_Run"
            }
            
            metadata_template_2 = metadata_template_1.copy()
            metadata_template_2["reasoning_summary"] = reasoning_2

            task_aggregated_data.append({
                "attempt_1": {
                    "answer": attempt_1, 
                    "correct": correct_1,
                    "metadata": metadata_template_1
                },
                "attempt_2": {
                    "answer": attempt_2, 
                    "correct": correct_2,
                    "metadata": metadata_template_2
                }
            })

        # Save task-level aggregated file (without timestamp)
        task_file = submission_dir / f"{task_id}.json"
        try:
            with open(task_file, "w") as f:
                json.dump(task_aggregated_data, f, indent=2)
        except Exception as e:
            print(f"Error saving task file {task_file}: {e}", file=sys.stderr)

    with open(submission_file, "w") as f:
        json.dump(formatted_submission, f)
        
    print(f"Submission file saved to: {submission_file}")

    # Generate results.json
    total_score = 0.0
    total_cost = 0.0
    total_attempts = 0
    total_output_tokens = 0
    total_tokens = 0
    total_duration = 0.0
    total_empty_attempts = 0
    
    task_results_map = {}
    
    unique_tasks = sorted(list(task_results.keys()))
    num_tasks = len(unique_tasks)
    
    for task_id in unique_tasks:
        tests = task_results[task_id]
        num_tests = len(tests)
        
        task_solved_tests = 0
        task_cost = 0.0
        task_output_tokens = 0
        task_total_tokens = 0
        task_duration = 0.0
        task_attempts_count = 0
        task_empty_attempts = 0
        
        # We need to iterate through all tests for this task
        # Max index logic was used above: max_idx = max(tests.keys())
        # But 'tests' dictionary only contains entries for tests that were actually run/returned.
        # If we ran a subset, we only count those.
        
        for i, preds_raw in tests.items():
            # Unpack preds
            solutions = preds_raw
            usage_stats = None
            if isinstance(preds_raw, tuple) and len(preds_raw) == 2 and isinstance(preds_raw[1], dict):
                solutions, usage_stats = preds_raw
            
            # Determine correctness and empty attempts
            is_solved = False
            
            # Re-extract attempts to check for equality with []
            # (Logic duplicated from formatted_submission generation for safety/clarity in this block)
            current_attempt_1 = [[0]]
            current_attempt_2 = [[0]]
            
            if solutions and isinstance(solutions, list):
                if len(solutions) > 0:
                    cand1 = solutions[0]
                    if isinstance(cand1, dict) and "grid" in cand1:
                        current_attempt_1 = cand1["grid"]
                        if cand1.get("is_correct") is True:
                            is_solved = True
                
                if len(solutions) > 1:
                    cand2 = solutions[1]
                    if isinstance(cand2, dict) and "grid" in cand2:
                        current_attempt_2 = cand2["grid"]
                        if cand2.get("is_correct") is True:
                            is_solved = True
                else:
                    # If only 1 solution, attempt 2 is copy of attempt 1
                    current_attempt_2 = current_attempt_1

            if is_solved:
                task_solved_tests += 1
            
            # Check for empty list []
            if current_attempt_1 == []:
                task_empty_attempts += 1
            if current_attempt_2 == []:
                task_empty_attempts += 1
            
            task_attempts_count += 2
            
            if usage_stats:
                task_cost += usage_stats.get("total_cost", 0.0) or 0.0
                task_output_tokens += usage_stats.get("completion_tokens", 0) or 0
                task_total_tokens += usage_stats.get("total_tokens", 0) or 0
                task_duration += usage_stats.get("total_duration", 0.0) or 0.0

        task_score = (task_solved_tests / num_tests) if num_tests > 0 else 0.0
        total_score += task_score
        
        total_cost += task_cost
        total_attempts += task_attempts_count
        total_output_tokens += task_output_tokens
        total_tokens += task_total_tokens
        total_duration += task_duration
        total_empty_attempts += task_empty_attempts
        
        task_results_map[task_id] = {
            "score": task_score,
            "cost": task_cost,
            "attempts": task_attempts_count,
            "output_tokens": task_output_tokens,
            "total_tokens": task_total_tokens,
            "duration": task_duration,
            "num_attempts_with_empty_list": task_empty_attempts
        }

    avg_cost_per_task = (total_cost / num_tasks) if num_tasks > 0 else 0.0
    avg_cost_per_attempt = (total_cost / total_attempts) if total_attempts > 0 else 0.0
    avg_output_tokens_per_task = (total_output_tokens / num_tasks) if num_tasks > 0 else 0.0
    avg_total_tokens_per_task = (total_tokens / num_tasks) if num_tasks > 0 else 0.0
    avg_duration_per_task = (total_duration / num_tasks) if num_tasks > 0 else 0.0

    results_data = {
        "score": total_score,
        "total_tasks": num_tasks,
        "total_cost": total_cost,
        "total_attempts": total_attempts,
        "avg_cost_per_task": avg_cost_per_task,
        "avg_cost_per_attempt": avg_cost_per_attempt,
        "avg_output_tokens_per_task": avg_output_tokens_per_task,
        "avg_total_tokens_per_task": avg_total_tokens_per_task,
        "avg_duration_per_task": avg_duration_per_task,
        "task_results": task_results_map,
        "num_attempts_with_empty_list": total_empty_attempts
    }
    
    results_file = submission_dir / "results.json"
    try:
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=4)
        print(f"Results file saved to: {results_file}")
    except Exception as e:
        print(f"Error saving results file: {e}", file=sys.stderr)
