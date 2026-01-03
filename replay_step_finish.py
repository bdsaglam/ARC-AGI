#!/usr/bin/env python3
import sys
import os
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add current directory to sys.path to ensure we can import from src
sys.path.append(os.getcwd())

from src.tasks import load_task
from src.audit_prompts import build_logic_prompt, build_consistency_prompt, build_duo_pick_prompt
from src.judges import run_judge, run_duo_pick_judge
from src.config import get_api_keys, get_http_client
from openai import OpenAI
from anthropic import Anthropic

def load_json_file(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}", file=sys.stderr)
        return None

def find_reasoning_for_candidate(candidate_run_ids, run_logs):
    """
    Searches through loaded run logs (step 1 and step 5) to find the reasoning
    associated with any of the run_ids that produced this candidate.
    """
    reasoning_map = {}
    
    for run_id in candidate_run_ids:
        # Search in all loaded log dictionaries
        for log_data in run_logs:
            if run_id in log_data:
                entry = log_data[run_id]
                # Try to get the full response
                if "Full raw LLM response" in entry:
                    reasoning_map[run_id] = entry["Full raw LLM response"]
                elif "detailed_logs" in entry and entry["detailed_logs"]:
                    # Fallback for deep thinking or other structured logs
                    for log_item in entry["detailed_logs"]:
                        if log_item.get("type") == "text":
                            reasoning_map[run_id] = log_item.get("content", "")
                            break
    return reasoning_map

def main():
    parser = argparse.ArgumentParser(description="Replay Step Finish Judging Logic")
    parser.add_argument("--logs-dir", required=True, help="Directory containing log files")
    parser.add_argument("--judge", required=True, choices=["vote", "logic", "consistency", "duo"], help="Judge strategy to replay")
    parser.add_argument("--model", default="gpt-5.2-low", help="Model to use for the judge (ignored for 'vote')")
    parser.add_argument("--task-test-selection", help="Comma-separated list of TaskID:TestIndex pairs (e.g. '4e34c42c:2,88e364bc:1')")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"Error: Directory '{logs_dir}' does not exist.")
        sys.exit(1)

    # Parse task selection if provided
    selected_tasks = None
    if args.task_test_selection:
        try:
            selected_tasks = set()
            for item in args.task_test_selection.split(","):
                parts = item.strip().split(":")
                if len(parts) != 2:
                    raise ValueError(f"Invalid format: {item}")
                selected_tasks.add((parts[0], int(parts[1])))
            print(f"Filtering for {len(selected_tasks)} specific task:test pairs.")
        except Exception as e:
            print(f"Error parsing --task-test-selection: {e}", file=sys.stderr)
            sys.exit(1)

    # 1. Setup API Clients
    openai_key, claude_key, google_keys = get_api_keys()
    http_client = get_http_client()
    openai_client = OpenAI(api_key=openai_key, http_client=http_client) if openai_key else None
    anthropic_client = Anthropic(api_key=claude_key, http_client=http_client) if claude_key else None
    
    # 2. Discovery
    finish_files = sorted(list(logs_dir.glob("*_step_finish.json")))
    if not finish_files:
        print(f"No step_finish.json files found in {logs_dir}")
        sys.exit(0)
        
    print(f"Found {len(finish_files)} total log files.")
    print(f"Replaying with Judge: {args.judge.upper()} (Model: {args.model})")
    print("-" * 80)
    print(f"{'Task ID':<15} | {'Test':<5} | {'Correct?':<10} | {'Score/Votes':<12} | {'Notes'}")
    print("-" * 80)

    total_tasks = 0
    total_correct = 0
    
    for finish_file in finish_files:
        # Parse filename: {timestamp}_{task_id}_{test_index}_step_finish.json
        parts = finish_file.name.split('_')
        # Handle variable length timestamp parts (date_time)
        # We know it ends with ..._{task_id}_{test_index}_step_finish.json
        # The timestamp is everything before the task_id
        
        try:
            # Assuming standard format: YYYY-MM-DD_HH-MM-SS_taskID_testIdx_step_finish.json
            # split by _ gives: [YYYY-MM-DD, HH-MM-SS, taskID, testIdx, step, finish.json]
            # so task_id is at index -4, test_index at -3
            if len(parts) < 4:
                continue
                
            task_id = parts[-4]
            test_index = int(parts[-3])
            timestamp_str = "_".join(parts[:-4]) 
            
            # Apply Filter
            if selected_tasks and (task_id, test_index) not in selected_tasks:
                continue
            
        except Exception as e:
            if args.verbose:
                print(f"Skipping malformed filename {finish_file.name}: {e}")
            continue

        # Load finish data
        finish_data = load_json_file(finish_file)
        if not finish_data:
            continue
            
        candidates_object = finish_data.get("candidates_object", {})
        if not candidates_object:
            if args.verbose:
                print(f"No candidates for {task_id}")
            continue
            
        # 3. Data Reconstruction (Trace Gathering)
        # Load associated step logs
        run_logs = []
        possible_steps = ["step_1", "step_5"]
        
        for step in possible_steps:
            log_pattern = f"{timestamp_str}_{task_id}_{test_index}_{step}.json"
            log_path = logs_dir / log_pattern
            if log_path.exists():
                data = load_json_file(log_path)
                if data:
                    run_logs.append(data)
        
        # Reconstruct Candidates List
        candidates_list = []
        # The candidates_object in json has keys as stringified tuples, we need to parse them back or just use the values
        # The values contain "grid", "models", "count"
        
        for idx, (grid_str, val) in enumerate(candidates_object.items()):
            # Parse grid from value, don't rely on key string parsing
            grid = val.get("grid")
            if grid is None: 
                continue
                
            models = val.get("models", [])
            
            # Find Reasoning
            reasoning = find_reasoning_for_candidate(models, run_logs)
            
            candidates_list.append({
                "id": idx,
                "grid": grid,
                "models": models,
                "count": val.get("count", 0),
                "is_correct": val.get("is_correct"),
                "reasoning": reasoning
            })

        if not candidates_list:
            continue
            
        # Load Actual Task for Prompt Building / Verification
        try:
            task = load_task(task_id)
            test_input = task.test[test_index-1].input
            train_examples = task.train
            ground_truth = task.test[test_index-1].output
        except Exception as e:
            print(f"Error loading task {task_id}: {e}")
            continue

        # 4. Judge Execution
        winner = None
        score_info = "N/A"
        
        if args.judge == "vote":
            # Sort by count descending
            candidates_list.sort(key=lambda x: x["count"], reverse=True)
            winner = candidates_list[0]
            score_info = f"{winner['count']} votes"
            
        elif args.judge == "duo":
            # Reconstruct reasoning_store for build_duo_pick_prompt
            reasoning_store = {}
            for c in candidates_list:
                reasoning_store.update(c["reasoning"])
            
            # Use total attempts from candidates counts
            total_attempts = sum(c["count"] for c in candidates_list)
            
            prompt = build_duo_pick_prompt(train_examples, test_input, candidates_list, reasoning_store, total_attempts)
            
            # Hack: Create a dummy duo_data dict to capture results
            duo_data = {"prompt": prompt}
            
            # Run Judge
            res_grids = run_duo_pick_judge(
                prompt, 
                args.model, 
                openai_client, 
                anthropic_client, 
                google_keys, 
                duo_data, 
                args.verbose, 
                openai_background=False
            )
            
            if res_grids and len(res_grids) > 0:
                # The judge returns GRIDS. We need to match them back to candidates to verify correctness
                # taking the first one as the "Winner"
                winner_grid = res_grids[0]
                winner_tuple = tuple(tuple(r) for r in winner_grid)
                
                # Find matching candidate
                for c in candidates_list:
                    cand_tuple = tuple(tuple(r) for r in c["grid"])
                    if cand_tuple == winner_tuple:
                        winner = c
                        break
                
                if not winner:
                    # Judge generated a NEW grid? Treat as new candidate
                    winner = {"grid": winner_grid, "is_correct": None}
                    # Check correctness manually against ground truth
                    if ground_truth is not None:
                         winner["is_correct"] = (winner_grid == ground_truth)
                         
                score_info = "Duo Pick"
            else:
                 score_info = "Failed"

        elif args.judge in ["logic", "consistency"]:
            # Prepare candidates for judging (filter if needed, here we take all unique)
            # Reconstruct reasoning_store for individual items? 
            # No, build_logic_prompt takes candidates_list and extracts reasoning from it if structured right?
            # Actually build_logic_prompt expects candidates list where each item has 'reasoning' dict. 
            # We built that in step 3.
            
            prompt = None
            judge_type = ""
            
            if args.judge == "logic":
                prompt = build_logic_prompt(train_examples, test_input, candidates_list)
                judge_type = "Logic"
            else:
                prompt = build_consistency_prompt(train_examples, test_input, candidates_list)
                judge_type = "Consistency"
                
            judge_data = {"prompt": prompt}
            
            res = run_judge(
                judge_type,
                prompt,
                args.model,
                openai_client,
                anthropic_client,
                google_keys,
                judge_data,
                args.verbose,
                openai_background=False
            )
            
            # Parse scores
            best_score = -1
            best_cand_id = None
            
            if res and "candidates" in res:
                for c_res in res["candidates"]:
                    cid = c_res.get("candidate_id")
                    score = c_res.get("score", 0)
                    if score > best_score:
                        best_score = score
                        best_cand_id = cid
            
            if best_cand_id is not None:
                # Find candidate with this ID
                for c in candidates_list:
                    if c["id"] == best_cand_id:
                        winner = c
                        break
                score_info = f"Score: {best_score}"
            else:
                score_info = "No valid scores"

        # 5. Result
        is_correct = False
        notes = ""
        
        if winner:
            # If we haven't verified correctness yet (e.g. Duo generated new grid)
            if winner.get("is_correct") is None:
                 if ground_truth is not None:
                     winner["is_correct"] = (winner["grid"] == ground_truth)
            
            is_correct = winner.get("is_correct", False)
        else:
            notes = "No winner selected"
            
        total_tasks += 1
        if is_correct:
            total_correct += 1
            
        status_icon = "✅" if is_correct else "❌"
        
        print(f"{task_id:<15} | {test_index:<5} | {status_icon:<10} | {score_info:<12} | {notes}")

    print("-" * 80)
    if total_tasks > 0:
        acc = (total_correct / total_tasks) * 100
        print(f"Summary: {total_correct}/{total_tasks} Correct ({acc:.2f}%)")
    else:
        print("No tasks processed.")

if __name__ == "__main__":
    main()
