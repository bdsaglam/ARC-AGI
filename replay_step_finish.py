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

def find_reasoning_for_candidate(candidate_run_ids, run_logs, verbose=False):
    """
    Searches through loaded run logs (step 1 and step 5) to find the reasoning
    associated with any of the run_ids that produced this candidate.
    Handles both flat (Step 1) and nested (Step 5) log structures.
    """
    reasoning_map = {}
    
    def search_in_dict(d, target_id):
        # 1. Direct match (fast path)
        if target_id in d:
            return d[target_id]
        
        # 2. Prefix match (for timestamped keys)
        # We look for key that starts with target_id + "_"
        prefix = target_id + "_"
        for k, v in d.items():
            if k.startswith(prefix):
                return v
            
            # 3. Nested search
            if isinstance(v, dict):
                # Check nested keys
                if target_id in v:
                    return v[target_id]
                
                # Check nested keys for prefix match
                for sub_k, sub_v in v.items():
                    if sub_k.startswith(prefix):
                        return sub_v
                    if isinstance(sub_v, dict) and target_id in sub_v: # Handle deeper nesting if needed
                         return sub_v[target_id]
        return None

    if verbose:
        print(f"DEBUG: Looking for run_ids: {candidate_run_ids}")
        print(f"DEBUG: Loaded {len(run_logs)} log files.")

    for run_id in candidate_run_ids:
        found_entry = None
        for log_data in run_logs:
            found_entry = search_in_dict(log_data, run_id)
            if found_entry:
                break
        
        if found_entry:
            if "Full raw LLM response" in found_entry:
                reasoning_map[run_id] = found_entry["Full raw LLM response"]
            elif "detailed_logs" in found_entry and found_entry["detailed_logs"]:
                for log_item in found_entry["detailed_logs"]:
                    if log_item.get("type") == "text":
                        reasoning_map[run_id] = log_item.get("content", "")
                        break
        elif verbose:
            print(f"DEBUG: Could not find entry for {run_id}")
    
    return reasoning_map

def resolve_task_path(task_id: str) -> Path:
    # Known data directories
    data_dirs = [
        Path("data/evaluation-arc-agi-1"),
        Path("data/evaluation-arc-agi-2"),
        Path("data/training-arc-agi-2"),
        # Fallback to what src/run_utils might have expected if different
        Path("data/arc-agi-2-evaluation"),
        Path("data/arc-agi-2-training"),
    ]
    
    for d in data_dirs:
        if d.exists():
            candidate = d / f"{task_id}.json"
            if candidate.exists():
                return candidate
                
    # If not found, try src.run_utils.find_task_path as a last resort
    # but wrap it in try/except because it might fail or raise error
    try:
        from src.run_utils import find_task_path
        return find_task_path(task_id)
    except Exception:
        pass
        
    return None

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
    print("-" * 90)
    print(f"{'Task ID':<12} | {'Test':<5} | {'Candidate #1':<35} | {'Candidate #2':<35}")
    print("-" * 90)

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
            reasoning = find_reasoning_for_candidate(models, run_logs, verbose=args.verbose)
            
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
            task_path = resolve_task_path(task_id)
            if not task_path:
                print(f"Error: Could not find task file for {task_id}")
                continue
                
            task = load_task(task_path)
            test_input = task.test[test_index-1].input
            train_examples = task.train
            ground_truth = task.test[test_index-1].output
        except Exception as e:
            print(f"Error loading task {task_id}: {e}")
            continue

        # 4. Judge Execution
        top_candidates = [] # Format: { "score_label": str, "is_correct": bool, "grid": ... }
        selection_metadata = {"judges": {}, "selection_process": {}}
        
        if args.judge == "vote":
            # Sort by count descending
            candidates_list.sort(key=lambda x: x["count"], reverse=True)
            picks = candidates_list[:2]
            
            selection_metadata["selection_process"] = {"type": "Vote Replay"}
            for i, c in enumerate(picks):
                top_candidates.append({
                    "score_label": f"{c['count']} votes",
                    "is_correct": c.get("is_correct"), # Will verify later if None
                    "grid": c["grid"]
                })
                selection_metadata["selection_process"][f"attempt_{i+1}"] = {"votes": c['count']}
            
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
                use_background=False
            )
            
            selection_metadata["judges"]["duo_pick"] = duo_data
            selection_metadata["selection_process"] = {"type": "Duo Pick Judge Replay"}

            if res_grids:
                for i, grid in enumerate(res_grids):
                    # Match back to candidate
                    grid_tuple = tuple(tuple(r) for r in grid)
                    found_cand = None
                    for c in candidates_list:
                        cand_tuple = tuple(tuple(r) for r in c["grid"])
                        if cand_tuple == grid_tuple:
                            found_cand = c
                            break
                    
                    if found_cand:
                        top_candidates.append({
                            "score_label": f"Existing ({found_cand['count']} votes)",
                            "is_correct": found_cand.get("is_correct"),
                            "grid": found_cand["grid"]
                        })
                    else:
                        # New grid generated by Duo
                        top_candidates.append({
                            "score_label": "Generated",
                            "is_correct": None, # Verify later
                            "grid": grid
                        })
                    selection_metadata["selection_process"][f"attempt_{i+1}"] = f"Judge Pick {i+1}"
            else:
                 # Failed
                 selection_metadata["selection_process"]["error"] = "Duo Pick Judge Failed"

        elif args.judge in ["logic", "consistency"]:
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
                use_background=False
            )
            
            selection_metadata["judges"][args.judge] = judge_data
            selection_metadata["selection_process"] = {"type": f"{judge_type} Judge Replay"}

            # Parse scores
            scores = {} # map id -> score
            if res and "candidates" in res:
                for c_res in res["candidates"]:
                    cid = c_res.get("candidate_id")
                    scores[cid] = c_res.get("score", 0)
            
            # Sort by score descending
            for c in candidates_list:
                c["judge_score"] = scores.get(c["id"], 0)
                
            candidates_list.sort(key=lambda x: x["judge_score"], reverse=True)
            picks = candidates_list[:2]
            
            for i, c in enumerate(picks):
                 top_candidates.append({
                    "score_label": f"Score: {c['judge_score']}",
                    "is_correct": c.get("is_correct"),
                    "grid": c["grid"]
                })
                 selection_metadata["selection_process"][f"attempt_{i+1}"] = {"score": c['judge_score']}

        # 5. Result Formatting
        
        # Verify correctness if needed
        is_any_correct = False
        
        cand_1_str = "-"
        cand_2_str = "-"
        
        # Helper to format candidate string
        def format_cand(cand_data):
            label = cand_data["score_label"]
            correct = cand_data.get("is_correct")
            if correct is None:
                if ground_truth is not None:
                     correct = (cand_data["grid"] == ground_truth)
                     cand_data["is_correct"] = correct
            
            icon = "✅" if correct else "❌"
            return f"{icon} {label}"

        if len(top_candidates) >= 1:
            cand_1_str = format_cand(top_candidates[0])
            if top_candidates[0].get("is_correct"): is_any_correct = True
            
        if len(top_candidates) >= 2:
            cand_2_str = format_cand(top_candidates[1])
            if top_candidates[1].get("is_correct"): is_any_correct = True

        total_tasks += 1
        if is_any_correct:
            total_correct += 1
            
        print(f"{task_id:<12} | {test_index:<5} | {cand_1_str:<35} | {cand_2_str:<35}")

        # 6. Output Replay Log
        output_dir = Path("tmp_replay")
        output_dir.mkdir(exist_ok=True)
        
        outcome = "PASS" if is_any_correct else "FAIL"
        if ground_truth is None:
            outcome = "SUBMITTED"
            
        finish_log = {
            "candidates_object": candidates_object,
            "selection_details": selection_metadata,
            "picked_solutions": [c["grid"] for c in top_candidates],
            "correct_solution": ground_truth,
            "result": outcome
        }
        
        output_filename = f"{timestamp_str}_{task_id}_{test_index}_step_finish.json"
        output_path = output_dir / output_filename
        with open(output_path, "w") as f:
            json.dump(finish_log, f, indent=4, default=lambda o: '<not serializable>')

    print("-" * 90)
    if total_tasks > 0:
        acc = (total_correct / total_tasks) * 100
        print(f"Summary: {total_correct}/{total_tasks} Correct ({acc:.2f}%)")
    else:
        print("No tasks processed.")

if __name__ == "__main__":
    main()
