import statistics
import json
try:
    from .utils import normalize_model_name
except ImportError:
    from utils import normalize_model_name

def determine_strategies_status(task_data_entry):
    """
    Determines the solved status for Standard, Vote only, and Score only strategies.
    Returns a dictionary with booleans.
    """
    finish_data = task_data_entry.get("finish_data")
    finish_status_val = task_data_entry.get("finish_status")
    
    # --- Standard Strategy ---
    standard_solved = False
    if finish_status_val:
        if finish_status_val == "PASS" or finish_status_val == "SOLVED":
            standard_solved = True


    # --- Vote only Strategy ---
    vote_only_solved = False
    if finish_data and isinstance(finish_data, dict) and "candidates_object" in finish_data:
        candidates_obj = finish_data["candidates_object"]
        if isinstance(candidates_obj, dict):
            sorted_candidates = sorted(
                [val for val in candidates_obj.values() if isinstance(val, dict)],
                key=lambda x: x.get("count", 0),
                reverse=True
            )
            top_two_candidates = sorted_candidates[:2]
            for candidate in top_two_candidates:
                if candidate.get("is_correct") is True:
                    vote_only_solved = True
                    break

    # --- Score only Strategy ---
    score_only_solved = False
    if finish_data and isinstance(finish_data, dict) and "selection_details" in finish_data:
        sel_details = finish_data["selection_details"]
        if isinstance(sel_details, dict) and "selection_process" in sel_details:
            sel_process = sel_details["selection_process"]
            if isinstance(sel_process, dict) and "candidates_summary" in sel_process:
                candidates_summary = sel_process["candidates_summary"]
                if isinstance(candidates_summary, list):
                    sorted_scored_candidates = sorted(
                        [c for c in candidates_summary if isinstance(c, dict) and "score" in c],
                        key=lambda x: x.get("score", 0),
                        reverse=True
                    )
                    top_two_scored_candidates = sorted_scored_candidates[:2]
                    
                    if finish_data and "candidates_object" in finish_data:
                        all_candidates_obj = finish_data["candidates_object"]
                        all_candidate_keys = list(all_candidates_obj.keys())
                        
                        for scored_candidate in top_two_scored_candidates:
                            cid = scored_candidate.get("id")
                            if cid is not None and isinstance(cid, int):
                                if 0 <= cid < len(all_candidate_keys):
                                    candidate_key = all_candidate_keys[cid]
                                    if all_candidates_obj[candidate_key].get("is_correct") is True:
                                        score_only_solved = True
                                        break
            else:
                 pass # print("DEBUG: No candidates_summary found")
        else:
             pass # print("DEBUG: No selection_process found")
    else:
         pass # print("DEBUG: No selection_details found")
    
    return {
        "standard": standard_solved,
        "vote": vote_only_solved,
        "score": score_only_solved
    }

def calculate_model_stats(task_data):
    """
    Aggregates statistics for all models across all tasks.
    """
    model_stats = {}

    for key in task_data:
        steps = task_data[key]["steps"]
        for step_name, calls in steps.items():
            for call in calls:
                raw_name = call["name"]
                model_name = normalize_model_name(raw_name)

                if model_name not in model_stats:
                    model_stats[model_name] = {
                        "solver_attempts": 0, 
                        "total_calls": 0,
                        "passes": 0,
                        "zero_duration_calls": 0,
                        "durations": [], 
                        "costs": [],
                        "input_tokens": [],
                        "output_tokens": [],
                        "cached_tokens": []
                    }

                model_stats[model_name]["total_calls"] += 1

                status = call["status"]
                if status in ["PASS", "FAIL"]:
                    model_stats[model_name]["solver_attempts"] += 1
                    if status == "PASS":
                        model_stats[model_name]["passes"] += 1
                
                # Collect valid durations and count zero durations
                duration = call.get("duration")
                if isinstance(duration, (int, float)):
                    if duration > 0:
                        model_stats[model_name]["durations"].append(duration)
                    elif duration == 0:
                        model_stats[model_name]["zero_duration_calls"] += 1

                # Collect valid costs
                if isinstance(call.get("cost"), (int, float)):
                     model_stats[model_name]["costs"].append(call["cost"])

                # Collect token stats
                if isinstance(call.get("input_tokens"), (int, float)):
                    model_stats[model_name]["input_tokens"].append(call["input_tokens"])
                
                if isinstance(call.get("output_tokens"), (int, float)):
                    model_stats[model_name]["output_tokens"].append(call["output_tokens"])
                
                if isinstance(call.get("cached_tokens"), (int, float)):
                    model_stats[model_name]["cached_tokens"].append(call["cached_tokens"])
    
    return model_stats

def calculate_percentile(data, percentile=0.95):
    if not data:
        return 0
    sorted_data = sorted(data)
    k = percentile * (len(sorted_data) - 1)
    f = int(k)
    c = int(k) + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    elif f == c:
         return sorted_data[int(k)]
    else:
         return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)
