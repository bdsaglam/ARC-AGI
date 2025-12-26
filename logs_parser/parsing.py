import json

def check_correctness(call_val, task_id, test_id, answers):
    is_correct = call_val.get("is_correct")
    # Fallback to ground truth check ONLY if is_correct is missing
    if is_correct is None and "Extracted grid" in call_val:
        extracted = call_val["Extracted grid"]
        if task_id in answers:
            idx = int(test_id) - 1
            if 0 <= idx < len(answers[task_id]):
                correct_grid = answers[task_id][idx]
                if extracted == correct_grid:
                    is_correct = True
                else:
                    is_correct = False
    return is_correct

def create_call_info(name, data, task_id, test_id, answers, generator=None, run_id=None):
    duration = 0
    cost = 0
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    status_str = ""
    timing_breakdown = []
    
    if isinstance(data, dict):
        duration = data.get("duration_seconds", 0)
        cost = data.get("total_cost", 0)
        input_tokens = data.get("input_tokens", 0)
        output_tokens = data.get("output_tokens", 0)
        cached_tokens = data.get("cached_tokens", 0)
        timing_breakdown = data.get("timing_breakdown", [])
        is_correct = check_correctness(data, task_id, test_id, answers)
        
        if is_correct is True:
            status_str = "PASS"
        elif is_correct is False:
            status_str = "FAIL"
            
    extracted_grid_failed = False
    bad_grid = False
    verification_details = {}
    llm_response = ""
    extracted_grid = None
    if isinstance(data, dict):
        llm_response = data.get("Full raw LLM response", "")
        verification_details = data.get("verification_details", {})
        if "Extracted grid" in data:
            extracted_grid = data["Extracted grid"]
            if extracted_grid is None:
                extracted_grid_failed = True
            elif isinstance(extracted_grid, list):
                height = len(extracted_grid)
                width = 0
                if height > 0 and isinstance(extracted_grid[0], list):
                    width = len(extracted_grid[0])
                
                if height == 1 or width == 1:
                    bad_grid = True

    return {
        "name": name,
        "run_id": run_id if run_id else name,
        "duration": duration,
        "cost": cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "timing_breakdown": timing_breakdown,
        "status": status_str,
        "generator": generator,
        "extracted_grid_failed": extracted_grid_failed,
        "bad_grid": bad_grid,
        "verification_details": verification_details,
        "llm_response": llm_response,
        "extracted_grid": extracted_grid
    }

def parse_finish_step(content):
    result = {
        "finish_data": content,
        "finish_status": None,
        "calls": [],
        "judge_stats": {}
    }
    
    if isinstance(content, dict):
        if "result" in content:
            result["finish_status"] = content["result"]
            
        candidates_obj = content.get("candidates_object", {})
        candidates_keys = list(candidates_obj.keys())

        # Extract judges info
        if "selection_details" in content:
            sel_details = content["selection_details"]
            if isinstance(sel_details, dict) and "judges" in sel_details:
                judges = sel_details["judges"]
                if isinstance(judges, dict):
                    for judge_name, judge_data in judges.items():
                        if not isinstance(judge_data, dict):
                            continue

                        duration = judge_data.get("duration_seconds", 0)
                        cost = judge_data.get("total_cost", 0)
                        input_tokens = judge_data.get("input_tokens", 0)
                        output_tokens = judge_data.get("output_tokens", 0)
                        cached_tokens = judge_data.get("cached_tokens", 0)
                        timing_breakdown = judge_data.get("timing_breakdown", [])
                        model = judge_data.get("model", "")
                        
                        display_name = f"Judge ({judge_name.capitalize()})"
                        if model:
                            display_name += f" - {model}"
                            
                        result["calls"].append({
                            "name": display_name,
                            "duration": duration,
                            "cost": cost,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "cached_tokens": cached_tokens,
                            "timing_breakdown": timing_breakdown,
                            "status": ""
                        })
                        
                        # Extract detailed stats
                        parsed = judge_data.get("parsed", {})
                        if parsed:
                            evaluations = parsed.get("candidates", [])
                            ranking = parsed.get("final_ranking_by_candidate", [])
                            
                            # Fallback: if ranking is missing, sort by score
                            if not ranking and evaluations:
                                sorted_evals = sorted(evaluations, key=lambda x: x.get("score", 0), reverse=True)
                                ranking = [e.get("candidate_id") for e in sorted_evals if "candidate_id" in e]
                            
                            stats_list = []
                            for eval_item in evaluations:
                                cid = eval_item.get("candidate_id")
                                score = eval_item.get("score", 0)
                                tier = eval_item.get("tier", "")
                                
                                if isinstance(cid, int) and 0 <= cid < len(candidates_keys):
                                    cand_key = candidates_keys[cid]
                                    actual_cand = candidates_obj[cand_key]
                                    is_correct = actual_cand.get("is_correct", False)
                                    
                                    # Determine rank
                                    rank = -1
                                    if cid in ranking:
                                        rank = ranking.index(cid) + 1
                                    
                                    stats_list.append({
                                        "is_correct": is_correct,
                                        "score": score,
                                        "rank": rank,
                                        "tier": tier
                                    })
                            
                            result["judge_stats"][judge_name] = {
                                "evaluations": stats_list,
                                "cost": cost,
                                "duration": duration
                            }
    return result

def parse_nested_step(content, task_id, test_id, answers):
    result = {
        "steps": {}, # sub-steps
        "solved": False
    }
    
    if isinstance(content, dict) and content.get("is_solved") is True:
        result["solved"] = True

    # Handle nested structure for step 5
    # content is { "sub-step": { "call": ... }, ... }
    for sub_step, calls_dict in content.items():
        if not isinstance(calls_dict, dict):
            continue 

        new_step_name = f"5-{sub_step}"
        cleaned_calls = []
        
        nested_containers = ["gemini_gen", "opus_gen"]
        
        for call_key, call_val in calls_dict.items():
            if call_key == "hint_generation" and isinstance(call_val, dict):
                # Handle hint_generation (direct call object)
                model = call_val.get("model", "")
                name = "Hint Generation"
                if model:
                    name += f" ({model})"
                
                # Check for cost/duration directly in call_val or if they are missing
                # Some logs might put stats in a wrapper, but schema says it's direct.
                cleaned_calls.append(create_call_info(name, call_val, task_id, test_id, answers, run_id=call_key))
                continue

            if call_key in nested_containers and isinstance(call_val, dict):
                # Determine generator
                generator_name = None
                if call_key == "gemini_gen":
                    generator_name = "Gemini"
                elif call_key == "opus_gen":
                    generator_name = "Opus"

                # Process nested calls
                for inner_call, inner_val in call_val.items():
                    if not isinstance(inner_val, dict):
                        continue
                        
                    if "_step_" in inner_call:
                        cleaned_name = inner_call.split("_step_")[0]
                    else:
                        cleaned_name = inner_call
                    
                    model = inner_val.get("model", "")
                    if model:
                        cleaned_name += f" ({model})"
                    
                    cleaned_calls.append(create_call_info(cleaned_name, inner_val, task_id, test_id, answers, generator=generator_name, run_id=inner_call))
            else:
                # Process normal call
                if "_step_" in call_key:
                    cleaned_name = call_key.split("_step_")[0]
                else:
                    cleaned_name = call_key
                
                # Try to infer generator from call key string for objects_pipeline variants
                gen_name = None
                if "gemini_gen" in call_key:
                    gen_name = "Gemini"
                elif "opus_gen" in call_key:
                    gen_name = "Opus"
                
                cleaned_calls.append(create_call_info(cleaned_name, call_val, task_id, test_id, answers, generator=gen_name, run_id=call_key))
        
        result["steps"][new_step_name] = cleaned_calls
        
    return result

def parse_generic_step(content, task_id, test_id, answers):
    result = {
        "calls": [],
        "solved": False
    }

    if isinstance(content, dict) and content.get("is_solved") is True:
         result["solved"] = True

    if not isinstance(content, dict):
        return result

    for call_key, call_val in content.items():
        if call_key == "is_solved": continue 

        if "_step_" in call_key:
            cleaned_name = call_key.split("_step_")[0]
        else:
            cleaned_name = call_key
        
        result["calls"].append(create_call_info(cleaned_name, call_val, task_id, test_id, answers, run_id=call_key))
        
    return result

def parse_log_file(filepath, task_id, test_id, step_name, answers):
    """
    Parses a single log file and returns a structured dictionary of results.
    """
    try:
        with open(filepath, 'r') as f:
            content = json.load(f)
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return None

    if step_name == "finish":
        return {"type": "finish", "data": parse_finish_step(content)}
    elif step_name == "5":
        return {"type": "nested", "data": parse_nested_step(content, task_id, test_id, answers)}
    else:
        return {"type": "generic", "data": parse_generic_step(content, task_id, test_id, answers)}