import argparse
import os
import re
import json

def load_answers(base_dir):
    answers = {}
    # Try looking for answers/ in the current working directory first
    answers_dir = os.path.join(base_dir, "answers")
    if not os.path.isdir(answers_dir):
        # Fallback: look in the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        answers_dir = os.path.join(script_dir, "answers")
        if not os.path.isdir(answers_dir):
            return answers

    for filename in os.listdir(answers_dir):
        if filename.endswith(".json"):
            task_id = filename.split(".")[0]
            filepath = os.path.join(answers_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if "test" in data:
                        answers[task_id] = [t.get("output") for t in data["test"]]
            except Exception as e:
                print(f"Warning: Could not read answer file {filename}: {e}")
    return answers

def parse_logs(directory):
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Load answers relative to current working directory
    answers = load_answers(os.getcwd())

    files = os.listdir(directory)
    # Regex to capture task_id, test_id, and step from .json files
    # Matches: ...<task>_<test>_step_<step_name>.json OR ...<task>_<test>_step_finish.json
    pattern = re.compile(r'([a-f0-9]{8})_(\d+)_step_([a-zA-Z0-9]+)\.json$')

    # Structure: {(task, test): {"steps": {step_name: [calls]}, "finish_data": {...}, "model_statuses": {}, "step_statuses": {}}}
    task_data = {}

    for filename in files:
        match = pattern.search(filename)
        if match:
            task_id = match.group(1)
            test_id = match.group(2)
            step_name = match.group(3) # '1', '3', '5', 'finish' etc.
            
            # Skip steps 2 and 4
            if step_name in ["2", "4"]:
                continue

            key = (task_id, int(test_id))
            if key not in task_data:
                task_data[key] = {"steps": {}, "finish_data": None, "model_statuses": {}, "step_statuses": {}}
            
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r') as f:
                    content = json.load(f)
                    
                    if step_name == "finish":
                        task_data[key]["finish_data"] = content
                        
                        if isinstance(content, dict):
                            # Check for root-level result
                            if "result" in content:
                                task_data[key]["finish_status"] = content["result"]
                        
                        # Extract judges info for display
                        cleaned_calls = []
                        if isinstance(content, dict) and "selection_details" in content:
                            sel_details = content["selection_details"]
                            if isinstance(sel_details, dict) and "judges" in sel_details:
                                judges = sel_details["judges"]
                                if isinstance(judges, dict):
                                    for judge_name, judge_data in judges.items():
                                        if isinstance(judge_data, dict):
                                            duration = judge_data.get("duration_seconds", 0)
                                            cost = judge_data.get("total_cost", 0)
                                            model = judge_data.get("model", "")
                                            
                                            display_name = f"Judge ({judge_name.capitalize()})"
                                            if model:
                                                display_name += f" - {model}"
                                                
                                            cleaned_calls.append({
                                                "name": display_name,
                                                "duration": duration,
                                                "cost": cost,
                                                "status": ""
                                            })
                        
                        task_data[key]["steps"]["finish"] = cleaned_calls
                        continue # Don't process finish as a regular step
                    
                    if step_name == "5":
                        if isinstance(content, dict) and content.get("is_solved") is True:
                             task_data[key]["step_statuses"]["5"] = True

                        # Handle nested structure for step 5
                        # content is { "sub-step": { "call": ... }, ... }
                        for sub_step, calls_dict in content.items():
                            if not isinstance(calls_dict, dict):
                                continue # Skip non-dict sub-step data

                            new_step_name = f"5-{sub_step}"
                            cleaned_calls = []
                            
                            # Define keys that act as nested containers
                            nested_containers = ["hint_generation", "gemini_gen", "opus_gen"]
                            
                            for call_key, call_val in calls_dict.items():
                                if call_key in nested_containers and isinstance(call_val, dict):
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
                                        
                                        duration = inner_val.get("duration_seconds", 0)
                                        cost = inner_val.get("total_cost", 0)
                                        
                                        is_correct = inner_val.get("is_correct")
                                        
                                        # Fallback to ground truth check ONLY if is_correct is missing
                                        if is_correct is None and "Extracted grid" in inner_val:
                                            extracted = inner_val["Extracted grid"]
                                            if task_id in answers:
                                                idx = int(test_id) - 1
                                                if 0 <= idx < len(answers[task_id]):
                                                    correct_grid = answers[task_id][idx]
                                                    if extracted == correct_grid:
                                                        is_correct = True
                                                    else:
                                                        is_correct = False

                                        status_str = ""
                                        if is_correct is True:
                                            status_str = "PASS"
                                        elif is_correct is False:
                                            status_str = "FAIL"
                                        
                                        cleaned_calls.append({
                                            "name": cleaned_name,
                                            "duration": duration,
                                            "cost": cost,
                                            "status": status_str
                                        })
                                else:
                                    # Process normal call
                                    if "_step_" in call_key:
                                        cleaned_name = call_key.split("_step_")[0]
                                    else:
                                        cleaned_name = call_key
                                    
                                    if isinstance(call_val, dict):
                                        duration = call_val.get("duration_seconds", 0)
                                        cost = call_val.get("total_cost", 0)
                                        
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

                                        status_str = ""
                                        if is_correct is True:
                                            status_str = "PASS"
                                        elif is_correct is False:
                                            status_str = "FAIL"
                                    else:
                                        duration = 0
                                        cost = 0
                                        status_str = ""
                                    
                                    cleaned_calls.append({
                                        "name": cleaned_name,
                                        "duration": duration,
                                        "cost": cost,
                                        "status": status_str
                                    })
                            
                            task_data[key]["steps"][new_step_name] = cleaned_calls
                    else:
                        # Handle flat structure for other steps (1, 3, etc.)
                        if isinstance(content, dict) and content.get("is_solved") is True:
                             task_data[key]["step_statuses"][step_name] = True

                        cleaned_calls = []
                        for call_key, call_val in content.items():
                            if call_key == "is_solved": continue # Skip the status key itself

                            if "_step_" in call_key:
                                cleaned_name = call_key.split("_step_")[0]
                            else:
                                cleaned_name = call_key
                            
                            if isinstance(call_val, dict): # Defensive check
                                duration = call_val.get("duration_seconds", 0)
                                cost = call_val.get("total_cost", 0)
                                
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

                                status_str = ""
                                if is_correct is True:
                                    status_str = "PASS"
                                elif is_correct is False:
                                    status_str = "FAIL"
                            else: # Not a dict, set defaults
                                duration = 0
                                cost = 0
                                status_str = ""
                            
                            cleaned_calls.append({
                                "name": cleaned_name,
                                "duration": duration,
                                "cost": cost,
                                "status": status_str
                            })
                        
                        task_data[key]["steps"][step_name] = cleaned_calls

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {filename}")
            except Exception as e:
                print(f"Warning: Could not read {filename}: {e}")

    # Sort pairs: task (string), test (int)
    sorted_keys = sorted(task_data.keys(), key=lambda x: (x[0], x[1]))

    for task, test in sorted_keys:
        status = "FAILED" # Default status
        finish_data = task_data[(task, test)]["finish_data"]
        
        # Check explicit result from finish file first
        if "finish_status" in task_data[(task, test)]:
             status_val = task_data[(task, test)]["finish_status"]
             if status_val == "PASS":
                 status = "SOLVED"
             elif status_val == "FAIL":
                 status = "FAILED"
             else:
                 status = status_val # Use raw value if unknown
        # Fallback to candidates check
        elif finish_data and isinstance(finish_data, dict) and "candidates_object" in finish_data:
            candidates_obj = finish_data["candidates_object"]
            if isinstance(candidates_obj, dict):
                for candidate_key, candidate_val in candidates_obj.items():
                    if isinstance(candidate_val, dict) and candidate_val.get("is_correct") is True:
                        status = "SOLVED"
                        break
        
        print(f"{task}:{test} {status}")
        
        steps_dict = task_data[(task, test)]["steps"]
        step_statuses = task_data[(task, test)]["step_statuses"]
        
        # Sort steps: integers first, then strings (like 'finish')
        sorted_steps = sorted(steps_dict.keys(), key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
        
        for step in sorted_steps:
            step_solved_mark = ""
            # Check if this step (or the parent step for substeps) is marked as solved
            # For 5-image, 5-hint etc, we look for "5" in step_statuses
            lookup_step = step
            if "-" in step:
                lookup_step = step.split("-")[0]
            
            if step_statuses.get(lookup_step) is True:
                step_solved_mark = " [SOLVED]"

            print(f"  {step}{step_solved_mark}")
            for call_info in steps_dict[step]:
                name = call_info["name"]
                duration = call_info["duration"]
                cost = call_info["cost"]
                status_val = call_info["status"]
                
                print(f"    {name:<50}\t{duration:6.2f}s\t${cost:7.4f}\t{status_val}")

def main():
    parser = argparse.ArgumentParser(description="Parse log files to extract task and test IDs.")
    parser.add_argument("directory", help="Path to the logs directory")
    args = parser.parse_args()

    parse_logs(args.directory)

if __name__ == "__main__":
    main()