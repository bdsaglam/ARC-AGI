import argparse
import os
import re
import json
import sys

try:
    from .utils import load_answers, normalize_model_name
    from .parsing import parse_log_file
    from .stats import calculate_model_stats, calculate_timing_stats_v2
    from .reporting import print_full_report
except ImportError:
    # Fallback for running as script directly
    from utils import load_answers, normalize_model_name
    from parsing import parse_log_file
    from stats import calculate_model_stats, calculate_timing_stats_v2
    from reporting import print_full_report

def extract_code_from_llm_response(llm_code: str) -> str | None:
    """
    Extracts Python code from LLM response.
    Handles Markdown blocks and JSON tool-call formats.
    """
    if not llm_code:
        return None

    # Stage 0: Attempt to parse as JSON first (handling tool calls)
    try:
        if llm_code.strip().startswith("{") and llm_code.strip().endswith("}"):
            data = json.loads(llm_code)
            if isinstance(data, dict):
                for key in ["python", "code", "solution", "content"]:
                    if key in data and isinstance(data[key], str) and "def solver" in data[key]:
                        # Recursive call to handle potential markdown inside the JSON field
                        return extract_code_from_llm_response(data[key])
    except json.JSONDecodeError:
        pass

    code = llm_code
    code_search_area = None
    
    # Stage 1: Explicit marker search
    if "### FINAL SOLUTION ###" in llm_code:
        parts = llm_code.split("### FINAL SOLUTION ###")
        # Take the part after the marker
        code_search_area = parts[-1]
    
    # Stage 2: Fallback - Search all markdown blocks in reverse for 'def solver'
    pattern = r"```python(.*?)```"
    if not code_search_area:
        blocks = re.findall(pattern, llm_code, re.DOTALL)
        for block in reversed(blocks):
            if "def solver" in block:
                code = block.strip()
                code_search_area = "FOUND_IN_BLOCK"
                break
    
    # Stage 3: If we have a search area (from marker or default), extract the block
    if code_search_area and code_search_area != "FOUND_IN_BLOCK":
        match = re.search(pattern, code_search_area, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            # Heuristic: If marker exists but no markdown after it, or no marker and no markdown found
            if "def solver" in code_search_area:
                lines = code_search_area.splitlines()
                start_idx = -1
                for i, line in enumerate(lines):
                    if "def solver" in line:
                        start_idx = i
                        break
                if start_idx != -1:
                    code = "\n".join(lines[start_idx:])
    
    # Stage 4: Ultimate fallback - search entire raw response if nothing found yet
    if not code_search_area:
        if "def solver" in llm_code:
            lines = llm_code.splitlines()
            start_idx = -1
            for i, line in enumerate(lines):
                if "def solver" in line:
                    start_idx = i
                    break
            if start_idx != -1:
                code = "\n".join(lines[start_idx:])
                
    return code if code and "def solver" in code else None

def parse_logs(directory, codegen_analysis=None, all_analysis=None):
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Filter variables
    target_task = None
    target_test = None
    analysis_mode = codegen_analysis or all_analysis
    if analysis_mode:
        try:
            target_task, target_test = analysis_mode.split(':')
        except ValueError:
            print(f"Error: Format should be task_id:test_id")
            return

    # Load answers relative to current working directory
    answers = load_answers(os.getcwd())

    files = os.listdir(directory)
    # Regex to capture task_id, test_id, and step from .json files
    pattern = re.compile(r'([a-f0-9]{8})_(\d+)_step_([a-zA-Z0-9]+)\.json$')

    # Structure: {(task, test): {"steps": {}, "finish_data": None, "finish_status": None, "step_statuses": {}}}
    task_data = {}

    for filename in files:
        match = pattern.search(filename)
        if match:
            task_id = match.group(1)
            test_id_str = match.group(2)
            step_name = match.group(3) # '1', '3', '5', 'finish' etc.
            
            # Filter if analysis requested
            if target_task and (task_id != target_task or test_id_str != target_test):
                continue

            # Skip steps 2 and 4 as per original logic
            if step_name in ["2", "4"]:
                continue

            key = (task_id, int(test_id_str))
            if key not in task_data:
                task_data[key] = {
                    "steps": {},
                    "finish_data": None,
                    "finish_status": None,
                    "step_statuses": {}
                }
            
            filepath = os.path.join(directory, filename)
            
            result = parse_log_file(filepath, task_id, test_id_str, step_name, answers)
            if not result:
                continue

            res_type = result["type"]
            data = result["data"]

            if res_type == "finish":
                task_data[key]["finish_data"] = data["finish_data"]
                if "judge_stats" in data:
                    if isinstance(task_data[key]["finish_data"], dict):
                        task_data[key]["finish_data"]["judge_stats"] = data["judge_stats"]
                
                task_data[key]["finish_status"] = data["finish_status"]
                # Finish step might have "calls" (judges) to be displayed
                if data["calls"]:
                    task_data[key]["steps"]["finish"] = data["calls"]
            
            elif res_type == "nested":
                # Step 5
                if data["solved"]:
                    task_data[key]["step_statuses"][step_name] = True
                
                # Merge sub-steps
                for sub_step_name, calls in data["steps"].items():
                    task_data[key]["steps"][sub_step_name] = calls
            
            elif res_type == "generic":
                # Step 1, 3 etc.
                if data["solved"]:
                    task_data[key]["step_statuses"][step_name] = True
                
                task_data[key]["steps"][step_name] = data["calls"]

    # If Analysis requested, perform it and exit
    if analysis_mode:
        key = (target_task, int(target_test))
        if key in task_data:
            count = 0
            data = task_data[key]
            
            # Variables for prompt.txt generation
            solutions = []
            original_prompt = None
            total_runs = 0
            
            # Sort steps to ensure deterministic order
            sorted_steps = sorted(data.get("steps", {}).keys(), key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
            
            for step_name in sorted_steps:
                calls = data["steps"][step_name]
                for call in calls:
                    name = call.get("name", "")
                    
                    # Ignore Judge calls
                    if "judge" in name.lower():
                        continue
                        
                    is_codegen = "codegen" in name.lower()
                    
                    # Counting and filtering logic
                    if codegen_analysis and not is_codegen:
                        continue
                    
                    total_runs += 1
                    
                    verification_status = "-"
                    details = call.get("verification_details")
                    if details and isinstance(details, dict):
                        verification_status = details.get("status", "-")
                    
                    # Logic for including in STDOUT and prompt.txt
                    include_in_prompt = False
                    sol_content = ""
                    
                    if is_codegen:
                        if verification_status == "PASS":
                            code = extract_code_from_llm_response(call.get("llm_response", ""))
                            if code:
                                include_in_prompt = True
                                sol_content = code
                                count += 1
                                test_status = call.get("status") or "UNKNOWN"
                                print(f"[Codegen] {name}: {verification_status} (Test: {test_status})")
                    else: # Non-codegen (text) call
                        include_in_prompt = True
                        sol_content = call.get("llm_response", "")
                        count += 1
                        test_status = call.get("status") or "UNKNOWN"
                        print(f"[Text] {name} (Test: {test_status})")
                    
                    if include_in_prompt and sol_content:
                        if sol_content not in solutions:
                            solutions.append(sol_content)

            print(f"Total PASS/Included: {count}")
            
            # --- Generate prompt.txt ---
            # We need to find the prompt by re-reading files for this specific task
            pattern_file = re.compile(rf'.*{target_task}_{target_test}_step_([a-zA-Z0-9]+)\.json$')
            for filename in sorted(files):
                if not pattern_file.match(filename):
                    continue
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r') as f:
                        file_content = json.load(f)
                    
                    def find_prompt_recursive(d):
                        if isinstance(d, dict):
                            for k, v in d.items():
                                if isinstance(v, dict):
                                    p = v.get("Full raw LLM call")
                                    if p:
                                        # Prefer non-codegen, non-judge for original prompt
                                        if "codegen" not in k.lower() and "judge" not in k.lower():
                                            return p
                                        # Fallback to any prompt found (likely codegen)
                                        if not hasattr(find_prompt_recursive, 'fallback') or find_prompt_recursive.fallback is None:
                                            find_prompt_recursive.fallback = p
                                if isinstance(v, (dict, list)):
                                    p = find_prompt_recursive(v)
                                    if p: return p
                        elif isinstance(d, list):
                            for item in d:
                                p = find_prompt_recursive(item)
                                if p: return p
                        return None

                    find_prompt_recursive.fallback = None
                    found = find_prompt_recursive(file_content)
                    original_prompt = found or find_prompt_recursive.fallback
                    if original_prompt: break
                except Exception:
                    pass
            
            if original_prompt:
                with open("prompt.txt", "w") as f:
                    f.write(f"Below is a prompt that was run {total_runs} times:\n\n")
                    f.write(f"<PROMPT START>\n{original_prompt}\n<PROMPT STOP>\n\n\n")
                    
                    if all_analysis:
                        f.write(f"Solutions were generated {len(solutions)} times, using different types of solvers. All solutions are represented below:\n\n")
                    else:
                        # Original codegen_analysis wording
                        f.write(f"For {count} of these prompts, the code generated solved all the training examples, but the code is different and generates different results on the test input data. Below are the {len(solutions)} different solutions:\n\n")
                    
                    for i, sol in enumerate(solutions, 1):
                        f.write(f"<SOLUTION {i} START>\n{sol}\n<SOLUTION {i} STOP>\n\n")
                    
                    if all_analysis:
                        f.write(f"Your task is to understand these {len(solutions)} solutions, and assess how well they've understood the problem, and how likely their solutions are to provide the correct solution to the test input. Often, new mechanics are introduced in the test example for which the solutions do not generalize well. Please output two solutions that you think represent the right mechanic for solving the problem. Output your two solutions as grids (in code blocks), and explain how you came to these two solutions being the two most likely. In coming up with your two solutions, study all the provided solutions and their reasoning to come up with a meta-conclusion about how to solve the problem.\n")
                    else:
                        # Original codegen_analysis wording
                        f.write(f"Your task is to understand these {len(solutions)} solutions, and assess how well they've understood the problem, and how likely their solutions are to provide the correct solution to the test input. Often, new mechanics are introduced in the test example for which the solutions do not generalize well.\n\n")
                        f.write("Please output what you think is the right mechanic for solving the problem, and express it both in natural language and in a complete solver() function implementation.\n")
            else:
                print("Warning: Could not find original prompt to generate prompt.txt")

        else:
            print("Total: 0")
        return

    # Calculate model stats
    model_stats = calculate_model_stats(task_data)
    
    # Calculate timing stats v2
    timing_stats_v2 = calculate_timing_stats_v2(task_data)

    # Check for failures file
    failure_count = 0
    max_token_failure_count = 0
    timeout_failure_count = 0
    server_failure_count = 0
    error_403_failure_count = 0
    rate_limit_failure_count = 0
    network_failure_count = 0
    connection_failure_count = 0
    content_filter_failure_count = 0
    other_failure_count = 0
    overlap_failure_count = 0

    for f in files:
        if f.endswith("_failures.jsonl"):
            try:
                with open(os.path.join(directory, f), 'r') as fp:
                    for line in fp:
                        if line.strip():
                            failure_count += 1
                            try:
                                record = json.loads(line)
                                error_msg = record.get("error_message", "")
                                
                                is_max_token = "max_output_tokens" in error_msg
                                is_timeout = "timed out after 3600s" in error_msg or "timed out. Falling back" in error_msg or "Timeout after 3600s" in error_msg
                                is_server_error = "server_error" in error_msg
                                is_403 = "Error code: 403" in error_msg
                                is_429 = "Error code: 429" in error_msg or "rate_limit_exceeded" in error_msg or "RESOURCE_EXHAUSTED" in error_msg
                                is_network = "Network/Protocol Error" in error_msg or "503 UNAVAILABLE" in error_msg
                                is_connection = "Connection error" in error_msg
                                is_content_filter = "content filtering policy" in error_msg or "Output blocked" in error_msg
                                
                                if is_max_token:
                                    max_token_failure_count += 1
                                if is_timeout:
                                    timeout_failure_count += 1
                                if is_server_error:
                                    server_failure_count += 1
                                if is_403:
                                    error_403_failure_count += 1
                                if is_429:
                                    rate_limit_failure_count += 1
                                if is_network:
                                    network_failure_count += 1
                                if is_connection:
                                    connection_failure_count += 1
                                if is_content_filter:
                                    content_filter_failure_count += 1
                                
                                if is_max_token and is_timeout:
                                    overlap_failure_count += 1
                                elif not (is_max_token or is_timeout or is_server_error or is_403 or is_429 or is_network or is_connection or is_content_filter):
                                    other_failure_count += 1
                                    
                            except json.JSONDecodeError:
                                other_failure_count += 1 # Count as 'other' (parse error)
            except Exception as e:
                print(f"Warning: Could not read failure file {f}: {e}")
            break

    # Print Report
    print_full_report(task_data, model_stats, failure_count, max_token_failure_count, timeout_failure_count, other_failure_count, overlap_failure_count, timing_stats_v2, server_failure_count, error_403_failure_count, network_failure_count, rate_limit_failure_count, connection_failure_count, content_filter_failure_count)

    print("\n--- Task Status Table ---")
    print("Task:Test,Status,Solved By")
    for (task_id, test_id), data in sorted(task_data.items()):
        status_val = data.get("finish_status")
        status = "SOLVED" if status_val in ["PASS", "SOLVED"] else "FAILED"
        
        solved_count = 0
        for step_name, calls in data.get("steps", {}).items():
            for call in calls:
                if call.get("status") == "PASS":
                    solved_count += 1
        
        print(f"{task_id}:{test_id},{status},{solved_count}")


def main():
    parser = argparse.ArgumentParser(description="Parse log files to extract task and test IDs.")
    parser.add_argument("directory", help="Path to the logs directory")
    parser.add_argument("--codegen_analysis", help="Perform specialized analysis for a specific task:test (e.g. '221dfab4:1'), only including codegen PASS solutions.", default=None)
    parser.add_argument("--all_analysis", help="Perform comprehensive analysis for a specific task:test, including all solvers (codegen filtered to PASS).", default=None)
    args = parser.parse_args()

    parse_logs(args.directory, args.codegen_analysis, args.all_analysis)

if __name__ == "__main__":
    main()
