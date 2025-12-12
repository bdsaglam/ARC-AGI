import argparse
import os
import re

try:
    from .utils import load_answers
    from .parsing import parse_log_file
    from .stats import calculate_model_stats
    from .reporting import print_full_report
except ImportError:
    # Fallback for running as script directly
    from utils import load_answers
    from parsing import parse_log_file
    from stats import calculate_model_stats
    from reporting import print_full_report

def parse_logs(directory):
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
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
            test_id = match.group(2)
            step_name = match.group(3) # '1', '3', '5', 'finish' etc.
            
            # Skip steps 2 and 4 as per original logic
            if step_name in ["2", "4"]:
                continue

            key = (task_id, int(test_id))
            if key not in task_data:
                task_data[key] = {
                    "steps": {}, 
                    "finish_data": None, 
                    "finish_status": None, 
                    "step_statuses": {}
                }
            
            filepath = os.path.join(directory, filename)
            
            result = parse_log_file(filepath, task_id, test_id, step_name, answers)
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

    # Calculate model stats
    model_stats = calculate_model_stats(task_data)

    # Print Report
    print_full_report(task_data, model_stats)

def main():
    parser = argparse.ArgumentParser(description="Parse log files to extract task and test IDs.")
    parser.add_argument("directory", help="Path to the logs directory")
    args = parser.parse_args()

    parse_logs(args.directory)

if __name__ == "__main__":
    main()
