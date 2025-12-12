try:
    from .stats import determine_strategies_status
    from .utils import normalize_model_name
except ImportError:
    from stats import determine_strategies_status
    from utils import normalize_model_name

def print_task_summary(task_data):
    print("\n" + "-" * 80)
    print("Task Summary (Failed Only)")
    print("-" * 80)
    
    header = f"{ 'Task:Test':<15} {'Outcome':<8} {'Max Step':<8} {'Pass Cnt':<8} {'Strategies (Std/Vote/Score)':<30} {'Passing Models'}"
    print(header)
    
    sorted_keys = sorted(task_data.keys(), key=lambda x: (x[0], x[1]))
    
    for task, test in sorted_keys:
        entry = task_data[(task, test)]
        
        # Strategies
        strategies = determine_strategies_status(entry)
        std_status = "PASS" if strategies["standard"] else "FAIL"
        vote_status = "PASS" if strategies["vote"] else "FAIL"
        score_status = "PASS" if strategies["score"] else "FAIL"
        
        outcome = "SOLVED" if strategies["standard"] else "FAILED"
        
        if outcome != "FAILED":
            continue

        strategies_str = f"{std_status}/{vote_status}/{score_status}"
        
        # Max Step
        steps = entry["steps"].keys()
        max_step_num = 0
        
        for s in steps:
            if s == "finish":
                continue
            # Handle "5-image", "1", "3" etc.
            part = s.split("-")[0]
            if part.isdigit():
                val = int(part)
                if val > max_step_num:
                    max_step_num = val
        
        max_step_str = str(max_step_num)
            
        # Passing Models
        passing_models = set()
        for step_name, calls in entry["steps"].items():
            for call in calls:
                if call["status"] == "PASS":
                    passing_models.add(call["name"])
        
        pass_count = len(passing_models)
        passing_models_list = sorted(list(passing_models))
        passing_models_str = ", ".join(passing_models_list)

        if outcome == "SOLVED":
            passing_models_str = "NA"
        
        print(f"{f'{task}:{test}':<15} {outcome:<8} {max_step_str:<8} {pass_count:<8} {strategies_str:<30} {passing_models_str}")

def print_failed_task_model_stats(task_data):
    print("\n" + "-" * 80)
    print("Pass Frequency by Model Type (Failed Tasks Only)")
    print("-" * 80)
    
    model_counts = {}

    for task_key, entry in task_data.items():
        # Check if failed
        strategies = determine_strategies_status(entry)
        outcome = "SOLVED" if strategies["standard"] else "FAILED"
        if outcome != "FAILED":
            continue
            
        # Get passing unique run IDs
        passing_run_ids = set()
        for step_name, calls in entry["steps"].items():
            for call in calls:
                if call["status"] == "PASS":
                    passing_run_ids.add(call["name"])
        
        # Tally by model type
        for run_id in passing_run_ids:
            model_type = normalize_model_name(run_id)
            model_counts[model_type] = model_counts.get(model_type, 0) + 1

    print(f"{ 'Model Type':<40} {'Frequency'}")
    
    sorted_counts = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)
    for model, count in sorted_counts:
        print(f"{model:<40} {count}")
