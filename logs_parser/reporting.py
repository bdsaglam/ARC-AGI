import statistics
try:
    from .stats import determine_strategies_status, calculate_percentile
    from .utils import normalize_model_name
except ImportError:
    from stats import determine_strategies_status, calculate_percentile
    from utils import normalize_model_name

def print_model_summary(model_stats, max_model_len):
    print("-" * 80)
    print("Model Summary")
    print("-" * 80)

    print(f"{ 'Model':<{max_model_len}}  {'Solver Attempts':<15}  {'Total Calls':<12}  {'Passed':<8}  {'Pass Rate'}")

    sorted_models = sorted(model_stats.keys())
    for m in sorted_models:
        stats = model_stats[m]
        attempts = stats["solver_attempts"]
        total = stats["total_calls"]
        passed = stats["passes"]
        rate = (passed / attempts) * 100 if attempts > 0 else 0
        print(f"{m:<{max_model_len}}  {attempts:<15}  {total:<12}  {passed:<8}  {rate:6.2f}%")

def print_timing_stats(model_stats, max_model_len, sorted_models):
    print("\n" + "-" * 80)
    print("Model Timing Statistics")
    print("-" * 80)
    
    print(f"{ 'Model':<{max_model_len}}  {'Avg (s)':<10}  {'95% (s)':<10}  {'Max (s)'}")
    
    for m in sorted_models:
        stats = model_stats[m]
        durations = sorted(stats["durations"])
        
        if not durations:
            print(f"{m:<{max_model_len}}  {'-':<10}  {'-':<10}  {'-'}")
            continue
            
        avg_time = statistics.mean(durations)
        max_time = max(durations)
        p95 = calculate_percentile(durations, 0.95)

        print(f"{m:<{max_model_len}}  {avg_time:<10.2f}  {p95:<10.2f}  {max_time:.2f}")

def print_cost_stats(model_stats, max_model_len, sorted_models):
    print("\n" + "-" * 80)
    print("Model Cost Statistics")
    print("-" * 80)

    grand_total_cost = sum(sum(stats["costs"]) for stats in model_stats.values())

    print(f"{ 'Model':<{max_model_len}}  {'Avg Cost':<10}  {'Total Cost':<12}  {'% of Total'}")

    for m in sorted_models:
        stats = model_stats[m]
        costs = stats["costs"]
        
        if not costs:
             print(f"{m:<{max_model_len}}  {'-':<10}  {'-':<12}  {'-'}")
             continue
        
        avg_cost = statistics.mean(costs)
        total_cost = sum(costs)
        
        percentage_of_total = (total_cost / grand_total_cost) * 100 if grand_total_cost > 0 else 0
        
        print(f"{m:<{max_model_len}}  ${avg_cost:<9.4f}  ${total_cost:<11.4f}  {percentage_of_total:8.2f}%")

def print_strategy_stats(total_tasks_count, solved_tasks_count, vote_only_solved_count, score_only_solved_count):
    print("\n" + "-" * 80)
    print("Strategy Performance")
    print("-" * 80)

    print(f"{ 'Strategy':<20}  {'Solved':<8}  {'Failed':<8}  {'Success Rate'}")
    
    # Standard Strategy
    failed_count = total_tasks_count - solved_tasks_count
    rate = (solved_tasks_count / total_tasks_count) * 100 if total_tasks_count > 0 else 0
    print(f"{ 'Standard':<20}  {solved_tasks_count:<8}  {failed_count:<8}  {rate:6.2f}%")

    # Vote only Strategy
    vote_only_failed = total_tasks_count - vote_only_solved_count
    vote_only_rate = (vote_only_solved_count / total_tasks_count) * 100 if total_tasks_count > 0 else 0
    print(f"{ 'Vote only':<20}  {vote_only_solved_count:<8}  {vote_only_failed:<8}  {vote_only_rate:6.2f}%")

    # Score only Strategy
    score_only_failed = total_tasks_count - score_only_solved_count
    score_only_rate = (score_only_solved_count / total_tasks_count) * 100 if total_tasks_count > 0 else 0
    print(f"{ 'Score only':<20}  {score_only_solved_count:<8}  {score_only_failed:<8}  {score_only_rate:6.2f}%")

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

    print(f"{'Model Type':<40} {'Frequency'}")
    
    sorted_counts = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)
    for model, count in sorted_counts:
        print(f"{model:<40} {count}")

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

def print_full_report(task_data, model_stats):
    # Determine max name length for pretty printing
    max_name_len = 0
    for key in task_data:
        for step_name in task_data[key]["steps"]:
            for call_info in task_data[key]["steps"][step_name]:
                if len(call_info["name"]) > max_name_len:
                    max_name_len = len(call_info["name"])
    max_name_len = max(max_name_len, 20) + 2 

    sorted_keys = sorted(task_data.keys(), key=lambda x: (x[0], x[1]))

    total_tasks_count = 0
    solved_tasks_count = 0
    vote_only_solved_count = 0
    score_only_solved_count = 0

    for task, test in sorted_keys:
        total_tasks_count += 1
        entry = task_data[(task, test)]
        
        strategies = determine_strategies_status(entry)
        
        status = "FAILED"
        if strategies["standard"]:
            status = "SOLVED"
            solved_tasks_count += 1
        
        if strategies["vote"]:
            vote_only_solved_count += 1
            
        if strategies["score"]:
            score_only_solved_count += 1

        print(f"{task}:{test} {status}")
        
        steps_dict = entry["steps"]
        step_statuses = entry["step_statuses"]
        
        sorted_steps = sorted(steps_dict.keys(), key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
        
        for step in sorted_steps:
            step_solved_mark = ""
            lookup_step = step.split("-")[0]
            
            if step_statuses.get(lookup_step) is True:
                step_solved_mark = " [SOLVED]"

            print(f"  {step}{step_solved_mark}")
            for call_info in steps_dict[step]:
                name = call_info["name"]
                duration = call_info["duration"]
                cost = call_info["cost"]
                status_val = call_info["status"]
                
                print(f"    {name:<{max_name_len}} {duration:8.2f}s  ${cost:9.4f}  {status_val}")

    # Model Stats
    max_model_len = 0
    for m in model_stats:
        max_model_len = max(max_model_len, len(m))
    max_model_len = max(max_model_len, 10)
    
    sorted_models = sorted(model_stats.keys())

    print_task_summary(task_data)
    print_model_summary(model_stats, max_model_len)
    print_timing_stats(model_stats, max_model_len, sorted_models)
    print_cost_stats(model_stats, max_model_len, sorted_models)
    print_strategy_stats(total_tasks_count, solved_tasks_count, vote_only_solved_count, score_only_solved_count)
    print_failed_task_model_stats(task_data)