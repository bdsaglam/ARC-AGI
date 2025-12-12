try:
    from .stats import determine_strategies_status
    from .report_models import print_model_summary, print_timing_stats, print_cost_stats
    from .report_tasks import print_task_summary, print_failed_task_model_stats
    from .report_strategies import print_strategy_stats, print_methodology_stats
    from .report_judges import print_judge_performance
except ImportError:
    from stats import determine_strategies_status
    from report_models import print_model_summary, print_timing_stats, print_cost_stats
    from report_tasks import print_task_summary, print_failed_task_model_stats
    from report_strategies import print_strategy_stats, print_methodology_stats
    from report_judges import print_judge_performance

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
    print_methodology_stats(task_data)
    print_judge_performance(task_data)
    print_failed_task_model_stats(task_data)