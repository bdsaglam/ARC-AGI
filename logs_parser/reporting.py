try:
    from .stats import determine_strategies_status
    from .report_models import print_model_summary, print_timing_stats, print_cost_stats, print_zero_duration_stats, print_failed_grid_stats, print_bad_grid_stats, print_timing_stats_v2
    from .report_tasks import print_task_summary, print_failed_task_model_stats, print_codegen_analysis
    from .report_strategies import print_strategy_stats, print_methodology_stats
    from .report_judges import print_judge_performance
except ImportError:
    from stats import determine_strategies_status
    from report_models import print_model_summary, print_timing_stats, print_cost_stats, print_zero_duration_stats, print_failed_grid_stats, print_bad_grid_stats, print_timing_stats_v2
    from report_tasks import print_task_summary, print_failed_task_model_stats, print_codegen_analysis
    from report_strategies import print_strategy_stats, print_methodology_stats
    from report_judges import print_judge_performance

def print_full_report(task_data, model_stats, failure_count=0, max_token_failure_count=0, timeout_failure_count=0, other_failure_count=0, overlap_failure_count=0, timing_stats_v2=None, server_failure_count=0, error_403_failure_count=0, network_failure_count=0, rate_limit_failure_count=0, connection_failure_count=0, content_filter_failure_count=0):
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
    
    if timing_stats_v2:
        print_timing_stats_v2(timing_stats_v2, max_model_len)

    print_cost_stats(model_stats, max_model_len, sorted_models)
    print_strategy_stats(total_tasks_count, solved_tasks_count, vote_only_solved_count, score_only_solved_count)
    print_methodology_stats(task_data)
    print_judge_performance(task_data)
    print_failed_task_model_stats(task_data)
    print_zero_duration_stats(model_stats, max_model_len, sorted_models)
    print_failed_grid_stats(model_stats, max_model_len, sorted_models)
    print_bad_grid_stats(model_stats, max_model_len, sorted_models)
    print_codegen_analysis(task_data)

    if failure_count > 0:
        print("\n" + "-" * 80)
        print("Failures")
        print("-" * 80)
        print(f"{'Total Errors':<25} {failure_count}")
        print(f"{'Max Token Errors':<25} {max_token_failure_count}")
        print(f"{'Timeout Errors':<25} {timeout_failure_count}")
        print(f"{'Server Errors':<25} {server_failure_count}")
        print(f"{'Network/503 Errors':<25} {network_failure_count}")
        print(f"{'Rate Limit Errors (429)':<25} {rate_limit_failure_count}")
        print(f"{'Connection Errors':<25} {connection_failure_count}")
        print(f"{'Content Filter Errors':<25} {content_filter_failure_count}")
        print(f"{'Error Code 403':<25} {error_403_failure_count}")
        print(f"{'Other Errors':<25} {other_failure_count}")
        print(f"{'Overlapping Errors':<25} {overlap_failure_count}")