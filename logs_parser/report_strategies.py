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

def print_methodology_stats(task_data):
    print("\n" + "-" * 80)
    print("Methodology Performance (Tasks reaching Step 5)")
    print("-" * 80)
    
    # Global stats: method_name -> {attempts, tasks_solved, total_passes, total_cost, total_duration}
    stats = {}
    
    for key, entry in task_data.items():
        steps_dict = entry["steps"]
        
        # Filter: Only consider tasks that reached Step 5
        has_step_5 = any(s.startswith("5-") for s in steps_dict.keys())
        if not has_step_5:
            continue
        
        # Track usage per task to avoid double counting attempts
        task_methods = {} # name -> {solved: bool, cost: 0, duration: 0}

        for step_name, calls in steps_dict.items():
            base_method_name = ""
            if step_name == "1":
                base_method_name = "Step 1"
            elif step_name == "3":
                base_method_name = "Step 3"
            elif step_name.startswith("5-"):
                base_method_name = step_name[2:] # Remove "5-" prefix
            else:
                continue
            
            for call in calls:
                # 1. Track the base method (combined)
                if base_method_name not in task_methods:
                    task_methods[base_method_name] = {"solved": False, "cost": 0, "duration": 0}
                
                tm_base = task_methods[base_method_name]
                tm_base["cost"] += call.get("cost", 0)
                tm_base["duration"] += call.get("duration", 0)
                
                is_pass = (call["status"] == "PASS")
                if is_pass:
                    tm_base["solved"] = True
                    if base_method_name not in stats:
                        stats[base_method_name] = {"attempts": 0, "tasks_solved": 0, "total_passes": 0, "total_cost": 0, "total_duration": 0}
                    stats[base_method_name]["total_passes"] += 1

                # 2. Track the split method if applicable (objects_pipeline variants)
                if base_method_name == "objects_pipeline":
                    gen = call.get("generator")
                    if gen:
                        split_name = f"{base_method_name} ({gen})"
                        
                        if split_name not in task_methods:
                            task_methods[split_name] = {"solved": False, "cost": 0, "duration": 0}
                        
                        tm_split = task_methods[split_name]
                        tm_split["cost"] += call.get("cost", 0)
                        tm_split["duration"] += call.get("duration", 0)
                        
                        if is_pass:
                            tm_split["solved"] = True
                            if split_name not in stats:
                                stats[split_name] = {"attempts": 0, "tasks_solved": 0, "total_passes": 0, "total_cost": 0, "total_duration": 0}
                            stats[split_name]["total_passes"] += 1

        # Aggregate task results to global stats
        for method, info in task_methods.items():
            if method not in stats:
                stats[method] = {"attempts": 0, "tasks_solved": 0, "total_passes": 0, "total_cost": 0, "total_duration": 0}
            
            s = stats[method]
            s["attempts"] += 1
            s["total_cost"] += info["cost"]
            s["total_duration"] += info["duration"]
            if info["solved"]:
                s["tasks_solved"] += 1

    print(f"{ 'Methodology':<30} {'Attempts':<10} {'Solved':<8} {'Rate':<8} {'Yield':<8} {'Avg Cost':<10} {'Avg Time'}")
    
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]["tasks_solved"], reverse=True)
    
    for method, s in sorted_stats:
        attempts = s["attempts"]
        solved = s["tasks_solved"]
        rate = (solved / attempts) * 100 if attempts > 0 else 0
        yield_count = s["total_passes"]
        avg_cost = s["total_cost"] / attempts if attempts > 0 else 0
        avg_time = s["total_duration"] / attempts if attempts > 0 else 0
        
        print(f"{method:<30} {attempts:<10} {solved:<8} {rate:6.1f}%  {yield_count:<8} ${avg_cost:<9.4f} {avg_time:.1f}s")