import statistics
try:
    from .stats import calculate_percentile
except ImportError:
    from stats import calculate_percentile

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
