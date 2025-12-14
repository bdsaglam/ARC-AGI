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
    print("\n" + "-" * 130)
    print("Model Cost Statistics")
    print("-" * 130)

    grand_total_cost = sum(sum(stats["costs"]) for stats in model_stats.values())

    print(f"{ 'Model':<{max_model_len}}  {'Avg Cost':<10}  {'Total Cost':<12}  {'% of Total':<12} {'Avg Input':<10} {'Avg Output':<10} {'Avg Cached':<10}")

    for m in sorted_models:
        stats = model_stats[m]
        costs = stats["costs"]
        
        if not costs:
             print(f"{m:<{max_model_len}}  {'-':<10}  {'-':<12}  {'-':<12} {'-':<10} {'-':<10} {'-':<10}")
             continue
        
        avg_cost = statistics.mean(costs)
        total_cost = sum(costs)
        
        percentage_of_total = (total_cost / grand_total_cost) * 100 if grand_total_cost > 0 else 0

        input_tokens = stats.get("input_tokens", [])
        output_tokens = stats.get("output_tokens", [])
        cached_tokens = stats.get("cached_tokens", [])

        avg_input = statistics.mean(input_tokens) if input_tokens else 0
        avg_output = statistics.mean(output_tokens) if output_tokens else 0
        avg_cached = statistics.mean(cached_tokens) if cached_tokens else 0
        
        print(f"{m:<{max_model_len}}  ${avg_cost:<9.4f}  ${total_cost:<11.4f}  {percentage_of_total:8.2f}%    {avg_input:<10.1f} {avg_output:<10.1f} {avg_cached:<10.1f}")

def print_zero_duration_stats(model_stats, max_model_len, sorted_models):
    print("\n" + "-" * 80)
    print("Zero Duration Calls")
    print("-" * 80)
    
    print(f"{ 'Model':<{max_model_len}}  {'Count'}")
    
    for m in sorted_models:
        stats = model_stats[m]
        zero_count = stats.get("zero_duration_calls", 0)
        
        print(f"{m:<{max_model_len}}  {zero_count}")

def print_failed_grid_stats(model_stats, max_model_len, sorted_models):
    print("\n" + "-" * 120)
    print("Failed Grid Extractions")
    print("-" * 120)
    
    print(f"{ 'Model':<{max_model_len}}  {'Count':<6}  {'Examples (First 5)'}")
    
    for m in sorted_models:
        stats = model_stats[m]
        fail_count = stats.get("failed_grid_extractions", 0)
        examples = stats.get("failed_grid_examples", [])[:5]
        ex_str = ", ".join(examples)
        
        print(f"{m:<{max_model_len}}  {fail_count:<6}  {ex_str}")

def print_bad_grid_stats(model_stats, max_model_len, sorted_models):
    print("\n" + "-" * 120)
    print("Bad Grids (1xN or Nx1)")
    print("-" * 120)
    
    print(f"{ 'Model':<{max_model_len}}  {'Count':<6}  {'Examples (First 5)'}")
    
    for m in sorted_models:
        stats = model_stats[m]
        bad_count = stats.get("bad_grid_count", 0)
        examples = stats.get("bad_grid_examples", [])[:5]
        ex_str = ", ".join(examples)
        
        print(f"{m:<{max_model_len}}  {bad_count:<6}  {ex_str}")

def print_timing_stats_v2(timing_stats, max_model_len):
    print("\n" + "-" * 80)
    print("Model Timing Statistics V2 (Detailed Breakdown)")
    print("-" * 80)
    
    print(f"{ 'Model':<{max_model_len}}  {'Count':<8}  {'Avg (s)':<10}  {'95% (s)':<10}  {'Max (s)'}")
    
    sorted_models = sorted(timing_stats.keys())
    
    for m in sorted_models:
        stats = timing_stats[m]
        durations = sorted(stats["durations"])
        count = stats["count"]
        
        if not durations:
            print(f"{m:<{max_model_len}}  {count:<8}  {'-':<10}  {'-':<10}  {'-'}")
            continue
            
        avg_time = statistics.mean(durations)
        max_time = max(durations)
        p95 = calculate_percentile(durations, 0.95)

        print(f"{m:<{max_model_len}}  {count:<8}  {avg_time:<10.2f}  {p95:<10.2f}  {max_time:.2f}")