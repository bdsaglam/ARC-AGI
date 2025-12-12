def print_judge_performance(task_data):
    print("\n" + "-" * 80)
    print("Judge Performance Comparison")
    print("-" * 80)
    
    stats = {
        "logic": {"tasks": 0, "cost": 0, "duration": 0, "top1_correct": 0, "top3_recall_hits": 0, "top3_recall_opportunities": 0, "score_correct_sum": 0, "score_correct_count": 0, "score_incorrect_sum": 0, "score_incorrect_count": 0},
        "consistency": {"tasks": 0, "cost": 0, "duration": 0, "top1_correct": 0, "top3_recall_hits": 0, "top3_recall_opportunities": 0, "score_correct_sum": 0, "score_correct_count": 0, "score_incorrect_sum": 0, "score_incorrect_count": 0}
    }
    
    for key, entry in task_data.items():
        finish_data = entry.get("finish_data")
        if not finish_data or not isinstance(finish_data, dict):
            continue
            
        judge_stats = finish_data.get("judge_stats", {})
        
        for judge_name in ["logic", "consistency"]:
            if judge_name in judge_stats:
                j_data = judge_stats[judge_name]
                s = stats[judge_name]
                
                s["tasks"] += 1
                s["cost"] += j_data.get("cost", 0)
                s["duration"] += j_data.get("duration", 0)
                
                evals = j_data.get("evaluations", [])
                if not evals:
                    continue
                
                # Check Top-1 Accuracy
                # Find evaluation with rank 1
                top1_cand = next((e for e in evals if e["rank"] == 1), None)
                if top1_cand and top1_cand["is_correct"]:
                    s["top1_correct"] += 1
                    
                # Check Top-3 Recall
                correct_candidates_in_pool = [e for e in evals if e["is_correct"]]
                if correct_candidates_in_pool:
                    s["top3_recall_opportunities"] += 1
                    hit = any(e["rank"] <= 3 and e["is_correct"] for e in evals if e["rank"] > 0)
                    if hit:
                        s["top3_recall_hits"] += 1
                
                # Scores
                for e in evals:
                    if e["is_correct"]:
                        s["score_correct_sum"] += e["score"]
                        s["score_correct_count"] += 1
                    else:
                        s["score_incorrect_sum"] += e["score"]
                        s["score_incorrect_count"] += 1

    print(f"{ 'Metric':<25} { 'Logic Judge':<20} { 'Consistency Judge':<20}")
    
    # Calculate Metrics
    metrics = [
        ("Tasks Evaluated", lambda s: f"{s['tasks']}"),
        ("Total Cost", lambda s: f"${s['cost']:.4f}"),
        ("Avg Duration", lambda s: f"{s['duration'] / s['tasks']:.2f}s" if s['tasks'] else "-"),
        ("Top-1 Accuracy", lambda s: f"{(s['top1_correct'] / s['tasks']) * 100:.1f}%" if s['tasks'] else "-"),
        ("Top-3 Recall", lambda s: f"{(s['top3_recall_hits'] / s['top3_recall_opportunities']) * 100:.1f}%" if s['top3_recall_opportunities'] else "-"),
        ("Avg Score (Correct)", lambda s: f"{s['score_correct_sum'] / s['score_correct_count']:.2f}" if s['score_correct_count'] else "-"),
        ("Avg Score (Incorrect)", lambda s: f"{s['score_incorrect_sum'] / s['score_incorrect_count']:.2f}" if s['score_incorrect_count'] else "-"),
    ]
    
    for label, func in metrics:
        l_val = func(stats["logic"])
        c_val = func(stats["consistency"])
        print(f"{label:<25} {l_val:<20} {c_val:<20}")
