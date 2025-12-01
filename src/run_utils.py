import re
from pathlib import Path

def find_task_path(task_id: str) -> Path:
    if task_id.endswith(".json"):
        p = Path(task_id)
        if p.exists():
            return p
        task_id = p.stem
    candidate = Path("data/arc-agi-2-evaluation") / f"{task_id}.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Task file for '{task_id}' not found in data/arc-agi-2-evaluation/.")

def is_solved(candidates_object) -> bool:
    if not candidates_object:
        return False

    total_model_runs = sum(group['count'] for group in candidates_object.values())
    if total_model_runs == 0:
        return False

    sorted_groups = sorted(candidates_object.values(), key=lambda g: g['count'], reverse=True)
    top_group = sorted_groups[0]
    max_count = top_group['count']
    
    percentage = (max_count / total_model_runs)
    
    # Condition 1: count > 25%
    # Condition 2: count >= 4
    if not (percentage > 0.25 and max_count >= 4):
        return False
        
    # Condition 3: all other groups have exactly count=1
    for group in sorted_groups[1:]:
        if group['count'] != 1:
            return False
            
    return True

def pick_solution(candidates_object):
    # Temporary implementation
    sorted_groups = sorted(candidates_object.values(), key=lambda g: g['count'], reverse=True)
    
    print("\n" + "="*40)
    print("FINAL OUTCOME")
    print("="*40)
    
    is_solved_flag = False
    unknown_status = False
    top_groups = sorted_groups[:2]
    
    if len(top_groups) > 0:
        if top_groups[0].get("is_correct") is None:
            unknown_status = True
        elif top_groups[0]["is_correct"]:
            is_solved_flag = True
        elif len(top_groups) > 1 and top_groups[1]["is_correct"]:
            is_solved_flag = True
        
    if unknown_status:
        print("Outcome: SUBMITTED (No Ground Truth)")
    elif is_solved_flag:
        print("Outcome: SOLVED")
    else:
        print("Outcome: FAILED")

    print("\n--- Debug Info ---")
    if not top_groups:
        print("No solutions generated.")
    else:
        for i, group in enumerate(top_groups):
            correctness = group.get('is_correct')
            c_str = "Unknown" if correctness is None else str(correctness)
            print(f"Group {i+1}: Count={group['count']}, Correct={c_str}")
            print(f"  Models: {', '.join(group['models'])}")

    # Check for other correct groups
    other_correct = []
    for i, group in enumerate(sorted_groups):
        if i < 2:
            continue
        if group.get('is_correct') is True:
            other_correct.append((i + 1, group))
            
    if other_correct:
        print("\n--- Other Correct Groups ---")
        for rank, group in other_correct:
            print(f"Group {rank}: Count={group['count']}, Correct={group['is_correct']}")
            print(f"  Models: {', '.join(group['models'])}")
            
    return top_groups, is_solved_flag