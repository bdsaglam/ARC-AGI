import re
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from src.models import call_model
from src.audit_prompts import (
    PROMPT_LOGIC_SYSTEM_ROLE,
    PROMPT_LOGIC_INSTRUCTIONS,
    PROMPT_CONSISTENCY_SYSTEM_ROLE,
    PROMPT_CONSISTENCY_TASK_CONTEXT,
    PROMPT_CONSISTENCY_INSTRUCTIONS_FORMAT,
    PROMPT_CONSISTENCY_OUTPUT_FORMAT
)

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
    # Model priority mapping (higher number = higher priority)
    MODEL_PRIORITY = {
        "claude-opus-4.5-thinking-60000": 4,
        "gemini-3-high": 3,
        "gpt-5.1-high": 2,
        "claude-sonnet-4.5-thinking-60000": 1
    }

    def get_group_priority(group):
        max_priority = 0
        for run_id in group['models']:
            # run_id format is typically "model-name_count_step"
            # We try to match the longest possible prefix in our priority map
            for model_name, priority in MODEL_PRIORITY.items():
                if run_id.startswith(model_name):
                    if priority > max_priority:
                        max_priority = priority
        return max_priority

    # Sort by count (descending) and then by priority (descending)
    sorted_groups = sorted(
        candidates_object.values(), 
        key=lambda g: (g['count'], get_group_priority(g)), 
        reverse=True
    )
    
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
            priority = get_group_priority(group)
            print(f"Group {i+1}: Count={group['count']}, Priority={priority}, Correct={c_str}")
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
            
    return top_groups, is_solved_flag, {}

def grid_to_string(grid):
    """Formats grid for Prompt Logic (visual style)."""
    if not grid:
        return "(Empty Grid)"
    
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    lines = [f"Size: {rows}x{cols}"]
    for row in grid:
        lines.append("".join(str(c) for c in row))
    return "\n".join(lines)

def grid_to_csv_rows(grid):
    """Formats grid for Prompt Consistency (comma-separated rows)."""
    if not grid:
        return ""
    lines = []
    for row in grid:
        lines.append("      " + ",".join(map(str, row)))
    return "\n".join(lines)

def pick_solution_v2(candidates_object, reasoning_store, task, test_index, openai_client, anthropic_client, google_keys, judge_model="gemini-3-high"):
    """
    Advanced solution picker using TWO LLM Judges (Logic & Consistency).
    Replicates methodology from solution_handler.py:
    - Attempt 1: Consensus (Vote Count)
    - Attempt 2: Auditor Choice (Max Score of Logic/Consistency)
    """
    print("\n[pick_solution_v2] Starting Advanced Solution Picker (Multi-Judge)")
    
    # 0. Preparation
    train_examples = task.train
    test_input = task.test[test_index-1].input if task.test and len(task.test) >= test_index else None
    
    # Flatten candidates for easy indexing
    candidates_list = []
    for idx, (grid_tuple, val) in enumerate(candidates_object.items()):
        candidates_list.append({
            "id": idx,
            "grid": val.get("grid"),
            "models": val.get("models"),
            "count": val.get("count"),
            "is_correct": val.get("is_correct"),
            "reasoning": {} 
        })
    
    print(f"[pick_solution_v2] Total unique candidates found: {len(candidates_list)}")

    # Init Metadata
    selection_metadata = {
        "judges": {},
        "selection_process": {}
    }

    if not candidates_list:
        print("[pick_solution_v2] No candidates found.")
        return [], False, selection_metadata

    # 1. Extract Reasoning
    for cand in candidates_list:
        for model_id in cand["models"]:
            if model_id in reasoning_store:
                cand["reasoning"][model_id] = reasoning_store[model_id]

    # 3. Build Prompts
    # Logic Prompt
    logic_parts = []
    logic_parts.append(PROMPT_LOGIC_SYSTEM_ROLE)
    logic_parts.append("\n<INPUT_DATA>")
    
    logic_parts.append("1. {SOLVED_EXAMPLES}:")
    for i, example in enumerate(train_examples):
        logic_parts.append(f"<EXAMPLE_{i+1}>")
        logic_parts.append("<INPUT>")
        logic_parts.append(grid_to_string(example.input))
        logic_parts.append("</INPUT>")
        logic_parts.append("<OUTPUT>")
        logic_parts.append(grid_to_string(example.output))
        logic_parts.append("</OUTPUT>")
        logic_parts.append(f"</EXAMPLE_{i+1}>")
    
    logic_parts.append("\n2. {TEST_INPUT}:")
    if test_input:
        logic_parts.append(grid_to_string(test_input))
    else:
        logic_parts.append("(No Test Input)")

    logic_parts.append("\n3. {CANDIDATES}:")
    for cand in candidates_list:
        c_id = cand['id']
        logic_parts.append(f"<CANDIDATE {c_id}>")
        logic_parts.append("<PROPOSED_SOLUTION>")
        logic_parts.append(grid_to_string(cand['grid']))
        logic_parts.append("</PROPOSED_SOLUTION>")
        for j, model_id in enumerate(cand['models']): # Removed [:3] slice
            alias = chr(65 + j)
            logic_parts.append(f'<REASONING_MODEL_{alias} model_id="{model_id}">')
            reasoning = cand['reasoning'].get(model_id, "(Reasoning not found)")
            logic_parts.append(reasoning)
            logic_parts.append(f"</REASONING_MODEL_{alias}>")
        logic_parts.append(f"</CANDIDATE {c_id}>")

    logic_parts.append("</INPUT_DATA>\n")
    logic_parts.append(PROMPT_LOGIC_INSTRUCTIONS)
    full_prompt_logic = "\n".join(logic_parts)

    # Consistency Prompt
    cons_parts = []
    cons_parts.append(PROMPT_CONSISTENCY_SYSTEM_ROLE)
    cons_parts.append(PROMPT_CONSISTENCY_TASK_CONTEXT)
    cons_parts.append("\n<PROBLEM>")
    
    for i, ex in enumerate(train_examples):
        cons_parts.append(f'  <TRAIN_EXAMPLE index="{i+1}">')
        cons_parts.append("    <INPUT_GRID>")
        cons_parts.append(grid_to_csv_rows(ex.input))
        cons_parts.append("    </INPUT_GRID>")
        cons_parts.append("    <OUTPUT_GRID>")
        cons_parts.append(grid_to_csv_rows(ex.output))
        cons_parts.append("    </OUTPUT_GRID>")
        cons_parts.append("  </TRAIN_EXAMPLE>")
        
    if test_input:
        cons_parts.append("  <TEST_INPUT>")
        cons_parts.append("    <INPUT_GRID>")
        cons_parts.append(grid_to_csv_rows(test_input))
        cons_parts.append("    </INPUT_GRID>")
        cons_parts.append("  </TEST_INPUT>")
        
    cons_parts.append("</PROBLEM>\n")
    
    cons_parts.append("<CANDIDATES>")
    for cand in candidates_list:
        c_id = cand['id']
        cons_parts.append(f'  <CANDIDATE id="{c_id}">')
        for j, model_id in enumerate(cand['models']): # Removed [:3] slice
            alias = chr(65 + j)
            cons_parts.append(f'    <ANSWER id="{alias}" model_id="{model_id}">') # Replicated ANSWER tag structure
            cons_parts.append(f'      <EXPLANATION>')
            reasoning = cand['reasoning'].get(model_id, "(Reasoning not found)")
            cons_parts.append(reasoning)
            cons_parts.append(f'      </EXPLANATION>')
            cons_parts.append(f'      <OUTPUT_GRID>')
            cons_parts.append(grid_to_csv_rows(cand['grid']))
            cons_parts.append(f'      </OUTPUT_GRID>')
            cons_parts.append(f'    </ANSWER>')
        cons_parts.append(f'  </CANDIDATE>')
    cons_parts.append("</CANDIDATES>\n")
    
    cons_parts.append(PROMPT_CONSISTENCY_INSTRUCTIONS_FORMAT)
    cons_parts.append(PROMPT_CONSISTENCY_OUTPUT_FORMAT)
    full_prompt_cons = "\n".join(cons_parts)


    # 4. Run Judges
    scores = {c['id']: 0.0 for c in candidates_list}

    # Data Holders
    logic_data = { "prompt": full_prompt_logic, "response": None, "parsed": None }
    cons_data = { "prompt": full_prompt_cons, "response": None, "parsed": None }

    def run_judge(judge_name, prompt, data_holder):
        print(f"\n[pick_solution_v2] Running {judge_name} Judge ({judge_model})...")
        try:
            response_obj = call_model(openai_client, anthropic_client, google_keys, prompt, judge_model)
            data_holder["response"] = response_obj.text
            
            json_match = re.search(r"\{.*\}", response_obj.text, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group(0))
                data_holder["parsed"] = parsed_json
                return parsed_json
        except Exception as e:
            print(f"[pick_solution_v2] {judge_name} Judge Error: {e}")
            data_holder["error"] = str(e)
        return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_logic = executor.submit(run_judge, "Logic", full_prompt_logic, logic_data)
        future_cons = executor.submit(run_judge, "Consistency", full_prompt_cons, cons_data)
        
        logic_res = future_logic.result()
        cons_res = future_cons.result()

    # Update Scores based on results
    if logic_res and "candidates" in logic_res:
        for c in logic_res["candidates"]:
            cid = c.get("candidate_id")
            if cid in scores:
                scores[cid] = max(scores[cid], c.get("score", 0))
    
    if cons_res and "candidates" in cons_res:
        for c in cons_res["candidates"]:
            cid = c.get("candidate_id")
            if cid in scores:
                scores[cid] = max(scores[cid], c.get("score", 0))

    # 2. Determine Attempt 1 (Consensus)
    # Sort by count desc, then by score desc
    candidates_by_consensus = sorted(candidates_list, key=lambda c: (c['count'], scores[c['id']]), reverse=True)
    attempt_1_candidate = candidates_by_consensus[0]
    print(f"[pick_solution_v2] Attempt 1 (Consensus): Candidate {attempt_1_candidate['id']} (Votes: {attempt_1_candidate['count']}, Score: {scores[attempt_1_candidate['id']]})")

    # 5. Determine Attempt 2 (Auditor)
    # Sort candidates by Max Score
    sorted_by_score = sorted(candidates_list, key=lambda c: scores[c['id']], reverse=True)
    
    attempt_2_candidate = None
    # Pick the best scoring candidate that is NOT Attempt 1
    for cand in sorted_by_score:
        if cand['id'] != attempt_1_candidate['id']:
            attempt_2_candidate = cand
            break
            
    final_selection = [attempt_1_candidate]
    if attempt_2_candidate:
        print(f"[pick_solution_v2] Attempt 2 (Auditor): Candidate {attempt_2_candidate['id']} (Max Score: {scores[attempt_2_candidate['id']]})")
        final_selection.append(attempt_2_candidate)
    else:
        print("[pick_solution_v2] Attempt 2 (Auditor): No distinct candidate found. Using Consensus only.")
    
    # Debug Scoring
    print("\n[pick_solution_v2] Final Scores:")
    for cand in candidates_list:
        print(f"  ID {cand['id']}: Score={scores[cand['id']]}, Votes={cand['count']}")

    # 6. Populate Metadata
    selection_metadata["judges"]["logic"] = logic_data
    selection_metadata["judges"]["consistency"] = cons_data
    selection_metadata["selection_process"] = {
        "candidates_summary": [
            {"id": c['id'], "votes": c['count'], "score": scores[c['id']]} 
            for c in candidates_list
        ],
        "attempt_1": {"type": "Consensus", "candidate_id": attempt_1_candidate['id'], "vote_count": attempt_1_candidate['count']},
        "attempt_2": {
            "type": "Auditor", 
            "candidate_id": attempt_2_candidate['id'] if attempt_2_candidate else None, 
            "audit_score": scores[attempt_2_candidate['id']] if attempt_2_candidate else None
        }
    }

    # 7. Construct Return Output
    top_groups = []
    for cand in final_selection:
        grid_tuple = tuple(tuple(row) for row in cand['grid'])
        top_groups.append(candidates_object[grid_tuple])

    # 8. Final Success Check
    print("\n" + "="*40)
    print("FINAL OUTCOME (V2 - Multi-Judge)")
    print("="*40)
    
    is_solved_flag = False
    unknown_status = False
    
    for i, group in enumerate(top_groups):
        correctness = group.get("is_correct")
        role = "Consensus" if i == 0 else "Auditor"
        print(f"Attempt {i+1} ({role}) Correctness: {correctness}")
        
        if correctness is None:
            unknown_status = True
        elif correctness:
            is_solved_flag = True
            
    if unknown_status:
        print("Outcome: SUBMITTED (No Ground Truth)")
    elif is_solved_flag:
        print("Outcome: SOLVED")
    else:
        print("Outcome: FAILED")

    return top_groups, is_solved_flag, selection_metadata