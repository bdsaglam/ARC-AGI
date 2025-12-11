from concurrent.futures import ThreadPoolExecutor
from src.audit_prompts import build_logic_prompt, build_consistency_prompt
from src.judges import run_judge

def pick_solution_v2(candidates_object, reasoning_store, task, test_index, openai_client, anthropic_client, google_keys, judge_model="gemini-3-high", verbose: int = 0):
    """
    Advanced solution picker using TWO LLM Judges (Logic & Consistency).
    Replicates methodology from solution_handler.py:
    - Attempt 1: Consensus (Vote Count)
    - Attempt 2: Auditor Choice (Max Score of Logic/Consistency)
    """
    if verbose >= 1:
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
    
    if verbose >= 1:
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

    # 2. Filter Candidates for Judging
    # If there are two or more candidates that have two or more models/votes,
    # then the only candidates that should be considered by the judges are candidates that have score of two or more.
    multi_vote_candidates = [c for c in candidates_list if c['count'] >= 2]
    
    if len(multi_vote_candidates) >= 2:
        candidates_for_judging = multi_vote_candidates
        if verbose >= 1:
            print(f"[pick_solution_v2] Filtering enabled: Judging only {len(candidates_for_judging)} candidates with >= 2 votes.")
    else:
        candidates_for_judging = candidates_list
        if verbose >= 1:
            print(f"[pick_solution_v2] Filtering disabled: Judging all {len(candidates_for_judging)} candidates.")

    # 3. Build Prompts
    full_prompt_logic = build_logic_prompt(train_examples, test_input, candidates_for_judging)
    full_prompt_cons = build_consistency_prompt(train_examples, test_input, candidates_for_judging)

    # 4. Run Judges
    scores = {c['id']: 0.0 for c in candidates_list} # Init all scores to 0

    # Data Holders
    logic_data = { "prompt": full_prompt_logic, "response": None, "parsed": None }
    cons_data = { "prompt": full_prompt_cons, "response": None, "parsed": None }

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_logic = executor.submit(run_judge, "Logic", full_prompt_logic, judge_model, openai_client, anthropic_client, google_keys, logic_data, verbose)
        future_cons = executor.submit(run_judge, "Consistency", full_prompt_cons, judge_model, openai_client, anthropic_client, google_keys, cons_data, verbose)
        
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

    # 5. Determine Attempt 1 (Consensus)
    # Sort by count desc, then by score desc
    candidates_by_consensus = sorted(candidates_list, key=lambda c: (c['count'], scores[c['id']]), reverse=True)
    attempt_1_candidate = candidates_by_consensus[0]
    
    # Determine if it was a tie-break
    is_tie_break = False
    if len(candidates_by_consensus) > 1:
        second_best = candidates_by_consensus[1]
        if second_best['count'] == attempt_1_candidate['count']:
            is_tie_break = True
            
    label = "Consensus (Tie-Break by Score)" if is_tie_break else "Consensus (Vote)"
    if verbose >= 1:
        print(f"[pick_solution_v2] Attempt 1 ({label}): Candidate {attempt_1_candidate['id']} (Votes: {attempt_1_candidate['count']}, Score: {scores[attempt_1_candidate['id']]})")

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
        if verbose >= 1:
            print(f"[pick_solution_v2] Attempt 2 (Auditor): Candidate {attempt_2_candidate['id']} (Max Score: {scores[attempt_2_candidate['id']]})")
        final_selection.append(attempt_2_candidate)
    else:
        if verbose >= 1:
            print("[pick_solution_v2] Attempt 2 (Auditor): No distinct candidate found. Using Consensus only.")
    
    # Debug Scoring
    if verbose >= 1:
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
    if verbose >= 1:
        print("\n" + "="*40)
        print("FINAL OUTCOME (V2 - Multi-Judge)")
        print("="*40)
    
    is_solved_flag = False
    unknown_status = False
    
    for i, group in enumerate(top_groups):
        correctness = group.get("is_correct")
        role = "Auditor"
        if i == 0:
            role = label

        if verbose >= 1:
            print(f"Attempt {i+1} ({role}) Correctness: {correctness}")
        
        if correctness is None:
            unknown_status = True
        elif correctness:
            is_solved_flag = True
            
    if unknown_status:
        if verbose >= 1:
            print("Outcome: SUBMITTED (No Ground Truth)")
    elif is_solved_flag:
        if verbose >= 1:
            print("Outcome: SOLVED")
    else:
        if verbose >= 1:
            print("Outcome: FAILED")

    return top_groups, is_solved_flag, selection_metadata
