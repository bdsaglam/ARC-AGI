import sys
from concurrent.futures import ThreadPoolExecutor
from src.audit_prompts import build_logic_prompt, build_consistency_prompt, build_duo_pick_prompt
from src.judges import run_judge, run_duo_pick_judge

def pick_solution_v2(candidates_object, reasoning_store, task, test_index, openai_client, anthropic_client, google_keys, judge_model="gpt-5.2-xhigh", verbose: int = 0, openai_background: bool = False, judge_consistency_enable: bool = False, judge_duo_pick_enable: bool = True, total_attempts: int = 0):
    """
    Advanced solution picker using LLM Judges.
    - If judge_duo_pick_enable: Runs a Meta-Conclusion judge to pick top 2 solutions.
    - Fallback: Consensus (Vote Count) and Auditor Choice (Max Score).
    """
    if verbose >= 1:
        print("\n[pick_solution_v2] Starting Advanced Solution Picker")
    
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

    # 2. Duo Pick Judge (High Priority)
    if judge_duo_pick_enable:
        duo_data = { "prompt": None, "response": None, "picked_grids": None }
        duo_prompt = build_duo_pick_prompt(train_examples, test_input, candidates_list, reasoning_store, total_attempts)
        duo_data["prompt"] = duo_prompt
        
        duo_res_grids = run_duo_pick_judge(duo_prompt, judge_model, openai_client, anthropic_client, google_keys, duo_data, verbose, openai_background)
        selection_metadata["judges"]["duo_pick"] = duo_data
        
        if duo_res_grids and len(duo_res_grids) >= 2:
            if verbose >= 1:
                print(f"[pick_solution_v2] Duo Pick Judge successfully returned {len(duo_res_grids)} grids.")
            
            final_selection_groups = []
            for i, res_grid in enumerate(duo_res_grids):
                res_tuple = tuple(tuple(row) for row in res_grid)
                
                # Try to find if this grid matches an existing candidate
                match_found = False
                for cand in candidates_list:
                    cand_tuple = tuple(tuple(row) for row in cand['grid'])
                    if res_tuple == cand_tuple:
                        # Use existing candidate metadata
                        group = candidates_object[cand_tuple]
                        
                        # Add Judge Feedback to reasoning_summary
                        feedback = f"\n\n--- DUO PICK JUDGE CHOICE (Attempt {i+1}) ---"
                        feedback += duo_data["response"][:1000] # Include snippet of response
                        
                        existing_summary = group.get("reasoning_summary", "")
                        group["reasoning_summary"] = feedback + "\n\n" + existing_summary
                        
                        final_selection_groups.append(group)
                        match_found = True
                        break
                
                if not match_found:
                    # Create a new candidate for the judge's unique solution
                    new_group = {
                        "grid": res_grid,
                        "count": 0,
                        "models": ["duo_pick_judge"],
                        "is_correct": None, # Unknown
                        "reasoning_summary": f"--- DUO PICK JUDGE ORIGINAL SOLUTION (Attempt {i+1}) ---" + duo_data["response"]
                    }
                    final_selection_groups.append(new_group)
            
            selection_metadata["selection_process"] = {
                "type": "Duo Pick Judge",
                "attempt_1": "Judge Pick 1",
                "attempt_2": "Judge Pick 2" if len(final_selection_groups) > 1 else None
            }
            
            # Final Success Check for Duo Pick
            is_solved_flag = False
            unknown_status = False
            for group in final_selection_groups:
                correctness = group.get("is_correct")
                if correctness is None: unknown_status = True
                elif correctness: is_solved_flag = True
            
            return final_selection_groups, is_solved_flag, selection_metadata
        else:
            reason = "Unknown"
            if "error" in duo_data:
                reason = f"Error: {duo_data['error']}"
            elif duo_res_grids is None:
                reason = "No grids parsed (None returned)"
            else:
                reason = f"Only {len(duo_res_grids)} grid(s) found (needed 2)"
            
            print(f"[pick_solution_v2] WARNING: Duo Pick Judge failed. Falling back to Standard logic. Reason: {reason}", file=sys.stderr)

    # 3. Standard Logic (Fallback)
    # Filter Candidates for Judging
    multi_vote_candidates = [c for c in candidates_list if c['count'] >= 2]
    if len(multi_vote_candidates) >= 2:
        candidates_for_judging = multi_vote_candidates
    else:
        candidates_for_judging = candidates_list

    # Build Prompts for Standard Judges
    full_prompt_logic = build_logic_prompt(train_examples, test_input, candidates_for_judging)
    full_prompt_cons = build_consistency_prompt(train_examples, test_input, candidates_for_judging)

    # Run Standard Judges
    scores = {c['id']: 0.0 for c in candidates_list}
    logic_data = { "prompt": full_prompt_logic, "response": None, "parsed": None }
    cons_data = { "prompt": full_prompt_cons, "response": None, "parsed": None }

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_logic = executor.submit(run_judge, "Logic", full_prompt_logic, judge_model, openai_client, anthropic_client, google_keys, logic_data, verbose, openai_background)
        future_cons = None
        if judge_consistency_enable:
            future_cons = executor.submit(run_judge, "Consistency", full_prompt_cons, judge_model, openai_client, anthropic_client, google_keys, cons_data, verbose, openai_background)
        logic_res = future_logic.result()
        cons_res = future_cons.result() if future_cons else None

    # Update Scores
    if logic_res and "candidates" in logic_res:
        for c in logic_res["candidates"]:
            cid = c.get("candidate_id")
            if cid in scores: scores[cid] = max(scores[cid], c.get("score", 0))
    if cons_res and "candidates" in cons_res:
        for c in cons_res["candidates"]:
            cid = c.get("candidate_id")
            if cid in scores: scores[cid] = max(scores[cid], c.get("score", 0))

    # Determine Attempts
    candidates_by_consensus = sorted(candidates_list, key=lambda c: (c['count'], scores[c['id']]), reverse=True)
    attempt_1_candidate = candidates_by_consensus[0]
    
    sorted_by_score = sorted(candidates_list, key=lambda c: scores[c['id']], reverse=True)
    attempt_2_candidate = next((c for c in sorted_by_score if c['id'] != attempt_1_candidate['id']), None)
            
    final_selection = [attempt_1_candidate]
    if attempt_2_candidate: final_selection.append(attempt_2_candidate)
    
    # Populate Metadata
    selection_metadata["judges"]["logic"] = logic_data
    selection_metadata["judges"]["consistency"] = cons_data
    selection_metadata["selection_process"] = {
        "type": "Standard (Consensus/Auditor)",
        "attempt_1": {"candidate_id": attempt_1_candidate['id'], "votes": attempt_1_candidate['count']},
        "attempt_2": {"candidate_id": attempt_2_candidate['id'] if attempt_2_candidate else None}
    }

    # Prepare Judge Feedback Map
    judge_feedback_map = {}
    if logic_res and "candidates" in logic_res:
        for c in logic_res["candidates"]:
            cid = c.get("candidate_id")
            feedback = []
            if "rule_summary" in c: feedback.append(f"Judge Rule Summary: {c['rule_summary']}")
            if "example_audit" in c and "summary" in c["example_audit"]:
                feedback.append(f"Judge Audit Summary: {c['example_audit']['summary']}")
            if feedback: judge_feedback_map[cid] = "\n\n--- JUDGE FEEDBACK ---" + "\n".join(feedback)

    # Construct Return Output
    top_groups = []
    for cand in final_selection:
        grid_tuple = tuple(tuple(row) for row in cand['grid'])
        group = candidates_object[grid_tuple]
        
        final_summary_parts = []
        if cand['id'] in judge_feedback_map:
            final_summary_parts.append(judge_feedback_map[cand['id']])
        if cand["reasoning"]:
            first_model_id = next(iter(cand["reasoning"]))
            raw_reasoning = cand["reasoning"][first_model_id]
            final_summary_parts.append("\n\n--- EXAMPLE REASONING ---" + raw_reasoning)
        
        group["reasoning_summary"] = "".join(final_summary_parts).strip()
        top_groups.append(group)

    return top_groups, any(g.get("is_correct") for g in top_groups), selection_metadata