import re
import json
from pathlib import Path

from src.models import call_model

# --- PROMPT TEMPLATES (LOGIC) ---
PROMPT_LOGIC_SYSTEM_ROLE = """<SYSTEM_ROLE>
You are the **ARC LOGIC AUDITOR**.
You are NOT a creative solver. You are a skeptical Critic and Verifier.
Your task is to review a list of pre-grouped "Candidate Clusters" (proposed solutions for an ARC puzzle) and rank them based on logical validity.

Your Core Principle is **FALSIFICATION**:
1. Trust NO ONE.
2. Assume every candidate is "hallucinating" until they prove otherwise.
3. The "Ground Truth" (Solved Examples) is the absolute law.
</SYSTEM_ROLE>"""

PROMPT_LOGIC_INSTRUCTIONS = """<AUDIT_PROTOCOL>
Execute this pipeline sequentially for every Candidate. You must output your thinking process inside <AUDIT_LOG> tags.

### PHASE 1: LOGIC SELECTION & CRYSTALLIZATION
- **Selection:** If a Candidate contains multiple <REASONING> blocks, read them all and select the **single most detailed and logical** explanation to audit.
- **Crystallization:** Convert that text into a strict "IF-THEN" algorithm.
  - *Bad:* "The pattern involves moving blue pixels." (Too vague to audit)
  - *Good:* "IF a pixel is Blue, THEN move it 1 step Right. Else, preserve color."
- *Constraint:* If the reasoning is incoherent or vague, mark the Candidate as "INVALID - VAGUE".

### PHASE 2: THE GROUND TRUTH AUDIT (CRITICAL)
- You must "Back-Test" the Crystallized Rule against the {{SOLVED_EXAMPLESப்பட்டன}}.
- For **EACH** Solved Example pair (Input -> Output), you must strictly perform this 3-step check:
  1. **Hypothesis:** "If I apply the Candidate's Rule to this Input, exactly what *should* happen?"
  2. **Observation:** "Look at the actual Official Output. What *actually* happened?"
  3. **Verdict:** "Do they match?"
- **Fatal Contradictions to Watch For:**
  * **Scope Error:** Rule applies to specific colors (e.g., "Blue"), but the example changes "Red" pixels.
  * **Geometry Error:** Rule says "rotate 90," but example shows a flip.
  * **Object Error:** Rule treats pixels individually, but example shows objects moving as blocks.
- *Constraint:* Record exactly how many Solved Examples the candidate PASSED vs FAILED. **Do not stop at the first failure**; check all examples to determine the severity of the failure (e.g., "Passed 2/3 Examples").

### PHASE 3: EXECUTION CONSISTENCY
- For Candidates that survived Phase 2 (or passed at least one example):
- Look at the {{TEST_INPUT}} and the Candidate's <PROPOSED_SOLUTION> grid.
- Does the proposed output actually follow the Crystallized Rule?
- *Common Hallucination:* The text says "Move Blue," but the grid shows Blue staying still. Mark this as **INTERNAL_CONTRADICTION**.

### PHASE 4: STACK RANKING & TIE-BREAKING
- Rank ALL candidates from Best to Worst based on this hierarchy:
  1. **GOLD (Tier 1):** Passed ALL Solved Examples + Consistent Execution on Test Input.
  2. **SILVER (Tier 2):** Passed ALL Solved Examples + Minor Execution Error on Test Input.
  3. **BRONZE (Tier 3):** Passed MOST Solved Examples (Partial Logic).
  4. **INVALID (Tier 4):** Failed ALL/MOST Solved Examples, Vague Reasoning, or Severe Internal Contradictions.

- **Tie-Breaking:** If two candidates are in the same Tier, rank the one with the **Simplest Rule** (Occam's Razor) higher.
</AUDIT_PROTOCOL>

<OUTPUT_FORMAT>
Return a single JSON object with the following structure:

{
  "candidates": [
    {
      "candidate_id": 0,
      "score": 8.7,
      "tier": "GOLD",
      "example_audit": {
        "per_example": {
          "1": "Pass",
          "2": "Pass",
          "3": "Partial"
          /* add more keys if there are more training examples */
        },
        "summary": "Rule matches main behaviors across examples; minor ambiguity in example 3."
      },
      "test_grid_consistency": "Plausible",
      "rule_summary": "Short, 1–3 sentence description of this candidate's representative rule."
    },
    {
      "candidate_id": 1,
      "score": 6.0,
      "tier": "INVALID",
      "example_audit": {
        "per_example": {
          "1": "Partial",
          "2": "Fail"
        },
        "summary": "Contradiction in example 2; seems overfitted"
      }
    }
  ]
}
</OUTPUT_FORMAT>"""

# --- PROMPT TEMPLATES (CONSISTENCY) ---
PROMPT_CONSISTENCY_SYSTEM_ROLE = """<SYSTEM_ROLE>
You are an ARC Solution Auditor.

Your primary ability is NOT to solve new ARC tasks from scratch.
Instead, you are excellent at:
- Checking whether a proposed rule is logically consistent
- Verifying that a rule matches known solved examples
- Verifying that a candidate’s test output actually follows its own stated rule

You are skeptical and detail-oriented. If a candidate’s explanation says X
but the examples show not-X, you must call that out.
</SYSTEM_ROLE>"""

PROMPT_CONSISTENCY_TASK_CONTEXT = """<TASK_CONTEXT>
The problem consists of:
- One or more solved training examples (each with input + output grids)
- One test input grid (no ground-truth output given)
- One or more candidate solutions, each proposing:
  - One predicted output grid for the test input
  - One or more verbal explanations of the transformation

Your job is to AUDIT the candidates:
- You do NOT need to invent your own new rule.
- You must decide which candidates are most logically consistent with the
  training examples and with themselves, and rank them.
</TASK_CONTEXT>"""

PROMPT_CONSISTENCY_INSTRUCTIONS_FORMAT = """<INSTRUCTIONS>
You must behave as an AUDITOR, not a solver.

Your overall goal:
- For each candidate, choose a representative explanation and treat it as that
  candidate’s proposed rule.
- Audit that rule against all training examples.
- Check whether the candidate’s predicted test OUTPUT_GRID actually follows
  that rule.
- Assign each candidate a score from 0 to 10 and rank all candidates.

Follow these steps:

STEP 1 — CHOOSE A REPRESENTATIVE RULE PER CANDIDATE
For each CANDIDATE:

  1. It may have multiple ANSWER blocks, all with the same OUTPUT_GRID.
  2. Among its ANSWERs, choose a "representative explanation":
     - Prefer the explanation that is:
       - most complete,
       - least self-contradictory,
       - easiest to understand.
  3. Treat that explanation as the candidate’s rule. 
  4. Treat the OUTPUT_GRID from any ANSWER as the candidate’s predicted
     test output (they are guaranteed identical within that candidate).

STEP 2 — EXAMPLE CONSISTENCY AUDIT (DO NOT USE THE TEST INPUT HERE)
For each candidate’s representative rule:

  1. Using only the training examples:
     For each TRAIN_EXAMPLE (in index order: 1, 2, 3, ...):

       - Check whether the described rule correctly explains the transformation
         from that example's INPUT_GRID to OUTPUT_GRID.

       - Be strict:
         * If the explanation states a universal rule that is clearly violated
           by any training example, mark that as a serious contradiction.
         * If the explanation fails to mention an obvious systematic behavior
           visible in multiple examples, note that as a weakness.

     For each training example, assign:
       - "Pass": fits the example with no obvious contradictions.
       - "Partial": roughly fits but has ambiguities or minor mismatches.
       - "Fail": clear contradiction with that example.

  2. Summarize, across all training examples:
     - How well does this rule fit the *set* of training examples taken together?
     - Does the rule feel overfitted, overcomplicated, or ad hoc?

STEP 3 — RULE-TO-TEST-GRID CONSISTENCY
For each candidate:

  1. Take its representative rule (from STEP 1).
  2. Apply the rule *conceptually* to the TEST_INPUT:
     - You do not need to compute a perfect output from scratch;
       focus on key structural consequences of the rule.
  3. Check whether the candidate’s test OUTPUT_GRID is a reasonable outcome
     of that rule.
     - If the grid blatantly violates the described rule, mark this as a
       contradiction.
     - If the grid is broadly consistent, mark it as plausible.

STEP 4 — SCORING AND GLOBAL RANKING

For each candidate, assign a numeric SCORE from 0 to 10:
  - 10: Rule is simple and coherent, strongly consistent with all training
        examples, and the test grid fits the rule.
  - 7–9: Mostly consistent with examples; minor ambiguities or small issues.
  - 4–6: Some consistency, but noticeable contradictions or hand-wavy parts.
  - 1–3: Major contradictions with examples or test grid; rule not credible.
  - 0: Completely incompatible with examples; or explanation is nonsense.

Then:
  - Rank all candidates in descending order of SCORE.
</INSTRUCTIONS>"""

PROMPT_CONSISTENCY_OUTPUT_FORMAT = """<OUTPUT_FORMAT>
Return a single JSON object with the following structure:

{
  "candidates": [
    {
      "candidate_id": 0,
      "score": 8.7,
      "example_audit": {
        "per_example": {
          "1": "Pass",
          "2": "Pass",
          "3": "Partial"
          /* add more keys if there are more training examples */
        },
        "summary": "Rule matches main behaviors across examples; minor ambiguity in example 3."
      },
      "test_grid_consistency": "Plausible",
      "rule_summary": "Short, 1–3 sentence description of this candidate's representative rule."
    },
    {
      "candidate_id": 1,
      "score": 6.0,
      "example_audit": {
        "per_example": {
          "1": "Partial",
          "2": "Fail"
        },
        "summary": "Contradiction in example 2; seems overfitted."
      },
      "test_grid_consistency": "Contradictory",
      "rule_summary": "..."
    }
    /* etc. for all candidates */
  ],
  "final_ranking_by_candidate": [
    0,
    4,
    5,
    1
    /* list all candidate_ids, ordered from most to least plausible */
  ]
}

Constraints:
- Do not add any fields outside this schema.
- All candidate_id values must match the id attributes in the <CANDIDATE> tags.
</OUTPUT_FORMAT>"""

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

def pick_solution_v2(candidates_object, reasoning_store, task, test_index, openai_client, anthropic_client, google_keys):
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

    # 2. Determine Attempt 1 (Consensus)
    # Sort by count desc
    candidates_by_consensus = sorted(candidates_list, key=lambda c: c['count'], reverse=True)
    attempt_1_candidate = candidates_by_consensus[0]
    print(f"[pick_solution_v2] Attempt 1 (Consensus): Candidate {attempt_1_candidate['id']} (Votes: {attempt_1_candidate['count']})")

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
        for j, model_id in enumerate(cand['models'][:3]):
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
        for j, model_id in enumerate(cand['models'][:3]):
            alias = chr(65 + j)
            cons_parts.append(f'    <EXPLANATION_{alias}>')
            reasoning = cand['reasoning'].get(model_id, "(Reasoning not found)")
            cons_parts.append(reasoning)
            cons_parts.append(f'    </EXPLANATION_{alias}>')
            cons_parts.append(f'    <TEST_OUTPUT_GRID>')
            cons_parts.append(grid_to_csv_rows(cand['grid']))
            cons_parts.append(f'    </TEST_OUTPUT_GRID>')
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

    # -- Logic Judge --
    print("\n[pick_solution_v2] Running Logic Judge (gemini-3-low)...")
    try:
        resp_logic = call_model(openai_client, anthropic_client, google_keys, full_prompt_logic, "gemini-3-low").text
        print("\n=== LOGIC JUDGE RESPONSE ===\n" + resp_logic + "\n============================")
        logic_data["response"] = resp_logic
        
        json_match = re.search(r"\{.*\}", resp_logic, re.DOTALL)
        if json_match:
            logic_json = json.loads(json_match.group(0))
            logic_data["parsed"] = logic_json
            if "candidates" in logic_json:
                for c in logic_json["candidates"]:
                    cid = c.get("candidate_id")
                    if cid in scores:
                        scores[cid] = max(scores[cid], c.get("score", 0))
    except Exception as e:
        print(f"[pick_solution_v2] Logic Judge Error: {e}")
        logic_data["error"] = str(e)

    # -- Consistency Judge --
    print("\n[pick_solution_v2] Running Consistency Judge (gemini-3-low)...")
    try:
        resp_cons = call_model(openai_client, anthropic_client, google_keys, full_prompt_cons, "gemini-3-low").text
        print("\n=== CONSISTENCY JUDGE RESPONSE ===\n" + resp_cons + "\n==================================")
        cons_data["response"] = resp_cons
        
        json_match = re.search(r"\{.*\}", resp_cons, re.DOTALL)
        if json_match:
            cons_json = json.loads(json_match.group(0))
            cons_data["parsed"] = cons_json
            if "candidates" in cons_json:
                for c in cons_json["candidates"]:
                    cid = c.get("candidate_id")
                    if cid in scores:
                        scores[cid] = max(scores[cid], c.get("score", 0))
    except Exception as e:
        print(f"[pick_solution_v2] Consistency Judge Error: {e}")
        cons_data["error"] = str(e)

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
