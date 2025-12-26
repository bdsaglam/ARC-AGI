import re
import sys
import copy

def extract_and_run_solver(llm_code: str, test_input_grid: list, train_examples: list = None) -> tuple[list | None, dict | None]:
    """
    Extracts Python code from LLM response, executes it, calls solver(), 
    and returns the predicted grid.
    
    If train_examples is provided, it verifies the solver against all training pairs first.
    Returns: (predicted_grid, verification_log)
    """
    verification_log = {"train_results": [], "status": "UNKNOWN"}
    
    code = llm_code
    # Try to extract from markdown block
    pattern = r"```python(.*?)```"
    match = re.search(pattern, llm_code, re.DOTALL)
    if match:
        code = match.group(1).strip()
    
    local_scope = {}
    try:
        # Execute the code definition
        exec(code, {}, local_scope)
    except Exception as e:
        verification_log["status"] = "FAIL_DEFINE"
        verification_log["error"] = str(e)
        return None, verification_log
        
    if "solver" not in local_scope:
        verification_log["status"] = "FAIL_NO_SOLVER"
        return None, verification_log
        
    solver_func = local_scope["solver"]
    if not callable(solver_func):
        verification_log["status"] = "FAIL_SOLVER_NOT_CALLABLE"
        return None, verification_log

    # Verification Step
    if train_examples:
        for i, ex in enumerate(train_examples):
            entry = {
                "index": i, 
                "status": "UNKNOWN",
                # Store as plain lists/data for JSON logging
                "input": ex.input, 
                "expected": ex.output,
                "actual": None
            }
            try:
                # Deepcopy input to prevent mutation side-effects affecting subsequent runs
                input_copy = copy.deepcopy(ex.input)
                
                # Run solver on training input
                res = solver_func(input_copy)
                entry["actual"] = res
                
                # Check against training output
                if res != ex.output:
                    entry["status"] = "FAIL"
                    verification_log["train_results"].append(entry)
                    verification_log["status"] = "FAIL_VERIFICATION"
                    verification_log["failed_example_index"] = i
                    return None, verification_log
                else:
                    entry["status"] = "PASS"
                    verification_log["train_results"].append(entry)
                    
            except Exception as e:
                entry["status"] = "CRASH"
                entry["error"] = str(e)
                verification_log["train_results"].append(entry)
                verification_log["status"] = "FAIL_CRASH"
                verification_log["failed_example_index"] = i
                return None, verification_log
        
        verification_log["status"] = "PASS"
        
    try:
        # Run the solver against the test input
        # Note: test_input_grid is already a list of lists (Python object)
        result = solver_func(test_input_grid)
        
        # Basic validation: must be list of lists
        if isinstance(result, list):
             # Ensure it's not a flat list? Or just robustly handle
             if len(result) > 0 and isinstance(result[0], list):
                 return result, verification_log
             # Handle empty grid case or other variations if needed
             if len(result) == 0:
                 return result, verification_log
        
        verification_log["test_run_error"] = "Result validation failed (not list of lists)"
        return None, verification_log
    except Exception as e:
        verification_log["test_run_error"] = f"Test execution failed: {str(e)}"
        return None, verification_log
