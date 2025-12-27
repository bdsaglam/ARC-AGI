import re
import sys
import copy
import traceback
from src.augmentation import get_augmented_pairs

def sanitize_output(obj):
    """Recursively converts numpy types to standard Python types."""
    if isinstance(obj, list):
        return [sanitize_output(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_output(x) for x in obj)
    if isinstance(obj, dict):
        return {k: sanitize_output(v) for k, v in obj.items()}
    
    # Numpy checks (using string matching if numpy not imported, but here it is imported)
    try:
        import numpy as np
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return sanitize_output(obj.tolist())
    except ImportError:
        pass
        
    return obj

def extract_and_run_solver(llm_code: str, test_input_grid: list, train_examples: list = None, task_id: str = None, test_index: int = None) -> tuple[list | None, dict | None]:
    """
    Extracts Python code from LLM response, executes it, calls solver(), 
    and returns the predicted grid.
    
    If train_examples is provided, it verifies the solver against all training pairs first.
    Returns: (predicted_grid, verification_log)
    """
    verification_log = {"train_results": [], "status": "UNKNOWN"}
    
    # Prefix for logs
    log_prefix = f"[{task_id}:{test_index}]" if task_id else "[Unknown Task]"

    try:
        code = llm_code
        # Try to extract from markdown block
        pattern = r"```python(.*?)```"
        match = re.search(pattern, llm_code, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            # Heuristic: If no markdown, try to find the start of the function definition
            if "def solver" in llm_code:
                lines = llm_code.splitlines()
                start_idx = -1
                for i, line in enumerate(lines):
                    if "def solver" in line:
                        start_idx = i
                        break
                if start_idx != -1:
                    code = "\n".join(lines[start_idx:])

        # Inject common utilities into the execution scope
        import math
        import itertools
        from collections import Counter, deque, defaultdict
        from typing import List, Optional, Tuple, Any, Dict, Set
        
        # Safe imports for optional libraries
        try:
            import numpy as np
        except ImportError:
            np = None
            
        try:
            import scipy
            import scipy.ndimage
        except ImportError:
            scipy = None

        local_scope = {
            "np": np,
            "scipy": scipy,
            "Counter": Counter,
            "deque": deque,
            "defaultdict": defaultdict,
            "List": List,
            "Optional": Optional,
            "Tuple": Tuple,
            "Any": Any,
            "Dict": Dict,
            "Set": Set,
            "copy": copy.copy,
            "deepcopy": copy.deepcopy,
            "gcd": math.gcd,
            "math": math,
            "itertools": itertools,
            "Grid": List[List[int]]
        }

        try:
            # Execute the code definition using local_scope as globals
            exec(code, local_scope)
        except Exception as e:
            print(f"DEBUG {log_prefix}: Code Definition FAILED: {e}", file=sys.stderr)
            verification_log["status"] = "FAIL_DEFINE"
            verification_log["error"] = str(e)
            return None, verification_log
            
        if "solver" not in local_scope:
            print(f"DEBUG {log_prefix}: Solver function 'solver' not found in generated code.", file=sys.stderr)
            verification_log["status"] = "FAIL_NO_SOLVER"
            return None, verification_log
            
        solver_func = local_scope["solver"]
        if not callable(solver_func):
            print(f"DEBUG {log_prefix}: 'solver' was defined but is not callable.", file=sys.stderr)
            verification_log["status"] = "FAIL_SOLVER_NOT_CALLABLE"
            return None, verification_log

        # Verification Step
        if train_examples:
            all_passed = True
            first_fail_status = None
            first_fail_index = -1

            for i, ex in enumerate(train_examples):
                entry = {
                    "index": i, 
                    "status": "UNKNOWN",
                    "input": ex.input, 
                    "expected": ex.output,
                    "actual": None
                }
                try:
                    input_copy = copy.deepcopy(ex.input)
                    raw_res = solver_func(input_copy)
                    res = sanitize_output(raw_res)
                    entry["actual"] = res
                    
                    if res != ex.output:
                        entry["status"] = "FAIL"
                        all_passed = False
                        if first_fail_status is None:
                            first_fail_status = "FAIL_VERIFICATION"
                            first_fail_index = i
                    else:
                        entry["status"] = "PASS"
                        
                except Exception as e:
                    print(f"DEBUG {log_prefix}: Solver CRASHED on Train Example {i+1}: {e}", file=sys.stderr)
                    # traceback.print_exc(file=sys.stderr) # Reduce verbosity
                    entry["status"] = "CRASH"
                    entry["error"] = str(e)
                    all_passed = False
                    if first_fail_status is None:
                        first_fail_status = "FAIL_CRASH"
                        first_fail_index = i
                
                verification_log["train_results"].append(entry)
            
            if not all_passed:
                verification_log["status"] = first_fail_status
                verification_log["failed_example_index"] = first_fail_index
                return None, verification_log
            
            verification_log["status"] = "PASS"

            # --- Augmentation Verification (Soft Check) ---
            try:
                augmented_results = []
                counts = {"rotation": 0, "reflection": 0, "color": 0}
                passed = {"rotation": 0, "reflection": 0, "color": 0}
                
                for i, ex in enumerate(train_examples):
                    augmented_pairs = get_augmented_pairs(ex.input, ex.output)
                    
                    for pair in augmented_pairs:
                        aug_type = pair["type"]
                        aug_cat = "color" if "color" in aug_type else ("rotation" if "rotation" in aug_type else "reflection")
                        counts[aug_cat] += 1
                        
                        aug_entry = {
                            "original_index": i,
                            "type": aug_type,
                            "status": "UNKNOWN"
                        }
                        
                        try:
                            # Run solver on augmented input
                            raw_res_aug = solver_func(pair["input"])
                            res_aug = sanitize_output(raw_res_aug)
                            
                            if res_aug == pair["output"]:
                                aug_entry["status"] = "PASS"
                                passed[aug_cat] += 1
                            else:
                                aug_entry["status"] = "FAIL"
                                # Optional: Store actual vs expected if needed, but keeping it light
                        except Exception as e:
                            aug_entry["status"] = "CRASH"
                            aug_entry["error"] = str(e)
                        
                        augmented_results.append(aug_entry)
                
                # Calculate Rates
                stats = {}
                for cat in counts:
                    if counts[cat] > 0:
                        stats[f"{cat}_pass_rate"] = round(passed[cat] / counts[cat], 2)
                    else:
                        stats[f"{cat}_pass_rate"] = 0.0
                
                verification_log["augmented_stats"] = stats
                verification_log["augmented_results"] = augmented_results
                
            except Exception as e:
                # Augmentation logic itself shouldn't crash the whole run
                print(f"DEBUG {log_prefix}: Augmentation verification failed: {e}", file=sys.stderr)
                verification_log["augmented_error"] = str(e)
            
        try:
            raw_result = solver_func(test_input_grid)
            result = sanitize_output(raw_result)
            
            if isinstance(result, list):
                 if len(result) > 0 and isinstance(result[0], list):
                     return result, verification_log
                 if len(result) == 0:
                     return result, verification_log
            
            print(f"DEBUG {log_prefix}: Solver returned invalid type (not list of lists).", file=sys.stderr)
            verification_log["test_run_error"] = "Result validation failed (not list of lists)"
            return None, verification_log
        except Exception as e:
            print(f"DEBUG {log_prefix}: Solver CRASHED on Test Input: {e}", file=sys.stderr)
            verification_log["test_run_error"] = f"Test execution failed: {str(e)}"
            return None, verification_log

    except Exception as e:
        print(f"DEBUG {log_prefix}: Extractor System Crash: {e}", file=sys.stderr)
        verification_log["status"] = "FAIL_EXTRACTOR_CRASH"
        verification_log["error"] = f"{type(e).__name__}: {str(e)}"
        verification_log["traceback"] = traceback.format_exc()
        return None, verification_log