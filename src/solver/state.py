import time
from pathlib import Path
from openai import OpenAI
from anthropic import Anthropic

from src.config import get_api_keys, get_http_client
from src.tasks import load_task
from src.run_utils import find_task_path
from src.selection import pick_solution_v2, pick_solution
from src.reporting import print_solver_summary
from src.logging import setup_logging, write_step_log, PrefixedStdout

class SolverState:
    def __init__(self, task_id: str, test_index: int, verbose: int, is_testing: bool, run_timestamp: str, task_path: Path = None, answer_path: Path = None, judge_model: str = "gemini-3-high", old_pick_solution: bool = False, task_status=None):
        self.task_id = task_id
        self.test_index = test_index
        self.verbose = verbose
        self.is_testing = is_testing
        self.run_timestamp = run_timestamp
        self.answer_path = answer_path
        self.judge_model = judge_model
        self.old_pick_solution = old_pick_solution
        self.task_status = task_status if task_status is not None else {}
        self.task_status.setdefault('step', '0')
        self.task_status.setdefault('phase', 'Init')
        self.task_status.setdefault('start_time', time.time())
        
        setup_logging(verbose)
        
        self.start_time = time.time()
        self.total_cost = 0.0
        self.run_id_counts = {}
        self.candidates_object = {}
        self.reasoning_store = {}
        
        # Load Task
        if task_path is None:
            task_path = find_task_path(task_id)
        self.task_path = task_path
        
        # Initialize Clients
        openai_key, claude_key, google_keys = get_api_keys()
        self.http_client = get_http_client(timeout=3600.0)
        self.openai_client = OpenAI(api_key=openai_key, http_client=self.http_client) if openai_key else None
        self.anthropic_client = Anthropic(api_key=claude_key, http_client=self.http_client) if claude_key else None
        self.google_keys = google_keys
        
        self.task = load_task(task_path, answer_path=answer_path)
        
        test_idx = test_index - 1
        if test_idx < 0 or test_idx >= len(self.task.test):
            raise ValueError(f"Test index {test_index} is out of range.")
        self.test_example = self.task.test[test_idx]

    def set_status(self, step=None, phase=None):
        if step is not None:
            self.task_status['step'] = str(step)
        if phase is not None:
            self.task_status['phase'] = phase

    def close(self):
        self.http_client.close()

    def process_results(self, results, step_log):
        initial_solutions = len(self.candidates_object)
        for res in results:
            if res:
                self.total_cost += res.get("cost", 0)
                run_key = f"{res['run_id']}_{time.time()}"
                step_log[run_key] = {
                    "duration_seconds": round(res.get("duration", 0), 2),
                    "total_cost": res.get("cost", 0),
                    "input_tokens": res.get("input_tokens", 0),
                    "output_tokens": res.get("output_tokens", 0),
                    "cached_tokens": res.get("cached_tokens", 0),
                    "Full raw LLM call": res["prompt"],
                    "Full raw LLM response": res["full_response"],
                    "Extracted grid": res["grid"],
                }
                
                # Store reasoning for the Judge
                self.reasoning_store[res["run_id"]] = res["full_response"]
                
                if res["grid"] is not None:
                    grid_tuple = tuple(tuple(row) for row in res["grid"])
                    if grid_tuple not in self.candidates_object:
                        self.candidates_object[grid_tuple] = {"grid": res["grid"], "count": 0, "models": [], "is_correct": res["is_correct"]}
                    self.candidates_object[grid_tuple]["count"] += 1
                    self.candidates_object[grid_tuple]["models"].append(res["run_id"])
        
        new_solutions = len(self.candidates_object) - initial_solutions
        if self.verbose >= 1:
            print(f"Found {new_solutions} new unique solutions.")

    def log_step(self, step_name: str, data: dict):
        write_step_log(step_name, data, self.run_timestamp, self.task_id, self.test_index, self.verbose >= 2)

    def print_summary(self, outcome: str):
        duration = time.time() - self.start_time
        print_solver_summary(duration, self.total_cost, outcome)

    def finalize(self, step_log_name="step_finish"):
        # Check if we have ground truth
        has_ground_truth = self.test_example.output is not None
        
        if self.old_pick_solution:
            if self.verbose >= 1:
                print("\n[finalize] Using old pick_solution logic.")
            picked_solutions, result, selection_metadata = pick_solution(self.candidates_object, self.verbose)
        else:
            picked_solutions, result, selection_metadata = pick_solution_v2(
                self.candidates_object, 
                self.reasoning_store, 
                self.task, 
                self.test_index,
                            self.openai_client,
                            self.anthropic_client,
                            self.google_keys,
                            self.judge_model,
                            self.verbose
                        )        
        if not has_ground_truth:
            outcome = "SUBMITTED"
        else:
            outcome = "PASS" if result else "FAIL"
            
        finish_log = {
            "candidates_object": {str(k): v for k, v in self.candidates_object.items()},
            "selection_details": selection_metadata,
            "picked_solutions": picked_solutions,
            "correct_solution": self.test_example.output,
            "result": outcome
        }
        self.log_step(step_log_name, finish_log)
        
        self.set_status(phase="Finished")
        self.print_summary(outcome)
        
        self.close()
        return picked_solutions
