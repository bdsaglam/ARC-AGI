import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.logging import PrefixedStdout
from src.tasks import build_prompt
from src.image_generation import generate_and_save_image
from src.hint_generation import generate_hint
from src.parallel import run_models_in_parallel
from src.selection import is_solved
from src.solver.pipelines import run_objects_pipeline_variant

def run_step_1(state, models):
    state.set_status(step=1, phase="Shallow search")
    print(f"Broad search: {len(models)} left")
    step_1_log = {}
    if state.verbose >= 1:
        print(f"Running {len(models)} models...")
    prompt_step1 = build_prompt(state.task.train, state.test_example)
    results_step1 = run_models_in_parallel(models, state.run_id_counts, "step_1", prompt_step1, state.test_example, state.openai_client, state.anthropic_client, state.google_keys, state.verbose, run_timestamp=state.run_timestamp, task_id=state.task_id, test_index=state.test_index, completion_message="Broad search", use_background=state.openai_background)
    state.process_results(results_step1, step_1_log)
    state.log_step("step_1", step_1_log)

def run_step_3(state, models):
    state.set_status(step=3, phase="Extended search")
    print(f"Narrow search: {len(models)} left")
    step_3_log = {}
    if state.verbose >= 1:
        print(f"Running {len(models)} models...")
    prompt_step3 = build_prompt(state.task.train, state.test_example)
    results_step3 = run_models_in_parallel(models, state.run_id_counts, "step_3", prompt_step3, state.test_example, state.openai_client, state.anthropic_client, state.google_keys, state.verbose, run_timestamp=state.run_timestamp, task_id=state.task_id, test_index=state.test_index, completion_message="Narrow search", use_background=state.openai_background)
    state.process_results(results_step3, step_3_log)
    state.log_step("step_3", step_3_log)

def check_is_solved(state, step_name, force_finish=False, continue_if_solved=False):
    state.set_status(phase="Eval")
    solved = is_solved(state.candidates_object)
    log = {"candidates_object": {str(k): v for k, v in state.candidates_object.items()}, "is_solved": solved}
    state.log_step(step_name, log)

    if solved:
        if continue_if_solved:
             print("Likely solution, continuing")
             return False, True
        else:
            print("Likely solution, exiting")
            return True, True

    if force_finish:
        print("No solution, exiting (forced)")
        return True, True # (finish_now, success) - success status is determined by finalize
    
    print("No solution, continuing")
    return False, False

def run_step_5(state, models, hint_model, objects_only=False):
    state.set_status(step=5, phase="Full search")
    
    # Calculate tries
    if state.is_testing:
        gen_gemini = "gemini-3-low"
        gen_opus = "claude-opus-4.5-thinking-4000"
        gen_gpt = "gpt-5.1-none"
        unique_solvers = ["claude-opus-4.5-no-thinking", "gpt-5.1-none", "gemini-3-low"]
    else:
        gen_gemini = "gemini-3-high"
        gen_gpt = "gpt-5.2-xhigh"
        gen_opus = "claude-opus-4.5-thinking-60000"
        unique_solvers = ["claude-opus-4.5-thinking-60000", "gemini-3-high", "gpt-5.2-xhigh"]
        
    # Counters setup
    n_models = len(models)
    n_objects_models = len(unique_solvers)
    
    counters = {
        'deep': n_models,
        'image': n_models,
        'hint': n_models + 1,
        'objects': (n_objects_models + 2) * 1 # 1 variant * (Extract + Transform + Solvers)
    }
    
    if objects_only:
        counters['deep'] = 0
        counters['image'] = 0
        counters['hint'] = 0
        
    lock = threading.Lock()
    
    def update_progress(key):
        with lock:
            if key in counters:
                counters[key] -= 1
            # Deep/Image/Hint/Objects
            d = counters['deep']
            i = counters['image']
            h = counters['hint']
            o = counters['objects']
            print(f"Going DEEP: {d}/{i}/{h}/{o} left")

    # Initial Print
    d = counters['deep']
    i = counters['image']
    h = counters['hint']
    o = counters['objects']
    print(f"Going DEEP: {d}/{i}/{h}/{o} left")

    step_5_log = {"trigger-deep-thinking": {}, "image": {}, "generate-hint": {}, "objects_pipeline": {}}

    # Generate image once for both visual and hint steps to avoid matplotlib race conditions
    common_image_path = f"logs/{state.run_timestamp}_{state.task_id}_{state.test_index}_step_5_common.png"
    generate_and_save_image(state.task, common_image_path)

    def run_deep_thinking_step(on_complete=None):
        if state.verbose >= 1:
            print(f"Running {len(models)} models with deep thinking...")
        prompt_deep = build_prompt(state.task.train, state.test_example, trigger_deep_thinking=True)
        results_deep = run_models_in_parallel(models, state.run_id_counts, "step_5_deep_thinking", prompt_deep, state.test_example, state.openai_client, state.anthropic_client, state.google_keys, state.verbose, run_timestamp=state.run_timestamp, task_id=state.task_id, test_index=state.test_index, on_task_complete=on_complete, use_background=state.openai_background)
        return "trigger-deep-thinking", results_deep, None

    def run_image_step(img_path, on_complete=None):
        if state.verbose >= 1:
            print(f"Running {len(models)} models with image...")
        # Image is already generated
        prompt_image = build_prompt(state.task.train, state.test_example, image_path=img_path)
        results_image = run_models_in_parallel(models, state.run_id_counts, "step_5_image", prompt_image, state.test_example, state.openai_client, state.anthropic_client, state.google_keys, state.verbose, image_path=img_path, run_timestamp=state.run_timestamp, task_id=state.task_id, test_index=state.test_index, on_task_complete=on_complete, use_background=state.openai_background)
        return "image", results_image, None

    def run_hint_step(img_path, on_complete=None):
        if state.verbose >= 1:
            print(f"Running {len(models)} models with generated hint...")
        # Image is already generated
        hint_data = generate_hint(state.task, img_path, hint_model, state.verbose)
        extra_log = {}
        if hint_data and hint_data["hint"]:
            extra_log = {
                "model": hint_model,
                "Full raw LLM call": hint_data["prompt"],
                "Full raw LLM response": hint_data["full_response"],
                "Extracted hint": hint_data["hint"],
                "duration_seconds": round(hint_data.get("duration", 0), 2),
                "total_cost": hint_data.get("cost", 0),
                "input_tokens": hint_data.get("input_tokens", 0),
                "output_tokens": hint_data.get("output_tokens", 0),
                "cached_tokens": hint_data.get("cached_tokens", 0),
            }
            prompt_hint = build_prompt(state.task.train, state.test_example, strategy=hint_data["hint"])
            results_hint = run_models_in_parallel(models, state.run_id_counts, "step_5_generate_hint", prompt_hint, state.test_example, state.openai_client, state.anthropic_client, state.google_keys, state.verbose, run_timestamp=state.run_timestamp, task_id=state.task_id, test_index=state.test_index, on_task_complete=on_complete, use_background=state.openai_background)
            return "generate-hint", results_hint, extra_log
        
        # If no hint generated, manually drain counter
        if on_complete:
            for _ in range(len(models)):
                on_complete()
        return "generate-hint", [], extra_log

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        
        if objects_only:
             # Assuming pipelines.py updated or ignores extra arg. 
             # Actually I can't pass callback to run_objects_pipeline_variant easily without editing it.
             # I will skip passing it for objects for now, so objects counter won't decrement.
             # Wait, that breaks the "X left" logic.
             # I should just update pipelines.py next.
             # I'll pass it, assuming I will fix pipelines.py immediately after.
             futures.append(executor.submit(run_objects_pipeline_variant, state, gen_gpt, "gpt_gen", unique_solvers, lambda: update_progress('objects'), use_background=state.openai_background))
        else:
            futures = [
                executor.submit(run_deep_thinking_step, lambda: update_progress('deep')),
                executor.submit(run_image_step, common_image_path, lambda: update_progress('image')),
                executor.submit(run_hint_step, common_image_path, lambda: update_progress('hint')),
                executor.submit(run_objects_pipeline_variant, state, gen_gpt, "gpt_gen", unique_solvers, lambda: update_progress('objects'), use_background=state.openai_background),
            ]
            
        for future in as_completed(futures):
            step_name, results, extra_log = future.result()
            
            # Handle logging
            if step_name.startswith("objects_pipeline_"):
                # pipeline returns (name, results, log)
                state.process_results(results, step_5_log["objects_pipeline"])
                # extra_log here is the pipeline extraction/transform details
                step_5_log["objects_pipeline"][step_name.replace("objects_pipeline_", "")] = extra_log
            else:
                state.process_results(results, step_5_log[step_name])
                if step_name == "generate-hint" and extra_log:
                     step_5_log["generate-hint"]["hint_generation"] = extra_log

    state.log_step("step_5", step_5_log)
