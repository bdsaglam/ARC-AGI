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
    print("Broad search")
    state.reporter.emit("RUNNING", "(Shallow search)", event="STEP_CHANGE")
    step_1_log = {}
    if state.verbose >= 1:
        print(f"Running {len(models)} models...")
    prompt_step1 = build_prompt(state.task.train, state.test_example)
    results_step1 = run_models_in_parallel(models, state.run_id_counts, "step_1", prompt_step1, state.test_example, state.openai_client, state.anthropic_client, state.google_keys, state.verbose, run_timestamp=state.run_timestamp, progress_queue=state.progress_queue, task_id=state.task_id, test_index=state.test_index)
    state.process_results(results_step1, step_1_log)
    state.log_step("step_1", step_1_log)

def run_step_3(state, models):
    state.set_status(step=3, phase="Extended search")
    print("Narrowing in...")
    state.reporter.emit("RUNNING", "(Extended search)", event="STEP_CHANGE")
    step_3_log = {}
    if state.verbose >= 1:
        print(f"Running {len(models)} models...")
    prompt_step3 = build_prompt(state.task.train, state.test_example)
    results_step3 = run_models_in_parallel(models, state.run_id_counts, "step_3", prompt_step3, state.test_example, state.openai_client, state.anthropic_client, state.google_keys, state.verbose, run_timestamp=state.run_timestamp, progress_queue=state.progress_queue, task_id=state.task_id, test_index=state.test_index)
    state.process_results(results_step3, step_3_log)
    state.log_step("step_3", step_3_log)

def check_is_solved(state, step_name, force_finish=False, continue_if_solved=False):
    state.set_status(phase="Eval")
    state.reporter.emit("RUNNING", f"(Evaluation)", event="STEP_CHANGE")
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
    print("Going deep...")
    state.reporter.emit("RUNNING", "(Full search)", event="STEP_CHANGE")
    step_5_log = {"trigger-deep-thinking": {}, "image": {}, "generate-hint": {}, "objects_pipeline": {}}

    # Generate image once for both visual and hint steps to avoid matplotlib race conditions
    common_image_path = f"logs/{state.run_timestamp}_{state.task_id}_{state.test_index}_step_5_common.png"
    generate_and_save_image(state.task, common_image_path)

    def run_deep_thinking_step():
        if state.verbose >= 1:
            print(f"Running {len(models)} models with deep thinking...")
        prompt_deep = build_prompt(state.task.train, state.test_example, trigger_deep_thinking=True)
        results_deep = run_models_in_parallel(models, state.run_id_counts, "step_5_deep_thinking", prompt_deep, state.test_example, state.openai_client, state.anthropic_client, state.google_keys, state.verbose, run_timestamp=state.run_timestamp, progress_queue=state.progress_queue, task_id=state.task_id, test_index=state.test_index)
        return "trigger-deep-thinking", results_deep, None

    def run_image_step(img_path):
        if state.verbose >= 1:
            print(f"Running {len(models)} models with image...")
        # Image is already generated
        prompt_image = build_prompt(state.task.train, state.test_example, image_path=img_path)
        results_image = run_models_in_parallel(models, state.run_id_counts, "step_5_image", prompt_image, state.test_example, state.openai_client, state.anthropic_client, state.google_keys, state.verbose, image_path=img_path, run_timestamp=state.run_timestamp, progress_queue=state.progress_queue, task_id=state.task_id, test_index=state.test_index)
        return "image", results_image, None

    def run_hint_step(img_path):
        if state.verbose >= 1:
            print(f"Running {len(models)} models with generated hint...")
        # Image is already generated
        hint_data = generate_hint(state.task, img_path, hint_model, state.verbose)
        extra_log = {}
        if hint_data and hint_data["hint"]:
            extra_log = {
                "Full raw LLM call": hint_data["prompt"],
                "Full raw LLM response": hint_data["full_response"],
                "Extracted hint": hint_data["hint"],
            }
            prompt_hint = build_prompt(state.task.train, state.test_example, strategy=hint_data["hint"])
            results_hint = run_models_in_parallel(models, state.run_id_counts, "step_5_generate_hint", prompt_hint, state.test_example, state.openai_client, state.anthropic_client, state.google_keys, state.verbose, run_timestamp=state.run_timestamp, progress_queue=state.progress_queue, task_id=state.task_id, test_index=state.test_index)
            return "generate-hint", results_hint, extra_log
        return "generate-hint", [], extra_log

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        if state.is_testing:
            # Testing models
            gen_gemini = "gemini-3-low"
            gen_opus = "claude-opus-4.5-thinking-4000"
            unique_solvers = ["claude-opus-4.5-no-thinking", "gpt-5.1-none", "gemini-3-low"]
        else:
            # Production models
            gen_gemini = "gemini-3-high"
            gen_opus = "claude-opus-4.5-thinking-60000"
            unique_solvers = ["claude-opus-4.5-thinking-60000", "gpt-5.1-high", "gemini-3-high"]
            
        if objects_only:
             futures.append(executor.submit(run_objects_pipeline_variant, state, gen_gemini, "gemini_gen", unique_solvers))
             futures.append(executor.submit(run_objects_pipeline_variant, state, gen_opus, "opus_gen", unique_solvers))
        else:
            futures = [
                executor.submit(run_deep_thinking_step),
                executor.submit(run_image_step, common_image_path),
                executor.submit(run_hint_step, common_image_path),
                executor.submit(run_objects_pipeline_variant, state, gen_gemini, "gemini_gen", unique_solvers),
                executor.submit(run_objects_pipeline_variant, state, gen_opus, "opus_gen", unique_solvers)
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
