import argparse
import sys
import os
import httpx
from openai import OpenAI
from anthropic import Anthropic
import certifi

from src.config import get_api_keys
from src.tasks import load_task, build_prompt, build_objects_extraction_prompt, build_objects_transformation_prompt
from src.run_utils import find_task_path
from src.parallel import run_single_model

def main():
    parser = argparse.ArgumentParser(description="Run a specific model with a custom prompt prefix on a task.")
    parser.add_argument("--model", type=str, default="claude-opus-4.5-no-thinking", help="Model to run.")
    parser.add_argument("--task", type=str, required=True, help="Task ID (e.g., 38007db0) or path to JSON file.")
    parser.add_argument("--test", type=int, default=1, help="Test case index (1-based, default: 1).")
    parser.add_argument("--prompt", type=str, default="", help="Custom prompt prefix to prepend to the standard task prompt.")
    parser.add_argument("--objects-extraction", action="store_true", help="Run a custom prompt to describe objects in the task (overrides standard prompt).")
    parser.add_argument("--objects-insertion", type=str, default=None, help="Text describing objects to insert into the standard prompt.")
    parser.add_argument("--objects-transformation", type=str, default=None, help="Run a custom prompt to describe transformations, appending this text.")
    parser.add_argument("--objects", action="store_true", help="Run a 3-step pipeline: Extraction -> Transformation -> Solution.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    
    args = parser.parse_args()

    # Setup clients
    openai_key, claude_key, google_keys = get_api_keys()
    
    os.environ["SSL_CERT_FILE"] = certifi.where()
    # Configure http client to match run.py settings
    http_client = httpx.Client(
        timeout=3600.0, 
        transport=httpx.HTTPTransport(retries=3, verify=False), 
        limits=httpx.Limits(keepalive_expiry=3600), 
        verify=False
    )
    
    anthropic_client = None
    if claude_key:
        anthropic_client = Anthropic(api_key=claude_key, http_client=http_client)
    elif "claude" in args.model:
        print("Warning: Anthropic API key not found, but a Claude model was requested.", file=sys.stderr)
    
    openai_client = None
    if openai_key:
        openai_client = OpenAI(api_key=openai_key, http_client=http_client)
    elif "gpt" in args.model:
        print("Warning: OpenAI API key not found, but a GPT model was requested.", file=sys.stderr)

    if not google_keys and "gemini" in args.model:
        print("Warning: Google API keys not found, but a Gemini model was requested.", file=sys.stderr)

    # Load Task
    try:
        task_path = find_task_path(args.task)
        task = load_task(task_path)
    except Exception as e:
        print(f"Error loading task: {e}", file=sys.stderr)
        http_client.close()
        sys.exit(1)

    # Validate test index
    test_idx = args.test - 1
    if test_idx < 0 or test_idx >= len(task.test):
        print(f"Error: Test index {args.test} is out of range. Task has {len(task.test)} test cases.", file=sys.stderr)
        http_client.close()
        sys.exit(1)
    
    test_example = task.test[test_idx]

    def _execute_run(prompt, step_name):
        print(f"\n=== RUNNING: {step_name} ===")
        print(f"Model: {args.model}")
        try:
            return run_single_model(
                model_name=args.model,
                run_id=f"manual_run_{step_name}",
                prompt=prompt,
                test_example=test_example,
                openai_client=openai_client,
                anthropic_client=anthropic_client,
                google_keys=google_keys,
                verbose=args.verbose
            )
        except Exception as e:
            print(f"Execution error in {step_name}: {e}")
            http_client.close()
            sys.exit(1)

    def _extract_tag_content(text, tag_name):
        import re
        pattern = f"<{tag_name}>(.*?)</{tag_name}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    if args.objects:
        # Step 1: Extraction
        print("Starting Step 1/3: Objects Extraction")
        extraction_prompt = build_objects_extraction_prompt(task.train, test_example)
        
        print("\n=== FULL PROMPT (STEP 1: EXTRACTION) ===")
        print(extraction_prompt)
        print("========================================")
        
        extraction_result = _execute_run(extraction_prompt, "extraction")
        extraction_full_text = extraction_result.get("full_response", "")
        
        print("\n=== RESPONSE (STEP 1: EXTRACTION) ===")
        print(extraction_full_text)
        print("=====================================")
        
        extraction_summary = _extract_tag_content(extraction_full_text, "objects_summary")
        if extraction_summary:
            print(f"Extracted <objects_summary>: Yes ({len(extraction_summary)} chars)")
            step1_output = extraction_summary
        else:
            print("Warning: <objects_summary> tags not found. Using full response.")
            step1_output = extraction_full_text
        
        print("Step 1 Complete.")

        # Step 2: Transformation
        print("\nStarting Step 2/3: Objects Transformation")
        transformation_prompt = build_objects_transformation_prompt(task.train, test_example, step1_output)
        
        print("\n=== FULL PROMPT (STEP 2: TRANSFORMATION) ===")
        print(transformation_prompt)
        print("============================================")
        
        transformation_result = _execute_run(transformation_prompt, "transformation")
        transformation_full_text = transformation_result.get("full_response", "")
        
        print("\n=== RESPONSE (STEP 2: TRANSFORMATION) ===")
        print(transformation_full_text)
        print("=========================================")
        
        transformation_summary = _extract_tag_content(transformation_full_text, "transformation_summary")
        if transformation_summary:
            print(f"Extracted <transformation_summary>: Yes ({len(transformation_summary)} chars)")
            step2_output = transformation_summary
        else:
            print("Warning: <transformation_summary> tags not found. Using full response.")
            step2_output = transformation_full_text
        
        print("Step 2 Complete.")

        # Step 3: Solution (Insertion)
        print("\nStarting Step 3/3: Final Solution")
        insertion_text = f"## Objects Description\n\n{step1_output}\n\n## Transformation Description\n\n{step2_output}"
        
        standard_prompt = build_prompt(task.train, test_example, objects_insertion=insertion_text)
        full_prompt = args.prompt
        if full_prompt:
            full_prompt += "\n\n" 
        full_prompt += standard_prompt
        
        print("\n=== FULL PROMPT (STEP 3: SOLUTION) ===")
        print(full_prompt)
        print("======================================")
        
        final_result = _execute_run(full_prompt, "solution")
        
        # Display Final Result
        print("\n=== RESPONSE (STEP 3: SOLUTION) ===")
        print(final_result.get("full_response", "(No response text)"))
        print("===================================")
        
        if final_result.get("grid"):
            print("Parsed Grid:")
            for row in final_result["grid"]:
                print(row)
            
            is_correct = final_result.get("is_correct")
            if is_correct is True:
                print("Result: PASS")
            elif is_correct is False:
                print("Result: FAIL")
            else:
                print("Result: UNKNOWN (No Ground Truth)")
        else:
            print("Failed to parse grid.")

    else:
        # Standard Single Run Logic
        if args.objects_extraction:
            final_prompt = build_objects_extraction_prompt(task.train, test_example)
        elif args.objects_transformation:
            final_prompt = build_objects_transformation_prompt(task.train, test_example, args.objects_transformation)
        else:
            standard_prompt = build_prompt(task.train, test_example, objects_insertion=args.objects_insertion)
            full_prompt = args.prompt
            if full_prompt:
                full_prompt += "\n\n" 
            full_prompt += standard_prompt
            final_prompt = full_prompt

        print("=== FULL PROMPT ===")
        print(final_prompt)
        print("===================")
        
        result = _execute_run(final_prompt, "main")

        print("\n=== RESPONSE ===")
        print(result.get("full_response", "(No response text)"))
        print("================")
        
        if not args.objects_extraction and not args.objects_transformation:
            if result.get("grid"):
                print("Parsed Grid:")
                for row in result["grid"]:
                    print(row)
                
                is_correct = result.get("is_correct")
                if is_correct is True:
                    print("Result: PASS")
                elif is_correct is False:
                    print("Result: FAIL")
                else:
                    print("Result: UNKNOWN (No Ground Truth)")
            else:
                print("Failed to parse grid.")

    http_client.close()

if __name__ == "__main__":
    main()