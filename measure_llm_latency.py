import argparse
import sys
import time
from pathlib import Path

# Add the current directory to sys.path so we can import from src
sys.path.append(str(Path(__file__).parent))

from src.solver.state import SolverState
from src.tasks import build_prompt
from src.parallel import run_single_model
from src.run_utils import find_task_path

def main():
    parser = argparse.ArgumentParser(description="Measure LLM latency for Step 1 prompt.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gpt-5.1-high)")
    parser.add_argument("--task", type=str, default="cb2d8a2c", help="Task ID (default: cb2d8a2c)")
    parser.add_argument("--test", type=int, default=1, help="Test index (default: 1)")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument("--openai-background", action="store_true", help="Use OpenAI Background submission mode")
    
    args = parser.parse_args()

    try:
        # Initialize SolverState to handle setup (clients, task loading, etc.)
        # We dummy some values that aren't critical for this single test
        print(f"Initializing SolverState for task {args.task}...")
        state = SolverState(
            task_id=args.task,
            test_index=args.test,
            verbose=args.verbose,
            is_testing=True, # Avoid huge initializations if any specific to solver mode
            run_timestamp="latency_test",
        )
        
        # Build Step 1 Prompt
        print("Building Step 1 prompt...")
        prompt = build_prompt(state.task.train, state.test_example)
        
        if args.verbose >= 2:
            print("\n--- Prompt Preview (first 500 chars) ---")
            print(prompt[:500])
            print("...\n----------------------------------------\n")

        print(f"Running model {args.model}...")
        print(f"Mode: {'Background (Async)' if args.openai_background else 'Standard (Sync)'}")
        
        # Measure Wall Clock Time
        start_time = time.time()

        # Start heartbeat thread
        stop_heartbeat = False
        def heartbeat():
            while not stop_heartbeat:
                time.sleep(60)
                if not stop_heartbeat:
                    elapsed = int(time.time() - start_time)
                    print(f"[Heartbeat] {elapsed}s elapsed - Still waiting for response...", file=sys.stderr)

        import threading
        t = threading.Thread(target=heartbeat, daemon=True)
        t.start()

        try:
            # Run the model
            result = run_single_model(
                model_name=args.model,
                run_id="latency_test_run",
                prompt=prompt,
                test_example=state.test_example,
                openai_client=state.openai_client,
                anthropic_client=state.anthropic_client,
                google_keys=state.google_keys,
                verbose=args.verbose,
                task_id=state.task_id,
                test_index=state.test_index,
                use_background=args.openai_background
            )
        finally:
            stop_heartbeat = True
            t.join(timeout=1)
        
        end_time = time.time()
        wall_latency = end_time - start_time

        # Output Results
        print("\n=== Results ===")
        print(f"Model: {args.model}")
        print(f"Task: {args.task}")
        print(f"Reported Duration (API+Net): {result.get('duration', 0):.4f}s")
        print(f"Wall Clock Latency: {wall_latency:.4f}s")
        print(f"Cost: ${result.get('cost', 0):.6f}")
        print(f"Input Tokens: {result.get('input_tokens', 0)}")
        print(f"Output Tokens: {result.get('output_tokens', 0)}")
        print(f"Cached Tokens: {result.get('cached_tokens', 0)}")
        print(f"Correct: {result.get('is_correct')}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
