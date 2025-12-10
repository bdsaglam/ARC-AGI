import json
import re
import time
from src.models import call_model, calculate_cost, parse_model_arg

def extract_json(text):
    """
    Robustly extract a JSON object from text.
    Prioritizes objects containing 'candidates' key.
    """
    if not text:
        return None
    text = text.strip()
    
    # 1. Try to find JSON block within markdown fences
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict) and "candidates" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    # 2. Scan for any '{' and try to decode a valid JSON object
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        start_idx = match.start()
        try:
            # raw_decode parsing stops at the end of the valid object
            obj, _ = decoder.raw_decode(text, idx=start_idx)
            if isinstance(obj, dict) and "candidates" in obj:
                return obj
        except json.JSONDecodeError:
            continue
            
    return None

def run_judge(judge_name, prompt, judge_model, openai_client, anthropic_client, google_keys, result_container, verbose: int = 0):
    if verbose >= 1:
        print(f"\n[pick_solution_v2] Running {judge_name} Judge ({judge_model})...")
    
    try:
        start_ts = time.perf_counter()
        response_obj = call_model(openai_client, anthropic_client, google_keys, prompt, judge_model)
        duration = time.perf_counter() - start_ts
        
        result_container["response"] = response_obj.text
        
        # Calculate cost
        cost = 0.0
        try:
            model_config = parse_model_arg(judge_model)
            cost = calculate_cost(model_config, response_obj)
        except Exception:
            pass
        
        # Capture metrics
        result_container["model"] = judge_model
        result_container["duration_seconds"] = round(duration, 2)
        result_container["total_cost"] = cost
        result_container["input_tokens"] = response_obj.prompt_tokens
        result_container["output_tokens"] = response_obj.completion_tokens
        result_container["cached_tokens"] = response_obj.cached_tokens
        
        parsed_json = extract_json(response_obj.text)
        
        if parsed_json:
            result_container["parsed"] = parsed_json
            return parsed_json
        else:
            print(f"[pick_solution_v2] {judge_name} Judge: Could not parse JSON. Response start: {response_obj.text[:500]}")
            
    except Exception as e:
        print(f"[pick_solution_v2] {judge_name} Judge Error: {e}")
        result_container["error"] = str(e)
    return None