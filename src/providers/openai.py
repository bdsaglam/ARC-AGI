import sys
from typing import Optional

from openai import OpenAI

from src.types import ModelConfig, ModelResponse
from src.llm_utils import run_with_retry, orchestrate_two_stage

def call_openai_internal(
    client: OpenAI,
    prompt: str,
    config: ModelConfig,
    return_strategy: bool = False,
    verbose: bool = False,
) -> ModelResponse:
    
    model = config.base_model
    reasoning_effort = str(config.config) # Cast to string for safety

    def _should_retry(e: Exception) -> bool:
        err_str = str(e)
        return (
            "Connection error" in err_str
            or "500" in err_str
            or "server_error" in err_str
            or "upstream connect error" in err_str
            or "timed out" in err_str
        )

    def _solve(p: str) -> ModelResponse:
        kwargs = {
            "model": model,
            "input": [{"role": "user", "content": p}],
            "timeout": 3600,
        }
        if reasoning_effort != "none":
            kwargs["reasoning"] = {"effort": reasoning_effort}

        response = run_with_retry(lambda: client.responses.create(**kwargs), _should_retry)

        text_output = ""
        if hasattr(response, "output"):
            for item in response.output:
                if item.type == "message":
                    for content_part in item.content:
                        if content_part.type == "output_text":
                            text_output += content_part.text
        if not text_output:
            raise RuntimeError(f"OpenAI Responses API Step 1 did not return text output. Response: {response}")

        usage = getattr(response, "usage", None)
        # Store the raw response object for Step 2 linkage
        resp_obj = ModelResponse(
            text=text_output,
            prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            cached_tokens=0,
            completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            strategy=None
        )
        resp_obj._raw_response = response # Private attribute for state passing
        return resp_obj

    def _explain(p: str, prev_resp: ModelResponse) -> Optional[ModelResponse]:
        try:
            kwargs = {
                "model": model,
                "previous_response_id": prev_resp._raw_response.id,
                "input": [{"role": "user", "content": p}],
                "timeout": 3600
            }
            response = run_with_retry(lambda: client.responses.create(**kwargs), _should_retry)
            
            text_output = ""
            if hasattr(response, "output"):
                for item in response.output:
                    if item.type == "message":
                        for content_part in item.content:
                            if content_part.type == "output_text":
                                text_output += content_part.text
            
            usage = getattr(response, "usage", None)
            return ModelResponse(
                text=text_output,
                prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
                cached_tokens=0,
                completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            )
        except Exception as e:
            print(f"Step 2 strategy extraction failed: {e}", file=sys.stderr)
            return None

    return orchestrate_two_stage(_solve, _explain, prompt, return_strategy, verbose)
