import sys
from typing import Optional

from google import genai
from google.genai import types

from src.types import ModelConfig, ModelResponse
from src.llm_utils import run_with_retry, orchestrate_two_stage

def call_gemini(
    client: genai.Client,
    prompt: str,
    config: ModelConfig,
    return_strategy: bool = False,
    verbose: bool = False,
) -> ModelResponse:
    
    model = config.base_model
    thinking_level = str(config.config)

    def _should_retry(e: Exception) -> bool:
        err_str = str(e)
        return "500" in err_str or "UNAVAILABLE" in err_str or "overloaded" in err_str.lower()

    level_enum = types.ThinkingLevel.LOW if thinking_level == "low" else types.ThinkingLevel.HIGH
    gen_config = types.GenerateContentConfig(
        temperature=1.0,
        max_output_tokens=65536,
        thinking_config={"include_thoughts": True, "thinking_level": level_enum}
    )

    # Shared chat object for state
    chat = client.chats.create(model=model, config=gen_config)

    def _solve(p: str) -> ModelResponse:
        response = run_with_retry(lambda: chat.send_message(p), _should_retry)
        
        try:
            text_parts = []
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.text: text_parts.append(part.text)
            
            usage = response.usage_metadata
            return ModelResponse(
                text="".join(text_parts).strip(),
                prompt_tokens=usage.prompt_token_count,
                cached_tokens=0,
                completion_tokens=usage.candidates_token_count,
            )
        except Exception as e:
             raise RuntimeError(f"Failed to parse Gemini response: {e} - Raw: {response}")

    def _explain(p: str, prev_resp: ModelResponse) -> Optional[ModelResponse]:
        try:
            # Chat object maintains history automatically
            response = run_with_retry(lambda: chat.send_message(p), _should_retry)
            
            text_parts = []
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.text: text_parts.append(part.text)
            
            usage = response.usage_metadata
            return ModelResponse(
                text="".join(text_parts).strip(),
                prompt_tokens=usage.prompt_token_count,
                cached_tokens=0,
                completion_tokens=usage.candidates_token_count,
            )
        except Exception as e:
            print(f"Step 2 strategy extraction failed: {e}", file=sys.stderr)
            return None

    return orchestrate_two_stage(_solve, _explain, prompt, return_strategy, verbose)
