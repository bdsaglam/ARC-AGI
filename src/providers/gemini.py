import sys
import warnings
from typing import Optional
import PIL.Image

from google import genai
from google.genai import types

from src.types import ModelConfig, ModelResponse
from src.llm_utils import run_with_retry, orchestrate_two_stage
from src.logging import get_logger

logger = get_logger("providers.gemini")

def call_gemini(
    client: genai.Client,
    prompt: str,
    config: ModelConfig,
    image_path: str = None,
    return_strategy: bool = False,
    verbose: bool = False,
) -> ModelResponse:
    
    model = config.base_model
    thinking_level = str(config.config)

    def _should_retry(e: Exception) -> bool:
        err_str = str(e)
        return (
            "500" in err_str 
            or "UNAVAILABLE" in err_str 
            or "overloaded" in err_str.lower()
            or "Server disconnected" in err_str
            or "RemoteProtocolError" in err_str
            or "connection closed" in err_str.lower()
            or "peer closed connection" in err_str.lower()
            or "incomplete chunked read" in err_str.lower()
        )

    # Use thinking_level with string literals "LOW" or "HIGH" (case insensitive usually, but standard is upper/lower matching the enum)
    # Typically the SDK accepts "low" / "high" strings for this field if typed as ThinkingLevel
    level_val = "low" if thinking_level == "low" else "high"
    
    gen_config = types.GenerateContentConfig(
        temperature=1.0,
        max_output_tokens=65536,
        thinking_config=types.ThinkingConfig(
            include_thoughts=True, 
            thinking_level=level_val
        )
    )

    # Shared chat object for state
    chat = client.chats.create(model=model, config=gen_config)

    def _run_chat(message):
        # Suppress Pydantic serialization warnings from the SDK
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*Pydantic serializer warnings.*")
            return chat.send_message(message)

    def _solve(p: str) -> ModelResponse:
        # Pass raw string to avoid Pydantic warnings; SDK handles wrapping
        message = [p]
        if image_path:
            img = PIL.Image.open(image_path)
            message.append(img)

        response = run_with_retry(lambda: _run_chat(message), _should_retry)
        
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
            message = p
            response = run_with_retry(lambda: _run_chat(message), _should_retry)
            
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
            logger.error(f"Step 2 strategy extraction failed: {e}")
            return None

    return orchestrate_two_stage(_solve, _explain, prompt, return_strategy, verbose, image_path)