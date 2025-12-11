import sys
import base64
import mimetypes
from typing import Optional

import openai
from openai import OpenAI

from src.types import ModelConfig, ModelResponse
from src.llm_utils import run_with_retry, orchestrate_two_stage
from src.logging import get_logger
from src.errors import RetryableProviderError, NonRetryableProviderError, UnknownProviderError

logger = get_logger("providers.openai")

def call_openai_internal(
    client: OpenAI,
    prompt: str,
    config: ModelConfig,
    image_path: str = None,
    return_strategy: bool = False,
    verbose: bool = False,
    task_id: str = None,
    test_index: int = None,
) -> ModelResponse:
    
    model = config.base_model
    reasoning_effort = str(config.config) # Cast to string for safety

    def _map_exception(e: Exception):
        # 1. Known SDK Retryables
        if isinstance(e, (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)):
            raise RetryableProviderError(f"OpenAI Transient Error (Model: {model}): {e}") from e
        
        # 2. Known SDK Non-Retryables
        if isinstance(e, (openai.BadRequestError, openai.AuthenticationError, openai.PermissionDeniedError)):
            raise NonRetryableProviderError(f"OpenAI Fatal Error (Model: {model}): {e}") from e

        # 3. String matching for other transient network errors
        err_str = str(e)
        if (
            "Connection error" in err_str
            or "500" in err_str
            or "server_error" in err_str
            or "upstream connect error" in err_str
            or "timed out" in err_str
            or "Server disconnected" in err_str
            or "RemoteProtocolError" in err_str
            or "connection closed" in err_str.lower()
            or "peer closed connection" in err_str.lower()
            or "incomplete chunked read" in err_str.lower()
        ):
            raise RetryableProviderError(f"Network/Protocol Error (Model: {model}): {e}") from e

        # 4. True Unknowns -> Loud Retry
        raise UnknownProviderError(f"Unexpected OpenAI Error (Model: {model}): {e}") from e

    def _solve(p: str) -> ModelResponse:
        content = [{"type": "input_text", "text": p}]
        if image_path:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            mime_type, _ = mimetypes.guess_type(image_path)
            if mime_type is None:
                mime_type = 'application/octet-stream'
            content.append({
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{base64_image}",
            })

        kwargs = {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "timeout": 3600,
            "stream": True,
        }
        if reasoning_effort != "none":
            kwargs["reasoning"] = {"effort": reasoning_effort}

        def _call_and_accumulate():
            try:
                # 1. Create the stream
                stream = client.responses.create(**kwargs)
                collected_content = []
                usage_data = None
                response_id = None
                
                # 2. Consume the stream (protected by try/except)
                for chunk in stream:
                    chunk_type = getattr(chunk, "type", "")
                    
                    # 1. Capture Response ID
                    if chunk_type == "response.created":
                        if hasattr(chunk, "response") and hasattr(chunk.response, "id"):
                            response_id = chunk.response.id

                    # 2. Capture Text Delta
                    if chunk_type == "response.output_text.delta":
                        if hasattr(chunk, "delta") and chunk.delta:
                            collected_content.append(chunk.delta)
                    
                    # 3. Capture Usage
                    if chunk_type == "response.completed":
                        if hasattr(chunk, "response") and hasattr(chunk.response, "usage"):
                            usage_data = chunk.response.usage

                return {
                    "text": "".join(collected_content),
                    "usage": usage_data,
                    "id": response_id
                }

            except Exception as e:
                _map_exception(e)

        result = run_with_retry(
            lambda: _call_and_accumulate(),
            task_id=task_id,
            test_index=test_index
        )

        text_output = result["text"]
        if not text_output:
            # We construct a partial response string for debugging if empty
            raise RuntimeError(f"OpenAI Responses API Step 1 did not return text output. Result ID: {result.get('id')}")

        usage = result["usage"]
        # Store the raw response object for Step 2 linkage
        resp_obj = ModelResponse(
            text=text_output,
            prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            cached_tokens=0,
            completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            strategy=None
        )
        # Mock the raw response object to support .id access in _explain
        class MockRawResponse:
            def __init__(self, rid):
                self.id = rid
        
        resp_obj._raw_response = MockRawResponse(result["id"]) # Private attribute for state passing
        return resp_obj

    def _explain(p: str, prev_resp: ModelResponse) -> Optional[ModelResponse]:
        try:
            kwargs = {
                "model": model,
                "previous_response_id": prev_resp._raw_response.id,
                "input": [{"role": "user", "content": p}],
                "timeout": 3600
            }
            
            def _create_safe():
                try:
                    return client.responses.create(**kwargs)
                except Exception as e:
                    _map_exception(e)

            # We don't necessarily need retry on the explain step to be as noisy, but good to have
            response = run_with_retry(
                lambda: _create_safe(),
                task_id=task_id,
                test_index=test_index
            )
            
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
            logger.error(f"Step 2 strategy extraction failed: {e}")
            return None

    return orchestrate_two_stage(_solve, _explain, prompt, return_strategy, verbose, image_path)