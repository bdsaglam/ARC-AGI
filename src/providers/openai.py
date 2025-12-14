import sys
import base64
import mimetypes
from typing import Optional

import openai
from openai import OpenAI
from anthropic import Anthropic

from src.types import ModelConfig, ModelResponse, CLAUDE_OPUS_BASE
from src.llm_utils import run_with_retry, orchestrate_two_stage
from src.logging import get_logger, log_failure
from src.errors import RetryableProviderError, NonRetryableProviderError, UnknownProviderError
from src.providers.anthropic import call_anthropic

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
    step_name: str = None,
    use_background: bool = False,
    run_timestamp: str = None,
    anthropic_client: Anthropic = None,
) -> ModelResponse:
    
    model = config.base_model
    reasoning_effort = str(config.config) # Cast to string for safety
    full_model_name = f"{model}-{reasoning_effort}"
    last_failed_job_id = None
    is_downgraded_retry = False

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

    def _solve_background(p: str) -> ModelResponse:
        import time
        import random
        nonlocal reasoning_effort, last_failed_job_id, is_downgraded_retry

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
            "timeout": 60, # Timeout for the submit request itself
            "background": True,
            "store": True,
            "max_output_tokens": 120000,
        }
        if reasoning_effort != "none":
            kwargs["reasoning"] = {"effort": reasoning_effort}
        
        if last_failed_job_id:
            kwargs["previous_response_id"] = last_failed_job_id

        # 1. Submit Job
        def _submit():
            try:
                return client.responses.create(**kwargs)
            except Exception as e:
                _map_exception(e)
        
        job = run_with_retry(lambda: _submit(), task_id=task_id, test_index=test_index, run_timestamp=run_timestamp, model_name=full_model_name)
        job_id = job.id
        if verbose:
            print(f"[BACKGROUND] [{model}] Job submitted. ID: {job_id}")

        # 2. Poll for Completion
        max_wait_time = 60  # 1 minute (TEMPORARY)
        start_time = time.time()
        poll_interval_base = 2.0
        last_log_time = time.time()

        while True:
            # Check Timeout
            elapsed = time.time() - start_time
            context_str = f"[{task_id}:{test_index}] ({step_name})" if task_id and step_name else ""
            if elapsed > max_wait_time:
                if reasoning_effort == "xhigh":
                    logger.warning(f"[BACKGROUND] {context_str} OpenAI Job {job_id} timed out after {max_wait_time}s. Falling back to Claude Opus...")
                    
                    if run_timestamp:
                        log_failure(
                            run_timestamp=run_timestamp,
                            task_id=task_id if task_id else "UNKNOWN",
                            run_id="OPENAI_BG_TIMEOUT",
                            error=RetryableProviderError(f"OpenAI Job {job_id} timed out. Falling back to Claude Opus..."),
                            model=f"{model}-{reasoning_effort}",
                            step=step_name if step_name else (task_id if task_id else "UNKNOWN"),
                            test_index=test_index,
                            is_retryable=True
                        )

                    if not anthropic_client:
                        raise NonRetryableProviderError("Fallback to Claude Opus required but anthropic_client is missing.")
                    
                    fallback_config = ModelConfig("anthropic", CLAUDE_OPUS_BASE, 60000)
                    response = call_anthropic(
                        anthropic_client,
                        prompt,
                        fallback_config,
                        image_path=image_path,
                        return_strategy=False,
                        verbose=verbose,
                        task_id=task_id,
                        test_index=test_index,
                        run_timestamp=run_timestamp
                    )
                    response.model_name = "claude-opus-4.5-thinking-60000"
                    return response

                elif reasoning_effort == "low":
                    logger.warning(f"[BACKGROUND] {context_str} OpenAI Job {job_id} (low) timed out after {max_wait_time}s. Falling back to Claude Opus (no-thinking)...")
                    
                    if run_timestamp:
                        log_failure(
                            run_timestamp=run_timestamp,
                            task_id=task_id if task_id else "UNKNOWN",
                            run_id="OPENAI_BG_TIMEOUT",
                            error=RetryableProviderError(f"OpenAI Job {job_id} timed out. Falling back to Claude Opus..."),
                            model=f"{model}-{reasoning_effort}",
                            step=step_name if step_name else (task_id if task_id else "UNKNOWN"),
                            test_index=test_index,
                            is_retryable=True
                        )

                    if not anthropic_client:
                        raise NonRetryableProviderError("Fallback to Claude Opus required but anthropic_client is missing.")
                    
                    fallback_config = ModelConfig("anthropic", CLAUDE_OPUS_BASE, 0)
                    response = call_anthropic(
                        anthropic_client,
                        prompt,
                        fallback_config,
                        image_path=image_path,
                        return_strategy=False,
                        verbose=verbose,
                        task_id=task_id,
                        test_index=test_index,
                        run_timestamp=run_timestamp
                    )
                    response.model_name = "claude-opus-4.5-no-thinking"
                    return response
                
                if is_downgraded_retry:
                    raise NonRetryableProviderError(f"OpenAI Background Job {job_id} timed out after {max_wait_time}s (Downgraded Retry Failed)")

                raise RetryableProviderError(f"OpenAI Background Job {job_id} timed out after {max_wait_time}s")

            # Logging every ~30s
            if verbose and (time.time() - last_log_time > 30):
                print(f"[BACKGROUND] [{model}] Job {job_id} still processing... ({int(elapsed)}s elapsed)")
                last_log_time = time.time()

            # Retrieve Status
            def _retrieve():
                try:
                    return client.responses.retrieve(job_id)
                except Exception as e:
                    _map_exception(e)

            job = run_with_retry(lambda: _retrieve(), task_id=task_id, test_index=test_index, run_timestamp=run_timestamp, model_name=full_model_name)

            if job.status in ("queued", "in_progress"):
                # Sleep with jitter
                sleep_time = poll_interval_base + random.uniform(0, 1.0)
                time.sleep(sleep_time)
                continue
            
            # Terminal States
            if job.status == "completed":
                text_output = ""
                # Try to use convenience field first
                if hasattr(job, "output_text") and job.output_text:
                    text_output = job.output_text
                # Fallback to manual extraction if needed
                elif hasattr(job, "output"):
                     for item in job.output:
                        if item.type == "message":
                            for content_part in item.content:
                                if content_part.type == "output_text":
                                    text_output += content_part.text
                
                usage = getattr(job, "usage", None)
                return ModelResponse(
                    text=text_output,
                    prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
                    cached_tokens=0,
                    completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
                    strategy=None
                )
            
            elif job.status == "failed":
                err_msg = f"Code: {job.error.code}, Message: {job.error.message}" if job.error else "Unknown error"
                raise RetryableProviderError(f"OpenAI Background Job {job_id} FAILED: {err_msg}")
            
            elif job.status in ("cancelled", "incomplete"):
                 reason = getattr(job, 'incomplete_details', 'Unknown')
                 reason_str = str(reason)
                 if "max_output_tokens" in reason_str or "token_limit" in reason_str:
                     context_str = f"[{task_id}:{test_index}] ({step_name})" if task_id and step_name else ""
                     if reasoning_effort == "xhigh":
                         logger.warning(f"[BACKGROUND] {context_str} OpenAI Job {job_id} hit token limit: {reason}. Falling back to Claude Opus...")
                         
                         if run_timestamp:
                             log_failure(
                                run_timestamp=run_timestamp,
                                task_id=task_id if task_id else "UNKNOWN",
                                run_id="OPENAI_BG_TOKEN_LIMIT",
                                error=RetryableProviderError(f"OpenAI Job {job_id} hit token limit: {reason}. Falling back to Claude Opus..."),
                                model=f"{model}-{reasoning_effort}",
                                step=step_name if step_name else (task_id if task_id else "UNKNOWN"),
                                test_index=test_index,
                                is_retryable=True
                             )

                         if not anthropic_client:
                             raise NonRetryableProviderError("Fallback to Claude Opus required but anthropic_client is missing.")

                         fallback_config = ModelConfig("anthropic", CLAUDE_OPUS_BASE, 60000)
                         response = call_anthropic(
                             anthropic_client,
                             prompt,
                             fallback_config,
                             image_path=image_path,
                             return_strategy=False,
                             verbose=verbose,
                             task_id=task_id,
                             test_index=test_index,
                             run_timestamp=run_timestamp
                         )
                         response.model_name = "claude-opus-4.5-thinking-60000"
                         return response

                     elif reasoning_effort == "low":
                         logger.warning(f"[BACKGROUND] {context_str} OpenAI Job {job_id} hit token limit: {reason}. Falling back to Claude Opus (no-thinking)...")
                         
                         if run_timestamp:
                             log_failure(
                                run_timestamp=run_timestamp,
                                task_id=task_id if task_id else "UNKNOWN",
                                run_id="OPENAI_BG_TOKEN_LIMIT",
                                error=RetryableProviderError(f"OpenAI Job {job_id} hit token limit: {reason}. Falling back to Claude Opus..."),
                                model=f"{model}-{reasoning_effort}",
                                step=step_name if step_name else (task_id if task_id else "UNKNOWN"),
                                test_index=test_index,
                                is_retryable=True
                             )

                         if not anthropic_client:
                             raise NonRetryableProviderError("Fallback to Claude Opus required but anthropic_client is missing.")

                         fallback_config = ModelConfig("anthropic", CLAUDE_OPUS_BASE, 0)
                         response = call_anthropic(
                             anthropic_client,
                             prompt,
                             fallback_config,
                             image_path=image_path,
                             return_strategy=False,
                             verbose=verbose,
                             task_id=task_id,
                             test_index=test_index,
                             run_timestamp=run_timestamp
                         )
                         response.model_name = "claude-opus-4.5-no-thinking"
                         return response
                 
                 raise NonRetryableProviderError(f"OpenAI Background Job {job_id} ended with status={job.status}, reason={reason}")
            
            else:
                raise UnknownProviderError(f"OpenAI Background Job {job_id} ended in unexpected status={job.status}")

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
            test_index=test_index,
            run_timestamp=run_timestamp,
            model_name=full_model_name
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
                test_index=test_index,
                run_timestamp=run_timestamp,
                model_name=full_model_name
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

    if use_background:
        # Background mode implies solving only (strategy extraction in background mode is not yet implemented/needed)
        # We wrap it in run_with_retry at the job level, so if the job FAILS (RetryableProviderError), we submit a new one.
        return run_with_retry(
            lambda: _solve_background(prompt),
            task_id=task_id,
            test_index=test_index,
            run_timestamp=run_timestamp,
            model_name=full_model_name
        )

    return orchestrate_two_stage(_solve, _explain, prompt, return_strategy, verbose, image_path)