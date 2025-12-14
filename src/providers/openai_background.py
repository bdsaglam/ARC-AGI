import time
import random
from typing import Optional, TYPE_CHECKING

from src.types import ModelConfig, ModelResponse, CLAUDE_OPUS_BASE
from src.llm_utils import run_with_retry
from src.logging import get_logger, log_failure
from src.errors import RetryableProviderError, NonRetryableProviderError, UnknownProviderError
from src.providers.anthropic import call_anthropic
from src.providers.openai_utils import _map_openai_exception

if TYPE_CHECKING:
    from src.providers.openai_runner import OpenAIRequestRunner

logger = get_logger("providers.openai")

class OpenAIBackgroundSolver:
    """Handles the lifecycle of OpenAI Background (Batch/Async) jobs."""

    def __init__(self, runner: 'OpenAIRequestRunner'):
        self.runner = runner
        self.client = runner.client
        self.verbose = runner.verbose

    def _fallback_to_claude(
        self, 
        prompt: str, 
        image_path: Optional[str], 
        reason: str, 
        start_ts: float,
        thinking: bool
    ) -> ModelResponse:
        """Handles fallback to Claude Opus when OpenAI jobs fail/timeout."""
        if not self.runner.anthropic_client:
            raise NonRetryableProviderError("Fallback to Claude Opus required but anthropic_client is missing.")

        model_suffix = "thinking-60000" if thinking else "no-thinking"
        fallback_config = ModelConfig("anthropic", CLAUDE_OPUS_BASE, 60000 if thinking else 0)
        
        context_str = f"[{self.runner.task_id}:{self.runner.test_index}] ({self.runner.step_name})" if self.runner.task_id and self.runner.step_name else ""
        log_msg = f"{context_str} OpenAI Job failed: {reason}. Falling back to Claude Opus ({model_suffix})..."
        logger.warning(f"[BACKGROUND] {log_msg}")

        if self.runner.run_timestamp:
            log_failure(
                run_timestamp=self.runner.run_timestamp,
                task_id=self.runner.task_id if self.runner.task_id else "UNKNOWN",
                run_id="OPENAI_BG_FAILURE",
                error=RetryableProviderError(f"OpenAI Job failed: {reason}. Falling back..."),
                model=self.runner.full_model_name,
                step=self.runner.step_name if self.runner.step_name else (self.runner.task_id if self.runner.task_id else "UNKNOWN"),
                test_index=self.runner.test_index,
                is_retryable=True
            )

        duration = time.perf_counter() - start_ts
        if self.runner.timing_tracker is not None:
            self.runner.timing_tracker.append({
                "type": "attempt",
                "model": self.runner.full_model_name,
                "duration": duration,
                "status": "failed",
                "error": f"Failed: {reason}. Falling back."
            })

        response = call_anthropic(
            self.runner.anthropic_client,
            prompt,
            fallback_config,
            image_path=image_path,
            return_strategy=False,
            verbose=self.verbose,
            task_id=self.runner.task_id,
            test_index=self.runner.test_index,
            run_timestamp=self.runner.run_timestamp,
            model_alias=f"claude-opus-4.5-{model_suffix}",
            timing_tracker=self.runner.timing_tracker
        )
        response.model_name = f"claude-opus-4.5-{model_suffix}"
        return response

    def solve(self, prompt: str, image_path: Optional[str] = None) -> ModelResponse:
        start_attempt_ts = time.perf_counter()

        try:
            content = self.runner._prepare_content(prompt, image_path)
            
            kwargs = {
                "model": self.runner.model,
                "input": [{"role": "user", "content": content}],
                "timeout": 60,
                "background": True,
                "store": True,
                "max_output_tokens": 120000,
            }
            if self.runner.reasoning_effort != "none":
                kwargs["reasoning"] = {"effort": self.runner.reasoning_effort}
            
            if self.runner.last_failed_job_id:
                kwargs["previous_response_id"] = self.runner.last_failed_job_id

            # 1. Submit Job
            def _submit():
                try:
                    return self.client.responses.create(**kwargs)
                except Exception as e:
                    _map_openai_exception(e, self.runner.full_model_name)
            
            job = run_with_retry(
                lambda: _submit(), 
                task_id=self.runner.task_id, 
                test_index=self.runner.test_index, 
                run_timestamp=self.runner.run_timestamp, 
                model_name=self.runner.full_model_name, 
                timing_tracker=self.runner.timing_tracker, 
                log_success=False
            )
            job_id = job.id
            if self.verbose:
                print(f"[BACKGROUND] [{self.runner.model}] Job submitted. ID: {job_id}")

            # 2. Poll for Completion
            max_wait_time = 3600  # 60 minutes
            start_time = time.time()
            poll_interval_base = 2.0
            last_log_time = time.time()
            
            while True:
                # Check Timeout
                elapsed = time.time() - start_time
                if elapsed > max_wait_time:
                    if self.runner.reasoning_effort == "xhigh":
                        return self._fallback_to_claude(prompt, image_path, f"Timeout after {max_wait_time}s", start_attempt_ts, thinking=True)
                    elif self.runner.reasoning_effort == "low":
                        return self._fallback_to_claude(prompt, image_path, f"Timeout after {max_wait_time}s", start_attempt_ts, thinking=False)
                    
                    if self.runner.is_downgraded_retry:
                        raise NonRetryableProviderError(f"OpenAI Background Job {job_id} timed out after {max_wait_time}s (Downgraded Retry Failed)")

                    raise RetryableProviderError(f"OpenAI Background Job {job_id} timed out after {max_wait_time}s")

                # Logging every ~30s
                if self.verbose and (time.time() - last_log_time > 30):
                    print(f"[BACKGROUND] [{self.runner.model}] Job {job_id} still processing... ({int(elapsed)}s elapsed)")
                    last_log_time = time.time()

                # Retrieve Status
                def _retrieve():
                    try:
                        return self.client.responses.retrieve(job_id)
                    except Exception as e:
                        _map_openai_exception(e, self.runner.full_model_name)

                try:
                    job = run_with_retry(
                        lambda: _retrieve(), 
                        task_id=self.runner.task_id, 
                        test_index=self.runner.test_index, 
                        run_timestamp=self.runner.run_timestamp, 
                        model_name=self.runner.full_model_name, 
                        timing_tracker=self.runner.timing_tracker, 
                        log_success=False
                    )
                except NonRetryableProviderError as e:
                    # Fallback on 403 Fatal Error (seen in background jobs)
                    if "OpenAI Fatal Error" in str(e) and "403" in str(e):
                        if self.runner.reasoning_effort == "xhigh":
                            return self._fallback_to_claude(prompt, image_path, f"OpenAI 403 Forbidden: {e}", start_attempt_ts, thinking=True)
                        elif self.runner.reasoning_effort == "low":
                            return self._fallback_to_claude(prompt, image_path, f"OpenAI 403 Forbidden: {e}", start_attempt_ts, thinking=False)
                    raise e

                if job.status in ("queued", "in_progress"):
                    sleep_time = poll_interval_base + random.uniform(0, 1.0)
                    time.sleep(sleep_time)
                    continue
                
                # Terminal States
                if job.status == "completed":
                    text_output = ""
                    if hasattr(job, "output_text") and job.output_text:
                        text_output = job.output_text
                    elif hasattr(job, "output"):
                        for item in job.output:
                            if item.type == "message":
                                for content_part in item.content:
                                    if content_part.type == "output_text":
                                        text_output += content_part.text
                    
                    usage = getattr(job, "usage", None)
                    
                    duration = time.perf_counter() - start_attempt_ts
                    if self.runner.timing_tracker is not None:
                        self.runner.timing_tracker.append({
                            "type": "attempt",
                            "model": self.runner.full_model_name,
                            "duration": duration,
                            "status": "success"
                        })

                    return ModelResponse(
                        text=text_output,
                        prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
                        cached_tokens=0,
                        completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
                        strategy=None
                    )
                
                elif job.status == "failed":
                    err_msg = f"Code: {job.error.code}, Message: {job.error.message}" if job.error else "Unknown error"
                    
                    # Fallback on server_error
                    if job.error and job.error.code == "server_error":
                        if self.runner.reasoning_effort == "xhigh":
                            return self._fallback_to_claude(prompt, image_path, f"OpenAI Server Error: {err_msg}", start_attempt_ts, thinking=True)
                        elif self.runner.reasoning_effort == "low":
                            return self._fallback_to_claude(prompt, image_path, f"OpenAI Server Error: {err_msg}", start_attempt_ts, thinking=False)

                    raise RetryableProviderError(f"OpenAI Background Job {job_id} FAILED: {err_msg}")
                
                elif job.status in ("cancelled", "incomplete"):
                    reason = getattr(job, 'incomplete_details', 'Unknown')
                    reason_str = str(reason)
                    if "max_output_tokens" in reason_str or "token_limit" in reason_str:
                        if self.runner.reasoning_effort == "xhigh":
                            return self._fallback_to_claude(prompt, image_path, f"Token limit: {reason}", start_attempt_ts, thinking=True)
                        elif self.runner.reasoning_effort == "low":
                            return self._fallback_to_claude(prompt, image_path, f"Token limit: {reason}", start_attempt_ts, thinking=False)
                    
                    raise NonRetryableProviderError(f"OpenAI Background Job {job_id} ended with status={job.status}, reason={reason}")
                
                else:
                    raise UnknownProviderError(f"OpenAI Background Job {job_id} ended in unexpected status={job.status}")

        except Exception as e:
            raise e
