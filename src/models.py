import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Union, Callable, Any

from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types

# Constants
GPT_5_1_BASE = "gpt-5.1"
CLAUDE_SONNET_BASE = "claude-sonnet-4-5-20250929"
CLAUDE_OPUS_BASE = "claude-opus-4-5-20251101"
GEMINI_3_BASE = "gemini-3-pro-preview"

PRICING_PER_1M_TOKENS = {
    GPT_5_1_BASE: {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    CLAUDE_SONNET_BASE: {
        "input": 3.00,
        "cached_input": 0.30,
        "output": 15.00,
    },
    CLAUDE_OPUS_BASE: {
        "input": 5.00,
        "cached_input": 0.50,
        "output": 25.00,
    },
    GEMINI_3_BASE: {
        "input": 2.00,
        "cached_input": 0.0,
        "output": 12.00,
    },
}

ORDERED_MODELS = [
    "gpt-5.1-none",
    "gpt-5.1-low",
    "gpt-5.1-medium",
    "gpt-5.1-high",
    "claude-sonnet-4.5-no-thinking",
    "claude-sonnet-4.5-thinking-1024",
    "claude-sonnet-4.5-thinking-4000",
    "claude-sonnet-4.5-thinking-16000",
    "claude-sonnet-4.5-thinking-60000",
    "claude-opus-4.5-low",
    "claude-opus-4.5-medium",
    "claude-opus-4.5-high",
    "gemini-3-low",
    "gemini-3-high",
]
SUPPORTED_MODELS = set(ORDERED_MODELS)

ResultRecord = Tuple[Path, int, bool, str, float, float, Optional[str]]

@dataclass
class ModelResponse:
    text: str
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    strategy: Optional[str] = None

@dataclass
class ModelConfig:
    provider: str
    base_model: str
    config: object  # can be int (budget), str (effort), or something else

def parse_model_arg(model_arg: str) -> ModelConfig:
    if model_arg not in SUPPORTED_MODELS:
        raise ValueError(f"Model '{model_arg}' not supported. Choose from {SUPPORTED_MODELS}")

    if model_arg.startswith("gpt-5.1-"):
        parts = model_arg.split("-")
        effort = parts[-1]
        return ModelConfig("openai", GPT_5_1_BASE, effort)

    if model_arg.startswith("claude-sonnet-4.5-"):
        suffix = model_arg.replace("claude-sonnet-4.5-", "")
        if suffix == "no-thinking":
            return ModelConfig("anthropic", CLAUDE_SONNET_BASE, 0)
        if suffix.startswith("thinking-"):
            try:
                budget = int(suffix.split("-")[1])
                return ModelConfig("anthropic", CLAUDE_SONNET_BASE, budget)
            except (IndexError, ValueError):
                pass

    if model_arg.startswith("claude-opus-4.5-"):
        parts = model_arg.split("-")
        effort = parts[-1]
        return ModelConfig("anthropic", CLAUDE_OPUS_BASE, effort)

    if model_arg.startswith("gemini-3-"):
        parts = model_arg.split("-")
        effort = parts[-1]
        return ModelConfig("google", GEMINI_3_BASE, effort)

    raise ValueError(f"Unknown model format: {model_arg}")

def run_with_retry(
    func: Callable[[], Any],
    retry_predicate: Callable[[Exception], bool],
    max_retries: int = 3
) -> Any:
    """Generic retry loop helper."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if retry_predicate(e):
                if attempt < max_retries - 1:
                    delay = 5 if attempt == 0 else 30
                    time.sleep(delay)
                    continue
            raise e

def _orchestrate_two_stage(
    solve_func: Callable[[str], ModelResponse],
    explain_func: Callable[[str, ModelResponse], Optional[ModelResponse]],
    prompt: str,
    return_strategy: bool,
    verbose: bool
) -> ModelResponse:
    """
    Orchestrates the Solve -> Explain workflow.
    """
    # Step 1: Solve
    if verbose:
        print(f"--- REAL PROMPT STEP 1 (Solve) ---\n{prompt}\n--- END REAL PROMPT STEP 1 ---", file=sys.stderr)
    
    response1 = solve_func(prompt)
    
    if not return_strategy:
        return response1

    # Step 2: Explain
    step2_input = "Explain the strategy you used in broad terms such that it can be applied on other similar examples and other input data."
    if verbose:
        print(f"--- REAL PROMPT STEP 2 (Explain) ---\n{step2_input}\n--- END REAL PROMPT STEP 2 ---", file=sys.stderr)

    response2 = explain_func(step2_input, response1)

    if response2:
        # Combine Usage
        return ModelResponse(
            text=response1.text,
            prompt_tokens=response1.prompt_tokens + response2.prompt_tokens,
            cached_tokens=response1.cached_tokens + response2.cached_tokens,
            completion_tokens=response1.completion_tokens + response2.completion_tokens,
            strategy=response2.text # Step 2 text IS the strategy
        )
    else:
        # Step 2 failed, return Step 1 only
        return response1


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

    return _orchestrate_two_stage(_solve, _explain, prompt, return_strategy, verbose)

def call_anthropic(
    client: Anthropic,
    prompt: str,
    config: ModelConfig,
    return_strategy: bool = False,
    verbose: bool = False,
) -> ModelResponse:
    MODEL_MAX_TOKENS = 64000
    
    model = config.base_model
    cfg_val = config.config

    def _should_retry(e: Exception) -> bool:
        err_str = str(e)
        return (
            "500" in err_str
            or "Internal server error" in err_str
            or "Connection reset" in err_str
            or "Connection error" in err_str
        )

    kwargs = {
        "model": model,
        "max_tokens": 8192
    }
    if isinstance(cfg_val, int) and cfg_val > 0:
        budget = cfg_val
        max_tokens = min(budget + 4096, MODEL_MAX_TOKENS)
        if budget >= max_tokens: budget = max_tokens - 2048
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        kwargs["max_tokens"] = max_tokens
    elif isinstance(cfg_val, str):
        kwargs["extra_headers"] = {"anthropic-beta": "effort-2025-11-24"}
        kwargs["extra_body"] = {"output_config": {"effort": cfg_val}}
        kwargs["max_tokens"] = 60000

    def _solve(p: str) -> ModelResponse:
        kw = kwargs.copy()
        kw["messages"] = [{"role": "user", "content": p}]
        
        def _run():
            with client.messages.stream(**kw) as stream:
                for _ in stream.text_stream: pass
                return stream.get_final_message()
        
        final = run_with_retry(_run, _should_retry)
        
        text_parts = []
        for block in final.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
        
        resp = ModelResponse(
            text="".join(text_parts).strip(),
            prompt_tokens=final.usage.input_tokens,
            cached_tokens=getattr(final.usage, "cache_read_input_tokens", 0) or 0,
            completion_tokens=final.usage.output_tokens,
        )
        resp._raw_content = final.content # Store for context
        return resp

    def _explain(p: str, prev_resp: ModelResponse) -> Optional[ModelResponse]:
        try:
            kw = kwargs.copy()
            kw["messages"] = [
                {"role": "user", "content": prompt}, # Original Prompt
                {"role": "assistant", "content": prev_resp._raw_content},
                {"role": "user", "content": p}
            ]
            
            def _run():
                with client.messages.stream(**kw) as stream:
                    for _ in stream.text_stream: pass
                    return stream.get_final_message()

            final = run_with_retry(_run, _should_retry)
            
            text_parts = []
            for block in final.content:
                if getattr(block, "type", None) == "text":
                    text_parts.append(block.text)

            return ModelResponse(
                text="".join(text_parts).strip(),
                prompt_tokens=final.usage.input_tokens,
                cached_tokens=getattr(final.usage, "cache_read_input_tokens", 0) or 0,
                completion_tokens=final.usage.output_tokens,
            )
        except Exception as e:
            print(f"Step 2 strategy extraction failed: {e}", file=sys.stderr)
            return None

    return _orchestrate_two_stage(_solve, _explain, prompt, return_strategy, verbose)


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

    return _orchestrate_two_stage(_solve, _explain, prompt, return_strategy, verbose)


def call_model(
    openai_client: OpenAI,
    anthropic_client: Anthropic,
    google_client: genai.Client,
    prompt: str,
    model_arg: str,
    return_strategy: bool = False,
    verbose: bool = False,
) -> ModelResponse:
    config = parse_model_arg(model_arg)

    if config.provider == "openai":
        return call_openai_internal(
            openai_client,
            prompt,
            config,
            return_strategy=return_strategy,
            verbose=verbose,
        )
    elif config.provider == "anthropic":
        if not anthropic_client:
            raise RuntimeError("Anthropic client not initialized.")
        return call_anthropic(
            anthropic_client,
            prompt,
            config,
            return_strategy=return_strategy,
            verbose=verbose,
        )
    elif config.provider == "google":
        if not google_client:
            raise RuntimeError("Google client not initialized.")
        return call_gemini(
            google_client,
            prompt,
            config,
            return_strategy=return_strategy,
            verbose=verbose,
        )

    raise ValueError(f"Unknown provider {config.provider}")