import os
import sys
import time
import json
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Union

from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types

PRICING_PER_1M_TOKENS = {
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "claude-sonnet-4-5-20250929": {
        "input": 3.00,
        "cached_input": 0.30,
        "output": 15.00,
    },
    "claude-opus-4-5-20251101": {
        "input": 5.00,
        "cached_input": 0.50,
        "output": 25.00,
    },
    "gemini-3-pro-preview": {
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

ResultRecord = Tuple[Path, int, bool, str, float, float]

@dataclass
class ModelResponse:
    text: str
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    strategy: Optional[str] = None

def parse_model_arg(model_arg: str) -> Tuple[str, str, object]:
    if model_arg not in SUPPORTED_MODELS:
        raise ValueError(f"Model '{model_arg}' not supported. Choose from {SUPPORTED_MODELS}")

    if model_arg.startswith("gpt-5.1-"):
        parts = model_arg.split("-")
        effort = parts[-1]
        base = "-".join(parts[:-1])
        return "openai", base, effort

    if model_arg.startswith("claude-sonnet-4.5-"):
        base = "claude-sonnet-4-5-20250929"
        suffix = model_arg.replace("claude-sonnet-4.5-", "")
        if suffix == "no-thinking":
            return "anthropic", base, 0
        if suffix.startswith("thinking-"):
            try:
                budget = int(suffix.split("-")[1])
                return "anthropic", base, budget
            except (IndexError, ValueError):
                pass

    if model_arg.startswith("claude-opus-4.5-"):
        base = "claude-opus-4-5-20251101"
        parts = model_arg.split("-")
        effort = parts[-1]
        return "anthropic", base, effort

    if model_arg.startswith("gemini-3-"):
        parts = model_arg.split("-")
        effort = parts[-1]
        return "google", "gemini-3-pro-preview", effort

    raise ValueError(f"Unknown model format: {model_arg}")

import httpx

def call_openai_internal(
    client: OpenAI,
    prompt: str,
    model: str,
    reasoning_effort: str,
    return_strategy: bool = False,
    verbose: bool = False,
) -> ModelResponse:
    # Uniformly use the Responses API for all OpenAI calls
    # Step 1: Solve the grid
    kwargs = {
        "model": model,
        "input": [{"role": "user", "content": prompt}],
        "timeout": 3600,
    }

    if verbose:
        print(f"--- REAL PROMPT STEP 1 (Solve) ---\n{prompt}\n--- END REAL PROMPT STEP 1 ---", file=sys.stderr)

    # Configure reasoning parameters (Standard hidden reasoning for performance)
    if reasoning_effort != "none":
        kwargs["reasoning"] = {
            "effort": reasoning_effort
        }

    response = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.responses.create(**kwargs)
            break
        except Exception as e:
            err_str = str(e)
            if (
                "Connection error" in err_str
                or "500" in err_str
                or "server_error" in err_str
                or "upstream connect error" in err_str
                or "timed out" in err_str
            ):
                if attempt < max_retries - 1:
                    delay = 5 if attempt == 0 else 30
                    time.sleep(delay)
                    continue
            raise e

    # Extract Step 1 Output (The Grid)
    text_output = ""
    if hasattr(response, "output"):
        for item in response.output:
            if item.type == "message":
                for content_part in item.content:
                    if content_part.type == "output_text":
                        text_output += content_part.text

    if not text_output:
         raise RuntimeError(f"OpenAI Responses API Step 1 did not return text output. Response: {response}")

    # Calculate Step 1 Usage
    usage1 = getattr(response, "usage", None)
    prompt_tokens = getattr(usage1, "input_tokens", 0) if usage1 else 0
    completion_tokens = getattr(usage1, "output_tokens", 0) if usage1 else 0
    
    # If strategy extraction is not requested, return immediately
    if not return_strategy:
        return ModelResponse(
            text=text_output,
            prompt_tokens=prompt_tokens,
            cached_tokens=0,
            completion_tokens=completion_tokens,
            strategy=None,
        )

    # STEP 2: Extract Strategy (using previous_response_id)
    step1_id = response.id
    
    step2_input = "Explain the strategy you used in broad terms such that it can be applied on other similar examples and other input data."
    
    if verbose:
        print(f"--- REAL PROMPT STEP 2 (Strategy Extraction) ---\n{step2_input}\n--- END REAL PROMPT STEP 2 ---", file=sys.stderr)

    kwargs_step2 = {
        "model": model,
        "previous_response_id": step1_id,
        "input": [{"role": "user", "content": step2_input}],
        "timeout": 3600
    }

    response2 = None
    for attempt in range(max_retries):
        try:
            response2 = client.responses.create(**kwargs_step2)
            break
        except Exception as e:
             # If Step 2 fails, we return the grid but with no strategy
             print(f"Step 2 strategy extraction failed: {e}", file=sys.stderr)
             return ModelResponse(
                text=text_output,
                prompt_tokens=prompt_tokens,
                cached_tokens=0,
                completion_tokens=completion_tokens,
                strategy=None,
            )

    # Parse Step 2 Output (The Strategy)
    strategy_text = ""
    if hasattr(response2, "output"):
        for item in response2.output:
            if item.type == "message":
                for content_part in item.content:
                    if content_part.type == "output_text":
                        strategy_text += content_part.text

    # Accumulate Usage
    usage2 = getattr(response2, "usage", None)
    prompt_tokens += getattr(usage2, "input_tokens", 0) if usage2 else 0
    completion_tokens += getattr(usage2, "output_tokens", 0) if usage2 else 0

    return ModelResponse(
        text=text_output,
        prompt_tokens=prompt_tokens,
        cached_tokens=0,
        completion_tokens=completion_tokens,
        strategy=strategy_text,
    )

def call_anthropic(
    client: Anthropic,
    prompt: str,
    model: str,
    config: Union[int, str],
    return_strategy: bool = False,
    verbose: bool = False,
) -> ModelResponse:
    # Claude doesn't support native JSON schema enforcement in the same way as OpenAI/Gemini via API params yet
    # (or it requires tools which adds complexity). We rely on prompt instructions for now.
    MODEL_MAX_TOKENS = 64000
    max_tokens = 8192

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    if isinstance(config, int) and config > 0:
        # Thinking (Sonnet)
        budget = config
        max_tokens = min(budget + 4096, MODEL_MAX_TOKENS)
        if budget >= max_tokens:
            budget = max_tokens - 2048
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
    elif isinstance(config, str):
        # Effort (Opus)
        kwargs["extra_headers"] = {"anthropic-beta": "effort-2025-11-24"}
        kwargs["extra_body"] = {"output_config": {"effort": config}}
        # Keep max_tokens default or increase?
        # User example used 2048.
        max_tokens = 60000  # Increase slightly for reasoning output

    kwargs["max_tokens"] = max_tokens

    if verbose:
        print(f"--- REAL PROMPT STEP 1 (Anthropic Solve) ---\n{prompt}\n--- END REAL PROMPT STEP 1 ---", file=sys.stderr)

    # Use streaming to avoid timeouts on long requests
    final_message = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    pass
                final_message = stream.get_final_message()
            break
        except Exception as e:
            err_str = str(e)
            if (
                "500" in err_str
                or "Internal server error" in err_str
                or "Connection reset" in err_str
                or "Connection error" in err_str
            ):
                if attempt < max_retries - 1:
                    delay = 5 if attempt == 0 else 30
                    time.sleep(delay)
                    continue
            raise e

    # Extract Step 1 Text (The Grid)
    text_parts = []
    thinking_parts = [] # Capture thinking for log or single-stage case
    
    for block in final_message.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)
        elif getattr(block, "type", None) == "thinking":
            thinking_parts.append(block.thinking)

    grid_text = "".join(text_parts).strip()
    
    # Usage Step 1
    p_tokens = final_message.usage.input_tokens
    c_tokens = final_message.usage.output_tokens
    cached = getattr(final_message.usage, "cache_read_input_tokens", 0) or 0

    if not return_strategy:
        # Single stage: if we captured thinking (native), return it?
        # The previous logic did: if capture_thinking and thinking_parts: explanation = ...
        # Since return_strategy maps to that, if user asked for strategy but we are in single call mode (legacy?), 
        # we might return native thoughts.
        # But the "Solve-Then-Explain" plan implies we perform Step 2.
        # Wait, if return_strategy is False, we return None.
        # If the user wants NATIVE thinking without Step 2, that's not currently exposed as a separate flag for Claude
        # in this new architecture (we deprecated capture-internal-thinking).
        # So we just return grid.
        return ModelResponse(
            text=grid_text,
            prompt_tokens=p_tokens,
            cached_tokens=cached,
            completion_tokens=c_tokens,
            strategy=None,
        )

    # STEP 2: Explain (Two-Stage)
    # We must feed back the FULL Step 1 content (Thinking + Text) to maintain context.
    
    step2_input = "Explain the strategy you used in broad terms such that it can be applied on other similar examples and other input data."
    
    if verbose:
        print(f"--- REAL PROMPT STEP 2 (Anthropic Explain) ---\n{step2_input}\n--- END REAL PROMPT STEP 2 ---", file=sys.stderr)

    # Construct history: User Prompt -> Assistant Step 1 (Thinking + Grid) -> User Step 2
    messages_history = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": final_message.content}, # Pass back the full content block list
        {"role": "user", "content": step2_input}
    ]
    
    kwargs_step2 = kwargs.copy()
    kwargs_step2["messages"] = messages_history
    # We can lower the budget for explanation if desired, but keeping it simple for now.
    
    final_message_2 = None
    for attempt in range(max_retries):
        try:
            with client.messages.stream(**kwargs_step2) as stream:
                for text in stream.text_stream:
                    pass
                final_message_2 = stream.get_final_message()
            break
        except Exception as e:
             print(f"Step 2 strategy extraction failed: {e}", file=sys.stderr)
             return ModelResponse(
                text=grid_text,
                prompt_tokens=p_tokens,
                cached_tokens=cached,
                completion_tokens=c_tokens,
                strategy=None,
            )

    # Extract Step 2 Text (The Strategy)
    strategy_parts = []
    for block in final_message_2.content:
        if getattr(block, "type", None) == "text":
            strategy_parts.append(block.text)
    strategy_text = "".join(strategy_parts).strip()
    
    # Accumulate Usage
    p_tokens += final_message_2.usage.input_tokens
    c_tokens += final_message_2.usage.output_tokens
    cached += (getattr(final_message_2.usage, "cache_read_input_tokens", 0) or 0)

    return ModelResponse(
        text=grid_text,
        prompt_tokens=p_tokens,
        cached_tokens=cached,
        completion_tokens=c_tokens,
        strategy=strategy_text,
    )

def call_gemini(
    client: genai.Client,
    prompt: str,
    model: str,
    thinking_level: str,
    return_strategy: bool = False,
    verbose: bool = False,
) -> ModelResponse:
    # Use REST API to bypass SDK limitation regarding thinking_level
    # Note: We are switching to the SDK's Chat object to preserve Thought Signatures (reasoning context)
    # effectively. The SDK 'client.chats.create' handles this state.
    
    # SDK 1.52.0+ uses 'thinking_level' (Enum)
    level_enum = types.ThinkingLevel.LOW if thinking_level == "low" else types.ThinkingLevel.HIGH

    generation_config = {
        "temperature": 1.0,
        "max_output_tokens": 65536,
        "thinking_config": {"include_thoughts": True, "thinking_level": level_enum},
    }

    # Initialize Chat Session
    chat = client.chats.create(
        model=model,
        config=types.GenerateContentConfig(**generation_config)
    )

    if verbose:
        print(f"--- REAL PROMPT STEP 1 (Gemini Solve) ---\n{prompt}\n--- END REAL PROMPT STEP 1 ---", file=sys.stderr)

    # STEP 1: Solve
    response = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = chat.send_message(prompt)
            break
        except Exception as e:
            err_str = str(e)
            if (
                "503" in err_str
                or "UNAVAILABLE" in err_str
                or "overloaded" in err_str.lower()
            ):
                if attempt < max_retries - 1:
                    delay = 5 if attempt == 0 else 30
                    time.sleep(delay)
                    continue
            raise e

    try:
        # Manually extract text to avoid SDK warning about 'thought_signature' parts
        text_parts = []
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.text:
                    text_parts.append(part.text)
        text = "".join(text_parts).strip()
        
        usage = response.usage_metadata
        p_tokens = usage.prompt_token_count
        c_tokens = usage.candidates_token_count
        # thoughts = usage.get("thoughtsTokenCount", 0) # SDK object might have different struct, verify if needed.
        # For now, standard usage object usually sums them or has them.
        
    except (KeyError, AttributeError, IndexError) as e:
        raise RuntimeError(f"Failed to parse Gemini response: {e} - Raw: {response}")

    if not return_strategy:
        return ModelResponse(
            text=text,
            prompt_tokens=p_tokens,
            cached_tokens=0,
            completion_tokens=c_tokens,
            strategy=None,
        )

    # STEP 2: Explain (Strategy Extraction)
    step2_input = "Explain the strategy you used in broad terms such that it can be applied on other similar examples and other input data."
    
    if verbose:
        print(f"--- REAL PROMPT STEP 2 (Gemini Explain) ---\n{step2_input}\n--- END REAL PROMPT STEP 2 ---", file=sys.stderr)

    response2 = None
    for attempt in range(max_retries):
        try:
            response2 = chat.send_message(step2_input)
            break
        except Exception as e:
             print(f"Step 2 strategy extraction failed: {e}", file=sys.stderr)
             return ModelResponse(
                text=text,
                prompt_tokens=p_tokens,
                cached_tokens=0,
                completion_tokens=c_tokens,
                strategy=None,
            )

    # Manually extract Step 2 text
    strategy_parts = []
    if response2.candidates and response2.candidates[0].content and response2.candidates[0].content.parts:
        for part in response2.candidates[0].content.parts:
            if part.text:
                strategy_parts.append(part.text)
    strategy_text = "".join(strategy_parts).strip()
    
    # Accumulate Usage
    usage2 = response2.usage_metadata
    p_tokens += usage2.prompt_token_count
    c_tokens += usage2.candidates_token_count

    return ModelResponse(
        text=text,
        prompt_tokens=p_tokens,
        cached_tokens=0,
        completion_tokens=c_tokens,
        strategy=strategy_text,
    )

def call_model(
    openai_client: OpenAI,
    anthropic_client: Anthropic,
    google_client: genai.Client,
    prompt: str,
    model_arg: str,
    return_strategy: bool = False,
    verbose: bool = False,
) -> ModelResponse:
    provider, base_model, config = parse_model_arg(model_arg)

    if provider == "openai":
        return call_openai_internal(
            openai_client,
            prompt,
            base_model,
            config,
            return_strategy=return_strategy,
            verbose=verbose,
        )
    elif provider == "anthropic":
        if not anthropic_client:
            raise RuntimeError("Anthropic client not initialized.")
        return call_anthropic(
            anthropic_client,
            prompt,
            base_model,
            config,
            return_strategy=return_strategy,
            verbose=verbose,
        )
    elif provider == "google":
        if not google_client:
            raise RuntimeError("Google client not initialized.")
        return call_gemini(
            google_client,
            prompt,
            base_model,
            config,
            return_strategy=return_strategy,
            verbose=verbose,
        )

    raise ValueError(f"Unknown provider {provider}")
