import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

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

    if model_arg.startswith("gemini-3-"):
        parts = model_arg.split("-")
        effort = parts[-1]
        return "google", "gemini-3-pro-preview", effort

    raise ValueError(f"Unknown model format: {model_arg}")

def call_openai_internal(
    client: OpenAI, prompt: str, model: str, reasoning_effort: str
) -> ModelResponse:
    request_kwargs = {"model": model, "input": prompt}
    if reasoning_effort != "none":
        request_kwargs["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**request_kwargs)
    text_output = None
    for item in response.output or []:
        contents = getattr(item, "content", None)
        if not contents:
            continue
        for content in contents:
            if getattr(content, "type", None) in {"text", "output_text"}:
                text_output = content.text.strip()
                break
        if text_output:
            break

    if not text_output:
        raise RuntimeError("OpenAI response did not contain text output.")

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0

    prompt_details = getattr(usage, "input_tokens_details", None)
    cached_tokens = getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0

    return ModelResponse(
        text=text_output,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
    )

def call_anthropic(
    client: Anthropic, prompt: str, model: str, budget: int
) -> ModelResponse:
    MODEL_MAX_TOKENS = 64000
    max_tokens = 8192

    if budget and budget > 0:
        max_tokens = min(budget + 4096, MODEL_MAX_TOKENS)
        # Ensure budget is less than max_tokens to leave room for response
        if budget >= max_tokens:
            budget = max_tokens - 2048  # Reserve 2k for final answer

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }

    if budget and budget > 0:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

    # Use streaming to avoid timeouts on long requests
    final_message = None
    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            pass
        final_message = stream.get_final_message()

    text_parts = []
    for block in final_message.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)

    text = "".join(text_parts).strip()

    p_tokens = final_message.usage.input_tokens
    c_tokens = final_message.usage.output_tokens
    cached = getattr(final_message.usage, "cache_read_input_tokens", 0) or 0

    return ModelResponse(
        text=text,
        prompt_tokens=p_tokens,
        cached_tokens=cached,
        completion_tokens=c_tokens,
    )

def call_gemini(
    client: genai.Client, prompt: str, model: str, thinking_level: str
) -> ModelResponse:
    # Map thinking_level to thinking_budget since SDK 1.47.0 lacks thinking_level
    budget = 2048
    if thinking_level == "high":
        budget = 16384

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_budget=budget),
        temperature=0.7,
        max_output_tokens=65536,
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    
    text_parts = []
    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.text:
                text_parts.append(part.text)
    text = "".join(text_parts).strip()

    usage = getattr(response, "usage_metadata", None)
    p_tokens = usage.prompt_token_count if usage else 0
    c_tokens = usage.candidates_token_count if usage else 0
    thoughts = getattr(usage, "thoughts_token_count", 0) or 0
    c_tokens += thoughts

    return ModelResponse(
        text=text,
        prompt_tokens=p_tokens,
        cached_tokens=0,
        completion_tokens=c_tokens,
    )

def call_model(
    openai_client: OpenAI,
    anthropic_client: Anthropic,
    google_client: genai.Client,
    prompt: str,
    model_arg: str,
) -> ModelResponse:
    provider, base_model, config = parse_model_arg(model_arg)

    if provider == "openai":
        return call_openai_internal(openai_client, prompt, base_model, config)
    elif provider == "anthropic":
        if not anthropic_client:
            raise RuntimeError("Anthropic client not initialized.")
        return call_anthropic(anthropic_client, prompt, base_model, config)
    elif provider == "google":
        if not google_client:
            raise RuntimeError("Google client not initialized.")
        return call_gemini(google_client, prompt, base_model, config)

    raise ValueError(f"Unknown provider {provider}")