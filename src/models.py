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

def call_openai_internal(
    client: OpenAI, prompt: str, model: str, reasoning_effort: str
) -> ModelResponse:
    request_kwargs = {"model": model, "input": prompt, "timeout": 3600}
    if reasoning_effort != "none":
        request_kwargs["reasoning"] = {"effort": reasoning_effort}

    response = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.responses.create(**request_kwargs)
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
    client: Anthropic, prompt: str, model: str, config: Union[int, str]
) -> ModelResponse:
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
        max_tokens = 4096 # Increase slightly for reasoning output
    
    kwargs["max_tokens"] = max_tokens

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
    # Use REST API to bypass SDK limitation regarding thinking_level
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Google API Key not found in environment")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    level_enum = "LOW" if thinking_level == "low" else "HIGH"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 1.0,
            "maxOutputTokens": 65536,
            "thinkingConfig": {"includeThoughts": True, "thinkingLevel": level_enum},
        },
    }

    response = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url, json=payload, headers={"Content-Type": "application/json"}
            )
            if resp.status_code != 200:
                if resp.status_code == 503 or "503" in resp.text:
                    raise Exception(f"503 Unavailable: {resp.text}")
                raise Exception(f"API Error {resp.status_code}: {resp.text}")

            response = resp.json()
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
        candidate = response["candidates"][0]
        parts = candidate["content"]["parts"]
        text_parts = [p["text"] for p in parts if "text" in p]
        text = "".join(text_parts).strip()

        usage = response.get("usageMetadata", {})
        p_tokens = usage.get("promptTokenCount", 0)
        c_tokens = usage.get("candidatesTokenCount", 0)
        thoughts = usage.get("thoughtsTokenCount", 0)
        c_tokens += thoughts

    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Failed to parse Gemini response: {e} - Raw: {response}")

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
