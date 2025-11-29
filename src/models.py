from openai import OpenAI
from anthropic import Anthropic
from google import genai

from src.types import (
    ModelConfig, 
    ModelResponse, 
    SUPPORTED_MODELS, 
    PRICING_PER_1M_TOKENS,
    GPT_5_1_BASE,
    CLAUDE_SONNET_BASE,
    CLAUDE_OPUS_BASE,
    GEMINI_3_BASE
)
from src.providers.openai import call_openai_internal
from src.providers.anthropic import call_anthropic
from src.providers.gemini import call_gemini

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
        suffix = model_arg.replace("claude-opus-4.5-", "")
        if suffix == "no-thinking":
            return ModelConfig("anthropic", CLAUDE_OPUS_BASE, 0)
        if suffix.startswith("thinking-"):
            try:
                budget = int(suffix.split("-")[1])
                return ModelConfig("anthropic", CLAUDE_OPUS_BASE, budget)
            except (IndexError, ValueError):
                pass

    if model_arg.startswith("gemini-3-"):
        parts = model_arg.split("-")
        effort = parts[-1]
        return ModelConfig("google", GEMINI_3_BASE, effort)

    raise ValueError(f"Unknown model format: {model_arg}")

def calculate_cost(model_config: ModelConfig, response: ModelResponse) -> float:
    """Calculates the cost of a model response based on pricing constants."""
    base_model = model_config.base_model
    pricing = PRICING_PER_1M_TOKENS.get(
        base_model, {"input": 0, "cached_input": 0, "output": 0}
    )

    # Special pricing case for Gemini 3 Pro Preview high context
    if (
        base_model == GEMINI_3_BASE
        and response.prompt_tokens > 200000
    ):
        pricing = {"input": 4.00, "cached_input": 0.0, "output": 18.00}

    non_cached_input = max(
        0, response.prompt_tokens - response.cached_tokens
    )

    cost = (
        (non_cached_input / 1_000_000 * pricing["input"])
        + (
            response.cached_tokens
            / 1_000_000
            * pricing.get("cached_input", 0)
        )
        + (response.completion_tokens / 1_000_000 * pricing["output"])
    )
    return cost

def call_model(
    openai_client: OpenAI,
    anthropic_client: Anthropic,
    google_client: genai.Client,
    prompt: str,
    model_arg: str,
    image_path: str = None,
    return_strategy: bool = False,
    verbose: bool = False,
) -> ModelResponse:
    config = parse_model_arg(model_arg)

    if config.provider == "openai":
        return call_openai_internal(
            openai_client,
            prompt,
            config,
            image_path=image_path,
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
            image_path=image_path,
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
            image_path=image_path,
            return_strategy=return_strategy,
            verbose=verbose,
        )

    raise ValueError(f"Unknown provider {config.provider}")