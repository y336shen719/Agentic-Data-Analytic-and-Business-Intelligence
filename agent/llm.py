"""Reusable ChatOpenAI factory for the agent."""

from __future__ import annotations

from typing import Optional

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from agent.config import ConfigError, LLMConfig, load_llm_config


def create_llm(config: Optional[LLMConfig] = None) -> ChatOpenAI:
    """Create a configured ``ChatOpenAI`` client.

    Args:
        config: Optional pre-built config; defaults to ``load_llm_config()``.

    Raises:
        ConfigError: if OPENAI_API_KEY is missing.

    Note: ``temperature`` and other sampling params are deliberately not set, since
    reasoning-capable models may reject them. The client construction itself makes
    no network call.
    """
    cfg = config or load_llm_config()
    if not cfg.api_key:
        raise ConfigError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and set it, "
            "or export it in your shell."
        )
    return ChatOpenAI(
        model=cfg.model,
        api_key=SecretStr(cfg.api_key),
        timeout=cfg.timeout_seconds,
        max_retries=cfg.max_retries,
    )
