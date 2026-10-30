"""OpenAI model configuration: environment parsing and validation only.

No network access and no client construction happens here (see ``agent/llm.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Mapping, Optional

from dotenv import load_dotenv

# Default per https://platform.openai.com/docs/models ("most efficient model for
# focused, high-volume tasks"). Override with OPENAI_MODEL.
DEFAULT_OPENAI_MODEL: str = "gpt-6-luna"
DEFAULT_TIMEOUT_SECONDS: float = 30.0
DEFAULT_MAX_RETRIES: int = 2

MAX_TIMEOUT_SECONDS: float = 120.0
MAX_RETRIES_LIMIT: int = 5


class ConfigError(ValueError):
    """Raised when LLM configuration is invalid or incomplete."""


@dataclass(frozen=True)
class LLMConfig:
    """Validated LLM settings. The API key is excluded from repr to avoid leaks."""

    model: str = DEFAULT_OPENAI_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    api_key: Optional[str] = field(default=None, repr=False)


def _parse_float(name: str, raw: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got: {raw!r}") from exc


def _parse_int(name: str, raw: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got: {raw!r}") from exc


def load_llm_config(env: Optional[Mapping[str, str]] = None) -> LLMConfig:
    """Build an ``LLMConfig`` from environment variables.

    Reads OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TIMEOUT_SECONDS, OPENAI_MAX_RETRIES.
    When ``env`` is None, a local ``.env`` file is loaded first (without overriding
    variables already set in the process) and ``os.environ`` is used.

    A missing API key is NOT an error here; ``create_llm`` enforces it so config
    can be inspected/tested without credentials.

    Raises:
        ConfigError: on a blank model or out-of-range/non-numeric limits.
    """
    if env is None:
        load_dotenv(override=False)
        env = os.environ

    model = (env.get("OPENAI_MODEL") or "").strip() or DEFAULT_OPENAI_MODEL

    raw_timeout = (env.get("OPENAI_TIMEOUT_SECONDS") or "").strip()
    timeout = _parse_float("OPENAI_TIMEOUT_SECONDS", raw_timeout) if raw_timeout else DEFAULT_TIMEOUT_SECONDS
    if not (0 < timeout <= MAX_TIMEOUT_SECONDS) or timeout != timeout:
        raise ConfigError(
            f"OPENAI_TIMEOUT_SECONDS must be in (0, {MAX_TIMEOUT_SECONDS}], got: {timeout}"
        )

    raw_retries = (env.get("OPENAI_MAX_RETRIES") or "").strip()
    retries = _parse_int("OPENAI_MAX_RETRIES", raw_retries) if raw_retries else DEFAULT_MAX_RETRIES
    if not (0 <= retries <= MAX_RETRIES_LIMIT):
        raise ConfigError(
            f"OPENAI_MAX_RETRIES must be between 0 and {MAX_RETRIES_LIMIT}, got: {retries}"
        )

    api_key = (env.get("OPENAI_API_KEY") or "").strip() or None

    return LLMConfig(model=model, timeout_seconds=timeout, max_retries=retries, api_key=api_key)
