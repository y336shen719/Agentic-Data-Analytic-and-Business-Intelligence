"""LangSmith tracing configuration, environment synchronization, and run metadata helpers.

Provides opt-in LangSmith / LangChain tracing integration with strict privacy redaction,
graceful error handling, and session-aware metadata propagation for LangGraph executions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
import re
from typing import Any, Dict, List, Mapping, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_LANGSMITH_PROJECT: str = "agentic-data-analyst"
DEFAULT_LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

# Sensitive key patterns that should be redacted from metadata
SENSITIVE_KEY_PATTERN = re.compile(
    r"(?i)(api[_-]?key|secret|token|password|auth|credential|cert)"
)


@dataclass(frozen=True)
class LangSmithConfig:
    """Validated LangSmith configuration. The API key is excluded from repr to prevent leakage."""

    enabled: bool = False
    api_key: Optional[str] = field(default=None, repr=False)
    project: str = DEFAULT_LANGSMITH_PROJECT
    endpoint: str = DEFAULT_LANGSMITH_ENDPOINT


def _is_truthy(val: Optional[str]) -> bool:
    """Return True if string represents a boolean true value."""
    if not val:
        return False
    return val.strip().lower() in ("true", "1", "yes", "y", "t", "on")


def load_langsmith_config(env: Optional[Mapping[str, str]] = None) -> LangSmithConfig:
    """Load LangSmith configuration from environment variables.

    Reads:
    - LANGSMITH_TRACING (or LANGCHAIN_TRACING_V2) -> bool (default: False)
    - LANGSMITH_API_KEY (or LANGCHAIN_API_KEY) -> Optional[str]
    - LANGSMITH_PROJECT (or LANGCHAIN_PROJECT) -> str (default: 'agentic-data-analyst')
    - LANGSMITH_ENDPOINT (or LANGCHAIN_ENDPOINT) -> str (default: 'https://api.smith.langchain.com')

    When env is None, load_dotenv() is called first without overriding existing variables.
    Tracing is strictly opt-in and disabled by default.
    """
    if env is None:
        load_dotenv(override=False)
        env = os.environ

    raw_tracing = env.get("LANGSMITH_TRACING") or env.get("LANGCHAIN_TRACING_V2") or "false"
    is_enabled = _is_truthy(raw_tracing)

    raw_key = (
        (env.get("LANGSMITH_API_KEY") or "").strip()
        or (env.get("LANGCHAIN_API_KEY") or "").strip()
        or None
    )

    project = (
        (env.get("LANGSMITH_PROJECT") or "").strip()
        or (env.get("LANGCHAIN_PROJECT") or "").strip()
        or DEFAULT_LANGSMITH_PROJECT
    )

    endpoint = (
        (env.get("LANGSMITH_ENDPOINT") or "").strip()
        or (env.get("LANGCHAIN_ENDPOINT") or "").strip()
        or DEFAULT_LANGSMITH_ENDPOINT
    )

    # If user enabled tracing but provided no API key, log warning and gracefully disable to prevent crashes
    if is_enabled and not raw_key:
        logger.warning(
            "LangSmith tracing is enabled via LANGSMITH_TRACING=true, but no LANGSMITH_API_KEY was provided. "
            "Tracing will be deactivated to prevent invocation failures."
        )
        is_enabled = False

    return LangSmithConfig(
        enabled=is_enabled,
        api_key=raw_key,
        project=project,
        endpoint=endpoint,
    )


def setup_langsmith_environment(config: Optional[LangSmithConfig] = None) -> bool:
    """Synchronize process environment variables for LangChain / LangSmith native tracing.

    Sets both LANGSMITH_* and LANGCHAIN_* environment variables to ensure compatibility
    across all versions of the LangChain and LangGraph SDKs.

    Returns:
        True if tracing is actively enabled and configured with an API key, False otherwise.
    """
    cfg = config or load_langsmith_config()

    if cfg.enabled and cfg.api_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = cfg.api_key
        os.environ["LANGCHAIN_API_KEY"] = cfg.api_key
        os.environ["LANGSMITH_PROJECT"] = cfg.project
        os.environ["LANGCHAIN_PROJECT"] = cfg.project
        os.environ["LANGSMITH_ENDPOINT"] = cfg.endpoint
        os.environ["LANGCHAIN_ENDPOINT"] = cfg.endpoint
        logger.info(
            "LangSmith native tracing is ACTIVE for project '%s'.", cfg.project
        )
        return True
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False


def redact_sensitive_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize metadata dictionary by redacting credentials and sensitive keys."""
    clean: Dict[str, Any] = {}
    for k, v in metadata.items():
        if SENSITIVE_KEY_PATTERN.search(str(k)):
            clean[k] = "[REDACTED]"
        elif isinstance(v, dict):
            clean[k] = redact_sensitive_metadata(v)
        elif isinstance(v, (list, tuple)):
            clean[k] = [
                redact_sensitive_metadata(item) if isinstance(item, dict) else item
                for item in v
            ]
        elif isinstance(v, str) and ("sk-" in v or "lsv2_" in v):
            clean[k] = "[REDACTED_API_KEY]"
        else:
            clean[k] = v
    return clean


def create_runnable_tracing_config(
    source: str = "streamlit",
    session_id: Optional[str] = None,
    scenario_id: Optional[str] = None,
    model_name: Optional[str] = None,
    turn_index: Optional[int] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    extra_tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate a standard RunnableConfig for LangGraph invocation.

    Constructs a readable run_name, contextual tags, and sanitized metadata.
    """
    safe_session = (session_id[:8] if session_id else "unknown")

    # Meaningful run_name
    if scenario_id:
        run_name = f"Agent Turn ({scenario_id})"
    else:
        run_name = f"Agent Turn ({source} - {safe_session})"

    tags: List[str] = ["agentic-data-analyst", source]
    if scenario_id:
        tags.append(scenario_id)
    if extra_tags:
        tags.extend([t for t in extra_tags if t not in tags])

    raw_metadata: Dict[str, Any] = {
        "application": "agentic-data-analyst",
        "source": source,
        "session_id": safe_session,
    }
    if scenario_id:
        raw_metadata["scenario_id"] = scenario_id
    if model_name:
        raw_metadata["model"] = model_name
    if turn_index is not None:
        raw_metadata["turn_index"] = turn_index

    if extra_metadata:
        raw_metadata.update(extra_metadata)

    clean_metadata = redact_sensitive_metadata(raw_metadata)

    return {
        "run_name": run_name,
        "tags": tags,
        "metadata": clean_metadata,
    }
