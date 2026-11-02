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
DEFAULT_FLUSH_TIMEOUT_SECONDS: float = 3.0

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
    workspace_id: Optional[str] = None


def _is_truthy(val: Optional[str]) -> bool:
    """Return True if string represents a boolean true value."""
    if not val:
        return False
    return str(val).strip().lower() in ("true", "1", "yes", "y", "t", "on")


def _resolve_val_from_env_or_secrets(
    keys: Sequence[str],
    env: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Retrieve first non-empty configuration value from environment or Streamlit secrets."""
    # 1. Check provided env mapping or os.environ
    effective_env = env if env is not None else os.environ
    for k in keys:
        v = effective_env.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()

    # 2. Check Streamlit secrets if running inside Streamlit Cloud / App context
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            for k in keys:
                if k in st.secrets:
                    sec_val = str(st.secrets[k]).strip()
                    if sec_val:
                        return sec_val
    except Exception:
        pass

    return None


def load_langsmith_config(env: Optional[Mapping[str, str]] = None) -> LangSmithConfig:
    """Load LangSmith configuration from environment variables or Streamlit secrets.

    Reads:
    - LANGSMITH_TRACING (or LANGCHAIN_TRACING_V2) -> bool (default: False)
    - LANGSMITH_API_KEY (or LANGCHAIN_API_KEY) -> Optional[str]
    - LANGSMITH_PROJECT (or LANGCHAIN_PROJECT) -> str (default: 'agentic-data-analyst')
    - LANGSMITH_ENDPOINT (or LANGCHAIN_ENDPOINT) -> str (default: 'https://api.smith.langchain.com')
    - LANGSMITH_WORKSPACE_ID (or LANGCHAIN_WORKSPACE_ID) -> Optional[str]

    When env is None, load_dotenv() is called first without overriding existing variables.
    Tracing is strictly opt-in and disabled by default.
    """
    if env is None:
        load_dotenv(override=False)

    raw_tracing = _resolve_val_from_env_or_secrets(
        ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"),
        env=env,
    )
    is_enabled = _is_truthy(raw_tracing)

    raw_key = _resolve_val_from_env_or_secrets(
        ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"),
        env=env,
    )

    project = (
        _resolve_val_from_env_or_secrets(
            ("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"),
            env=env,
        )
        or DEFAULT_LANGSMITH_PROJECT
    )

    endpoint = (
        _resolve_val_from_env_or_secrets(
            ("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT"),
            env=env,
        )
        or DEFAULT_LANGSMITH_ENDPOINT
    )

    workspace_id = _resolve_val_from_env_or_secrets(
        ("LANGSMITH_WORKSPACE_ID", "LANGCHAIN_WORKSPACE_ID"),
        env=env,
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
        workspace_id=workspace_id,
    )


# Module-level cached tracer and client to prevent redundant connection setup
_CACHED_TRACER: Optional[Any] = None
_CACHED_CLIENT: Optional[Any] = None


def get_langsmith_client(config: Optional[LangSmithConfig] = None) -> Optional[Any]:
    """Obtain or initialize a singleton LangSmith Client instance."""
    global _CACHED_CLIENT
    cfg = config or load_langsmith_config()
    if not (cfg.enabled and cfg.api_key):
        return None

    if _CACHED_CLIENT is None or getattr(_CACHED_CLIENT, "api_key", None) != cfg.api_key:
        try:
            from langsmith import Client

            _CACHED_CLIENT = Client(
                api_key=cfg.api_key,
                api_url=cfg.endpoint,
                workspace_id=cfg.workspace_id,
            )
        except Exception as exc:
            logger.warning("Failed to initialize LangSmith Client: %s", exc)
            return None

    return _CACHED_CLIENT


def get_langchain_tracer(config: Optional[LangSmithConfig] = None) -> Optional[Any]:
    """Obtain or initialize a configured LangChainTracer bound to the current project and client."""
    global _CACHED_TRACER
    cfg = config or load_langsmith_config()
    if not (cfg.enabled and cfg.api_key):
        return None

    client = get_langsmith_client(cfg)
    if client is None:
        return None

    if (
        _CACHED_TRACER is None
        or getattr(_CACHED_TRACER, "project_name", None) != cfg.project
        or getattr(_CACHED_TRACER, "client", None) is not client
    ):
        try:
            from langchain_core.tracers import LangChainTracer

            _CACHED_TRACER = LangChainTracer(
                project_name=cfg.project,
                client=client,
            )
        except Exception as exc:
            logger.warning("Failed to initialize LangChainTracer: %s", exc)
            return None

    return _CACHED_TRACER


def setup_langsmith_environment(config: Optional[LangSmithConfig] = None) -> bool:
    """Synchronize process environment variables for LangChain / LangSmith native tracing.

    Sets both LANGSMITH_* and LANGCHAIN_* environment variables to ensure compatibility
    across all versions of the LangChain and LangGraph SDKs.

    Returns:
        True if tracing is actively enabled and configured with an API key, False otherwise.
    """
    global _CACHED_TRACER, _CACHED_CLIENT
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
        if cfg.workspace_id:
            os.environ["LANGSMITH_WORKSPACE_ID"] = cfg.workspace_id
            os.environ["LANGCHAIN_WORKSPACE_ID"] = cfg.workspace_id

        # Invalidate stale global client in run_trees if it was initialized before env setup
        try:
            import langsmith.run_trees as rt

            if hasattr(rt, "_CLIENT") and rt._CLIENT is not None:
                rt._CLIENT = None
        except Exception:
            pass

        _CACHED_TRACER = None
        _CACHED_CLIENT = None

        key_len = len(cfg.api_key)
        masked_key = f"{cfg.api_key[:8]}...{cfg.api_key[-4:]}" if key_len > 12 else "[CONFIGURED]"
        logger.info(
            "LangSmith native tracing is ACTIVE for project '%s' (endpoint: %s, workspace: %s, key: %s).",
            cfg.project,
            cfg.endpoint,
            cfg.workspace_id or "default",
            masked_key,
        )
        return True
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        _CACHED_TRACER = None
        _CACHED_CLIENT = None
        return False


def flush_tracing(
    tracer: Optional[Any] = None,
    timeout: float = DEFAULT_FLUSH_TIMEOUT_SECONDS,
) -> bool:
    """Synchronously flush pending buffered traces to LangSmith before UI reruns or thread termination.

    Ensures that asynchronously batched runs in LangSmith's queue/buffer are transmitted
    over HTTP before Streamlit calls st.rerun() or exits script context.

    Args:
        tracer: Optional LangChainTracer instance. Defaults to singleton tracer if None.
        timeout: Maximum seconds to wait for pending HTTP uploads to drain.

    Returns:
        True if traces were flushed successfully or tracing was inactive; False on timeout/error.
    """
    import time

    cfg = load_langsmith_config()
    target_tracer = tracer or (get_langchain_tracer(cfg) if (cfg.enabled and cfg.api_key) else None)
    target_client = getattr(target_tracer, "client", None) or (
        get_langsmith_client(cfg) if (cfg.enabled and cfg.api_key) else None
    )

    if target_tracer is None and target_client is None:
        return True

    start_time = time.monotonic()
    project_label = getattr(target_tracer, "project_name", None) or cfg.project
    try:
        # 1. Drain pending tracer futures if available
        if target_tracer is not None and hasattr(target_tracer, "wait_for_futures"):
            target_tracer.wait_for_futures()

        # 2. Flush client ops buffer and batch queue
        if hasattr(target_client, "flush"):
            target_client.flush(timeout=timeout)

        elapsed = time.monotonic() - start_time
        logger.info(
            "[LangSmith] Flushed pending traces for project '%s' in %.3fs.",
            project_label,
            elapsed,
        )
        return True
    except Exception as exc:
        elapsed = time.monotonic() - start_time
        logger.warning(
            "[LangSmith] Trace flush encountered an issue after %.3fs for project '%s': %s",
            elapsed,
            project_label,
            str(exc),
        )
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
    extra_callbacks: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Generate a standard RunnableConfig for LangGraph invocation.

    Constructs a readable run_name, contextual tags, sanitized metadata,
    and explicitly binds the LangChainTracer to callbacks when tracing is active.
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

    callbacks: List[Any] = list(extra_callbacks) if extra_callbacks else []

    # Bind explicit LangChainTracer if tracing is enabled
    cfg = load_langsmith_config()
    if cfg.enabled and cfg.api_key:
        tracer = get_langchain_tracer(cfg)
        if tracer is not None and tracer not in callbacks:
            callbacks.append(tracer)

    config: Dict[str, Any] = {
        "run_name": run_name,
        "tags": tags,
        "metadata": clean_metadata,
    }
    if callbacks:
        config["callbacks"] = callbacks

    return config
