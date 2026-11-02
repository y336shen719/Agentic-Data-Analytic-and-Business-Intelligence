"""Streamlit session management, state persistence, and artifact extraction.

Provides session-scoped dependencies (DatasetStore, ChartRegistry, tools, compiled LangGraph),
multi-turn conversation handling, tool execution extraction, and conversation reset.
Decoupled from Streamlit UI runtime to enable deterministic automated testing.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import uuid

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from agent.config import DEFAULT_OPENAI_MODEL, LLMConfig
from agent.graph import (
    DEFAULT_MAX_PYTHON_RETRIES,
    DEFAULT_MAX_SQL_RETRIES,
    DEFAULT_MAX_STEPS,
    create_agent_graph,
)
from agent.llm import create_llm
from agent.state import create_initial_state
from agent.tool_wrappers import (
    create_chart_generation_tools,
    create_python_analysis_tools,
    create_sql_tools,
)
from agent.tracing import (
    create_runnable_tracing_config,
    setup_langsmith_environment,
)
from data.prepare_data import DEFAULT_DB_PATH
from tools.chart_registry import ChartRegistry
from tools.dataset_store import DatasetStore

# Ensure environment is configured for LangSmith tracing if enabled
setup_langsmith_environment()

DEFAULT_MAX_TURNS_CONTEXT: int = 10
MAX_HISTORY_MESSAGES: int = 100
DEFAULT_INCOMPLETE_MESSAGE: str = (
    "I was unable to complete the analysis. No final response was generated."
)

logger = logging.getLogger(__name__)


def extract_text_from_message_content(content: Any) -> str:
    """Extract plain text narrative from message content.

    Handles plain strings, lists of string fragments, and structured content block dictionaries
    (e.g. OpenAI/Anthropic format `[{'type': 'text', 'text': '...'}]` or `[{'text': '...'}]`).
    Filters out non-text blocks such as tool use, function call args, or reasoning markers.

    Args:
        content: The raw `content` field of a BaseMessage (str, list, dict, or None).

    Returns:
        Stripped string representing the extracted narrative.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif block_type is None and isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return "".join(parts).strip()
    return ""


def sanitize_user_facing_narrative(text: str) -> str:
    """Sanitize technical identifiers (chart_*, ds_*) from user-facing narrative.

    Narrowly targets leaked internal IDs and explicit 'Chart: chart_...' mentions
    while strictly preserving business values, metrics, and explanatory text.
    """
    if not text:
        return ""
    # 1. Markdown links referencing internal chart IDs: [Title](chart_...) -> Title
    cleaned = re.sub(r"\[([^\]]+)\]\(chart_[a-f0-9]+\)", r"\1", text)
    # 2. Markdown links referencing internal dataset IDs: [Title](ds_...) -> Title
    cleaned = re.sub(r"\[([^\]]+)\]\(ds_[a-f0-9]+\)", r"\1", text)
    # 3. Explicit labels like "Chart: chart_002c486a1a20", "Chart ID: `chart_...`", "Figure: chart_..."
    cleaned = re.sub(r"(?i)\b(?:chart|figure)(?:\s*id)?\s*:\s*`?chart_[a-f0-9]{8,32}`?", "", cleaned)
    # 4. Explicit labels like "Dataset: ds_...", "Dataset ID: `ds_...`"
    cleaned = re.sub(r"(?i)\b(?:dataset)(?:\s*id)?\s*:\s*`?ds_[a-f0-9]{16,32}`?", "", cleaned)
    # 5. Standalone or backticked chart IDs: `chart_002c486a1a20` or chart_002c486a1a20
    cleaned = re.sub(r"`?chart_[a-f0-9]{8,32}`?", "", cleaned)
    # 6. Standalone or backticked dataset IDs: `ds_...` or ds_...
    cleaned = re.sub(r"`?ds_[a-f0-9]{16,32}`?", "", cleaned)
    # 7. Normalize leftover empty lines and trailing whitespaces
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


@dataclass
class AgentSession:
    """Session-scoped container for stateful analytics agent components."""

    session_id: str
    dataset_store: DatasetStore
    chart_registry: ChartRegistry
    tools: list[BaseTool]
    graph: CompiledStateGraph
    llm: Any
    configured_model: str = DEFAULT_OPENAI_MODEL
    configured_api_key: Optional[str] = field(default=None, repr=False)
    agent_messages: list[BaseMessage] = field(default_factory=list)
    ui_messages: list[Dict[str, Any]] = field(default_factory=list)
    db_path: Path = field(default_factory=lambda: Path(DEFAULT_DB_PATH))
    max_steps: int = DEFAULT_MAX_STEPS
    max_sql_retries: int = DEFAULT_MAX_SQL_RETRIES
    max_python_retries: int = DEFAULT_MAX_PYTHON_RETRIES


def resolve_api_key(user_input_key: Optional[str] = None) -> Optional[str]:
    """Resolve OpenAI API key from user input, Streamlit secrets, or environment.

    Precedence:
    1. Explicit nonempty user-provided key.
    2. Streamlit secrets (if available).
    3. Environment variable OPENAI_API_KEY.
    4. Otherwise, None.
    """
    if user_input_key and user_input_key.strip():
        return user_input_key.strip()

    try:
        import streamlit as st

        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            sec = str(st.secrets["OPENAI_API_KEY"]).strip()
            if sec:
                return sec
    except Exception:
        pass

    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    return None


def trim_messages_for_context(
    messages: Sequence[BaseMessage],
    max_turns: int = DEFAULT_MAX_TURNS_CONTEXT,
    max_messages: int = MAX_HISTORY_MESSAGES,
) -> list[BaseMessage]:
    """Trim conversation history preserving tool-call and tool-message pairing.

    Identifies HumanMessage turn boundaries and retains the maximum number of recent
    complete user turns such that:
    1. The number of retained turns is <= `max_turns`.
    2. The total count of messages is <= `max_messages` (soft limit with an explicit
       oversized-turn exception: if the single most recent turn alone exceeds `max_messages`,
       that entire turn is preserved intact to prevent breaking tool-call/ToolMessage pairing).

    Guarantees that the returned list always starts with a HumanMessage and that
    AIMessage tool_calls are never split from their matching ToolMessages.
    """
    if not messages:
        return []

    human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if not human_indices:
        return list(messages[-max_messages:])

    # Candidate start indices for the last max_turns turns
    candidate_turn_starts = human_indices[-max_turns:]

    # Select the earliest candidate turn start where total messages <= max_messages
    chosen_start = None
    for start_idx in candidate_turn_starts:
        turn_len = len(messages) - start_idx
        if turn_len <= max_messages:
            chosen_start = start_idx
            break

    # If even the single most recent turn exceeds max_messages, retain that entire newest turn
    # as an explicit oversized-turn exception to preserve complete tool-call pairing.
    if chosen_start is None:
        chosen_start = candidate_turn_starts[-1]

    return list(messages[chosen_start:])


def extract_turn_details(
    new_messages: Sequence[BaseMessage],
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """Extract final assistant response text, generated chart IDs, and tool execution steps.

    Robustly parses ToolMessages (handling dicts, lists, scalars, nulls, and invalid JSON),
    classifies tool-level errors, prevents raw exception leaks, and collects valid chart IDs.

    Args:
        new_messages: Messages produced by LangGraph during the current turn.

    Returns:
        Tuple of (assistant_text, chart_ids, tool_execution_summaries).
    """
    chart_ids: List[str] = []
    tool_executions: List[Dict[str, Any]] = []
    assistant_text: str = ""

    # Map tool_call_id to invocation args
    tool_calls_by_id: Dict[str, Dict[str, Any]] = {}
    for msg in new_messages:
        if isinstance(msg, AIMessage):
            tcs = getattr(msg, "tool_calls", None) or []
            for tc in tcs:
                if isinstance(tc, dict) and "id" in tc:
                    tool_calls_by_id[tc["id"]] = tc

    for msg in new_messages:
        if isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "") or ""
            tcid = getattr(msg, "tool_call_id", "") or ""
            call_info = tool_calls_by_id.get(tcid, {})
            call_args = call_info.get("args", {})

            # Robust JSON parsing (handles dict, list, string, scalar, null, and invalid syntax)
            raw_parsed: Any = None
            tool_content = msg.content
            if isinstance(tool_content, list):
                parts = []
                for block in tool_content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append(block["text"])
                tool_content = "".join(parts)

            if isinstance(tool_content, str) and tool_content.strip():
                try:
                    raw_parsed = json.loads(tool_content)
                except Exception:
                    try:
                        raw_parsed = ast.literal_eval(tool_content)
                    except Exception:
                        raw_parsed = None
            elif isinstance(tool_content, dict):
                raw_parsed = tool_content

            data: Dict[str, Any] = raw_parsed if isinstance(raw_parsed, dict) else {}

            # Status resolution: ToolMessage-level error takes precedence
            msg_status = getattr(msg, "status", None)
            if msg_status == "error":
                status = "error"
            elif "status" in data and isinstance(data["status"], str):
                status = data["status"]
            else:
                status = "completed"

            if tool_name == "execute_sql":
                query = call_args.get("query", "N/A") if isinstance(call_args, dict) else "N/A"
                row_cnt = data.get("row_count", "N/A")
                ds_id = data.get("dataset_id", "N/A")
                tool_executions.append({
                    "tool": tool_name,
                    "title": "🔍 SQL Query Execution",
                    "details": query,
                    "language": "sql",
                    "summary": f"Status: {status} | Rows: {row_cnt} | Dataset ID: {ds_id}",
                    "status": status,
                })

            elif tool_name == "get_database_schema":
                tbls = call_args.get("table_names") if isinstance(call_args, dict) else None
                tbl_desc = ", ".join(tbls) if isinstance(tbls, list) else "All Tables"
                tool_executions.append({
                    "tool": tool_name,
                    "title": "📋 Schema Inspection",
                    "details": f"Tables: {tbl_desc}",
                    "language": "text",
                    "summary": f"Status: {status}",
                    "status": status,
                })

            elif tool_name == "execute_python_analysis":
                op = call_args.get("operation", "N/A") if isinstance(call_args, dict) else "N/A"
                ds_in = call_args.get("dataset_id", "N/A") if isinstance(call_args, dict) else "N/A"
                derived_id = data.get("derived_dataset_id", "None")
                tool_executions.append({
                    "tool": tool_name,
                    "title": f"🐍 Python Analysis ({op})",
                    "details": json.dumps(call_args, indent=2) if isinstance(call_args, dict) else str(call_args),
                    "language": "json",
                    "summary": f"Status: {status} | Source: {ds_in} | Derived: {derived_id}",
                    "status": status,
                })

            elif tool_name == "generate_chart":
                ctype = call_args.get("chart_type", "chart") if isinstance(call_args, dict) else "chart"
                cid = data.get("chart_id")
                if not cid and isinstance(tool_content, str):
                    m = re.search(r"['\"]chart_id['\"]\s*:\s*['\"]([^'\"]+)['\"]", tool_content)
                    if m:
                        cid = m.group(1)
                        if status != "error":
                            status = "success"

                is_valid_cid = isinstance(cid, str) and bool(cid.strip())

                if status == "success" and is_valid_cid:
                    chart_ids.append(cid.strip())
                    summary_text = f"Status: success | Chart ID: {cid.strip()}"
                else:
                    summary_text = f"Status: {status} | Chart generation failed"

                tool_executions.append({
                    "tool": tool_name,
                    "title": f"📊 Chart Generation ({str(ctype).title()})",
                    "details": json.dumps(call_args, indent=2) if isinstance(call_args, dict) else str(call_args),
                    "language": "json",
                    "summary": summary_text,
                    "status": status,
                })

            else:
                tool_executions.append({
                    "tool": tool_name,
                    "title": f"⚙️ Tool: {tool_name}",
                    "details": str(call_args),
                    "language": "text",
                    "summary": f"Status: {status}",
                    "status": status,
                })

    # Track latest assistant narrative, prioritizing the final user-facing response
    # 1. Search in reverse for a final non-tool-calling AIMessage with narrative text
    for msg in reversed(new_messages):
        if isinstance(msg, AIMessage):
            has_tool_calls = bool(getattr(msg, "tool_calls", None))
            if not has_tool_calls:
                text = extract_text_from_message_content(msg.content)
                if text:
                    assistant_text = text
                    break

    # 2. Fallback: if no non-tool-calling AIMessage has text, check if any AIMessage
    # has text (supports single-turn fixtures where AIMessage includes tool calls and text)
    if not assistant_text:
        for msg in reversed(new_messages):
            if isinstance(msg, AIMessage):
                text = extract_text_from_message_content(msg.content)
                if text:
                    assistant_text = text
                    break

    # 3. Clean user-facing narrative and fallback if empty
    sanitized_text = sanitize_user_facing_narrative(assistant_text)
    if not sanitized_text:
        sanitized_text = DEFAULT_INCOMPLETE_MESSAGE

    return sanitized_text, chart_ids, tool_executions


def create_agent_session(
    db_path: Optional[Union[str, Path]] = None,
    llm: Optional[Any] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_sql_retries: int = DEFAULT_MAX_SQL_RETRIES,
    max_python_retries: int = DEFAULT_MAX_PYTHON_RETRIES,
) -> AgentSession:
    """Construct an isolated, session-scoped agent container with fresh stores and tools.

    Args:
        db_path: Path to target DuckDB database. Defaults to DEFAULT_DB_PATH.
        llm: Optional custom or preconfigured LLM instance.
        api_key: Optional OpenAI API key to construct LLM if llm is None.
        model: Optional model name override.
        max_steps: Maximum LLM step limit.
        max_sql_retries: Maximum SQL correction attempts.
        max_python_retries: Maximum Python analysis correction attempts.

    Returns:
        Configured AgentSession instance.
    """
    effective_db = Path(db_path) if db_path else Path(DEFAULT_DB_PATH)
    session_id = f"sess_{uuid.uuid4().hex[:10]}"

    # Session-scoped stores
    store = DatasetStore()
    registry = ChartRegistry()

    # Shared tools bound to session-scoped stores
    sql_tools = create_sql_tools(db_path=effective_db, dataset_store=store)
    py_tools = create_python_analysis_tools(dataset_store=store)
    chart_tools = create_chart_generation_tools(dataset_store=store, chart_registry=registry)
    all_tools = list(sql_tools) + list(py_tools) + list(chart_tools)

    effective_model = model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    resolved_key = resolve_api_key(api_key)

    # LLM configuration
    effective_llm = llm
    if effective_llm is None:
        cfg = LLMConfig(
            model=effective_model,
            timeout_seconds=30.0,
            max_retries=2,
            api_key=resolved_key,
        )
        effective_llm = create_llm(config=cfg)

    # Compiled LangGraph
    graph = create_agent_graph(
        llm=effective_llm,
        tools=all_tools,
        max_steps=max_steps,
        max_sql_retries=max_sql_retries,
        max_python_retries=max_python_retries,
    )

    return AgentSession(
        session_id=session_id,
        dataset_store=store,
        chart_registry=registry,
        tools=all_tools,
        graph=graph,
        llm=effective_llm,
        configured_model=effective_model,
        configured_api_key=resolved_key,
        agent_messages=[],
        ui_messages=[],
        db_path=effective_db,
        max_steps=max_steps,
        max_sql_retries=max_sql_retries,
        max_python_retries=max_python_retries,
    )


def reconfigure_session(
    session: AgentSession,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    llm: Optional[Any] = None,
) -> None:
    """Rebuild the active LLM and LangGraph when model or credentials change.

    Preserves existing DatasetStore, ChartRegistry, tools, and message histories.
    """
    effective_model = model or session.configured_model or DEFAULT_OPENAI_MODEL
    effective_key = resolve_api_key(api_key) if api_key is not None else session.configured_api_key

    if llm is not None:
        effective_llm = llm
    else:
        cfg = LLMConfig(
            model=effective_model,
            timeout_seconds=30.0,
            max_retries=2,
            api_key=effective_key,
        )
        effective_llm = create_llm(config=cfg)

    new_graph = create_agent_graph(
        llm=effective_llm,
        tools=session.tools,
        max_steps=session.max_steps,
        max_sql_retries=session.max_sql_retries,
        max_python_retries=session.max_python_retries,
    )

    session.llm = effective_llm
    session.graph = new_graph
    session.configured_model = effective_model
    session.configured_api_key = effective_key


def execute_session_turn(
    session: AgentSession,
    user_query: str,
    max_turns_context: int = DEFAULT_MAX_TURNS_CONTEXT,
    source: str = "streamlit",
    scenario_id: Optional[str] = None,
    turn_index: Optional[int] = None,
    run_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a single multi-turn interaction within the session.

    Appends the user query, prepares bounded context, resets per-turn counters,
    invokes the session's LangGraph with tracing metadata, extracts artifacts, and updates UI history.
    Maintains semantic consistency between agent conversation history and UI history
    even when graph execution fails unexpectedly.

    Args:
        session: Active AgentSession.
        user_query: User natural language prompt.
        max_turns_context: Number of preceding turns retained for LLM context.
        source: Calling source ('streamlit' or 'evaluation').
        scenario_id: Optional evaluation benchmark scenario identifier.
        turn_index: 0-indexed turn position within the conversation.
        run_config: Optional additional LangChain RunnableConfig parameters.

    Returns:
        Dictionary representing the assistant's response record for the UI.
    """
    clean_query = user_query.strip()
    user_msg = HumanMessage(content=clean_query)
    session.agent_messages.append(user_msg)

    # UI message for user
    session.ui_messages.append({
        "role": "user",
        "content": clean_query,
    })

    # Prepare context bounded by turn boundaries and message limits
    context_messages = trim_messages_for_context(
        session.agent_messages,
        max_turns=max_turns_context,
        max_messages=MAX_HISTORY_MESSAGES,
    )
    prior_len = len(context_messages)

    # Reset per-turn execution counters
    initial_turn_state = create_initial_state(
        messages=context_messages,
        step_count=0,
        last_sql=None,
        last_error=None,
        sql_retry_count=0,
        last_error_category=None,
        python_retry_count=0,
        pending_python_recovery=False,
    )

    # Prepare LangSmith / LangGraph invocation config with safe metadata
    effective_config = create_runnable_tracing_config(
        source=source,
        session_id=session.session_id,
        scenario_id=scenario_id,
        model_name=session.configured_model,
        turn_index=turn_index,
    )
    if run_config:
        effective_config.update(run_config)

    try:
        final_state = session.graph.invoke(initial_turn_state, config=effective_config)
        all_graph_messages = final_state.get("messages", [])
        new_messages = all_graph_messages[prior_len:]

        # Preserve newly generated messages into conversation history
        session.agent_messages.extend(new_messages)

        # Enforce total message history bound on session.agent_messages
        session.agent_messages = trim_messages_for_context(
            session.agent_messages,
            max_turns=DEFAULT_MAX_TURNS_CONTEXT * 2,
            max_messages=MAX_HISTORY_MESSAGES * 2,
        )

        assistant_text, chart_ids, tool_executions = extract_turn_details(new_messages)
        if logger.isEnabledFor(logging.DEBUG):
            ai_types = [type(m).__name__ for m in new_messages]
            last_ai = next((m for m in reversed(new_messages) if isinstance(m, AIMessage)), None)
            logger.debug(
                "execute_session_turn: input_len=%d, output_len=%d, new_len=%d, "
                "types=%s, last_ai_content_type=%s, last_ai_tool_calls=%d, "
                "step_count=%s, error_category=%s",
                prior_len,
                len(all_graph_messages),
                len(new_messages),
                ai_types,
                type(last_ai.content).__name__ if last_ai else "None",
                len(getattr(last_ai, "tool_calls", []) or []) if last_ai else 0,
                final_state.get("step_count"),
                final_state.get("last_error_category"),
            )

        assistant_record = {
            "role": "assistant",
            "content": assistant_text,
            "chart_ids": chart_ids,
            "tool_executions": tool_executions,
        }

    except Exception:
        safe_msg = "I encountered an unexpected error while processing your request. Please try again."
        # Maintain consistent agent conversation history on failure
        session.agent_messages.append(AIMessage(content=safe_msg))
        assistant_record = {
            "role": "assistant",
            "content": safe_msg,
            "chart_ids": [],
            "tool_executions": [],
            "error": True,
        }

    session.ui_messages.append(assistant_record)
    return assistant_record


def reset_agent_session(session: AgentSession) -> None:
    """Clear conversation history, dataset store, and chart registry."""
    session.dataset_store.clear()
    session.chart_registry.clear()
    session.agent_messages.clear()
    session.ui_messages.clear()


def get_or_create_session(
    state_dict: Dict[str, Any],
    db_path: Optional[Union[str, Path]] = None,
    llm: Optional[Any] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    session_key: str = "agent_session",
) -> AgentSession:
    """Retrieve existing session from state dict or instantiate a new one.

    If the session already exists, detects whether model or credentials changed
    and reconfigures the active LLM and graph while preserving session history and stores.
    """
    effective_model = model or DEFAULT_OPENAI_MODEL
    if session_key not in state_dict or state_dict[session_key] is None:
        session = create_agent_session(
            db_path=db_path,
            llm=llm,
            api_key=api_key,
            model=effective_model,
        )
        state_dict[session_key] = session
    else:
        session = state_dict[session_key]
        resolved_key = resolve_api_key(api_key) if api_key is not None else session.configured_api_key
        model_changed = bool(model and session.configured_model != model)
        key_changed = (resolved_key != session.configured_api_key)
        llm_changed = bool(llm is not None and session.llm is not llm)
        if model_changed or key_changed or llm_changed:
            reconfigure_session(
                session,
                model=model,
                api_key=api_key,
                llm=llm,
            )
    return state_dict[session_key]
