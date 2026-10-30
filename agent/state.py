"""Typed state definition for the Agentic Data Analyst workflow.

Defines the central `AgentState` schema passed across LangGraph nodes and edges.
Message history uses the `add_messages` reducer for message preservation and ID-based
upserts, while scalar metadata fields use LangGraph's default overwrite semantics.
"""

from __future__ import annotations

from typing import Annotated, Optional, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Core state schema for the agentic analytical workflow.

    Fields:
        messages: Conversation history, managed via LangGraph's `add_messages` reducer
            (appends new messages or updates existing messages sharing the same ID).
        step_count: Cumulative count of completed LLM node invocations.
        last_sql: Most recently requested SQL query string, or None.
        last_error: Latest SQL execution error ("<error_type>: <message>"), or None
            after a successful SQL execution.
        sql_retry_count: Number of SQL correction retries actually executed for this
            request, excluding the initial attempt. A retry is a SQL execution that
            happens while ``last_error`` is set. Never reset within a request.
        last_error_category: ``classify_sql_result`` category of ``last_error``
            (None when ``last_error`` is None). Used for deterministic retry routing.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int
    last_sql: Optional[str]
    last_error: Optional[str]
    sql_retry_count: int
    last_error_category: Optional[str]


def create_initial_state(
    messages: Optional[Sequence[BaseMessage]] = None,
    step_count: int = 0,
    last_sql: Optional[str] = None,
    last_error: Optional[str] = None,
    sql_retry_count: int = 0,
    last_error_category: Optional[str] = None,
) -> AgentState:
    """Helper to construct a fresh, typed initial AgentState dictionary.

    Args:
        messages: Optional initial sequence of BaseMessage instances. Defaults to empty list.
        step_count: Initial LLM invocation step count (default 0).
        last_sql: Initial SQL string (default None).
        last_error: Initial error string (default None).
        sql_retry_count: Initial retry count (default 0).
        last_error_category: Initial error category (default None).

    Returns:
        AgentState dictionary initialized with clean values.
    """
    return {
        "messages": list(messages) if messages is not None else [],
        "step_count": step_count,
        "last_sql": last_sql,
        "last_error": last_error,
        "sql_retry_count": sql_retry_count,
        "last_error_category": last_error_category,
    }
