"""LangGraph orchestration graph for the Agentic Data Analyst workflow.

Topology::

    START --(step_count >= max_steps)--> limit_exceeded --> END
      |
      v
     llm --(no tool_calls)--------------------------------> END
      |  --(max_steps reached, tool_calls pending)--------> limit_exceeded --> END
      |  --(SQL calls would exceed retry budget)----------> recovery_stop --> END
      v
    tools (ToolNode) --> inspect_tool_results
                              |--(success / retry allowed / non-SQL)--> llm
                              |--(non-retryable error / budget spent)-> recovery_stop --> END

Counters:
    * ``step_count``       - number of completed LLM invocations; bounded by ``max_steps``.
    * ``sql_retry_count``  - SQL executions that ran while ``last_error`` was set (i.e. after a
                             failed attempt), excluding the initial attempt; bounded by
                             ``max_sql_retries``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from agent.llm import create_llm
from agent.prompts import SYSTEM_PROMPT
from agent.sql_error_classifier import RECOVERABLE_SQL_ERROR, classify_sql_result
from agent.state import AgentState
from agent.tool_wrappers import create_sql_tools

DEFAULT_MAX_STEPS: int = 12
DEFAULT_MAX_SQL_RETRIES: int = 2
SQL_TOOL_NAME: str = "execute_sql"
_MAX_ERROR_CHARS: int = 500


def _parse_tool_content(msg: ToolMessage) -> Any:
    """Return the structured SQL result carried by a ToolMessage, or None if unusable.

    ToolNode serialises dict results to a JSON string. Content-block lists (text blocks) are
    also accepted. ToolNode-level failures (``status == "error"``, e.g. invalid arguments)
    and anything that is not valid JSON yield None, which the classifier treats as malformed.
    """
    if getattr(msg, "status", "success") == "error":
        return None

    content = msg.content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            else:
                return None
        content = "".join(parts)

    if not isinstance(content, str):
        return None
    try:
        return json.loads(content)
    except ValueError:
        return None


def _latest_tool_batch(messages: Sequence[BaseMessage]) -> tuple[Optional[AIMessage], Dict[str, ToolMessage]]:
    """Return the AIMessage that requested the latest ToolNode run and its ToolMessages by id."""
    idx = len(messages)
    while idx > 0 and isinstance(messages[idx - 1], ToolMessage):
        idx -= 1
    ai_msg = messages[idx - 1] if idx > 0 and isinstance(messages[idx - 1], AIMessage) else None
    return ai_msg, {m.tool_call_id: m for m in messages[idx:]}


def _sql_call_count(message: Any) -> int:
    calls = getattr(message, "tool_calls", None) or []
    return sum(1 for tc in calls if tc.get("name") == SQL_TOOL_NAME)


def _unexecuted_tool_messages(message: Any, reason: str) -> list[BaseMessage]:
    """One error ToolMessage per pending tool call, keeping the original tool_call_id."""
    out: list[BaseMessage] = []
    for tc in getattr(message, "tool_calls", None) or []:
        name = tc.get("name", "unknown_tool")
        out.append(
            ToolMessage(
                content=f"Tool '{name}' was not executed because {reason}.",
                name=name,
                tool_call_id=tc.get("id") or "unknown_call_id",
                status="error",
            )
        )
    return out


def create_agent_graph(
    llm: Optional[Any] = None,
    tools: Optional[Sequence[BaseTool]] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    system_prompt: Optional[str] = None,
    max_sql_retries: int = DEFAULT_MAX_SQL_RETRIES,
) -> CompiledStateGraph:
    """Build and compile the stateful LangGraph agent workflow.

    Args:
        llm: Optional ChatOpenAI or custom chat model. If None, defaults to `create_llm()`.
        tools: Optional sequence of BaseTool instances. If None, defaults to `create_sql_tools()`.
        max_steps: Maximum LLM invocations (strictly positive int). Defaults to 12.
        system_prompt: Optional system instruction prompt. If None, defaults to SYSTEM_PROMPT.
        max_sql_retries: SQL correction retries allowed after the initial attempt
            (non-negative int; 0 disables correction). Defaults to 2.

    Raises:
        ValueError: If ``max_steps`` or ``max_sql_retries`` is invalid.
    """
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError(
            f"max_steps must be a strictly positive integer, got: {max_steps!r} "
            f"(type: {type(max_steps).__name__})"
        )
    if isinstance(max_sql_retries, bool) or not isinstance(max_sql_retries, int) or max_sql_retries < 0:
        raise ValueError(
            f"max_sql_retries must be a non-negative integer, got: {max_sql_retries!r} "
            f"(type: {type(max_sql_retries).__name__})"
        )

    effective_llm = llm if llm is not None else create_llm()
    approved_tools: list[BaseTool] = list(tools) if tools is not None else list(create_sql_tools())
    effective_system_prompt: str = system_prompt if system_prompt is not None else SYSTEM_PROMPT

    # Bind approved tools to the LLM if the model supports tool binding
    if hasattr(effective_llm, "bind_tools"):
        bound_llm = effective_llm.bind_tools(approved_tools)
    else:
        bound_llm = effective_llm

    # 1. LLM Node: Invokes model with history, increments step_count, tracks SQL
    def llm_node(state: AgentState) -> Dict[str, Any]:
        current_steps = state.get("step_count", 0)
        messages: list[BaseMessage] = list(state.get("messages", []))

        # Prepend system message if not already present in history
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=effective_system_prompt)] + messages

        response: BaseMessage = bound_llm.invoke(messages)

        # Track latest requested SQL query if execute_sql was invoked in tool_calls
        last_sql = state.get("last_sql")
        tool_calls = getattr(response, "tool_calls", None) or []
        for tc in tool_calls:
            if isinstance(tc, dict) and tc.get("name") == SQL_TOOL_NAME:
                query_arg = tc.get("args", {}).get("query")
                if isinstance(query_arg, str):
                    last_sql = query_arg

        updates: Dict[str, Any] = {
            "messages": [response],
            "step_count": current_steps + 1,
        }
        if last_sql is not None:
            updates["last_sql"] = last_sql

        return updates

    # 2. Inspection Node: classify SQL results of the latest ToolNode run (no SQL is executed here)
    def inspect_tool_results(state: AgentState) -> Dict[str, Any]:
        ai_msg, tool_msgs = _latest_tool_batch(state.get("messages", []))
        if ai_msg is None:
            return {}

        running_error = state.get("last_error")
        running_category = state.get("last_error_category")
        retries = state.get("sql_retry_count", 0)
        saw_sql = False

        # Deterministic order: the order of the originating tool calls, SQL calls only.
        for tc in ai_msg.tool_calls:
            if tc.get("name") != SQL_TOOL_NAME:
                continue
            saw_sql = True

            msg = tool_msgs.get(tc.get("id"))
            # A missing or mismatched ToolMessage is never treated as success.
            result = _parse_tool_content(msg) if msg is not None and msg.name in (None, SQL_TOOL_NAME) else None
            outcome = classify_sql_result(result)

            # A SQL execution that happens while an earlier failure is outstanding is a retry.
            if running_error is not None:
                retries += 1

            if outcome["status"] == "success":
                running_error, running_category = None, None
                continue

            running_error = f"{outcome['error_type']}: {outcome['error_message']}"[:_MAX_ERROR_CHARS]
            running_category = outcome["category"]
            if not outcome["retryable"]:
                break  # fail closed: first non-retryable failure decides; later results can't mask it

        if not saw_sql:
            return {}
        return {
            "last_error": running_error,
            "last_error_category": running_category,
            "sql_retry_count": retries,
        }

    # 3. Recovery Stop Node: explain why the workflow terminated (SQL failure / retry budget)
    def recovery_stop_node(state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        pending = _unexecuted_tool_messages(last_message, f"the SQL correction retry limit ({max_sql_retries}) would be exceeded")

        category = state.get("last_error_category")
        error = state.get("last_error")
        if pending:
            text = (
                f"Execution stopped: the requested SQL calls would exceed the SQL correction retry "
                f"limit ({max_sql_retries}). No pending tools were executed."
            )
        elif category == RECOVERABLE_SQL_ERROR:
            text = (
                f"Execution stopped: SQL still failed after {state.get('sql_retry_count', 0)} correction "
                f"retries (limit {max_sql_retries}). Last error: {error}"
            )
        elif category == "non_retryable_security_error":
            text = f"Execution stopped: the SQL query was rejected by security policy and will not be retried. Error: {error}"
        elif category == "non_retryable_resource_error":
            text = f"Execution stopped: SQL execution failed with an operational error and will not be retried. Error: {error}"
        else:
            text = f"Execution stopped: SQL returned an unexpected or unrecognised result and will not be retried. Error: {error}"

        return {"messages": [*pending, AIMessage(content=text)]}

    # 4. Limit Exceeded Node: pending ToolMessages + warning when max_steps is reached
    def limit_exceeded_node(state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        pending = _unexecuted_tool_messages(
            last_message,
            f"the workflow reached the maximum LLM invocation limit ({max_steps})",
        )
        warning_msg = AIMessage(
            content=(
                f"Execution stopped: maximum LLM invocation limit ({max_steps}) was reached before "
                "the agent concluded tool calling."
            )
        )
        return {"messages": [*pending, warning_msg]}

    # 5. Routers
    def route_start(state: AgentState) -> str:
        if state.get("step_count", 0) >= max_steps:
            return "limit_exceeded"
        return "llm"

    def route_llm_output(state: AgentState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return END

        last_message = messages[-1]
        if not getattr(last_message, "tool_calls", None):
            return END

        if state.get("step_count", 0) >= max_steps:
            return "limit_exceeded"

        # Pre-execution guard: SQL calls in one response cannot exceed the remaining budget.
        # Remaining executions = retries left, plus one initial attempt if no failure is outstanding.
        allowed = max_sql_retries - state.get("sql_retry_count", 0)
        if state.get("last_error") is None:
            allowed += 1
        if _sql_call_count(last_message) > allowed:
            return "recovery_stop"

        return "tools"

    def route_after_inspection(state: AgentState) -> str:
        if state.get("last_error") is None:
            return "llm"
        if state.get("last_error_category") != RECOVERABLE_SQL_ERROR:
            return "recovery_stop"
        if state.get("sql_retry_count", 0) >= max_sql_retries:
            return "recovery_stop"
        return "llm"

    # 6. Construct the Graph
    builder = StateGraph(AgentState)

    builder.add_node("llm", llm_node)
    builder.add_node("tools", ToolNode(approved_tools, handle_tool_errors=True))
    builder.add_node("inspect_tool_results", inspect_tool_results)
    builder.add_node("recovery_stop", recovery_stop_node)
    builder.add_node("limit_exceeded", limit_exceeded_node)

    builder.add_conditional_edges(
        START,
        route_start,
        {"llm": "llm", "limit_exceeded": "limit_exceeded"},
    )
    builder.add_conditional_edges(
        "llm",
        route_llm_output,
        {
            "tools": "tools",
            "limit_exceeded": "limit_exceeded",
            "recovery_stop": "recovery_stop",
            END: END,
        },
    )
    builder.add_edge("tools", "inspect_tool_results")
    builder.add_conditional_edges(
        "inspect_tool_results",
        route_after_inspection,
        {"llm": "llm", "recovery_stop": "recovery_stop"},
    )
    builder.add_edge("recovery_stop", END)
    builder.add_edge("limit_exceeded", END)

    return builder.compile()
