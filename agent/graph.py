"""LangGraph orchestration graph for the Agentic Data Analyst workflow.

Topology::

    START --(step_count >= max_steps)--> limit_exceeded --> END
      |
      v
     llm --(no tool_calls)--------------------------------> END
      |  --(max_steps reached, tool_calls pending)--------> limit_exceeded --> END
      |  --(SQL/Python calls would exceed retry budget)---> recovery_stop --> END
      v
    tools (ToolNode) --> inspect_tool_results
                              |--(success / retry allowed)-----------> llm
                              |--(non-retryable error / budget spent)-> recovery_stop --> END

Counters and State:
    * ``step_count``              - number of completed LLM invocations; bounded by ``max_steps``.
    * ``sql_retry_count``         - SQL executions that ran while an earlier SQL error was outstanding;
                                    bounded by ``max_sql_retries``.
    * ``python_retry_count``      - corrective retries for the active Python analysis task;
                                    resets to 0 upon successful recovery; bounded by ``max_python_retries``.
    * ``pending_python_recovery`` - boolean flag tracking whether an unresolved recoverable Python
                                    failure is active. Preserved across intermediate SQL regeneration calls.

Batch Precedence & Semantics:
    1. Fail Closed: Non-retryable errors immediately determine batch outcome and route to recovery_stop.
    2. No Silent Masking: Recoverable errors in a batch are never cleared by unrelated successes in the same batch.
    3. Independent Budgets: SQL and Python retry counters operate independently.
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
from agent.python_error_classifier import (
    CATEGORY_INTERNAL_FAILURE,
    CATEGORY_INVALID_DATASET_STRUCTURE,
    CATEGORY_MISSING_DATASET,
    CATEGORY_POLICY_VIOLATION,
    CATEGORY_UNKNOWN_ERROR,
    MALFORMED_RESULT,
    RECOVERABLE_PYTHON_CATEGORIES,
    classify_python_result,
)
from agent.sql_error_classifier import RECOVERABLE_SQL_ERROR, classify_sql_result
from agent.state import AgentState
from agent.tool_wrappers import create_sql_tools

DEFAULT_MAX_STEPS: int = 12
DEFAULT_MAX_SQL_RETRIES: int = 2
DEFAULT_MAX_PYTHON_RETRIES: int = 2
SQL_TOOL_NAME: str = "execute_sql"
PYTHON_TOOL_NAME: str = "execute_python_analysis"
CHART_TOOL_NAME: str = "generate_chart"
NON_RETRYABLE_CHART_ERROR: str = "non_retryable_chart_error"
_MAX_ERROR_CHARS: int = 500


def classify_chart_result(result: Any) -> Dict[str, Any]:
    """Deterministically classify a chart generation tool result.

    Outcomes:
    - Success: status="success", category=None, retryable=False.
    - Structured error: status="error", category=NON_RETRYABLE_CHART_ERROR, retryable=False.
    - Malformed/missing: status="error", category=NON_RETRYABLE_CHART_ERROR, retryable=False.
    """
    if result is None:
        return {
            "status": "error",
            "category": NON_RETRYABLE_CHART_ERROR,
            "retryable": False,
            "error_type": "MalformedChartResultError",
            "error_message": "Chart tool result is missing or failed during execution.",
        }
    if not isinstance(result, dict):
        return {
            "status": "error",
            "category": NON_RETRYABLE_CHART_ERROR,
            "retryable": False,
            "error_type": "MalformedChartResultError",
            "error_message": f"Expected dict from chart tool, got {type(result).__name__}.",
        }

    status = result.get("status")
    if status == "success":
        chart_id = result.get("chart_id")
        if not isinstance(chart_id, str) or not chart_id.strip():
            return {
                "status": "error",
                "category": NON_RETRYABLE_CHART_ERROR,
                "retryable": False,
                "error_type": "MalformedChartResultError",
                "error_message": "Chart tool reported success without a valid chart_id.",
            }
        return {
            "status": "success",
            "category": None,
            "retryable": False,
            "chart_id": chart_id,
        }

    if status == "error":
        error_type = result.get("error_type") or "ChartGenerationError"
        error_msg = result.get("error_message") or "Chart generation failed."
        return {
            "status": "error",
            "category": NON_RETRYABLE_CHART_ERROR,
            "retryable": False,
            "error_type": str(error_type),
            "error_message": str(error_msg),
        }

    return {
        "status": "error",
        "category": NON_RETRYABLE_CHART_ERROR,
        "retryable": False,
        "error_type": "MalformedChartResultError",
        "error_message": f"Unexpected chart tool status: {status!r}.",
    }


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


def _python_call_count(message: Any) -> int:
    calls = getattr(message, "tool_calls", None) or []
    return sum(1 for tc in calls if tc.get("name") == PYTHON_TOOL_NAME)


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


def inspect_tool_results(state: AgentState) -> Dict[str, Any]:
    """Classify tool results of the latest ToolNode run (SQL and Python analysis)."""
    ai_msg, tool_msgs = _latest_tool_batch(state.get("messages", []))
    if ai_msg is None:
        return {}

    running_error = state.get("last_error")
    running_category = state.get("last_error_category")
    sql_retries = state.get("sql_retry_count", 0)
    python_retries = state.get("python_retry_count", 0)
    pending_py_recovery = state.get("pending_python_recovery", False)
    saw_relevant_tool = False

    tool_events: list[dict[str, Any]] = []

    # Deterministic order: the order of originating tool calls
    for tc in ai_msg.tool_calls:
        tool_name = tc.get("name")
        if tool_name not in (SQL_TOOL_NAME, PYTHON_TOOL_NAME, CHART_TOOL_NAME):
            continue
        saw_relevant_tool = True
        msg = tool_msgs.get(tc.get("id"))

        if tool_name == SQL_TOOL_NAME:
            result = _parse_tool_content(msg) if msg is not None and msg.name in (None, SQL_TOOL_NAME) else None
            outcome = classify_sql_result(result)
            tool_events.append({"tool": SQL_TOOL_NAME, "call_id": tc.get("id"), "outcome": outcome})
        elif tool_name == PYTHON_TOOL_NAME:
            result = _parse_tool_content(msg) if msg is not None and msg.name in (None, PYTHON_TOOL_NAME) else None
            outcome = classify_python_result(result)
            tool_events.append({"tool": PYTHON_TOOL_NAME, "call_id": tc.get("id"), "outcome": outcome})
        elif tool_name == CHART_TOOL_NAME:
            if msg is not None and getattr(msg, "status", "success") == "error":
                outcome = {
                    "status": "error",
                    "category": NON_RETRYABLE_CHART_ERROR,
                    "retryable": False,
                    "error_type": "ChartExecutionError",
                    "error_message": "Chart tool execution failed.",
                }
            else:
                result = _parse_tool_content(msg) if msg is not None and msg.name in (None, CHART_TOOL_NAME) else None
                outcome = classify_chart_result(result)
            tool_events.append({"tool": CHART_TOOL_NAME, "call_id": tc.get("id"), "outcome": outcome})

    if not saw_relevant_tool:
        return {}

    # Update retry counters for invocations that ran during recovery
    curr_sql_err = (running_category == RECOVERABLE_SQL_ERROR)
    curr_py_pending = pending_py_recovery

    for e in tool_events:
        if e["tool"] == SQL_TOOL_NAME:
            if curr_sql_err:
                sql_retries += 1
            if e["outcome"]["status"] == "error":
                curr_sql_err = bool(e["outcome"]["retryable"])
            else:
                curr_sql_err = False
        elif e["tool"] == PYTHON_TOOL_NAME:
            if curr_py_pending:
                python_retries += 1
            if e["outcome"]["status"] == "error":
                if e["outcome"]["retryable"]:
                    curr_py_pending = True
                else:
                    curr_py_pending = False
        elif e["tool"] == CHART_TOOL_NAME:
            # Chart calls never consume SQL or Python retry budgets
            pass

    # 1. Non-retryable failures take absolute precedence (Fail Closed)
    non_retryable_event = next(
        (e for e in tool_events if e["outcome"]["status"] == "error" and not e["outcome"]["retryable"]),
        None,
    )
    if non_retryable_event is not None:
        running_error = f"{non_retryable_event['outcome']['error_type']}: {non_retryable_event['outcome']['error_message']}"[:_MAX_ERROR_CHARS]
        running_category = non_retryable_event["outcome"]["category"]
        if non_retryable_event["tool"] == PYTHON_TOOL_NAME:
            pending_py_recovery = False
        return {
            "last_error": running_error,
            "last_error_category": running_category,
            "sql_retry_count": sql_retries,
            "python_retry_count": python_retries,
            "pending_python_recovery": pending_py_recovery,
        }

    # 2. Recoverable errors must not be masked by successes in the same batch
    recoverable_py_events = [
        e for e in tool_events if e["tool"] == PYTHON_TOOL_NAME and e["outcome"]["status"] == "error" and e["outcome"]["retryable"]
    ]
    recoverable_sql_events = [
        e for e in tool_events if e["tool"] == SQL_TOOL_NAME and e["outcome"]["status"] == "error" and e["outcome"]["retryable"]
    ]

    if recoverable_py_events:
        # A Python analysis failed: mark recovery pending, preserve error
        first_err = recoverable_py_events[0]
        running_error = f"{first_err['outcome']['error_type']}: {first_err['outcome']['error_message']}"[:_MAX_ERROR_CHARS]
        running_category = first_err["outcome"]["category"]
        pending_py_recovery = True
        return {
            "last_error": running_error,
            "last_error_category": running_category,
            "sql_retry_count": sql_retries,
            "python_retry_count": python_retries,
            "pending_python_recovery": pending_py_recovery,
        }

    if recoverable_sql_events:
        # Check if this was a pure SQL batch where a later SQL query succeeded (intra-batch SQL retry)
        all_sql = all(e["tool"] == SQL_TOOL_NAME for e in tool_events)
        has_later_sql_success = all_sql and any(
            e["outcome"]["status"] == "success"
            for e in tool_events[tool_events.index(recoverable_sql_events[0]):]
        )
        if has_later_sql_success:
            running_error, running_category = None, None
            # If no pending Python recovery was active, error is clear
        else:
            first_err = recoverable_sql_events[0]
            running_error = f"{first_err['outcome']['error_type']}: {first_err['outcome']['error_message']}"[:_MAX_ERROR_CHARS]
            running_category = first_err["outcome"]["category"]
            if any(e["tool"] == PYTHON_TOOL_NAME and e["outcome"]["status"] == "success" for e in tool_events):
                pending_py_recovery = False
                python_retries = 0
            return {
                "last_error": running_error,
                "last_error_category": running_category,
                "sql_retry_count": sql_retries,
                "python_retry_count": python_retries,
                "pending_python_recovery": pending_py_recovery,
            }

    # 3. All tools in the batch succeeded
    saw_python_success = any(e["tool"] == PYTHON_TOOL_NAME and e["outcome"]["status"] == "success" for e in tool_events)
    saw_sql_success = any(e["tool"] == SQL_TOOL_NAME and e["outcome"]["status"] == "success" for e in tool_events)

    if saw_python_success:
        # Successful Python execution completes and resets Python recovery
        python_retries = 0
        pending_py_recovery = False
        if running_category in RECOVERABLE_PYTHON_CATEGORIES or running_category == CATEGORY_MISSING_DATASET:
            running_error, running_category = None, None

    if saw_sql_success:
        # Successful SQL execution completes SQL recovery
        if running_category == RECOVERABLE_SQL_ERROR:
            running_error, running_category = None, None

    # If neither Python nor SQL was in recoverable state, or both resolved:
    if not pending_py_recovery and running_category != RECOVERABLE_SQL_ERROR:
        running_error, running_category = None, None

    return {
        "last_error": running_error,
        "last_error_category": running_category,
        "sql_retry_count": sql_retries,
        "python_retry_count": python_retries,
        "pending_python_recovery": pending_py_recovery,
    }


def create_agent_graph(
    llm: Optional[Any] = None,
    tools: Optional[Sequence[BaseTool]] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    system_prompt: Optional[str] = None,
    max_sql_retries: int = DEFAULT_MAX_SQL_RETRIES,
    max_python_retries: int = DEFAULT_MAX_PYTHON_RETRIES,
) -> CompiledStateGraph:
    """Build and compile the stateful LangGraph agent workflow.

    Args:
        llm: Optional ChatOpenAI or custom chat model. If None, defaults to `create_llm()`.
        tools: Optional sequence of BaseTool instances. If None, defaults to `create_sql_tools()`.
        max_steps: Maximum LLM invocations (strictly positive int). Defaults to 12.
        system_prompt: Optional system instruction prompt. If None, defaults to SYSTEM_PROMPT.
        max_sql_retries: SQL correction retries allowed after the initial attempt
            (non-negative int; 0 disables correction). Defaults to 2.
        max_python_retries: Python analysis correction retries allowed after the initial attempt
            (non-negative int; 0 disables correction). Defaults to 2.

    Raises:
        ValueError: If ``max_steps``, ``max_sql_retries``, or ``max_python_retries`` is invalid.
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
    if isinstance(max_python_retries, bool) or not isinstance(max_python_retries, int) or max_python_retries < 0:
        raise ValueError(
            f"max_python_retries must be a non-negative integer, got: {max_python_retries!r} "
            f"(type: {type(max_python_retries).__name__})"
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

    # 3. Recovery Stop Node: explain why the workflow terminated (SQL/Python failure / retry budget)
    def recovery_stop_node(state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        # Pre-execution guard checks
        py_calls = _python_call_count(last_message)
        sql_calls = _sql_call_count(last_message)

        allowed_py = max_python_retries - state.get("python_retry_count", 0)
        if not state.get("pending_python_recovery", False):
            allowed_py += 1

        allowed_sql = max_sql_retries - state.get("sql_retry_count", 0)
        if state.get("last_error_category") != RECOVERABLE_SQL_ERROR:
            allowed_sql += 1

        if py_calls > allowed_py or allowed_py < 0:
            reason = f"the Python analysis correction retry limit ({max_python_retries}) would be exceeded"
            pending = _unexecuted_tool_messages(last_message, reason)
            text = (
                f"Execution stopped: the requested Python analysis calls would exceed the Python "
                f"analysis correction retry limit ({max_python_retries}). No pending tools were executed."
            )
        elif sql_calls > allowed_sql:
            reason = f"the SQL correction retry limit ({max_sql_retries}) would be exceeded"
            pending = _unexecuted_tool_messages(last_message, reason)
            text = (
                f"Execution stopped: the requested SQL calls would exceed the SQL correction retry "
                f"limit ({max_sql_retries}). No pending tools were executed."
            )
        else:
            pending = []
            category = state.get("last_error_category")
            error = state.get("last_error")
            if category == NON_RETRYABLE_CHART_ERROR:
                text = (
                    f"Execution stopped: chart generation failed and will not be automatically "
                    f"retried. Error: {error}"
                )
            elif category == RECOVERABLE_SQL_ERROR:
                text = (
                    f"Execution stopped: SQL still failed after {state.get('sql_retry_count', 0)} correction "
                    f"retries (limit {max_sql_retries}). Last error: {error}"
                )
            elif category in RECOVERABLE_PYTHON_CATEGORIES or state.get("pending_python_recovery", False):
                text = (
                    f"Execution stopped: Python analysis still failed after {state.get('python_retry_count', 0)} correction "
                    f"retries (limit {max_python_retries}). Last error: {error}"
                )
            elif category == "non_retryable_security_error":
                text = f"Execution stopped: the SQL query was rejected by security policy and will not be retried. Error: {error}"
            elif category == "non_retryable_resource_error":
                text = f"Execution stopped: SQL execution failed with an operational error and will not be retried. Error: {error}"
            elif category == CATEGORY_INTERNAL_FAILURE:
                text = f"Execution stopped: Python analysis encountered an internal failure and will not be retried. Error: {error}"
            elif category == CATEGORY_POLICY_VIOLATION:
                text = f"Execution stopped: Python analysis violated execution policy and will not be retried. Error: {error}"
            elif category == CATEGORY_UNKNOWN_ERROR:
                text = f"Execution stopped: Python analysis returned an unknown error and will not be retried. Error: {error}"
            elif category == MALFORMED_RESULT:
                text = f"Execution stopped: Python analysis returned a malformed result and will not be retried. Error: {error}"
            else:
                text = f"Execution stopped: Tool execution returned an unexpected or unrecognised result and will not be retried. Error: {error}"

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

        # Pre-execution guard: SQL calls in one response cannot exceed remaining budget.
        allowed_sql = max_sql_retries - state.get("sql_retry_count", 0)
        if state.get("last_error_category") != RECOVERABLE_SQL_ERROR:
            allowed_sql += 1
        if _sql_call_count(last_message) > allowed_sql:
            return "recovery_stop"

        # Pre-execution guard: Python calls in one response cannot exceed remaining budget.
        allowed_py = max_python_retries - state.get("python_retry_count", 0)
        if not state.get("pending_python_recovery", False):
            allowed_py += 1
        if _python_call_count(last_message) > allowed_py or allowed_py < 0:
            return "recovery_stop"

        return "tools"

    def route_after_inspection(state: AgentState) -> str:
        if state.get("last_error") is None:
            return "llm"
        category = state.get("last_error_category")
        if category == NON_RETRYABLE_CHART_ERROR:
            return "recovery_stop"
        if category == RECOVERABLE_SQL_ERROR:
            if state.get("sql_retry_count", 0) >= max_sql_retries:
                return "recovery_stop"
            return "llm"
        if category in RECOVERABLE_PYTHON_CATEGORIES or state.get("pending_python_recovery", False):
            if state.get("python_retry_count", 0) >= max_python_retries:
                return "recovery_stop"
            return "llm"
        return "recovery_stop"

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
