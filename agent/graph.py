"""LangGraph orchestration graph for the Agentic Data Analyst workflow.

Implements the stateful agent workflow:
    START -> (conditional: step_count < max_steps?) -> llm -> (conditional: tool_calls?) -> tools -> llm
         |                                                 |
         |                                                 +-> (no tool_calls) -> END
         |                                                 |
         +-------------------------------------------------+-> (limit reached) -> limit_exceeded -> END
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from agent.llm import create_llm
from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState
from agent.tool_wrappers import create_sql_tools

DEFAULT_MAX_STEPS: int = 12


def create_agent_graph(
    llm: Optional[Any] = None,
    tools: Optional[Sequence[BaseTool]] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    system_prompt: Optional[str] = None,
) -> CompiledStateGraph:
    """Build and compile the stateful LangGraph agent workflow.

    Args:
        llm: Optional ChatOpenAI or custom chat model. If None, defaults to `create_llm()`.
        tools: Optional sequence of BaseTool instances. If None, defaults to `create_sql_tools()`.
        max_steps: Maximum allowed LLM invocation steps before terminating. Must be a
            strictly positive integer. Defaults to 12.
        system_prompt: Optional system instruction prompt. If None, defaults to SYSTEM_PROMPT.

    Returns:
        CompiledStateGraph ready for execution via `.invoke()`.

    Raises:
        ValueError: If `max_steps` is not a strictly positive integer.
    """
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError(
            f"max_steps must be a strictly positive integer, got: {max_steps!r} "
            f"(type: {type(max_steps).__name__})"
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

        # Track latest SQL query if execute_sql was invoked in tool_calls
        last_sql = state.get("last_sql")
        tool_calls = getattr(response, "tool_calls", None) or []
        for tc in tool_calls:
            if isinstance(tc, dict) and tc.get("name") == "execute_sql":
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

    # 2. Limit Exceeded Node: Synthesizes pending ToolMessages and appends explanation warning
    def limit_exceeded_node(state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        output_messages: list[BaseMessage] = []

        # If the last message contains pending tool calls, synthesize matching ToolMessages
        if messages:
            last_message = messages[-1]
            tool_calls = getattr(last_message, "tool_calls", None) or []
            for tc in tool_calls:
                tc_id = tc.get("id")
                tc_name = tc.get("name", "unknown_tool")
                output_messages.append(
                    ToolMessage(
                        content=(
                            f"Tool '{tc_name}' was not executed because the workflow reached "
                            f"the maximum LLM invocation limit ({max_steps})."
                        ),
                        name=tc_name,
                        tool_call_id=tc_id if tc_id else "unknown_call_id",
                        status="error",
                    )
                )

        warning_msg = AIMessage(
            content=(
                f"Execution stopped: maximum LLM invocation limit ({max_steps}) was reached before "
                "the agent concluded tool calling."
            )
        )
        output_messages.append(warning_msg)
        return {"messages": output_messages}

    # 3. Entry Router: Inspects whether step_count already meets or exceeds max_steps
    def route_start(state: AgentState) -> str:
        if state.get("step_count", 0) >= max_steps:
            return "limit_exceeded"
        return "llm"

    # 4. Conditional Router: Inspects the latest message and LLM invocation limit
    def route_llm_output(state: AgentState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return END

        last_message = messages[-1]
        has_tool_calls = bool(getattr(last_message, "tool_calls", None))

        # If LLM finished with a normal response, terminate directly
        if not has_tool_calls:
            return END

        # If LLM requested tool calls but step limit has been reached, route to limit_exceeded
        if state.get("step_count", 0) >= max_steps:
            return "limit_exceeded"

        return "tools"

    # 5. Construct the Graph
    builder = StateGraph(AgentState)

    builder.add_node("llm", llm_node)
    builder.add_node("tools", ToolNode(approved_tools, handle_tool_errors=True))
    builder.add_node("limit_exceeded", limit_exceeded_node)

    builder.add_conditional_edges(
        START,
        route_start,
        {
            "llm": "llm",
            "limit_exceeded": "limit_exceeded",
        },
    )
    builder.add_conditional_edges(
        "llm",
        route_llm_output,
        {
            "tools": "tools",
            "limit_exceeded": "limit_exceeded",
            END: END,
        },
    )
    builder.add_edge("tools", "llm")
    builder.add_edge("limit_exceeded", END)

    return builder.compile()
