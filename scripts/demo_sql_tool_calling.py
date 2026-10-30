"""Educational demo of SQL tool calling using LangChain wrappers.

Manual, billable live run:  python -m scripts.demo_sql_tool_calling

Demonstrates a bounded, multi-step tool-calling interaction:
1. User asks an analytical question.
2. LLM requests schema discovery (get_database_schema).
3. Tool result is returned as a ToolMessage.
4. LLM requests an analytical SQL query (execute_sql).
5. Tool result is returned as a ToolMessage.
6. LLM provides the final business summary.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from agent.config import ConfigError
from agent.llm import create_llm
from agent.tool_wrappers import create_sql_tools

QUESTION = "How many orders are in the database? First check the schema, then execute a SQL query."

# Explicit upper bound on conversation rounds to prevent infinite loops
MAX_ROUNDS = 4


def _dispatch_tool(
    tools_by_name: Dict[str, BaseTool],
    tool_call: Dict[str, Any],
    log: Callable[[str], None] = print,
) -> ToolMessage:
    """Validate and execute one tool call requested by the model."""
    call_id = str(tool_call.get("id") or "")
    name = tool_call.get("name")
    args = tool_call.get("args") or {}

    log(f"\n--- Tool Call Requested ---")
    log(f"Tool Name   : {name}")
    log(f"Tool Args   : {args}")
    log(f"Tool Call ID: {call_id}")

    tool_obj = tools_by_name.get(name) if isinstance(name, str) else None
    if tool_obj is None:
        error_msg = f"Error: Tool '{name}' is not recognized."
        log(f"Tool Result : {error_msg}")
        return ToolMessage(content=error_msg, tool_call_id=call_id, status="error")

    try:
        raw_result = tool_obj.invoke(args)
        # Format result nicely for logging and ToolMessage content
        if isinstance(raw_result, dict):
            serialized_content = json.dumps(raw_result, indent=2, ensure_ascii=False)
        else:
            serialized_content = str(raw_result)
        
        # Log concise summary for large schemas or results
        if len(serialized_content) > 500:
            log(f"Tool Result : {serialized_content[:300]}... [truncated {len(serialized_content)} chars]")
        else:
            log(f"Tool Result : {serialized_content}")

        return ToolMessage(content=serialized_content, tool_call_id=call_id)
    except Exception as exc:
        err_text = f"Error executing tool '{name}': {type(exc).__name__}: {exc}"
        log(f"Tool Result : {err_text}")
        return ToolMessage(content=err_text, tool_call_id=call_id, status="error")


def run_sql_tool_calling_demo(
    llm: Any,
    question: str = QUESTION,
    log: Callable[[str], None] = print,
) -> List[BaseMessage]:
    """Execute a bounded tool-calling interaction with SQL tools."""
    schema_tool, sql_tool = create_sql_tools()
    tools = [schema_tool, sql_tool]
    tools_by_name = {t.name: t for t in tools}

    llm_with_tools = llm.bind_tools(tools)

    messages: List[BaseMessage] = [
        SystemMessage(
            content=(
                "You are an analytical assistant for an e-commerce database. "
                "Always discover the database schema first before writing SQL queries. "
                "Only write read-only SELECT queries. Once you have the data, provide a clear answer."
            )
        ),
        HumanMessage(content=question),
    ]

    log(f"User Question: {question}")

    for round_num in range(1, MAX_ROUNDS + 1):
        log(f"\n>>> Model Invocation (Round {round_num}/{MAX_ROUNDS})")
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            log("\n=== Final LLM Business Answer ===")
            log(str(ai_msg.content))
            return messages

        # Execute each requested tool call and append matching ToolMessage
        for call in ai_msg.tool_calls:
            tool_msg = _dispatch_tool(tools_by_name, call, log=log)
            messages.append(tool_msg)

    # If loop ended due to round limit, do one final response without tools
    log("\nReached maximum interaction rounds. Requesting final answer...")
    final_msg = llm.invoke(messages)
    messages.append(final_msg)
    log("\n=== Final LLM Business Answer ===")
    log(str(final_msg.content))
    return messages


def main() -> int:
    try:
        llm = create_llm()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        run_sql_tool_calling_demo(llm)
    except Exception as exc:
        print(f"Demo failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
