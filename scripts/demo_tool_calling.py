"""Educational demo of OpenAI tool calling (one request/response cycle, no agent loop).

Manual, billable live run:  python -m scripts.demo_tool_calling

Cycle: HumanMessage -> LLM returns AIMessage.tool_calls -> WE execute the tool
-> ToolMessage(tool_call_id) -> LLM writes the final answer.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Callable, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agent.config import ConfigError
from agent.llm import create_llm

QUESTION = "What is 123 multiplied by 456? Use the multiplication tool."

# Hard bound on tool calls executed from a single model response.
MAX_TOOL_CALLS_PER_RESPONSE = 5


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the product."""
    return a * b


# Allowlist: the model can only trigger tools registered here.
TOOLS: Dict[str, Any] = {multiply.name: multiply}


def _error_message(call_id: str, text: str) -> ToolMessage:
    return ToolMessage(content=f"Error: {text}", tool_call_id=call_id, status="error")


def execute_tool_call(tool_call: Dict[str, Any]) -> ToolMessage:
    """Validate and run one model-requested tool call; always returns a ToolMessage.

    OpenAI requires a ToolMessage for every tool_call id, so failures are reported
    back as error ToolMessages instead of raising.
    """
    call_id = str(tool_call.get("id") or "")
    name = tool_call.get("name")
    args = tool_call.get("args")

    selected = TOOLS.get(name) if isinstance(name, str) else None
    if selected is None:
        return _error_message(call_id, f"unknown tool {name!r}")

    expected = set(selected.args)  # parameter names from the tool schema
    if not isinstance(args, dict) or set(args) != expected:
        return _error_message(call_id, f"arguments must be exactly {sorted(expected)}")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in args.values()):
        return _error_message(call_id, "all arguments must be integers")

    try:
        result = selected.invoke(args)  # runs OUR Python function, with pydantic validation
    except Exception as exc:
        return _error_message(call_id, f"tool execution failed: {type(exc).__name__}")
    return ToolMessage(content=str(result), tool_call_id=call_id)


def run_tool_calling_cycle(
    llm: Any,
    question: str = QUESTION,
    log: Callable[[str], None] = print,
) -> List[BaseMessage]:
    """Run exactly one tool-calling round trip and return the full message history."""
    messages: List[BaseMessage] = [HumanMessage(content=question)]
    log(f"User question: {question}")

    ai_msg = llm.bind_tools(list(TOOLS.values())).invoke(messages)
    messages.append(ai_msg)

    if not ai_msg.tool_calls:
        log("Model requested no tools. Response: " + str(ai_msg.content))
        return messages

    for i, call in enumerate(ai_msg.tool_calls):
        log(f"LLM-requested tool : {call.get('name')}")
        log(f"LLM-generated args : {call.get('args')}")
        log(f"Tool call ID       : {call.get('id')}")
        if i >= MAX_TOOL_CALLS_PER_RESPONSE:
            tool_msg = _error_message(str(call.get("id") or ""), "too many tool calls in one response")
        else:
            tool_msg = execute_tool_call(call)
        log(f"Python tool result : {tool_msg.content}")
        messages.append(tool_msg)

    # Final synthesis uses the unbound llm so no further tool calls are possible.
    final = llm.invoke(messages)
    messages.append(final)
    log(f"Final LLM response : {final.content}")
    return messages


def main() -> int:
    try:
        llm = create_llm()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    try:
        run_tool_calling_cycle(llm)
    except Exception as exc:  # API/network errors; never print secrets
        print(f"LLM call failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
