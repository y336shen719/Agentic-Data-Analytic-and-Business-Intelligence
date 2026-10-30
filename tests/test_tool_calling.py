"""Deterministic tests for the tool-calling demo. No API key, no network."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, List

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.demo_tool_calling import (
    MAX_TOOL_CALLS_PER_RESPONSE,
    QUESTION,
    execute_tool_call,
    multiply,
    run_tool_calling_cycle,
)


def _call(name: str = "multiply", args: Any = None, call_id: str = "call_abc") -> dict:
    return {"name": name, "args": {"a": 123, "b": 456} if args is None else args, "id": call_id, "type": "tool_call"}


class FakeLLM:
    """Scripted stand-in for ChatOpenAI; records every invoke() input."""

    def __init__(self, responses: List[AIMessage]) -> None:
        self._responses = list(responses)
        self.bound_tools: list = []
        self.invocations: List[List[BaseMessage]] = []

    def bind_tools(self, tools: list) -> "FakeLLM":
        self.bound_tools = tools
        return self

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        self.invocations.append(list(messages))
        return self._responses.pop(0)


@pytest.mark.unit
def test_multiply_tool_result() -> None:
    assert multiply.invoke({"a": 123, "b": 456}) == 56088
    assert multiply.invoke({"a": -3, "b": 4}) == -12


@pytest.mark.unit
def test_tool_schema_name_and_parameters() -> None:
    assert multiply.name == "multiply"
    assert "Multiply two integers" in multiply.description
    schema = multiply.args
    assert set(schema) == {"a", "b"}
    assert schema["a"]["type"] == "integer"
    assert schema["b"]["type"] == "integer"


@pytest.mark.unit
def test_execute_tool_call_success_keeps_call_id() -> None:
    msg = execute_tool_call(_call(call_id="call_xyz"))
    assert isinstance(msg, ToolMessage)
    assert msg.content == "56088"
    assert msg.tool_call_id == "call_xyz"
    assert msg.status == "success"


@pytest.mark.unit
@pytest.mark.parametrize(
    "tool_call",
    [
        _call(name="os.system"),
        _call(name="delete_everything"),
        {"name": None, "args": {}, "id": "call_abc"},
        _call(args={"a": 1}),
        _call(args={"a": 1, "b": 2, "c": 3}),
        _call(args={"a": "12", "b": 3}),
        _call(args={"a": 1.5, "b": 3}),
        _call(args={"a": True, "b": 3}),
        _call(args="123 * 456"),
    ],
)
def test_unknown_tool_or_invalid_args_return_error_toolmessage(tool_call: dict) -> None:
    msg = execute_tool_call(tool_call)
    assert isinstance(msg, ToolMessage)
    assert msg.status == "error"
    assert msg.content.startswith("Error:")
    assert msg.tool_call_id == tool_call["id"]


@pytest.mark.unit
def test_full_cycle_message_order_and_ids() -> None:
    llm = FakeLLM([
        AIMessage(content="", tool_calls=[_call(call_id="call_1")]),
        AIMessage(content="123 x 456 = 56,088."),
    ])
    logs: List[str] = []
    history = run_tool_calling_cycle(llm, log=logs.append)

    assert [type(m) for m in history] == [HumanMessage, AIMessage, ToolMessage, AIMessage]
    assert history[0].content == QUESTION
    assert history[2].tool_call_id == history[1].tool_calls[0]["id"] == "call_1"
    assert history[2].content == "56088"
    assert history[3].content == "123 x 456 = 56,088."
    assert llm.bound_tools == [multiply]
    # Second LLM call receives Human -> AI -> Tool.
    assert [type(m) for m in llm.invocations[1]] == [HumanMessage, AIMessage, ToolMessage]
    assert any("call_1" in line for line in logs)


@pytest.mark.unit
def test_multiple_tool_calls_each_get_matching_toolmessage() -> None:
    llm = FakeLLM([
        AIMessage(content="", tool_calls=[
            _call(args={"a": 2, "b": 3}, call_id="c1"),
            _call(args={"a": 4, "b": 5}, call_id="c2"),
        ]),
        AIMessage(content="6 and 20"),
    ])
    history = run_tool_calling_cycle(llm, log=lambda _: None)
    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    assert [(m.tool_call_id, m.content) for m in tool_msgs] == [("c1", "6"), ("c2", "20")]
    assert [type(m) for m in history] == [HumanMessage, AIMessage, ToolMessage, ToolMessage, AIMessage]


@pytest.mark.unit
def test_unknown_tool_in_cycle_still_answers_every_call_id() -> None:
    llm = FakeLLM([
        AIMessage(content="", tool_calls=[_call(name="rm_rf", call_id="bad1")]),
        AIMessage(content="sorry"),
    ])
    history = run_tool_calling_cycle(llm, log=lambda _: None)
    assert history[2].tool_call_id == "bad1"
    assert history[2].status == "error"


@pytest.mark.unit
def test_excess_tool_calls_are_bounded_not_executed() -> None:
    n = MAX_TOOL_CALLS_PER_RESPONSE + 2
    calls = [_call(args={"a": i, "b": 1}, call_id=f"c{i}") for i in range(n)]
    llm = FakeLLM([AIMessage(content="", tool_calls=calls), AIMessage(content="done")])
    history = run_tool_calling_cycle(llm, log=lambda _: None)
    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == n  # every id still answered
    assert [m.status for m in tool_msgs[:MAX_TOOL_CALLS_PER_RESPONSE]] == ["success"] * MAX_TOOL_CALLS_PER_RESPONSE
    assert all(m.status == "error" for m in tool_msgs[MAX_TOOL_CALLS_PER_RESPONSE:])


@pytest.mark.unit
def test_no_tool_call_returns_after_single_llm_call() -> None:
    llm = FakeLLM([AIMessage(content="It is 56088.")])
    history = run_tool_calling_cycle(llm, log=lambda _: None)
    assert [type(m) for m in history] == [HumanMessage, AIMessage]
    assert len(llm.invocations) == 1
