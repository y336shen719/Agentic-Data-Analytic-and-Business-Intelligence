"""Offline tests for SQL error recovery wired into the agent graph.

Real compiled LangGraph invocations with a scripted fake LLM and fake SQL tools.
No OpenAI calls, no DuckDB, no network.
"""

from __future__ import annotations

import uuid
from typing import Any, List, Optional

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

from agent.graph import create_agent_graph
from agent.sql_error_classifier import (
    MALFORMED_RESULT,
    NON_RETRYABLE_RESOURCE_ERROR,
    NON_RETRYABLE_SECURITY_ERROR,
    NON_RETRYABLE_UNKNOWN_ERROR,
    RECOVERABLE_SQL_ERROR,
)
from agent.state import create_initial_state


class ScriptedLLM(BaseChatModel):
    """Replays scripted AIMessages; fails loudly if invoked more often than scripted."""

    script: List[AIMessage]
    invocations: List[Any] = Field(default_factory=list)

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        if len(self.invocations) >= len(self.script):
            raise AssertionError("LLM invoked more times than scripted")
        resp = self.script[len(self.invocations)]
        self.invocations.append(list(messages))
        msg = AIMessage(
            content=resp.content,
            tool_calls=[dict(tc) for tc in resp.tool_calls],
            id=str(uuid.uuid4()),
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedLLM":
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted-llm"


def ai(*calls: dict, text: str = "") -> AIMessage:
    return AIMessage(content=text, tool_calls=list(calls))


def sql_call(query: str, call_id: str) -> dict:
    return {"name": "execute_sql", "args": {"query": query}, "id": call_id}


def ok(rows: list) -> dict:
    return {
        "status": "success",
        "columns": ["n"],
        "rows": rows,
        "row_count": len(rows),
        "truncated": False,
        "execution_time_ms": 1.0,
    }


def err(error_type: str, message: str = "boom") -> dict:
    return {"status": "error", "error_type": error_type, "error_message": message, "execution_time_ms": 1.0}


BINDER = err("BinderException", 'Binder Error: Referenced column "zz" not found')
SECURITY_TABLE = err("SQLValidationError", "Table or view 'secrets' is not in the allowed tables list.")
SECURITY_DDL = err("SQLValidationError", "Disallowed DDL/DML/administrative operation detected in AST: Drop")
TIMEOUT = err("TimeoutError", "Query execution exceeded timeout limit of 10.0 seconds.")
UNKNOWN = err("WeirdNewError", "???")


@tool
def get_database_schema() -> dict:
    """Fake schema tool."""
    return {"tables": [{"name": "orders"}]}


def make_sql_tool(results: list):
    """Fake execute_sql returning scripted results in order; records executed queries."""
    executed: list[str] = []

    @tool
    def execute_sql(query: str) -> Any:
        """Fake SQL tool."""
        executed.append(query)
        return results[len(executed) - 1]

    return execute_sql, executed


def run(
    script: list,
    results: list,
    *,
    max_sql_retries: int = 2,
    max_steps: int = 12,
    initial: Optional[dict] = None,
    graph_llm: Optional[ScriptedLLM] = None,
):
    llm = graph_llm or ScriptedLLM(script=script)
    sql_tool, executed = make_sql_tool(results)
    graph = create_agent_graph(
        llm=llm,
        tools=[get_database_schema, sql_tool],
        max_steps=max_steps,
        max_sql_retries=max_sql_retries,
    )
    state = initial or create_initial_state(messages=[HumanMessage(content="How many orders?")])
    return graph.invoke(state), llm, executed


def tool_messages(state: dict) -> list[ToolMessage]:
    return [m for m in state["messages"] if isinstance(m, ToolMessage)]


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize("value", [0, 1, 2, 10])
def test_valid_max_sql_retries_accepted(value):
    graph = create_agent_graph(llm=ScriptedLLM(script=[]), tools=[get_database_schema], max_sql_retries=value)
    assert graph is not None


@pytest.mark.parametrize("value", [-1, True, False, 1.5, "2", None, [], {}])
def test_invalid_max_sql_retries_rejected(value):
    with pytest.raises(ValueError, match="max_sql_retries"):
        create_agent_graph(llm=ScriptedLLM(script=[]), tools=[get_database_schema], max_sql_retries=value)


def test_positional_compatibility_preserved():
    graph = create_agent_graph(ScriptedLLM(script=[]), [get_database_schema], 5, "prompt")
    assert graph is not None


# --------------------------------------------------------------------------- success paths


def test_sql_success_updates_state():
    state, llm, executed = run(
        [ai(sql_call("SELECT 1", "c1")), ai(text="There are 5 orders.")],
        [ok([[5]])],
    )
    assert executed == ["SELECT 1"]
    assert state["last_sql"] == "SELECT 1"
    assert state["last_error"] is None
    assert state["last_error_category"] is None
    assert state["sql_retry_count"] == 0
    assert state["step_count"] == 2
    assert state["messages"][-1].content == "There are 5 orders."


def test_empty_successful_result_is_not_an_error():
    state, llm, executed = run(
        [ai(sql_call("SELECT n FROM t WHERE 1=0", "c1")), ai(text="No matching rows.")],
        [ok([])],
    )
    assert state["last_error"] is None
    assert state["sql_retry_count"] == 0
    assert len(llm.invocations) == 2
    assert state["messages"][-1].content == "No matching rows."


def test_content_block_tool_output_is_parsed():
    """ToolMessage content given as text blocks is supported."""
    import json

    payload = json.dumps(ok([[1]]))

    @tool
    def execute_sql(query: str) -> Any:
        """Fake SQL tool returning content blocks."""
        return [{"type": "text", "text": payload}]

    llm = ScriptedLLM(script=[ai(sql_call("SELECT 1", "c1")), ai(text="done")])
    graph = create_agent_graph(llm=llm, tools=[execute_sql])
    state = graph.invoke(create_initial_state(messages=[HumanMessage(content="q")]))
    assert isinstance(tool_messages(state)[0].content, list)
    assert state["last_error"] is None
    assert len(llm.invocations) == 2


# --------------------------------------------------------------------------- retry counting


def test_initial_failure_is_not_counted_as_retry():
    state, llm, executed = run(
        [ai(sql_call("SELECT zz", "c1")), ai(text="I could not run the query.")],
        [BINDER],
    )
    assert state["sql_retry_count"] == 0
    assert state["last_error"].startswith("BinderException:")
    assert state["last_error_category"] == RECOVERABLE_SQL_ERROR
    assert len(llm.invocations) == 2  # LLM was given another opportunity


def test_one_recoverable_error_then_corrected_success():
    state, llm, executed = run(
        [ai(sql_call("SELECT zz", "c1")), ai(sql_call("SELECT n", "c2")), ai(text="5 orders")],
        [BINDER, ok([[5]])],
    )
    assert executed == ["SELECT zz", "SELECT n"]
    assert state["sql_retry_count"] == 1
    assert state["last_error"] is None
    assert state["last_error_category"] is None
    assert state["last_sql"] == "SELECT n"
    assert state["messages"][-1].content == "5 orders"


def test_two_recoverable_errors_then_success_on_second_retry():
    state, llm, executed = run(
        [
            ai(sql_call("q1", "c1")),
            ai(sql_call("q2", "c2")),
            ai(sql_call("q3", "c3")),
            ai(text="5 orders"),
        ],
        [BINDER, BINDER, ok([[5]])],
    )
    assert executed == ["q1", "q2", "q3"]
    assert state["sql_retry_count"] == 2
    assert state["last_error"] is None
    assert state["messages"][-1].content == "5 orders"
    assert state["step_count"] == 4


def test_exhausted_retry_budget_stops_correction():
    state, llm, executed = run(
        [ai(sql_call("q1", "c1")), ai(sql_call("q2", "c2")), ai(sql_call("q3", "c3"))],
        [BINDER, BINDER, BINDER],
    )
    assert executed == ["q1", "q2", "q3"]  # initial + 2 retries, nothing more
    assert len(llm.invocations) == 3  # no further LLM call after stop
    assert state["sql_retry_count"] == 2
    assert state["last_error_category"] == RECOVERABLE_SQL_ERROR
    last = state["messages"][-1]
    assert isinstance(last, AIMessage)
    assert "Execution stopped" in last.content
    assert "correction retries" in last.content
    assert "BinderException" in last.content


def test_max_sql_retries_zero_disables_correction():
    state, llm, executed = run(
        [ai(sql_call("q1", "c1"))],
        [BINDER],
        max_sql_retries=0,
    )
    assert executed == ["q1"]
    assert len(llm.invocations) == 1
    assert state["sql_retry_count"] == 0
    assert "Execution stopped" in state["messages"][-1].content


# --------------------------------------------------------------------------- non-retryable


@pytest.mark.parametrize(
    "result,category,snippet",
    [
        (SECURITY_TABLE, NON_RETRYABLE_SECURITY_ERROR, "security policy"),  # unauthorized table
        (SECURITY_DDL, NON_RETRYABLE_SECURITY_ERROR, "security policy"),  # forbidden operation
        (TIMEOUT, NON_RETRYABLE_RESOURCE_ERROR, "operational error"),
        (UNKNOWN, NON_RETRYABLE_UNKNOWN_ERROR, "unrecognised"),
    ],
)
def test_non_retryable_errors_never_retry(result, category, snippet):
    state, llm, executed = run([ai(sql_call("q1", "c1"))], [result])
    assert len(executed) == 1
    assert len(llm.invocations) == 1  # LLM never gets a chance to rephrase
    assert state["sql_retry_count"] == 0
    assert state["last_error_category"] == category
    assert snippet in state["messages"][-1].content
    # history stays valid: AI(tool_call) -> ToolMessage -> final AI
    assert [type(m).__name__ for m in state["messages"][-3:]] == ["AIMessage", "ToolMessage", "AIMessage"]


@pytest.mark.parametrize(
    "bad_result",
    ["not json at all", None, {}, {"status": "weird"}, {"status": "success"}, {"status": "error"}, [1, 2]],
)
def test_malformed_tool_payloads_fail_safely(bad_result):
    state, llm, executed = run([ai(sql_call("q1", "c1"))], [bad_result])
    assert len(llm.invocations) == 1
    assert state["sql_retry_count"] == 0
    assert state["last_error_category"] == MALFORMED_RESULT
    assert "Execution stopped" in state["messages"][-1].content


def test_toolnode_level_failure_is_malformed_and_stops():
    """execute_sql called without its required argument -> ToolNode error message, not success."""
    bad = {"name": "execute_sql", "args": {}, "id": "c1"}
    state, llm, executed = run([ai(bad)], [])
    assert executed == []
    assert tool_messages(state)[0].status == "error"
    assert state["last_error_category"] == MALFORMED_RESULT
    assert len(llm.invocations) == 1


# --------------------------------------------------------------------------- non-SQL tools / batches


def test_schema_results_do_not_touch_sql_counters():
    state, llm, executed = run(
        [
            ai({"name": "get_database_schema", "args": {}, "id": "s1"}),
            ai({"name": "get_database_schema", "args": {}, "id": "s2"}),
            ai(text="schema looked at"),
        ],
        [],
    )
    assert executed == []
    assert state["sql_retry_count"] == 0
    assert state["last_error"] is None
    assert state["last_error_category"] is None


def test_schema_call_between_failure_and_retry_does_not_count():
    state, llm, executed = run(
        [
            ai(sql_call("bad", "c1")),
            ai({"name": "get_database_schema", "args": {}, "id": "s1"}),
            ai(sql_call("good", "c2")),
            ai(text="done"),
        ],
        [BINDER, ok([[1]])],
    )
    assert state["sql_retry_count"] == 1  # only the SQL execution after the failure
    assert state["last_error"] is None


def test_multiple_tool_messages_in_one_run_failure_then_success():
    state, llm, executed = run(
        [
            ai(
                {"name": "get_database_schema", "args": {}, "id": "s1"},
                sql_call("bad", "c1"),
                sql_call("good", "c2"),
            ),
            ai(text="done"),
        ],
        [BINDER, ok([[1]])],
    )
    assert executed == ["bad", "good"]
    assert state["sql_retry_count"] == 1
    assert state["last_error"] is None
    assert len(llm.invocations) == 2


def test_multiple_tool_messages_in_one_run_success_then_failure():
    state, llm, executed = run(
        [ai(sql_call("good", "c1"), sql_call("bad", "c2")), ai(text="partial")],
        [ok([[1]]), BINDER],
    )
    assert state["sql_retry_count"] == 0  # second query ran with no failure outstanding
    assert state["last_error_category"] == RECOVERABLE_SQL_ERROR
    assert len(llm.invocations) == 2


def test_non_retryable_failure_in_batch_is_not_masked_by_later_success():
    state, llm, executed = run(
        [ai(sql_call("a", "c1"), sql_call("b", "c2"), sql_call("c", "c3"))],
        [SECURITY_TABLE, ok([[1]]), ok([[2]])],
    )
    assert state["last_error_category"] == NON_RETRYABLE_SECURITY_ERROR
    assert len(llm.invocations) == 1
    assert "security policy" in state["messages"][-1].content


def test_tool_call_ids_preserved_one_to_one():
    state, llm, executed = run(
        [
            ai({"name": "get_database_schema", "args": {}, "id": "call_schema"}, sql_call("q", "call_sql")),
            ai(text="done"),
        ],
        [ok([[1]])],
    )
    ai_msg = next(m for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls)
    expected = {tc["id"]: tc["name"] for tc in ai_msg.tool_calls}
    got = {m.tool_call_id: m.name for m in tool_messages(state)}
    assert got == expected
    assert len(tool_messages(state)) == len(expected)


# --------------------------------------------------------------------------- budget cannot be bypassed


def test_parallel_sql_calls_exceeding_budget_are_not_executed():
    calls = [sql_call(f"q{i}", f"c{i}") for i in range(4)]  # allowed: 1 initial + 2 retries = 3
    state, llm, executed = run([ai(*calls)], [])
    assert executed == []
    tms = tool_messages(state)
    assert [m.tool_call_id for m in tms] == ["c0", "c1", "c2", "c3"]
    assert all(m.status == "error" and "was not executed" in m.content for m in tms)
    assert "would exceed" in state["messages"][-1].content
    assert len(llm.invocations) == 1


def test_parallel_sql_calls_within_budget_execute_and_stop_when_all_fail():
    calls = [sql_call(f"q{i}", f"c{i}") for i in range(3)]
    state, llm, executed = run([ai(*calls)], [BINDER, BINDER, BINDER])
    assert executed == ["q0", "q1", "q2"]
    assert state["sql_retry_count"] == 2
    assert len(llm.invocations) == 1
    assert "Execution stopped" in state["messages"][-1].content


def test_budget_is_not_replenished_across_a_success():
    """sql_retry_count is cumulative for the request."""
    state, llm, executed = run(
        [
            ai(sql_call("a", "c1")),
            ai(sql_call("b", "c2")),  # retry 1 -> succeeds
            ai(sql_call("c", "c3")),  # new initial attempt, fails
            ai(sql_call("d", "c4")),  # retry 2 -> fails
        ],
        [BINDER, ok([[1]]), BINDER, BINDER],
    )
    assert executed == ["a", "b", "c", "d"]
    assert state["sql_retry_count"] == 2
    assert "Execution stopped" in state["messages"][-1].content


# --------------------------------------------------------------------------- limits still enforced


def test_max_steps_still_enforced_during_recovery():
    state, llm, executed = run(
        [ai(sql_call("q1", "c1")), ai(sql_call("q2", "c2"))],
        [BINDER],
        max_steps=2,
    )
    assert executed == ["q1"]  # q2 never executed
    assert len(llm.invocations) == 2
    assert state["step_count"] == 2
    tms = tool_messages(state)
    assert [m.tool_call_id for m in tms] == ["c1", "c2"]
    assert "was not executed because the workflow reached" in tms[-1].content
    assert "maximum LLM invocation limit (2)" in state["messages"][-1].content


def test_initial_step_count_at_limit_never_invokes_llm_or_sql():
    initial = create_initial_state(messages=[HumanMessage(content="q")], step_count=3)
    state, llm, executed = run([], [], max_steps=3, initial=initial)
    assert llm.invocations == []
    assert executed == []
    assert "maximum LLM invocation limit (3)" in state["messages"][-1].content


# --------------------------------------------------------------------------- isolation


def test_independent_invocations_do_not_share_retry_state():
    llm = ScriptedLLM(script=[ai(sql_call("bad", "c1")), ai(sql_call("good", "c2")), ai(text="done")])
    sql_tool, executed = make_sql_tool([SECURITY_TABLE, ok([[1]])])
    graph = create_agent_graph(llm=llm, tools=[get_database_schema, sql_tool], max_sql_retries=0)

    first = graph.invoke(create_initial_state(messages=[HumanMessage(content="q1")]))
    assert first["last_error_category"] == NON_RETRYABLE_SECURITY_ERROR

    second = graph.invoke(create_initial_state(messages=[HumanMessage(content="q2")]))
    assert second["last_error"] is None
    assert second["last_error_category"] is None
    assert second["sql_retry_count"] == 0
    assert second["step_count"] == 2
    assert len(second["messages"]) == 4  # no history leaked from the first run
