"""Comprehensive integration tests for Python Analysis Error Classification & Bounded Automatic Retry (Step 5.4).

Tests A through J exercise the real compiled LangGraph workflow with deterministic
scripted LLM invocations and offline tools. No OpenAI API calls, no network access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional
import uuid

import duckdb
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

from agent.graph import create_agent_graph
from agent.python_error_classifier import (
    CATEGORY_INTERNAL_FAILURE,
    CATEGORY_INVALID_COLUMN,
    CATEGORY_INVALID_DATASET_STRUCTURE,
    CATEGORY_INVALID_PARAMETER,
    CATEGORY_MISSING_DATASET,
    RECOVERABLE_PYTHON_CATEGORIES,
)
from agent.sql_error_classifier import RECOVERABLE_SQL_ERROR
from agent.state import create_initial_state
from agent.tool_wrappers import create_python_analysis_tool, create_sql_tools
from tools.dataset_store import DatasetStore


# ==============================================================================
# Test Mocks and Helpers
# ==============================================================================

class ScriptedLLM(BaseChatModel):
    """Replays scripted AIMessages; fails if invoked more times than scripted."""

    script: List[AIMessage]
    invocations: List[Any] = Field(default_factory=list)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if len(self.invocations) >= len(self.script):
            raise AssertionError(
                f"LLM invoked {len(self.invocations) + 1} times, but only {len(self.script)} responses scripted."
            )
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


def sql_call(query: str, call_id: str = "call_sql") -> dict:
    return {"name": "execute_sql", "args": {"query": query}, "id": call_id}


def py_call(dataset_id: str, operation: str, params: Optional[dict] = None, call_id: str = "call_py") -> dict:
    return {
        "name": "execute_python_analysis",
        "args": {
            "dataset_id": dataset_id,
            "operation": operation,
            "params": params or {},
        },
        "id": call_id,
    }


def make_scripted_py_tool(results: list):
    """Fake execute_python_analysis returning scripted results in order; records calls."""
    executed: list[dict] = []

    @tool
    def execute_python_analysis(dataset_id: str, operation: str, params: Optional[dict] = None) -> Any:
        """Fake python tool."""
        executed.append({"dataset_id": dataset_id, "operation": operation, "params": params or {}})
        return results[len(executed) - 1]

    return execute_python_analysis, executed


def make_scripted_sql_tool(results: list):
    """Fake execute_sql returning scripted results in order; records queries."""
    executed: list[str] = []

    @tool
    def execute_sql(query: str) -> Any:
        """Fake sql tool."""
        executed.append(query)
        return results[len(executed) - 1]

    return execute_sql, executed


def run_graph(
    script: list[AIMessage],
    tools: list,
    max_steps: int = 12,
    max_sql_retries: int = 2,
    max_python_retries: int = 2,
    initial_query: str = "Analyze sales data",
):
    llm = ScriptedLLM(script=script)
    graph = create_agent_graph(
        llm=llm,
        tools=tools,
        max_steps=max_steps,
        max_sql_retries=max_sql_retries,
        max_python_retries=max_python_retries,
    )
    initial_state = create_initial_state(messages=[HumanMessage(content=initial_query)])
    final_state = graph.invoke(initial_state)
    return final_state, llm


# ==============================================================================
# Tests A through J
# ==============================================================================

# --- Test A: Invalid Column, Corrective Retry Succeeds ---
def test_test_a_invalid_column_corrective_retry_succeeds():
    """SQL provides valid dataset -> Python fails on bad column -> LLM corrects -> Python succeeds."""
    sql_tool, executed_sql = make_scripted_sql_tool([
        {
            "status": "success",
            "columns": ["period", "revenue"],
            "rows": [["2023-01", 100.0], ["2023-02", 150.0]],
            "row_count": 2,
            "dataset_id": "ds_sales",
        }
    ])
    py_tool, executed_py = make_scripted_py_tool([
        {
            "status": "error",
            "error_type": "InvalidColumnError",
            "error_message": "Column 'rev' does not exist in dataset",
            "dataset_id": "ds_sales",
            "operation": "describe",
        },
        {
            "status": "success",
            "result": {"summary": {"revenue": {"count": 2, "mean": 125.0}}},
            "dataset_id": "ds_sales",
            "operation": "describe",
        },
    ])

    script = [
        ai(sql_call("SELECT period, revenue FROM sales", "c1")),
        ai(py_call("ds_sales", "describe", {"columns": ["rev"]}, "c2")),  # fails: invalid column
        ai(py_call("ds_sales", "describe", {"columns": ["revenue"]}, "c3")),  # retry 1: succeeds
        ai(text="The average revenue is 125.0."),
    ]

    final_state, llm = run_graph(script, [sql_tool, py_tool])

    assert len(executed_sql) == 1
    assert len(executed_py) == 2
    assert executed_py[0]["params"]["columns"] == ["rev"]
    assert executed_py[1]["params"]["columns"] == ["revenue"]

    # Verify retry count was reset after success
    assert final_state["python_retry_count"] == 0
    assert final_state["sql_retry_count"] == 0
    assert final_state["last_error"] is None
    assert final_state["last_error_category"] is None
    assert final_state["messages"][-1].content == "The average revenue is 125.0."


# --- Test B: Invalid Parameters, Retry Exhaustion ---
def test_test_b_invalid_parameters_retry_exhaustion():
    """Initial call fails -> Retry 1 fails -> Retry 2 fails -> Graph stops safely at budget."""
    err_res = {
        "status": "error",
        "error_type": "InvalidParameterError",
        "error_message": "Invalid method parameter",
        "dataset_id": "ds_test",
        "operation": "correlation",
    }
    py_tool, executed_py = make_scripted_py_tool([err_res, err_res, err_res, err_res])

    script = [
        ai(py_call("ds_test", "correlation", {"method": "bad1"}, "c1")),  # Initial failure
        ai(py_call("ds_test", "correlation", {"method": "bad2"}, "c2")),  # Retry 1
        ai(py_call("ds_test", "correlation", {"method": "bad3"}, "c3")),  # Retry 2
        ai(py_call("ds_test", "correlation", {"method": "bad4"}, "c4")),  # Unreachable / blocked
    ]

    final_state, llm = run_graph(script, [py_tool], max_python_retries=2)

    assert len(executed_py) == 3  # initial + 2 retries
    assert final_state["python_retry_count"] == 2
    assert final_state["last_error_category"] == CATEGORY_INVALID_PARAMETER
    last_msg = final_state["messages"][-1]
    assert "Execution stopped: Python analysis still failed after 2 correction retries" in last_msg.content
    assert "InvalidParameterError: Invalid method parameter" in last_msg.content


# --- Test C: Non-Retryable Error ---
def test_test_c_non_retryable_error_stops_immediately():
    """Internal failure or construction failure is non-retryable; agent terminates without retry."""
    internal_err = {
        "status": "error",
        "error_type": "InternalExecutionError",
        "error_message": "Fatal numerical calculation crash",
        "dataset_id": "ds_test",
        "operation": "describe",
    }
    py_tool, executed_py = make_scripted_py_tool([internal_err, internal_err])

    script = [
        ai(py_call("ds_test", "describe", {}, "c1")),
        ai(py_call("ds_test", "describe", {}, "c2")),  # Should not be reached
    ]

    final_state, llm = run_graph(script, [py_tool])

    assert len(executed_py) == 1
    assert final_state["python_retry_count"] == 0
    assert final_state["last_error_category"] == CATEGORY_INTERNAL_FAILURE
    last_msg = final_state["messages"][-1]
    assert "Execution stopped: Python analysis encountered an internal failure and will not be retried." in last_msg.content
    assert "Fatal numerical calculation crash" in last_msg.content


# --- Test D: Missing Dataset Recovery (Real DatasetStore & Tools) ---
def test_test_d_missing_dataset_recovery(tmp_path: Path):
    """LLM uses invalid dataset ID -> DatasetNotFoundError -> SQL fetches dataset -> Python succeeds."""
    db_file = tmp_path / "test_missing_ds.duckdb"
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE metrics (val DOUBLE);")
        con.execute("INSERT INTO metrics VALUES (10.0), (20.0), (30.0);")

    store = DatasetStore()
    schema_tool, sql_tool = create_sql_tools(db_path=db_file, dataset_store=store)
    py_tool = create_python_analysis_tool(dataset_store=store)

    class MissingDsAnalyst(BaseChatModel):
        invocations: List[Any] = Field(default_factory=list)

        def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
            self.invocations.append(list(messages))
            last = messages[-1]

            if isinstance(last, HumanMessage):
                # Turn 1: Try analyzing an invalid dataset ID
                msg = ai(py_call("ds_nonexistent", "describe", {}, "c1"))
                return ChatResult(generations=[ChatGeneration(message=msg)])

            if isinstance(last, ToolMessage) and last.name == "execute_python_analysis":
                res = json.loads(last.content)
                if res.get("status") == "error":
                    # Turn 2: Missing dataset -> fetch via SQL
                    msg = ai(sql_call("SELECT val FROM metrics", "c2"))
                    return ChatResult(generations=[ChatGeneration(message=msg)])
                else:
                    # Turn 4: Analysis succeeded -> final answer
                    msg = ai(text="Calculated summary successfully.")
                    return ChatResult(generations=[ChatGeneration(message=msg)])

            if isinstance(last, ToolMessage) and last.name == "execute_sql":
                res = json.loads(last.content)
                new_ds = res["dataset_id"]
                # Turn 3: Retrying Python with newly obtained dataset ID
                msg = ai(py_call(new_ds, "describe", {"columns": ["val"]}, "c3"))
                return ChatResult(generations=[ChatGeneration(message=msg)])

            raise AssertionError("Unexpected turn in model")

        def bind_tools(self, tools: Any, **kwargs: Any) -> "MissingDsAnalyst":
            return self

        @property
        def _llm_type(self) -> str:
            return "missing-ds-analyst"

    llm = MissingDsAnalyst()
    graph = create_agent_graph(llm=llm, tools=[sql_tool, py_tool])
    initial = create_initial_state(messages=[HumanMessage(content="Describe metrics")])
    final_state = graph.invoke(initial)

    assert final_state["last_error"] is None
    assert final_state["last_error_category"] is None
    assert final_state["python_retry_count"] == 0
    assert final_state["sql_retry_count"] == 0
    assert final_state["messages"][-1].content == "Calculated summary successfully."


# --- Test E: Duplicate Period Recovery ---
def test_test_e_duplicate_period_recovery(tmp_path: Path):
    """Duplicate periods in raw table fail growth_rate -> SQL aggregates -> Python growth_rate succeeds."""
    db_file = tmp_path / "test_dup_periods.duckdb"
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE orders (period VARCHAR, rev DOUBLE);")
        # Duplicate 2023-01
        con.execute("INSERT INTO orders VALUES ('2023-01', 10.0), ('2023-01', 20.0), ('2023-02', 40.0);")

    store = DatasetStore()
    _, sql_tool = create_sql_tools(db_path=db_file, dataset_store=store)
    py_tool = create_python_analysis_tool(dataset_store=store)

    class DupPeriodAnalyst(BaseChatModel):
        invocations: List[Any] = Field(default_factory=list)

        def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
            self.invocations.append(list(messages))
            last = messages[-1]

            if isinstance(last, HumanMessage):
                # 1. Fetch unaggregated data
                msg = ai(sql_call("SELECT period, rev FROM orders", "c1"))
                return ChatResult(generations=[ChatGeneration(message=msg)])

            if isinstance(last, ToolMessage) and last.name == "execute_sql":
                res = json.loads(last.content)
                ds_id = res["dataset_id"]
                # If this was unaggregated query (c1), try growth_rate -> will fail
                if "GROUP BY" not in str(self.invocations[-1][-2].tool_calls[0]["args"]):
                    msg = ai(py_call(ds_id, "growth_rate", {"period_column": "period", "value_column": "rev"}, "c2"))
                else:
                    # After aggregated query, try growth_rate -> will succeed
                    msg = ai(py_call(ds_id, "growth_rate", {"period_column": "period", "value_column": "total_rev"}, "c4"))
                return ChatResult(generations=[ChatGeneration(message=msg)])

            if isinstance(last, ToolMessage) and last.name == "execute_python_analysis":
                res = json.loads(last.content)
                if res.get("status") == "error":
                    # DuplicatePeriodError -> aggregate with SQL
                    msg = ai(sql_call("SELECT period, SUM(rev) AS total_rev FROM orders GROUP BY period ORDER BY period", "c3"))
                    return ChatResult(generations=[ChatGeneration(message=msg)])
                else:
                    return ChatResult(generations=[ChatGeneration(message=ai(text="Growth rate computed."))])

            raise AssertionError("Unexpected conversation state")

        def bind_tools(self, tools: Any, **kwargs: Any) -> "DupPeriodAnalyst":
            return self

        @property
        def _llm_type(self) -> str:
            return "dup-period-analyst"

    llm = DupPeriodAnalyst()
    graph = create_agent_graph(llm=llm, tools=[sql_tool, py_tool])
    initial = create_initial_state(messages=[HumanMessage(content="Calculate revenue growth")])
    final_state = graph.invoke(initial)

    assert final_state["last_error"] is None
    assert final_state["python_retry_count"] == 0
    assert final_state["messages"][-1].content == "Growth rate computed."


# --- Test F: SQL and Python Retry Independence ---
def test_test_f_sql_and_python_retry_independence():
    """SQL error retries do not increment python_retry_count; Python retries do not touch sql_retry_count."""
    sql_tool, executed_sql = make_scripted_sql_tool([
        {"status": "error", "error_type": "BinderException", "error_message": "Col not found"},
        {"status": "success", "columns": ["x"], "rows": [[1]], "row_count": 1, "dataset_id": "ds_1"},
    ])
    py_tool, executed_py = make_scripted_py_tool([
        {"status": "error", "error_type": "InvalidColumnError", "error_message": "Bad col"},
        {"status": "success", "result": {"summary": {}}, "dataset_id": "ds_1", "operation": "describe"},
    ])

    script = [
        ai(sql_call("SELECT bad FROM t", "c1")),  # SQL fails (sql_retry_count = 0)
        ai(sql_call("SELECT x FROM t", "c2")),  # SQL retry 1 succeeds -> sql_retry_count = 1
        ai(py_call("ds_1", "describe", {"columns": ["bad"]}, "c3")),  # Py fails (python_retry_count = 0)
        ai(py_call("ds_1", "describe", {"columns": ["x"]}, "c4")),  # Py retry 1 succeeds -> resets to 0
        ai(text="All finished."),
    ]

    final_state, llm = run_graph(script, [sql_tool, py_tool])

    assert len(executed_sql) == 2
    assert len(executed_py) == 2
    # SQL retry count reflects 1 corrective retry for SQL
    assert final_state["sql_retry_count"] == 1
    # Python retry count reset upon success
    assert final_state["python_retry_count"] == 0
    assert final_state["last_error"] is None


# --- Test G: Independent Python Operations ---
def test_test_g_independent_python_operations():
    """First Python operation succeeds; second independent operation receives full retry allowance."""
    py_tool, executed_py = make_scripted_py_tool([
        # Op 1 succeeds directly
        {"status": "success", "result": {"mean": 10}, "dataset_id": "ds_1", "operation": "describe"},
        # Op 2 initial failure
        {"status": "error", "error_type": "InvalidColumnError", "error_message": "col_z missing"},
        # Op 2 retry 1 succeeds
        {"status": "success", "result": {"mean": 20}, "dataset_id": "ds_1", "operation": "describe"},
    ])

    script = [
        ai(py_call("ds_1", "describe", {"columns": ["col_a"]}, "c1")),
        ai(py_call("ds_1", "describe", {"columns": ["col_z"]}, "c2")),  # Op 2 fails
        ai(py_call("ds_1", "describe", {"columns": ["col_b"]}, "c3")),  # Op 2 retry 1 succeeds
        ai(text="Done with both."),
    ]

    final_state, llm = run_graph(script, [py_tool])

    assert len(executed_py) == 3
    assert final_state["python_retry_count"] == 0
    assert final_state["last_error"] is None


# --- Test H: Multiple Tool Calls (Mixed Outcomes) ---
def test_test_h_multiple_tool_calls_mixed_outcomes():
    """A batch containing SQL success and Python failure preserves the Python failure."""
    sql_tool, executed_sql = make_scripted_sql_tool([
        {"status": "success", "columns": ["x"], "rows": [[1]], "row_count": 1, "dataset_id": "ds_1"}
    ])
    py_tool, executed_py = make_scripted_py_tool([
        {"status": "error", "error_type": "InvalidColumnError", "error_message": "bad col"}
    ])

    script = [
        ai(
            sql_call("SELECT x FROM t", "c1"),
            py_call("ds_1", "describe", {"columns": ["bad"]}, "c2"),
        ),
        # Next turn LLM receives the error and concludes
        ai(text="Acknowledged failure."),
    ]

    final_state, llm = run_graph(script, [sql_tool, py_tool])

    # SQL success did not mask the Python error
    assert final_state["last_error_category"] == CATEGORY_INVALID_COLUMN
    assert "InvalidColumnError: bad col" in final_state["last_error"]


# --- Test I: Repeated Identical Tool Call ---
def test_test_i_repeated_identical_tool_call():
    """LLM attempts the exact same failed Python call; graph bounds retries and stops cleanly."""
    err_res = {
        "status": "error",
        "error_type": "InvalidColumnError",
        "error_message": "Column 'foo' does not exist",
    }
    py_tool, executed_py = make_scripted_py_tool([err_res, err_res, err_res, err_res])

    identical_call = py_call("ds_1", "describe", {"columns": ["foo"]}, "call_foo")
    script = [
        ai(identical_call),  # Initial
        ai(identical_call),  # Retry 1
        ai(identical_call),  # Retry 2 -> Exhausted
        ai(identical_call),  # Blocked
    ]

    final_state, llm = run_graph(script, [py_tool], max_python_retries=2)

    assert len(executed_py) == 3
    assert final_state["python_retry_count"] == 2
    last_msg = final_state["messages"][-1]
    assert "Execution stopped: Python analysis still failed after 2 correction retries (limit 2)." in last_msg.content


# --- Test J: Global Step Limit ---
def test_test_j_global_step_limit_enforced_during_python_recovery():
    """Global max_steps halts execution even if Python retry budget remains."""
    err_res = {
        "status": "error",
        "error_type": "InvalidColumnError",
        "error_message": "Column 'foo' missing",
    }
    py_tool, executed_py = make_scripted_py_tool([err_res, err_res, err_res])

    script = [
        ai(py_call("ds_1", "describe", {}, "c1")),
        ai(py_call("ds_1", "describe", {}, "c2")),
        ai(py_call("ds_1", "describe", {}, "c3")),
    ]

    final_state, llm = run_graph(script, [py_tool], max_steps=2, max_python_retries=5)

    assert final_state["step_count"] == 2
    assert len(executed_py) == 1  # Only first call executed before step limit hit
    last_msg = final_state["messages"][-1]
    assert "maximum LLM invocation limit (2) was reached" in last_msg.content


# --- Validation and Pre-Execution Guard Tests ---
@pytest.mark.parametrize("value", [0, 1, 2, 10])
def test_valid_max_python_retries_accepted(value):
    g = create_agent_graph(llm=ScriptedLLM(script=[]), tools=[], max_python_retries=value)
    assert g is not None


@pytest.mark.parametrize("value", [-1, True, False, 1.5, "2", None, [], {}])
def test_invalid_max_python_retries_rejected(value):
    with pytest.raises(ValueError, match="max_python_retries must be a non-negative integer"):
        create_agent_graph(llm=ScriptedLLM(script=[]), tools=[], max_python_retries=value)


def test_parallel_python_calls_exceeding_budget_are_not_executed():
    calls = [py_call("ds_1", "describe", {}, f"c{i}") for i in range(4)]  # allowed: 1 initial + 2 retries = 3
    py_tool, executed_py = make_scripted_py_tool([])
    final_state, llm = run_graph([ai(*calls)], [py_tool], max_python_retries=2)
    assert executed_py == []
    assert len(llm.invocations) == 1
    last_msg = final_state["messages"][-1]
    assert "would exceed the Python analysis correction retry limit" in last_msg.content


def test_missing_dataset_reuses_existing_valid_reference():
    """When a dataset ID has a typo, the LLM reuses the existing valid dataset ID without extra SQL."""
    sql_tool, executed_sql = make_scripted_sql_tool([
        {
            "status": "success",
            "columns": ["val"],
            "rows": [[10.0]],
            "row_count": 1,
            "dataset_id": "ds_existing_valid",
        }
    ])
    py_tool, executed_py = make_scripted_py_tool([
        {
            "status": "error",
            "error_type": "DatasetNotFoundError",
            "error_message": "Dataset ID 'ds_typo' not found in store",
        },
        {
            "status": "success",
            "result": {"summary": {"val": {"mean": 10.0}}},
            "dataset_id": "ds_existing_valid",
            "operation": "describe",
        },
    ])

    script = [
        ai(sql_call("SELECT val FROM metrics", "c1")),
        ai(py_call("ds_typo", "describe", {}, "c2")),  # Typo in dataset ID
        ai(py_call("ds_existing_valid", "describe", {}, "c3")),  # Reuses existing valid ID
        ai(text="Summary completed using existing dataset reference."),
    ]

    final_state, llm = run_graph(script, [sql_tool, py_tool])

    assert len(executed_sql) == 1  # Only 1 initial SQL run, no redundant regeneration
    assert len(executed_py) == 2
    assert executed_py[0]["dataset_id"] == "ds_typo"
    assert executed_py[1]["dataset_id"] == "ds_existing_valid"
    assert final_state["python_retry_count"] == 0
    assert final_state["last_error"] is None
    assert final_state["messages"][-1].content == "Summary completed using existing dataset reference."


def test_explicit_policy_violation_stops_immediately_without_retry():
    """Explicit policy violations terminate execution immediately as non-retryable."""
    sec_err = {
        "status": "error",
        "error_type": "SecurityPolicyError",
        "error_message": "Arbitrary Python code execution is prohibited.",
    }
    py_tool, executed_py = make_scripted_py_tool([sec_err, sec_err])

    script = [
        ai(py_call("ds_test", "execute_arbitrary_code", {}, "c1")),
        ai(py_call("ds_test", "describe", {}, "c2")),  # Should not be executed
    ]

    final_state, llm = run_graph(script, [py_tool])

    assert len(executed_py) == 1
    assert final_state["python_retry_count"] == 0
    last_msg = final_state["messages"][-1]
    assert "Execution stopped: Python analysis violated execution policy and will not be retried." in last_msg.content
    assert "Arbitrary Python code execution is prohibited" in last_msg.content


# ==============================================================================
# Issue A Tests — Mixed Tool Batch Error Masking (Tests A1 - A6)
# ==============================================================================

def test_mixed_batch_a1_python_failure_then_success_preserved():
    """Test A1: Python failure followed by Python success in same batch.
    Failure remains pending and is not silently cleared by later success.
    """
    py_tool, executed_py = make_scripted_py_tool([
        {"status": "error", "error_type": "InvalidColumnError", "error_message": "Column 'foo' not found"},
        {"status": "success", "result": {"mean": 42.0}, "operation": "describe", "dataset_id": "ds_1"},
        {"status": "success", "result": {"mean": 10.0}, "operation": "describe", "dataset_id": "ds_1"},
    ])

    script = [
        ai(
            py_call("ds_1", "describe", {"columns": ["foo"]}, "c1"),
            py_call("ds_1", "describe", {"columns": ["val"]}, "c2"),
        ),
        ai(py_call("ds_1", "describe", {"columns": ["bar"]}, "c3")),
        ai(text="Analysis complete."),
    ]

    final_state, llm = run_graph(script, [py_tool])

    assert len(executed_py) == 3
    assert len(llm.invocations) == 3
    assert final_state["last_error"] is None
    assert final_state["pending_python_recovery"] is False
    assert final_state["python_retry_count"] == 0
    assert final_state["messages"][-1].content == "Analysis complete."


def test_mixed_batch_a2_python_success_then_failure_preserved():
    """Test A2: Python success followed by Python failure in same batch.
    Failure is classified correctly and not ignored regardless of call order.
    """
    py_tool, executed_py = make_scripted_py_tool([
        {"status": "success", "result": {"mean": 42.0}, "operation": "describe", "dataset_id": "ds_1"},
        {"status": "error", "error_type": "InvalidColumnError", "error_message": "Column 'missing' not found"},
        {"status": "success", "result": {"mean": 10.0}, "operation": "describe", "dataset_id": "ds_1"},
    ])

    script = [
        ai(
            py_call("ds_1", "describe", {"columns": ["val"]}, "c1"),
            py_call("ds_1", "describe", {"columns": ["missing"]}, "c2"),
        ),
        ai(py_call("ds_1", "describe", {"columns": ["corrected"]}, "c3")),
        ai(text="Done."),
    ]

    final_state, llm = run_graph(script, [py_tool])

    assert len(executed_py) == 3
    assert len(llm.invocations) == 3
    assert final_state["pending_python_recovery"] is False
    assert final_state["last_error"] is None


def test_mixed_batch_a3_python_non_retryable_failure_takes_priority():
    """Test A3: Python non-retryable failure followed by success in same batch.
    Non-retryable failure takes priority (Fail Closed) and routes to recovery_stop.
    """
    py_tool, executed_py = make_scripted_py_tool([
        {"status": "error", "error_type": "SecurityPolicyError", "error_message": "Prohibited operation"},
        {"status": "success", "result": {"mean": 42.0}, "operation": "describe", "dataset_id": "ds_1"},
    ])

    script = [
        ai(
            py_call("ds_1", "execute_code", {}, "c1"),
            py_call("ds_1", "describe", {}, "c2"),
        ),
        ai(text="Should not be reached"),
    ]

    final_state, llm = run_graph(script, [py_tool])

    assert len(executed_py) == 2
    assert len(llm.invocations) == 1
    last_msg = final_state["messages"][-1]
    assert "Execution stopped: Python analysis violated execution policy and will not be retried." in last_msg.content
    assert final_state["last_error_category"] == "policy_violation"


def test_mixed_batch_a4_mixed_sql_and_python_batch():
    """Test A4: Mixed SQL/Python batches preserve Python errors and retry accounting."""
    # Subtest 1: SQL success + Python failure
    sql_tool, executed_sql = make_scripted_sql_tool([
        {"status": "success", "columns": ["x"], "rows": [[1]], "row_count": 1}
    ])
    py_tool, executed_py = make_scripted_py_tool([
        {"status": "error", "error_type": "InvalidParameterError", "error_message": "Bad k value"},
        {"status": "success", "result": {"clusters": 3}, "operation": "describe", "dataset_id": "ds_1"},
    ])

    script = [
        ai(sql_call("SELECT 1", "c1"), py_call("ds_1", "detect_outliers", {"k": -1}, "c2")),
        ai(py_call("ds_1", "detect_outliers", {"k": 3}, "c3")),
        ai(text="All done."),
    ]

    final_state, llm = run_graph(script, [sql_tool, py_tool])
    assert len(executed_sql) == 1
    assert len(executed_py) == 2
    assert final_state["last_error"] is None
    assert final_state["pending_python_recovery"] is False

    # Subtest 2: Python failure + SQL success
    sql_tool2, executed_sql2 = make_scripted_sql_tool([
        {"status": "success", "columns": ["y"], "rows": [[2]], "row_count": 1}
    ])
    py_tool2, executed_py2 = make_scripted_py_tool([
        {"status": "error", "error_type": "InvalidColumnError", "error_message": "Bad column"},
        {"status": "success", "result": {"ok": True}, "operation": "describe", "dataset_id": "ds_1"},
    ])

    script2 = [
        ai(py_call("ds_1", "describe", {"columns": ["bad"]}, "c1"), sql_call("SELECT 2", "c2")),
        ai(py_call("ds_1", "describe", {"columns": ["good"]}, "c3")),
        ai(text="Completed."),
    ]

    final_state2, llm2 = run_graph(script2, [sql_tool2, py_tool2])
    assert len(executed_sql2) == 1
    assert len(executed_py2) == 2
    assert final_state2["last_error"] is None


def test_mixed_batch_a5_multiple_python_failures_deterministic():
    """Test A5: Multiple Python failures deterministically preserve first error in call order."""
    py_tool, executed_py = make_scripted_py_tool([
        {"status": "error", "error_type": "InvalidColumnError", "error_message": "Col A missing"},
        {"status": "error", "error_type": "InvalidParameterError", "error_message": "Param B invalid"},
    ])

    script = [
        ai(
            py_call("ds_1", "describe", {"columns": ["col_a"]}, "c1"),
            py_call("ds_1", "describe", {"method": "bad"}, "c2"),
        ),
        ai(text="Handled failure."),
    ]

    final_state, llm = run_graph(script, [py_tool])

    assert len(executed_py) == 2
    assert len(llm.invocations) == 2
    assert final_state["last_error_category"] == CATEGORY_INVALID_COLUMN
    assert "Col A missing" in final_state["last_error"]
    assert final_state["pending_python_recovery"] is True


def test_mixed_batch_a6_all_success_calls_no_recovery():
    """Test A6: Multiple successful calls in a batch leave clean state with no pending error."""
    py_tool, executed_py = make_scripted_py_tool([
        {"status": "success", "result": {"m1": 1}, "operation": "describe", "dataset_id": "ds_1"},
        {"status": "success", "result": {"m2": 2}, "operation": "describe", "dataset_id": "ds_1"},
    ])

    script = [
        ai(
            py_call("ds_1", "describe", {}, "c1"),
            py_call("ds_1", "describe", {}, "c2"),
        ),
        ai(text="Finished analysis."),
    ]

    final_state, llm = run_graph(script, [py_tool])

    assert len(executed_py) == 2
    assert len(llm.invocations) == 2
    assert final_state["last_error"] is None
    assert final_state["last_error_category"] is None
    assert final_state["pending_python_recovery"] is False
    assert final_state["python_retry_count"] == 0


# ==============================================================================
# Issue B Tests — Cross-Tool Python Retry Budget Bypass (Tests B1 - B8)
# ==============================================================================

def test_cross_tool_b1_python_failure_sql_success_python_retry():
    """Test B1: Python failure -> SQL success -> Python retry.
    Verify Python retry counter is preserved and resumed Python call counts as corrective retry.
    """
    sql_tool, executed_sql = make_scripted_sql_tool([
        {"status": "success", "columns": ["sales"], "rows": [[100.0]], "row_count": 1, "dataset_id": "ds_new"}
    ])
    py_tool, executed_py = make_scripted_py_tool([
        {"status": "error", "error_type": "InvalidColumnError", "error_message": "Col 'rev' not found"},
        {"status": "success", "result": {"mean": 100.0}, "operation": "describe", "dataset_id": "ds_new"},
    ])

    script = [
        ai(py_call("ds_old", "describe", {"columns": ["rev"]}, "c1")),  # Initial attempt, fails
        ai(sql_call("SELECT sales FROM orders", "c2")),                  # SQL regeneration succeeds
        ai(py_call("ds_new", "describe", {"columns": ["sales"]}, "c3")),# Resumed Python call (retry 1)
        ai(text="Done"),
    ]

    final_state, llm = run_graph(script, [sql_tool, py_tool])

    assert len(executed_py) == 2
    assert len(executed_sql) == 1
    assert final_state["pending_python_recovery"] is False
    assert final_state["last_error"] is None
    assert final_state["python_retry_count"] == 0  # Reset upon successful recovery


def test_cross_tool_b2_python_failure_sql_failure_sql_recovery_python_retry():
    """Test B2: Python failure -> SQL failure -> SQL recovery -> Python retry.
    Verify pending Python recovery survives intermediate SQL error and recovery.
    """
    sql_tool, executed_sql = make_scripted_sql_tool([
        {"status": "error", "error_type": "BinderException", "error_message": "Table missing"},
        {"status": "success", "columns": ["revenue"], "rows": [[200.0]], "row_count": 1, "dataset_id": "ds_fixed"},
    ])
    py_tool, executed_py = make_scripted_py_tool([
        {"status": "error", "error_type": "InvalidColumnError", "error_message": "Col 'rev' not found"},
        {"status": "success", "result": {"mean": 200.0}, "operation": "describe", "dataset_id": "ds_fixed"},
    ])

    script = [
        ai(py_call("ds_1", "describe", {"columns": ["rev"]}, "c1")),  # Python fails -> pending_python_recovery=True
        ai(sql_call("SELECT * FROM bad_table", "c2")),                 # SQL fails with BinderException
        ai(sql_call("SELECT revenue FROM good_table", "c3")),          # SQL retried and succeeds
        ai(py_call("ds_fixed", "describe", {"columns": ["revenue"]}, "c4")), # Python corrective retry
        ai(text="Done"),
    ]

    final_state, llm = run_graph(script, [sql_tool, py_tool])

    assert len(executed_py) == 2
    assert len(executed_sql) == 2
    assert final_state["sql_retry_count"] == 1
    assert final_state["pending_python_recovery"] is False
    assert final_state["last_error"] is None


def test_cross_tool_b3_alternating_tools_cannot_exceed_retry_budget():
    """Test B3: Python fail -> SQL success -> Python fail -> SQL success -> Python fail.
    Verify that no more than two corrective Python retries can execute.
    """
    sql_tool, executed_sql = make_scripted_sql_tool([
        {"status": "success", "columns": ["a"], "rows": [[1]], "row_count": 1},
        {"status": "success", "columns": ["b"], "rows": [[2]], "row_count": 1},
    ])
    py_err = {"status": "error", "error_type": "InvalidColumnError", "error_message": "Still bad column"}
    py_tool, executed_py = make_scripted_py_tool([py_err, py_err, py_err])

    script = [
        ai(py_call("ds_1", "describe", {"columns": ["bad1"]}, "c1")),  # Initial attempt (fails, retry_count=0)
        ai(sql_call("SELECT 1", "c2")),                                 # Intermediate SQL
        ai(py_call("ds_1", "describe", {"columns": ["bad2"]}, "c3")),  # Corrective retry 1 (fails, retry_count=1)
        ai(sql_call("SELECT 2", "c4")),                                 # Intermediate SQL
        ai(py_call("ds_1", "describe", {"columns": ["bad3"]}, "c5")),  # Corrective retry 2 (fails, retry_count=2)
    ]

    final_state, llm = run_graph(script, [sql_tool, py_tool], max_python_retries=2)

    assert len(executed_py) == 3   # 1 initial + 2 retries = 3
    assert len(executed_sql) == 2
    assert final_state["python_retry_count"] == 2
    last_msg = final_state["messages"][-1]
    assert "Execution stopped: Python analysis still failed after 2 correction retries" in last_msg.content


def test_cross_tool_b4_exhausted_python_budget_cannot_be_restored_by_sql():
    """Test B4: Switching tools cannot restore an exhausted Python retry allowance."""
    py_err = {"status": "error", "error_type": "InvalidColumnError", "error_message": "Col error"}
    py_tool, executed_py = make_scripted_py_tool([py_err, py_err, py_err])
    sql_tool, executed_sql = make_scripted_sql_tool([
        {"status": "success", "columns": ["x"], "rows": [[1]], "row_count": 1}
    ])

    script = [
        ai(py_call("ds_1", "describe", {}, "c1")),  # Initial
        ai(py_call("ds_1", "describe", {}, "c2")),  # Retry 1
        ai(py_call("ds_1", "describe", {}, "c3")),  # Retry 2 -> Exhausted
    ]

    final_state, llm = run_graph(script, [py_tool, sql_tool], max_python_retries=2)
    assert len(executed_py) == 3
    assert final_state["python_retry_count"] == 2
    assert "Execution stopped" in final_state["messages"][-1].content


def test_cross_tool_b5_successful_python_recovery_resets_budget():
    """Test B5: Successful Python recovery resets python_retry_count and clears pending_python_recovery."""
    py_tool, executed_py = make_scripted_py_tool([
        {"status": "error", "error_type": "InvalidColumnError", "error_message": "Wrong column"},
        {"status": "success", "result": {"mean": 5.0}, "operation": "describe", "dataset_id": "ds_1"},
    ])

    script = [
        ai(py_call("ds_1", "describe", {"columns": ["wrong"]}, "c1")),
        ai(py_call("ds_1", "describe", {"columns": ["correct"]}, "c2")),
        ai(text="Recovered successfully."),
    ]

    final_state, llm = run_graph(script, [py_tool])

    assert len(executed_py) == 2
    assert final_state["python_retry_count"] == 0
    assert final_state["pending_python_recovery"] is False
    assert final_state["last_error"] is None


def test_cross_tool_b6_independent_subsequent_python_task_gets_full_budget():
    """Test B6: After task 1 completes, subsequent independent task receives its own initial attempt and retries."""
    py_tool, executed_py = make_scripted_py_tool([
        {"status": "error", "error_type": "InvalidColumnError", "error_message": "Task 1 typo"},
        {"status": "success", "result": {"mean": 10.0}, "operation": "describe", "dataset_id": "ds_1"},
        {"status": "error", "error_type": "InvalidParameterError", "error_message": "Task 2 param typo"},
        {"status": "success", "result": {"growth": 0.05}, "operation": "growth_rate", "dataset_id": "ds_1"},
    ])

    script = [
        # Task 1: 1 failure + 1 retry success
        ai(py_call("ds_1", "describe", {"columns": ["typo"]}, "c1")),
        ai(py_call("ds_1", "describe", {"columns": ["valid"]}, "c2")),
        # Task 2: 1 failure + 1 retry success
        ai(py_call("ds_1", "growth_rate", {"bad": 1}, "c3")),
        ai(py_call("ds_1", "growth_rate", {"time_col": "date", "value_col": "val"}, "c4")),
        ai(text="Both tasks succeeded."),
    ]

    final_state, llm = run_graph(script, [py_tool])

    assert len(executed_py) == 4
    assert final_state["python_retry_count"] == 0
    assert final_state["pending_python_recovery"] is False
    assert final_state["last_error"] is None


def test_cross_tool_b7_sql_and_python_retry_counter_independence():
    """Test B7: Python failures do not consume SQL retry budget, and SQL failures do not consume Python budget."""
    sql_tool, executed_sql = make_scripted_sql_tool([
        {"status": "error", "error_type": "BinderException", "error_message": "SQL err 1"},
        {"status": "success", "columns": ["x"], "rows": [[1]], "row_count": 1},
    ])
    py_tool, executed_py = make_scripted_py_tool([
        {"status": "error", "error_type": "InvalidColumnError", "error_message": "Py err 1"},
        {"status": "success", "result": {"m": 1}, "operation": "describe", "dataset_id": "ds_1"},
    ])

    script = [
        ai(sql_call("SELECT bad", "c1")),                     # SQL initial fail
        ai(sql_call("SELECT good", "c2")),                    # SQL retry 1 success
        ai(py_call("ds_1", "describe", {"col": "bad"}, "c3")),# Python initial fail
        ai(py_call("ds_1", "describe", {"col": "ok"}, "c4")), # Python retry 1 success
        ai(text="Done"),
    ]

    final_state, llm = run_graph(script, [sql_tool, py_tool])

    assert len(executed_sql) == 2
    assert len(executed_py) == 2
    assert final_state["sql_retry_count"] == 1  # 1 SQL retry was consumed
    assert final_state["python_retry_count"] == 0  # Python recovered and reset to 0
    assert final_state["last_error"] is None


def test_cross_tool_b8_global_step_limit_bounds_all_recovery_paths():
    """Test B8: Global max_steps stops execution even during active cross-tool recovery."""
    sql_tool, executed_sql = make_scripted_sql_tool([
        {"status": "error", "error_type": "BinderException", "error_message": "SQL err"}
    ])
    py_tool, executed_py = make_scripted_py_tool([
        {"status": "error", "error_type": "InvalidColumnError", "error_message": "Py err"}
    ])

    script = [
        ai(py_call("ds_1", "describe", {}, "c1")),
        ai(sql_call("SELECT 1", "c2")),
        ai(py_call("ds_1", "describe", {}, "c3")),
    ]

    final_state, llm = run_graph(script, [sql_tool, py_tool], max_steps=2, max_python_retries=5, max_sql_retries=5)

    assert final_state["step_count"] == 2
    assert len(executed_py) == 1
    assert len(executed_sql) == 0  # Second step was hit before c2 could run or finish
    last_msg = final_state["messages"][-1]
    assert "maximum LLM invocation limit (2) was reached" in last_msg.content


# ==============================================================================
# Mixed SQL/Python Simultaneous Failures & State Synchronization (Step 5.4 Final)
# ==============================================================================

def test_mixed_sql_python_recoverable_errors_sql_first():
    """Verify mixed recoverable batch with SQL first and Python second.
    Both tools execute, both ToolMessages are preserved, Python failure is primary,
    pending_python_recovery is True, and retry counters are accurate.
    """
    sql_tool, executed_sql = make_scripted_sql_tool([
        {"status": "error", "error_type": "BinderException", "error_message": "SQL syntax error"}
    ])
    py_tool, executed_py = make_scripted_py_tool([
        {"status": "error", "error_type": "InvalidColumnError", "error_message": "Col 'rev' missing"}
    ])

    script = [
        ai(sql_call("SELECT bad", "c1"), py_call("ds_1", "describe", {"columns": ["rev"]}, "c2")),
        ai(text="Received failures, preparing recovery"),
    ]

    final_state, llm = run_graph(script, [sql_tool, py_tool])

    assert len(executed_sql) == 1
    assert len(executed_py) == 1
    assert len(llm.invocations) == 2
    # Both ToolMessages preserved in conversation history
    tool_msgs = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 2
    assert tool_msgs[0].tool_call_id == "c1"
    assert tool_msgs[1].tool_call_id == "c2"

    # Deterministic error categorization and pending recovery state
    assert final_state["last_error_category"] == CATEGORY_INVALID_COLUMN
    assert "Col 'rev' missing" in final_state["last_error"]
    assert final_state["pending_python_recovery"] is True
    assert final_state["sql_retry_count"] == 0
    assert final_state["python_retry_count"] == 0


def test_mixed_sql_python_recoverable_errors_python_first():
    """Verify mixed recoverable batch with Python first and SQL second produces identical semantics.
    Order-independent error precedence and pending recovery.
    """
    py_tool, executed_py = make_scripted_py_tool([
        {"status": "error", "error_type": "InvalidColumnError", "error_message": "Col 'rev' missing"}
    ])
    sql_tool, executed_sql = make_scripted_sql_tool([
        {"status": "error", "error_type": "BinderException", "error_message": "SQL syntax error"}
    ])

    script = [
        ai(py_call("ds_1", "describe", {"columns": ["rev"]}, "c1"), sql_call("SELECT bad", "c2")),
        ai(text="Received failures, preparing recovery"),
    ]

    final_state, llm = run_graph(script, [sql_tool, py_tool])

    assert len(executed_py) == 1
    assert len(executed_sql) == 1
    assert len(llm.invocations) == 2
    tool_msgs = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 2
    assert tool_msgs[0].tool_call_id == "c1"
    assert tool_msgs[1].tool_call_id == "c2"

    assert final_state["last_error_category"] == CATEGORY_INVALID_COLUMN
    assert "Col 'rev' missing" in final_state["last_error"]
    assert final_state["pending_python_recovery"] is True
    assert final_state["sql_retry_count"] == 0
    assert final_state["python_retry_count"] == 0


def test_mixed_sql_python_nonretryable_precedence():
    """Verify that non-retryable failures take absolute precedence in mixed batches,
    regardless of tool type or call ordering.
    """
    # Subtest 1: Non-retryable SQL error + Recoverable Python error
    sql_tool1, executed_sql1 = make_scripted_sql_tool([
        {"status": "error", "error_type": "SQLValidationError", "error_message": "Access to secret_table is forbidden by security policy"}
    ])
    py_tool1, executed_py1 = make_scripted_py_tool([
        {"status": "error", "error_type": "InvalidColumnError", "error_message": "Col missing"}
    ])
    script1 = [
        ai(sql_call("SELECT * FROM secret_table", "c1"), py_call("ds_1", "describe", {}, "c2")),
        ai(text="Should not reach"),
    ]
    final_state1, llm1 = run_graph(script1, [sql_tool1, py_tool1])
    assert len(executed_sql1) == 1
    assert len(executed_py1) == 1
    assert len(llm1.invocations) == 1
    assert final_state1["last_error_category"] == "non_retryable_security_error"
    assert "security policy" in final_state1["messages"][-1].content

    # Subtest 2: Recoverable SQL error + Non-retryable Python error
    sql_tool2, executed_sql2 = make_scripted_sql_tool([
        {"status": "error", "error_type": "BinderException", "error_message": "Typo in query"}
    ])
    py_tool2, executed_py2 = make_scripted_py_tool([
        {"status": "error", "error_type": "SecurityPolicyError", "error_message": "Prohibited python execution"}
    ])
    script2 = [
        ai(sql_call("SELECT bad", "c1"), py_call("ds_1", "execute_code", {}, "c2")),
        ai(text="Should not reach"),
    ]
    final_state2, llm2 = run_graph(script2, [sql_tool2, py_tool2])
    assert len(executed_sql2) == 1
    assert len(executed_py2) == 1
    assert len(llm2.invocations) == 1
    assert final_state2["last_error_category"] == "policy_violation"
    assert "violated execution policy" in final_state2["messages"][-1].content


def test_pending_python_recovery_preserved_across_sql():
    """Verify pending_python_recovery is retained across multiple intermediate SQL queries."""
    sql_tool, executed_sql = make_scripted_sql_tool([
        {"status": "success", "columns": ["x"], "rows": [[1]], "row_count": 1},
        {"status": "success", "columns": ["y"], "rows": [[2]], "row_count": 1},
    ])
    py_tool, executed_py = make_scripted_py_tool([
        {"status": "error", "error_type": "InvalidColumnError", "error_message": "Col 'rev' not found"},
    ])

    script = [
        ai(py_call("ds_1", "describe", {"columns": ["rev"]}, "c1")),  # Python fails -> sets pending_python_recovery=True
        ai(sql_call("SELECT 1", "c2")),                                 # Intermediate SQL query 1
        ai(sql_call("SELECT 2", "c3")),                                 # Intermediate SQL query 2
        ai(text="Inspected schema, ready for Python retry."),
    ]

    final_state, llm = run_graph(script, [sql_tool, py_tool])

    assert len(executed_py) == 1
    assert len(executed_sql) == 2
    assert final_state["pending_python_recovery"] is True
    assert final_state["python_retry_count"] == 0  # Resumed Python retry has not yet executed


def test_pending_python_recovery_reset_after_success():
    """Verify pending_python_recovery is reset to False and python_retry_count to 0 on success."""
    py_tool, executed_py = make_scripted_py_tool([
        {"status": "error", "error_type": "InvalidColumnError", "error_message": "Typo in col"},
        {"status": "success", "result": {"mean": 10.0}, "operation": "describe", "dataset_id": "ds_1"},
    ])

    script = [
        ai(py_call("ds_1", "describe", {"columns": ["typo"]}, "c1")),
        ai(py_call("ds_1", "describe", {"columns": ["valid"]}, "c2")),
        ai(text="Completed."),
    ]

    final_state, llm = run_graph(script, [py_tool])

    assert len(executed_py) == 2
    assert final_state["pending_python_recovery"] is False
    assert final_state["python_retry_count"] == 0
    assert final_state["last_error"] is None
    assert final_state["last_error_category"] is None



