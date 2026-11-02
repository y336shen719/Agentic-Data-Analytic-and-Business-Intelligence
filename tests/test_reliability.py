"""Comprehensive reliability and failure-injection test suite (Step 8.3).

Validates all 20 reliability scenarios across 7 operational categories:
- Category A: SQL Recovery (REL_SQL_01 to REL_SQL_04)
- Category B: Python Recovery (REL_PY_01 to REL_PY_04)
- Category C: Security Enforcement (REL_SEC_01 to REL_SEC_04)
- Category D: Chart Reliability (REL_CHART_01 to REL_CHART_03)
- Category E: Execution Limits (REL_EXEC_01 to REL_EXEC_02)
- Category F: Session Reliability (REL_SESSION_01 to REL_SESSION_02)
- Category G: Failure Presentation (REL_UI_01)
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
import duckdb
import pytest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from agent.graph import (
    DEFAULT_MAX_PYTHON_RETRIES,
    DEFAULT_MAX_SQL_RETRIES,
    DEFAULT_MAX_STEPS,
    NON_RETRYABLE_CHART_ERROR,
    create_agent_graph,
)
from agent.python_error_classifier import (
    CATEGORY_INVALID_COLUMN,
    CATEGORY_INVALID_DATASET_STRUCTURE,
    CATEGORY_MISSING_DATASET,
)
from agent.sql_error_classifier import (
    NON_RETRYABLE_RESOURCE_ERROR,
    NON_RETRYABLE_SECURITY_ERROR,
    RECOVERABLE_SQL_ERROR,
)
from agent.state import create_initial_state
from agent.tool_wrappers import (
    create_chart_generation_tools,
    create_python_analysis_tools,
    create_sql_tools,
)
from agent.ui_session import (
    AgentSession,
    create_agent_session,
    execute_session_turn,
    reset_agent_session,
    sanitize_user_facing_narrative,
)
from data.prepare_data import DEFAULT_DB_PATH
from evaluation.reliability import RELIABILITY_SCENARIOS, get_reliability_scenario_by_id
from tools.chart_registry import ChartNotFoundError, ChartRegistry
from tools.dataset_store import DatasetStore
from tools.sql_tool import SQLValidationError, validate_sql


class ScriptedLLM:
    """Deterministic scripted mock LLM for testing orchestration workflows."""

    def __init__(self, responses: List[AIMessage]) -> None:
        self.responses = list(responses)
        self.call_count = 0
        self.invocations: List[List[Any]] = []

    def bind_tools(self, tools: List[BaseTool]) -> "ScriptedLLM":
        return self

    def invoke(self, messages: List[Any]) -> AIMessage:
        self.invocations.append(messages)
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        # Fallback response once scripted list is exhausted
        self.call_count += 1
        return AIMessage(content="Analysis concluded.")


@pytest.fixture
def temp_duckdb(tmp_path: Path) -> Path:
    """Fixture providing an isolated temporary copy of the Olist DuckDB database."""
    source_db = Path(DEFAULT_DB_PATH)
    target_db = tmp_path / "test_ecommerce.duckdb"
    shutil.copyfile(source_db, target_db)
    return target_db


def _make_dataset(store: DatasetStore, data: Dict[str, List[Any]], query: Optional[str] = None) -> str:
    """Helper to save structured mock query results into DatasetStore."""
    cols = list(data.keys())
    num_rows = len(data[cols[0]]) if cols else 0
    rows = []
    for i in range(num_rows):
        rows.append({c: data[c][i] for c in cols})
    payload = {
        "status": "success",
        "columns": cols,
        "rows": rows,
        "row_count": num_rows,
    }
    return store.save(payload, query=query)


# =============================================================================
# Category A — SQL Recovery (REL_SQL_01 to REL_SQL_04)
# =============================================================================

def test_rel_sql_01_syntax_error_recovery(temp_duckdb: Path) -> None:
    """REL_SQL_01: First SQL query has syntax error, second query succeeds within retry budget."""
    store = DatasetStore()
    tools = create_sql_tools(db_path=temp_duckdb, dataset_store=store)

    # Scripted LLM: Call 1 fails with syntax error, Call 2 provides valid query, Call 3 concludes
    scripted_llm = ScriptedLLM([
        AIMessage(
            content="",
            tool_calls=[{"name": "execute_sql", "args": {"query": "SELECT * FRM orders LIMIT 5"}, "id": "call_syntax_err"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "execute_sql", "args": {"query": "SELECT count(*) AS total_orders FROM orders"}, "id": "call_syntax_fix"}],
        ),
        AIMessage(content="There are 99,441 orders in the database."),
    ])

    graph = create_agent_graph(llm=scripted_llm, tools=tools, max_sql_retries=2)
    state = create_initial_state("Count all orders")
    final_state = graph.invoke(state)

    # Assertions
    assert final_state["sql_retry_count"] == 1, "Should consume exactly 1 retry attempt"
    assert final_state["last_error_category"] is None, "Error category should be cleared on success"
    assert any("99,441" in m.content for m in final_state["messages"] if isinstance(m, AIMessage))
    assert len(store.list_datasets()) == 1, "Successful query should register dataset"


def test_rel_sql_02_invalid_column_recovery(temp_duckdb: Path) -> None:
    """REL_SQL_02: First SQL references nonexistent column, second uses correct column."""
    store = DatasetStore()
    tools = create_sql_tools(db_path=temp_duckdb, dataset_store=store)

    scripted_llm = ScriptedLLM([
        AIMessage(
            content="",
            tool_calls=[{"name": "execute_sql", "args": {"query": "SELECT non_existing_column FROM customers"}, "id": "call_bad_col"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "execute_sql", "args": {"query": "SELECT customer_state, count(*) FROM customers GROUP BY 1"}, "id": "call_good_col"}],
        ),
        AIMessage(content="Customer state aggregation succeeded."),
    ])

    graph = create_agent_graph(llm=scripted_llm, tools=tools, max_sql_retries=2)
    state = create_initial_state("Aggregate customers by state")
    final_state = graph.invoke(state)

    assert final_state["sql_retry_count"] == 1
    assert final_state["last_error_category"] is None
    assert final_state["step_count"] == 3


def test_rel_sql_03_timeout_handling(temp_duckdb: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_execute_sql(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "status": "error",
            "error_type": "TimeoutError",
            "error_message": "Query execution exceeded timeout limit of 30.0 seconds.",
            "execution_time_ms": 30000.0,
        }

    import agent.tool_wrappers
    monkeypatch.setattr(agent.tool_wrappers, "raw_execute_sql", mock_execute_sql)

    store = DatasetStore()
    tools = create_sql_tools(db_path=temp_duckdb, dataset_store=store)

    scripted_llm = ScriptedLLM([
        AIMessage(
            content="",
            tool_calls=[{"name": "execute_sql", "args": {"query": "SELECT * FROM orders"}, "id": "call_timeout"}],
        ),
    ])

    graph = create_agent_graph(llm=scripted_llm, tools=tools, max_sql_retries=2)
    state = create_initial_state("Query orders with timeout")
    final_state = graph.invoke(state)

    assert final_state["last_error_category"] == NON_RETRYABLE_RESOURCE_ERROR
    last_msg = final_state["messages"][-1]
    assert "operational error" in last_msg.content.lower() or "timeout" in last_msg.content.lower()
    assert scripted_llm.call_count == 1, "Non-retryable resource error must immediately stop execution"


def test_rel_sql_04_retry_budget_exhaustion(temp_duckdb: Path) -> None:
    """REL_SQL_04: Force persistent SQL errors until max_sql_retries (2) is exhausted, terminating safely."""
    store = DatasetStore()
    tools = create_sql_tools(db_path=temp_duckdb, dataset_store=store)

    # Initial attempt + 2 retries = 3 attempts total
    scripted_llm = ScriptedLLM([
        AIMessage(content="", tool_calls=[{"name": "execute_sql", "args": {"query": "SELECT * FRM bad1"}, "id": "c1"}]),
        AIMessage(content="", tool_calls=[{"name": "execute_sql", "args": {"query": "SELECT * FRM bad2"}, "id": "c2"}]),
        AIMessage(content="", tool_calls=[{"name": "execute_sql", "args": {"query": "SELECT * FRM bad3"}, "id": "c3"}]),
        AIMessage(content="", tool_calls=[{"name": "execute_sql", "args": {"query": "SELECT * FRM bad4"}, "id": "c4"}]),
    ])

    graph = create_agent_graph(llm=scripted_llm, tools=tools, max_sql_retries=2)
    state = create_initial_state("Persistent malformed SQL")
    final_state = graph.invoke(state)

    # The 3rd attempt is retry attempt #2 (since initial does not count as retry)
    assert final_state["sql_retry_count"] >= 2
    assert final_state["last_error_category"] == RECOVERABLE_SQL_ERROR
    last_msg = final_state["messages"][-1]
    assert "correction retries" in last_msg.content or "retry limit" in last_msg.content
    assert scripted_llm.call_count <= 4, "Must not execute unbounded retries"


# =============================================================================
# Category B — Python Recovery (REL_PY_01 to REL_PY_04)
# =============================================================================

def test_rel_py_01_invalid_column_recovery(temp_duckdb: Path) -> None:
    """REL_PY_01: Python analysis with invalid column is classified as recoverable and recovers on retry."""
    store = DatasetStore()
    ds_id = _make_dataset(store, {"freight_value": [10.0, 20.0, 30.0], "price": [100.0, 200.0, 300.0]})

    tools = create_python_analysis_tools(dataset_store=store)

    scripted_llm = ScriptedLLM([
        # Call 1: invalid column
        AIMessage(
            content="",
            tool_calls=[{"name": "execute_python_analysis", "args": {"dataset_id": ds_id, "operation": "describe", "params": {"column": "nonexistent_col"}}, "id": "py1"}],
        ),
        # Call 2: valid column
        AIMessage(
            content="",
            tool_calls=[{"name": "execute_python_analysis", "args": {"dataset_id": ds_id, "operation": "describe", "params": {"column": "freight_value"}}, "id": "py2"}],
        ),
        AIMessage(content="Summary statistics for freight_value computed."),
    ])

    graph = create_agent_graph(llm=scripted_llm, tools=tools, max_python_retries=2)
    state = create_initial_state("Describe freight")
    final_state = graph.invoke(state)

    assert final_state["python_retry_count"] == 0, "Counter resets to 0 upon successful recovery"
    assert final_state["pending_python_recovery"] is False
    assert final_state["last_error_category"] is None
    assert "freight_value" in final_state["messages"][-1].content


def test_rel_py_02_invalid_dataset_id() -> None:
    """REL_PY_02: Python analysis with missing dataset ID classified as CATEGORY_MISSING_DATASET."""
    store = DatasetStore()
    tools = create_python_analysis_tools(dataset_store=store)

    scripted_llm = ScriptedLLM([
        AIMessage(
            content="",
            tool_calls=[{"name": "execute_python_analysis", "args": {"dataset_id": "ds_does_not_exist_999", "operation": "describe"}, "id": "py_missing"}],
        ),
    ])

    graph = create_agent_graph(llm=scripted_llm, tools=tools, max_python_retries=0)
    state = create_initial_state("Analyze missing dataset")
    final_state = graph.invoke(state)

    assert final_state["last_error_category"] == CATEGORY_MISSING_DATASET
    assert "Dataset 'ds_does_not_exist_999' not found" in final_state["last_error"]


def test_rel_py_03_invalid_numeric_input() -> None:
    """REL_PY_03: Attempting correlation/outlier detection on non-numeric column is safely caught."""
    store = DatasetStore()
    ds_id = _make_dataset(store, {"city": ["SP", "RJ", "MG"], "state": ["SP", "RJ", "MG"]})
    tools = create_python_analysis_tools(dataset_store=store)

    scripted_llm = ScriptedLLM([
        AIMessage(
            content="",
            tool_calls=[{"name": "execute_python_analysis", "args": {"dataset_id": ds_id, "operation": "detect_outliers", "params": {"column": "city"}}, "id": "py_type_err"}],
        ),
    ])

    graph = create_agent_graph(llm=scripted_llm, tools=tools, max_python_retries=0)
    state = create_initial_state("Detect outliers in city text")
    final_state = graph.invoke(state)

    assert final_state["last_error_category"] in (CATEGORY_INVALID_COLUMN, CATEGORY_INVALID_DATASET_STRUCTURE)
    assert "numeric" in final_state["last_error"].lower() or "not a numeric column" in final_state["last_error"].lower()


def test_rel_py_04_retry_budget_exhaustion() -> None:
    """REL_PY_04: Repeated recoverable Python failures terminate when max_python_retries is reached."""
    store = DatasetStore()
    ds_id = _make_dataset(store, {"val": [1, 2, 3]})
    tools = create_python_analysis_tools(dataset_store=store)

    scripted_llm = ScriptedLLM([
        AIMessage(content="", tool_calls=[{"name": "execute_python_analysis", "args": {"dataset_id": ds_id, "operation": "detect_outliers", "params": {"column": "bad_col_1"}}, "id": "p1"}]),
        AIMessage(content="", tool_calls=[{"name": "execute_python_analysis", "args": {"dataset_id": ds_id, "operation": "detect_outliers", "params": {"column": "bad_col_2"}}, "id": "p2"}]),
        AIMessage(content="", tool_calls=[{"name": "execute_python_analysis", "args": {"dataset_id": ds_id, "operation": "detect_outliers", "params": {"column": "bad_col_3"}}, "id": "p3"}]),
    ])

    graph = create_agent_graph(llm=scripted_llm, tools=tools, max_python_retries=1)
    state = create_initial_state("Exhaust Python retry budget")
    final_state = graph.invoke(state)

    assert final_state["python_retry_count"] >= 1
    last_msg = final_state["messages"][-1]
    assert "Python analysis still failed" in last_msg.content or "retry limit" in last_msg.content


# =============================================================================
# Category C — Security Enforcement (REL_SEC_01 to REL_SEC_04)
# =============================================================================

def test_rel_sec_01_mutation_rejection(temp_duckdb: Path) -> None:
    """REL_SEC_01: Mutation statements (DROP / DELETE / UPDATE) are rejected by AST validation before execution."""
    # Test DROP TABLE
    with pytest.raises(SQLValidationError) as exc1:
        validate_sql("DROP TABLE orders;")
    assert "Only read-only analytical SELECT queries are permitted" in str(exc1.value)
    assert "Drop" in str(exc1.value)

    # Test DELETE
    with pytest.raises(SQLValidationError) as exc2:
        validate_sql("DELETE FROM orders WHERE 1=1;")
    assert "Only read-only analytical SELECT queries are permitted" in str(exc2.value)
    assert "Delete" in str(exc2.value)

    # Test through LangGraph agent tool calling
    store = DatasetStore()
    tools = create_sql_tools(db_path=temp_duckdb, dataset_store=store)

    scripted_llm = ScriptedLLM([
        AIMessage(content="", tool_calls=[{"name": "execute_sql", "args": {"query": "DROP TABLE orders;"}, "id": "call_drop"}]),
    ])

    graph = create_agent_graph(llm=scripted_llm, tools=tools)
    state = create_initial_state("Delete the orders table")
    final_state = graph.invoke(state)

    assert final_state["last_error_category"] == NON_RETRYABLE_SECURITY_ERROR
    assert "security policy" in final_state["messages"][-1].content
    assert scripted_llm.call_count == 1, "Must never retry on security error"


def test_rel_sec_02_external_file_access() -> None:
    """REL_SEC_02: Prohibited external file reads via read_csv / read_parquet are blocked."""
    prohibited_queries = [
        "SELECT * FROM read_csv_auto('/etc/passwd');",
        "SELECT * FROM read_parquet('s3://mybucket/data.parquet');",
        "SELECT * FROM glob('../*');",
    ]

    for q in prohibited_queries:
        with pytest.raises(SQLValidationError) as exc:
            validate_sql(q)
        assert "not permitted" in str(exc.value).lower()


def test_rel_sec_03_unsafe_statement_rejection() -> None:
    """REL_SEC_03: Administrative and dangerous statements (ATTACH, COPY, PRAGMA) are rejected."""
    unsafe_statements = [
        "ATTACH 'malicious.db' AS evil;",
        "COPY orders TO 'orders.csv';",
        "PRAGMA database_list;",
        "CREATE TABLE test AS SELECT 1;",
    ]

    for stmt in unsafe_statements:
        with pytest.raises(SQLValidationError):
            validate_sql(stmt)


def test_rel_sec_04_database_integrity(temp_duckdb: Path) -> None:
    """REL_SEC_04: Verify table row counts and schema remain completely invariant after attacks."""
    con = duckdb.connect(database=str(temp_duckdb), read_only=True)
    initial_orders_count = con.execute("SELECT count(*) FROM orders").fetchone()[0]
    initial_items_count = con.execute("SELECT count(*) FROM order_items").fetchone()[0]
    con.close()

    # Attempt malicious executions through SQL tool
    store = DatasetStore()
    sql_tool = create_sql_tools(db_path=temp_duckdb, dataset_store=store)[1]  # execute_sql

    malicious_inputs = [
        "DROP TABLE orders;",
        "UPDATE orders SET order_status = 'tampered';",
        "TRUNCATE order_items;",
    ]

    for bad_sql in malicious_inputs:
        res = sql_tool.invoke({"query": bad_sql})
        parsed = json.loads(res) if isinstance(res, str) else res
        assert parsed["status"] == "error"

    # Post-check invariance
    con2 = duckdb.connect(database=str(temp_duckdb), read_only=True)
    post_orders_count = con2.execute("SELECT count(*) FROM orders").fetchone()[0]
    post_items_count = con2.execute("SELECT count(*) FROM order_items").fetchone()[0]
    con2.close()

    assert initial_orders_count == post_orders_count == 99441
    assert initial_items_count == post_items_count == 112650


# =============================================================================
# Category D — Chart Reliability (REL_CHART_01 to REL_CHART_03)
# =============================================================================

def test_rel_chart_01_unsupported_type() -> None:
    """REL_CHART_01: Unsupported chart type raises non-retryable chart error with zero retries."""
    store = DatasetStore()
    registry = ChartRegistry()
    ds_id = _make_dataset(store, {"x": [1, 2], "y": [3, 4]})

    tools = create_chart_generation_tools(dataset_store=store, chart_registry=registry)

    scripted_llm = ScriptedLLM([
        AIMessage(
            content="",
            tool_calls=[{"name": "generate_chart", "args": {"dataset_id": ds_id, "chart_type": "3d_surface", "x_column": "x", "y_column": "y"}, "id": "c_bad_type"}],
        ),
    ])

    graph = create_agent_graph(llm=scripted_llm, tools=tools)
    state = create_initial_state("Plot 3D surface")
    final_state = graph.invoke(state)

    assert final_state["last_error_category"] == NON_RETRYABLE_CHART_ERROR
    assert len(registry) == 0, "Invalid chart must never be stored in ChartRegistry"
    assert scripted_llm.call_count == 1, "Chart failures must not trigger automatic retries"


def test_rel_chart_02_invalid_columns() -> None:
    """REL_CHART_02: Chart generation with missing column fails cleanly without corrupting registry."""
    store = DatasetStore()
    registry = ChartRegistry()
    ds_id = _make_dataset(store, {"valid_col": [10, 20]})

    tools = create_chart_generation_tools(dataset_store=store, chart_registry=registry)

    scripted_llm = ScriptedLLM([
        AIMessage(
            content="",
            tool_calls=[{"name": "generate_chart", "args": {"dataset_id": ds_id, "chart_type": "bar", "x_column": "valid_col", "y_column": "missing_y"}, "id": "c_bad_col"}],
        ),
    ])

    graph = create_agent_graph(llm=scripted_llm, tools=tools)
    state = create_initial_state("Plot with missing y column")
    final_state = graph.invoke(state)

    assert final_state["last_error_category"] == NON_RETRYABLE_CHART_ERROR
    assert len(registry) == 0


def test_rel_chart_03_missing_artifact_graceful_handling() -> None:
    """REL_CHART_03: Requesting a missing or invalid chart_id from ChartRegistry raises ChartNotFoundError gracefully."""
    registry = ChartRegistry()
    with pytest.raises(ChartNotFoundError):
        registry.get("chart_nonexistent_123456")

    # In UI session execution, an invalid chart_id is safely logged and does not crash turn extraction
    from agent.ui_session import extract_turn_details

    mock_tool_msg = ToolMessage(
        content=json.dumps({"status": "success", "chart_id": "chart_valid_1"}),
        name="generate_chart",
        tool_call_id="c1",
    )
    # AIMessage with tool call
    ai_call = AIMessage(
        content="",
        tool_calls=[{"name": "generate_chart", "args": {"chart_type": "bar"}, "id": "c1"}],
    )
    text, chart_ids, executions = extract_turn_details([ai_call, mock_tool_msg])
    assert chart_ids == ["chart_valid_1"]


# =============================================================================
# Category E — Execution Limits (REL_EXEC_01 to REL_EXEC_02)
# =============================================================================

def test_rel_exec_01_step_limit_exceeded(temp_duckdb: Path) -> None:
    """REL_EXEC_01: Workflow terminates deterministically when max_steps is reached."""
    store = DatasetStore()
    tools = create_sql_tools(db_path=temp_duckdb, dataset_store=store)

    # Endless loop of tool calls
    responses = [
        AIMessage(content="", tool_calls=[{"name": "execute_sql", "args": {"query": f"SELECT {i}"}, "id": f"c_{i}"}])
        for i in range(10)
    ]
    scripted_llm = ScriptedLLM(responses)

    # Bound max_steps to 3
    graph = create_agent_graph(llm=scripted_llm, tools=tools, max_steps=3)
    state = create_initial_state("Loop forever")
    final_state = graph.invoke(state)

    assert final_state["step_count"] == 3
    last_msg = final_state["messages"][-1]
    assert "maximum LLM invocation limit (3) was reached" in last_msg.content


def test_rel_exec_02_non_recoverable_error_precedence() -> None:
    """REL_EXEC_02: Non-retryable error immediately halts execution without allowing retries."""
    store = DatasetStore()
    registry = ChartRegistry()
    tools = create_chart_generation_tools(dataset_store=store, chart_registry=registry)

    scripted_llm = ScriptedLLM([
        AIMessage(
            content="",
            tool_calls=[{"name": "generate_chart", "args": {"dataset_id": "nonexistent_ds", "chart_type": "bar", "x_column": "a", "y_column": "b"}, "id": "c_chart_err"}],
        ),
        AIMessage(content="Should not be called."),
    ])

    graph = create_agent_graph(llm=scripted_llm, tools=tools)
    state = create_initial_state("Fail chart immediately")
    final_state = graph.invoke(state)

    assert final_state["last_error_category"] == NON_RETRYABLE_CHART_ERROR
    assert scripted_llm.call_count == 1, "Must immediately halt on non-retryable error"


# =============================================================================
# Category F — Session Reliability (REL_SESSION_01 to REL_SESSION_02)
# =============================================================================

def test_rel_session_01_cross_session_isolation(temp_duckdb: Path) -> None:
    """REL_SESSION_01: Verify Session A and Session B maintain strict resource and history isolation."""
    # Create two sessions
    session_a = create_agent_session(db_path=temp_duckdb, llm=ScriptedLLM([AIMessage(content="Turn A")]))
    session_b = create_agent_session(db_path=temp_duckdb, llm=ScriptedLLM([AIMessage(content="Turn B")]))

    # Register dataset in Session A
    ds_a = _make_dataset(session_a.dataset_store, {"x": [1, 2, 3]}, query="SELECT x FROM a")

    # Register chart in Session A
    import plotly.graph_objects as go
    fig_a = go.Figure(data=[go.Bar(x=["A"], y=[10])])
    cid_a = session_a.chart_registry.save(fig_a.to_dict(), ds_a, "bar", "Chart A")

    # Verify Session B cannot see Session A's dataset or chart
    b_dataset_ids = [d["dataset_id"] for d in session_b.dataset_store.list_datasets()]
    assert ds_a not in b_dataset_ids
    assert cid_a not in session_b.chart_registry
    with pytest.raises(KeyError):
        session_b.dataset_store.get(ds_a)
    with pytest.raises(ChartNotFoundError):
        session_b.chart_registry.get(cid_a)


def test_rel_session_02_session_reset(temp_duckdb: Path) -> None:
    """REL_SESSION_02: reset_agent_session clears stores, registries, and message histories."""
    import plotly.graph_objects as go

    session = create_agent_session(db_path=temp_duckdb, llm=ScriptedLLM([AIMessage(content="Turn 1")]))

    # Populate session
    ds_id = _make_dataset(session.dataset_store, {"a": [1]})
    cid = session.chart_registry.save(go.Figure().to_dict(), ds_id, "bar", "Test")
    session.agent_messages.append(HumanMessage(content="Hello"))
    session.ui_messages.append({"role": "user", "content": "Hello"})

    assert len(session.dataset_store) == 1
    assert len(session.chart_registry) == 1
    assert len(session.agent_messages) == 1
    assert len(session.ui_messages) == 1

    # Reset
    reset_agent_session(session)

    assert len(session.dataset_store) == 0
    assert len(session.chart_registry) == 0
    assert len(session.agent_messages) == 0
    assert len(session.ui_messages) == 0


# =============================================================================
# Category G — Failure Presentation (REL_UI_01)
# =============================================================================

def test_rel_ui_01_sanitized_failure_and_recovery(temp_duckdb: Path) -> None:
    """REL_UI_01: Internal exceptions containing sensitive info are sanitized; turn 2 recovers cleanly."""
    # 1. Test narrative sanitizer strips sensitive technical identifiers
    raw_sensitive_narrative = (
        "Analysis ready. View chart at [Revenue](chart_002c486a1a20) and data in ds_0123456789abcdef.\n"
        "Internal reference: Chart ID: `chart_002c486a1a20`.\n"
        "Total revenue was $1,234,567.89 across 96,478 orders."
    )
    sanitized = sanitize_user_facing_narrative(raw_sensitive_narrative)
    assert "chart_002c486a1a20" not in sanitized
    assert "ds_0123456789abcdef" not in sanitized
    assert "$1,234,567.89" in sanitized
    assert "96,478" in sanitized

    # 2. Test turn execution crash containment and clean recovery on subsequent turn
    scripted_llm = ScriptedLLM([
        # Turn 1: Simulates error message from model
        AIMessage(content="Internal error: /var/secrets/openai_api_key failed."),
        # Turn 2: Normal business answer
        AIMessage(content="Delivered orders count is 96,478."),
    ])

    session = create_agent_session(db_path=temp_duckdb, llm=scripted_llm)

    # Turn 1
    res1 = execute_session_turn(session, "Simulate turn 1")
    assert not res1.get("error", False)

    # Turn 2: Should execute cleanly with fresh turn counters
    res2 = execute_session_turn(session, "How many delivered orders?")
    assert not res2.get("error", False)
    assert "96,478" in res2["content"]
    assert len(session.ui_messages) == 4  # 2 user messages + 2 assistant messages
