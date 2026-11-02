"""Deterministic tests for Step 7: Lightweight Planning & Intelligent Tool Selection.

Covers test cases A through J:
  Case A: Simple aggregation -> SQL only
  Case B: Pearson correlation -> SQL only or SQL -> Python
  Case C: Outlier detection -> SQL -> Python
  Case D: Monthly revenue visualization -> SQL -> Chart
  Case E: Statistical analysis + visualization -> SQL -> Python -> Chart
  Case F: Multi-turn visualization -> Existing Dataset -> Chart
  Case G: Explicit histogram request -> SQL -> Chart
  Case H: SQL security rejection -> Safe termination
  Case I: Completed SQL-only answer -> No redundant Python call
  Case J: Chart request completion -> Chart artifact must exist
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import duckdb
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from agent.graph import (
    CHART_TOOL_NAME,
    PYTHON_TOOL_NAME,
    SQL_TOOL_NAME,
    create_agent_graph,
)
from agent.prompts import SYSTEM_PROMPT
from agent.ui_session import (
    create_agent_session,
    execute_session_turn,
)
from tools.dataset_store import DatasetStore
from tools.chart_registry import ChartRegistry


# ==============================================================================
# Deterministic Scripted LLM for Testing Orchestration & Planning Paths
# ==============================================================================

class DynamicScriptedLLM(BaseChatModel):
    """Dynamic scripted LLM that inspects conversation state or plays predefined turns."""

    responder: Any = None
    invocations: List[Any] = Field(default_factory=list)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.invocations.append(list(messages))
        if callable(self.responder):
            resp = self.responder(messages)
        elif isinstance(self.responder, list):
            idx = len(self.invocations) - 1
            if idx >= len(self.responder):
                raise AssertionError(
                    f"LLM invoked {idx + 1} times, but only {len(self.responder)} responses provided."
                )
            resp = self.responder[idx]
        else:
            raise TypeError(f"Unsupported responder type: {type(self.responder)}")

        msg = AIMessage(
            content=resp.content,
            tool_calls=[dict(tc) for tc in resp.tool_calls],
            id=str(uuid.uuid4()),
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "DynamicScriptedLLM":
        return self

    @property
    def _llm_type(self) -> str:
        return "dynamic-scripted-planning-llm"


def ai(*calls: dict, text: str = "") -> AIMessage:
    return AIMessage(content=text, tool_calls=list(calls))


def sql_call(query: str, call_id: str = "call_sql") -> dict:
    return {"name": SQL_TOOL_NAME, "args": {"query": query}, "id": call_id}


def py_call(
    dataset_id: str,
    operation: str,
    params: Optional[dict] = None,
    call_id: str = "call_py",
) -> dict:
    return {
        "name": PYTHON_TOOL_NAME,
        "args": {
            "dataset_id": dataset_id,
            "operation": operation,
            "params": params or {},
        },
        "id": call_id,
    }


def chart_call(
    dataset_id: str,
    chart_type: str,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    params: Optional[dict] = None,
    call_id: str = "call_chart",
) -> dict:
    return {
        "name": CHART_TOOL_NAME,
        "args": {
            "dataset_id": dataset_id,
            "chart_type": chart_type,
            "x_column": x_column,
            "y_column": y_column,
            "params": params or {},
        },
        "id": call_id,
    }


def _extract_dataset_id_from_messages(messages: List[BaseMessage]) -> Optional[str]:
    """Find the most recent dataset_id in tool results."""
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            try:
                data = json.loads(m.content)
                if isinstance(data, dict):
                    if "dataset_id" in data and data["dataset_id"]:
                        return data["dataset_id"]
                    if "derived_dataset_id" in data and data["derived_dataset_id"]:
                        return data["derived_dataset_id"]
            except Exception:
                pass
    return None


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    """Create a temporary DuckDB database containing sample Olist tables."""
    db_file = tmp_path / "planning_test.duckdb"
    con = duckdb.connect(str(db_file))
    con.execute("""
        CREATE TABLE orders AS
        SELECT 'ord_1' AS order_id, 'delivered' AS order_status, '2017-03-01' AS order_purchase_timestamp
        UNION ALL
        SELECT 'ord_2' AS order_id, 'delivered' AS order_status, '2017-03-15' AS order_purchase_timestamp
        UNION ALL
        SELECT 'ord_3' AS order_id, 'canceled' AS order_status, '2017-04-01' AS order_purchase_timestamp
    """)
    con.execute("""
        CREATE TABLE order_items AS
        SELECT 'ord_1' AS order_id, 10.0 AS price, 5.0 AS freight_value, 'health_beauty' AS product_category_name
        UNION ALL
        SELECT 'ord_2' AS order_id, 20.0 AS price, 10.0 AS freight_value, 'bed_bath' AS product_category_name
        UNION ALL
        SELECT 'ord_3' AS order_id, 30.0 AS price, 15.0 AS freight_value, 'sports' AS product_category_name
    """)
    con.execute("""
        CREATE TABLE order_summary AS
        SELECT 'ord_1' AS order_id, 'delivered' AS order_status, 100.0 AS item_subtotal, '2017-01-10' AS order_purchase_timestamp, 3.0 AS delivery_duration_days
        UNION ALL
        SELECT 'ord_2' AS order_id, 'delivered' AS order_status, 200.0 AS item_subtotal, '2017-02-12' AS order_purchase_timestamp, 5.0 AS delivery_duration_days
        UNION ALL
        SELECT 'ord_3' AS order_id, 'delivered' AS order_status, 300.0 AS item_subtotal, '2017-02-28' AS order_purchase_timestamp, 45.0 AS delivery_duration_days
    """)
    con.close()
    return db_file


# ==============================================================================
# Prompt Architecture & Protocol Verification
# ==============================================================================

def test_system_prompt_planning_protocol_content():
    """Verify that SYSTEM_PROMPT contains explicit lightweight adaptive planning rules."""
    assert "LIGHTWEIGHT ADAPTIVE PLANNING PROTOCOL" in SYSTEM_PROMPT
    assert "Analytical Objective" in SYSTEM_PROMPT
    assert "Data Requirements & Granularity" in SYSTEM_PROMPT
    assert "Minimal Tool Sequence" in SYSTEM_PROMPT
    assert "Adaptive Evaluation" in SYSTEM_PROMPT
    assert "DuckDB is an efficient analytical database engine" in SYSTEM_PROMPT
    assert "DO NOT invoke Python analysis" in SYSTEM_PROMPT
    assert "describe" in SYSTEM_PROMPT and "skewness" in SYSTEM_PROMPT
    assert "Explicit Visualization Requests" in SYSTEM_PROMPT or "generate_chart" in SYSTEM_PROMPT


# ==============================================================================
# Regression Test Cases A - J
# ==============================================================================

def test_case_a_simple_aggregation_sql_only(test_db: Path):
    """Case A: Simple aggregation question -> SQL only, no Python or Chart tool."""
    def responder(messages: List[BaseMessage]) -> AIMessage:
        last = messages[-1]
        if isinstance(last, HumanMessage):
            return ai(sql_call("SELECT count(*) AS delivered_count FROM orders WHERE order_status = 'delivered'"))
        elif isinstance(last, ToolMessage):
            return ai(text="There are 2 delivered orders in the database.")
        raise AssertionError("Unexpected state in Case A")

    llm = DynamicScriptedLLM(responder=responder)
    session = create_agent_session(db_path=test_db, llm=llm)

    res = execute_session_turn(session, "How many delivered orders are there?")
    assert "2 delivered orders" in res["content"]
    assert res["chart_ids"] == []

    # Verify only SQL was executed
    tools_called = [e["tool"] for e in res["tool_executions"]]
    assert tools_called == [SQL_TOOL_NAME]
    assert PYTHON_TOOL_NAME not in tools_called
    assert CHART_TOOL_NAME not in tools_called


def test_case_b_pearson_correlation_sql_only(test_db: Path):
    """Case B: Pearson correlation -> SQL only via corr(), without unnecessary Python call."""
    def responder(messages: List[BaseMessage]) -> AIMessage:
        last = messages[-1]
        if isinstance(last, HumanMessage):
            return ai(sql_call("SELECT corr(price, freight_value) AS r FROM order_items"))
        elif isinstance(last, ToolMessage):
            return ai(text="The Pearson correlation between product price and freight value is 1.0.")
        raise AssertionError("Unexpected state in Case B")

    llm = DynamicScriptedLLM(responder=responder)
    session = create_agent_session(db_path=test_db, llm=llm)

    res = execute_session_turn(session, "What is the correlation between product price and freight value?")
    assert "correlation" in res["content"].lower()
    tools_called = [e["tool"] for e in res["tool_executions"]]
    assert tools_called == [SQL_TOOL_NAME]
    assert PYTHON_TOOL_NAME not in tools_called


def test_case_c_outlier_detection_sql_then_python(test_db: Path):
    """Case C: Outlier detection -> SQL to extract data, then Python detect_outliers."""
    def responder(messages: List[BaseMessage]) -> AIMessage:
        last = messages[-1]
        if isinstance(last, HumanMessage):
            return ai(sql_call("SELECT delivery_duration_days FROM order_summary WHERE delivery_duration_days IS NOT NULL"))
        elif isinstance(last, ToolMessage) and last.name == SQL_TOOL_NAME:
            ds_id = _extract_dataset_id_from_messages(messages)
            assert ds_id is not None
            return ai(py_call(dataset_id=ds_id, operation="detect_outliers", params={"column": "delivery_duration_days"}))
        elif isinstance(last, ToolMessage) and last.name == PYTHON_TOOL_NAME:
            return ai(text="Identified 1 outlier in delivery duration with value 45.0 days.")
        raise AssertionError("Unexpected state in Case C")

    llm = DynamicScriptedLLM(responder=responder)
    session = create_agent_session(db_path=test_db, llm=llm)

    res = execute_session_turn(session, "Identify outliers in delivery duration.")
    assert "Identified 1 outlier" in res["content"]
    tools_called = [e["tool"] for e in res["tool_executions"]]
    assert tools_called == [SQL_TOOL_NAME, PYTHON_TOOL_NAME]
    assert CHART_TOOL_NAME not in tools_called


def test_case_d_monthly_revenue_visualization_sql_then_chart(test_db: Path):
    """Case D: Monthly revenue visual -> SQL aggregate directly to Chart tool, no Python."""
    def responder(messages: List[BaseMessage]) -> AIMessage:
        last = messages[-1]
        if isinstance(last, HumanMessage):
            return ai(sql_call("SELECT strftime('%Y-%m', CAST(order_purchase_timestamp AS TIMESTAMP)) AS month, SUM(item_subtotal) AS rev FROM order_summary GROUP BY month ORDER BY month"))
        elif isinstance(last, ToolMessage) and last.name == SQL_TOOL_NAME:
            ds_id = _extract_dataset_id_from_messages(messages)
            assert ds_id is not None
            return ai(chart_call(dataset_id=ds_id, chart_type="line", x_column="month", y_column="rev"))
        elif isinstance(last, ToolMessage) and last.name == CHART_TOOL_NAME:
            return ai(text="Here is the monthly delivered revenue trend for 2017.")
        raise AssertionError("Unexpected state in Case D")

    llm = DynamicScriptedLLM(responder=responder)
    session = create_agent_session(db_path=test_db, llm=llm)

    res = execute_session_turn(session, "Show monthly delivered revenue in 2017.")
    assert "monthly delivered revenue trend" in res["content"]
    assert len(res["chart_ids"]) == 1
    assert res["chart_ids"][0] in session.chart_registry

    tools_called = [e["tool"] for e in res["tool_executions"]]
    assert tools_called == [SQL_TOOL_NAME, CHART_TOOL_NAME]
    assert PYTHON_TOOL_NAME not in tools_called


def test_case_e_statistical_analysis_plus_visualization(test_db: Path):
    """Case E: Statistical analysis + visualization -> SQL -> Python -> Chart."""
    def responder(messages: List[BaseMessage]) -> AIMessage:
        last = messages[-1]
        if isinstance(last, HumanMessage):
            return ai(sql_call("SELECT freight_value FROM order_items"))
        elif isinstance(last, ToolMessage) and last.name == SQL_TOOL_NAME:
            ds_id = _extract_dataset_id_from_messages(messages)
            assert ds_id is not None
            return ai(py_call(dataset_id=ds_id, operation="detect_outliers", params={"column": "freight_value"}))
        elif isinstance(last, ToolMessage) and last.name == PYTHON_TOOL_NAME:
            ds_id = _extract_dataset_id_from_messages(messages)
            # Use original SQL dataset for box plot
            sql_msg = [m for m in messages if isinstance(m, ToolMessage) and m.name == SQL_TOOL_NAME][0]
            orig_ds_id = json.loads(sql_msg.content)["dataset_id"]
            return ai(chart_call(dataset_id=orig_ds_id, chart_type="box", y_column="freight_value"))
        elif isinstance(last, ToolMessage) and last.name == CHART_TOOL_NAME:
            return ai(text="The freight values range from 5.0 to 15.0 with no extreme outliers detected.")
        raise AssertionError("Unexpected state in Case E")

    llm = DynamicScriptedLLM(responder=responder)
    session = create_agent_session(db_path=test_db, llm=llm)

    res = execute_session_turn(session, "Analyze the distribution of freight values, identify outliers, and visualize the distribution.")
    assert "freight values range from" in res["content"]
    assert len(res["chart_ids"]) == 1

    tools_called = [e["tool"] for e in res["tool_executions"]]
    assert tools_called == [SQL_TOOL_NAME, PYTHON_TOOL_NAME, CHART_TOOL_NAME]


def test_case_f_multi_turn_dataset_reuse_chart(test_db: Path):
    """Case F: Turn 1 produces dataset; Turn 2 visualizes it by reusing existing dataset_id without re-querying SQL."""
    turn1_ds_id = None

    def responder(messages: List[BaseMessage]) -> AIMessage:
        nonlocal turn1_ds_id
        last = messages[-1]
        # Turn 1
        if isinstance(last, HumanMessage) and "top 5" in last.content.lower():
            return ai(sql_call("SELECT product_category_name, SUM(price) AS rev FROM order_items GROUP BY product_category_name ORDER BY rev DESC LIMIT 5"))
        elif isinstance(last, ToolMessage) and last.name == SQL_TOOL_NAME:
            turn1_ds_id = json.loads(last.content)["dataset_id"]
            return ai(text="The top categories are sports, bed_bath, and health_beauty.")
        # Turn 2
        elif isinstance(last, HumanMessage) and "visualize" in last.content.lower():
            assert turn1_ds_id is not None
            # Directly call chart tool reusing turn 1 dataset_id
            return ai(chart_call(dataset_id=turn1_ds_id, chart_type="bar", x_column="product_category_name", y_column="rev"))
        elif isinstance(last, ToolMessage) and last.name == CHART_TOOL_NAME:
            return ai(text="Here is the bar chart for the top product categories.")
        raise AssertionError(f"Unexpected message state: {last}")

    llm = DynamicScriptedLLM(responder=responder)
    session = create_agent_session(db_path=test_db, llm=llm)

    # Turn 1
    t1_res = execute_session_turn(session, "What are the top 5 product categories by revenue?")
    assert "top categories are sports" in t1_res["content"]
    assert t1_res["chart_ids"] == []
    assert [e["tool"] for e in t1_res["tool_executions"]] == [SQL_TOOL_NAME]

    # Turn 2
    t2_res = execute_session_turn(session, "Visualize them.")
    assert "bar chart for the top product categories" in t2_res["content"]
    assert len(t2_res["chart_ids"]) == 1
    # Ensure SQL was NOT invoked in Turn 2
    t2_tools = [e["tool"] for e in t2_res["tool_executions"]]
    assert t2_tools == [CHART_TOOL_NAME]
    assert SQL_TOOL_NAME not in t2_tools


def test_case_g_explicit_histogram_request(test_db: Path):
    """Case G: Explicit histogram request -> SQL retrieves raw observations, Chart Tool renders histogram."""
    def responder(messages: List[BaseMessage]) -> AIMessage:
        last = messages[-1]
        if isinstance(last, HumanMessage):
            return ai(sql_call("SELECT freight_value FROM order_items"))
        elif isinstance(last, ToolMessage) and last.name == SQL_TOOL_NAME:
            ds_id = _extract_dataset_id_from_messages(messages)
            assert ds_id is not None
            return ai(chart_call(dataset_id=ds_id, chart_type="histogram", x_column="freight_value"))
        elif isinstance(last, ToolMessage) and last.name == CHART_TOOL_NAME:
            return ai(text="The freight value distribution histogram is shown below.")
        raise AssertionError("Unexpected state in Case G")

    llm = DynamicScriptedLLM(responder=responder)
    session = create_agent_session(db_path=test_db, llm=llm)

    res = execute_session_turn(session, "Analyze the distribution of freight values with a histogram.")
    assert "histogram is shown below" in res["content"]
    assert len(res["chart_ids"]) == 1
    cid = res["chart_ids"][0]
    fig_rec = session.chart_registry.get(cid)
    assert fig_rec.chart_type == "histogram"


def test_case_h_sql_security_rejection_safe_termination(test_db: Path):
    """Case H: SQL security violation (e.g. DROP TABLE) halts safely without invoking other tools."""
    def responder(messages: List[BaseMessage]) -> AIMessage:
        last = messages[-1]
        if isinstance(last, HumanMessage):
            return ai(sql_call("DROP TABLE orders"))
        elif isinstance(last, ToolMessage):
            raise AssertionError("Model should not be reinvoked after non-retryable security rejection")
        raise AssertionError(f"Unexpected state in Case H: {last}")

    llm = DynamicScriptedLLM(responder=responder)
    session = create_agent_session(db_path=test_db, llm=llm)

    res = execute_session_turn(session, "Drop the orders table.")
    assert "Execution stopped" in res["content"]
    assert "rejected by security policy" in res["content"]
    assert res["chart_ids"] == []


def test_case_i_completed_sql_only_no_redundant_python(test_db: Path):
    """Case I: When SQL directly produces mean and median, no redundant Python describe call is made."""
    def responder(messages: List[BaseMessage]) -> AIMessage:
        last = messages[-1]
        if isinstance(last, HumanMessage):
            return ai(sql_call("SELECT avg(freight_value) AS mean_freight, median(freight_value) AS median_freight FROM order_items"))
        elif isinstance(last, ToolMessage) and last.name == SQL_TOOL_NAME:
            # Model immediately completes without invoking Python describe
            return ai(text="The mean freight is 10.0 and median freight is 10.0, indicating a symmetric distribution.")
        raise AssertionError("Unexpected state in Case I")

    llm = DynamicScriptedLLM(responder=responder)
    session = create_agent_session(db_path=test_db, llm=llm)

    res = execute_session_turn(session, "Compare the mean and median freight values across order items.")
    assert "mean freight is 10.0" in res["content"]
    tools_called = [e["tool"] for e in res["tool_executions"]]
    assert tools_called == [SQL_TOOL_NAME]
    assert PYTHON_TOOL_NAME not in tools_called


def test_case_j_chart_request_completion_must_have_artifact(test_db: Path):
    """Case J: When a chart is requested, execution is only complete if a Chart artifact is successfully registered."""
    def responder(messages: List[BaseMessage]) -> AIMessage:
        last = messages[-1]
        if isinstance(last, HumanMessage):
            return ai(sql_call("SELECT product_category_name, SUM(price) AS total_rev FROM order_items GROUP BY product_category_name"))
        elif isinstance(last, ToolMessage) and last.name == SQL_TOOL_NAME:
            ds_id = _extract_dataset_id_from_messages(messages)
            assert ds_id is not None
            return ai(chart_call(dataset_id=ds_id, chart_type="bar", x_column="product_category_name", y_column="total_rev"))
        elif isinstance(last, ToolMessage) and last.name == CHART_TOOL_NAME:
            return ai(text="Here is the category revenue bar chart.")
        raise AssertionError("Unexpected state in Case J")

    llm = DynamicScriptedLLM(responder=responder)
    session = create_agent_session(db_path=test_db, llm=llm)

    res = execute_session_turn(session, "Show me a bar chart of category revenue.")
    assert len(res["chart_ids"]) == 1
    cid = res["chart_ids"][0]
    # Verify artifact actually exists in session ChartRegistry
    assert cid in session.chart_registry
    rec = session.chart_registry.get(cid)
    assert rec.figure is not None
    assert rec.chart_type == "bar"
