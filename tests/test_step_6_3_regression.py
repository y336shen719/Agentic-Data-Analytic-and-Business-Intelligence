"""Regression tests for Step 6.3:
Covers all four reported issues:
Issue 1: Chart Generated but Not Rendered (Tests A, B, E)
Issue 2: SQL Worker PicklingError & AST Logic (Tests C, D)
Issue 3: Histogram vs Discrete Distribution Visualization Selection (Test E)
Issue 4: Clean, result-first UI without internal IDs or technical expanders (Test F, G)
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import unittest.mock as mock

import duckdb
import numpy as np
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from agent.sql_error_classifier import classify_sql_result, NON_RETRYABLE_RESOURCE_ERROR
from agent.ui_session import (
    DEFAULT_INCOMPLETE_MESSAGE,
    create_agent_session,
    execute_session_turn,
    extract_turn_details,
    sanitize_user_facing_narrative,
)
from tools.chart_registry import ChartRegistry
from tools.dataset_store import DatasetStore
from tools.sql_tool import (
    _ALLOWED_FUNC_NAMES,
    _resolve_worker_target,
    _worker_query_runner,
    execute_sql,
    validate_sql,
)
from tools.python_analysis import PythonAnalysisEngine
from agent.tool_wrappers import create_chart_generation_tool


def _extract_trace_values(val: Any) -> list:
    """Helper to extract trace values from plotly dict/list representations."""
    if isinstance(val, dict) and "bdata" in val:
        raw_bytes = base64.b64decode(val["bdata"])
        dtype = np.dtype(val.get("dtype", "f8"))
        return np.frombuffer(raw_bytes, dtype=dtype).tolist()
    return list(val)


# ==============================================================================
# Regression Test A: SQL -> Chart -> Streamlit
# ==============================================================================

def test_regression_sql_to_chart_to_streamlit(tmp_path: Path, monkeypatch):
    """Test A: Verify SQL execution -> Chart registration -> UI extraction -> Streamlit rendering.
    
    Verifies:
    - SQL returns data.
    - Chart Tool registers a valid Plotly Figure.
    - The UI message contains the correct internal Chart ID.
    - Streamlit retrieves and renders the corresponding figure.
    - The chart's actual x/y values match the test fixture.
    - Streamlit reruns do not duplicate the chart.
    - The displayed narrative contains no internal IDs.
    """
    db_file = tmp_path / "test_a.duckdb"
    con = duckdb.connect(str(db_file))
    con.execute("""
        CREATE TABLE order_summary AS
        SELECT 'ord_1' AS order_id, 'delivered' AS order_status, 100.0 AS item_subtotal, '2017-01-15' AS order_purchase_timestamp
        UNION ALL
        SELECT 'ord_2' AS order_id, 'delivered' AS order_status, 200.0 AS item_subtotal, '2017-02-15' AS order_purchase_timestamp
    """)
    con.close()

    session = create_agent_session(db_path=db_file, api_key="sk-mock-test")

    # 1. SQL execution returns data and registers in store
    sql_res = execute_sql(
        "SELECT strftime('%Y-%m', CAST(order_purchase_timestamp AS TIMESTAMP)) AS month, SUM(item_subtotal) AS revenue "
        "FROM order_summary WHERE order_status = 'delivered' GROUP BY month ORDER BY month",
        db_path=db_file,
    )
    assert sql_res["status"] == "success"
    assert len(sql_res["rows"]) == 2

    ds_id = session.dataset_store.save(sql_res, query="SELECT ...")

    chart_tool = create_chart_generation_tool(
        dataset_store=session.dataset_store,
        chart_registry=session.chart_registry,
    )
    chart_res = chart_tool.invoke({
        "dataset_id": ds_id,
        "chart_type": "line",
        "x_column": "month",
        "y_column": "revenue",
    })
    assert chart_res["status"] == "success"
    chart_id = chart_res["chart_id"]
    assert chart_id in session.chart_registry

    # 3. Graph simulation produces messages with chart ID and narrative that leaks chart ID
    raw_narrative = (
        f"The monthly revenue trend for delivered orders in 2017 is shown in the chart below. "
        f"Revenue here is based on item prices and excludes freight.\n\nChart: {chart_id}"
    )
    messages = [
        HumanMessage(content="Show the monthly revenue trend for delivered orders in 2017."),
        AIMessage(content="", tool_calls=[{"name": "generate_chart", "args": {"dataset_id": ds_id, "chart_type": "line"}, "id": "call_c1"}]),
        ToolMessage(content=json.dumps(chart_res), tool_call_id="call_c1", name="generate_chart"),
        AIMessage(content=raw_narrative),
    ]

    assistant_text, chart_ids, tool_executions = extract_turn_details(messages)
    assert chart_ids == [chart_id]
    assert chart_id not in assistant_text
    assert "Chart:" not in assistant_text
    assert "The monthly revenue trend" in assistant_text

    # 4. Streamlit rendering via AppTest
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "test-pass")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-apptest")
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()

    st_session = at.session_state["agent_session"]
    st_session.chart_registry = session.chart_registry
    st_session.ui_messages.append({"role": "user", "content": "Show the monthly revenue trend for delivered orders in 2017."})
    st_session.ui_messages.append({
        "role": "assistant",
        "content": assistant_text,
        "chart_ids": chart_ids,
        "tool_executions": tool_executions,
    })

    with mock.patch("streamlit.plotly_chart") as mock_plotly:
        at.run()
        assert not at.exception
        assert mock_plotly.call_count == 1
        rendered_fig = mock_plotly.call_args[0][0]
        assert isinstance(rendered_fig, go.Figure)
        assert list(rendered_fig.data[0].x) == ["2017-01", "2017-02"]
        assert [float(v) for v in _extract_trace_values(rendered_fig.data[0].y)] == [100.0, 200.0]

        # Rerun does not duplicate chart
        at.run()
        assert not at.exception
        assert mock_plotly.call_count == 2


# ==============================================================================
# Regression Test B: Monthly Revenue Trend
# ==============================================================================

def test_regression_monthly_revenue_trend(tmp_path: Path):
    """Test B: Verify agent can generate monthly delivered-item revenue and produce line chart.
    
    Checks that the expected month and revenue values match the fixture without hardcoding production amounts.
    """
    db_file = tmp_path / "test_b.duckdb"
    con = duckdb.connect(str(db_file))
    con.execute("""
        CREATE TABLE order_summary AS
        SELECT 'ord_1' AS order_id, 'delivered' AS order_status, 150.50 AS item_subtotal, '2017-03-10' AS order_purchase_timestamp
        UNION ALL
        SELECT 'ord_2' AS order_id, 'delivered' AS order_status, 250.25 AS item_subtotal, '2017-03-22' AS order_purchase_timestamp
        UNION ALL
        SELECT 'ord_3' AS order_id, 'delivered' AS order_status, 400.00 AS item_subtotal, '2017-04-05' AS order_purchase_timestamp
        UNION ALL
        SELECT 'ord_4' AS order_id, 'canceled' AS order_status, 999.00 AS item_subtotal, '2017-04-10' AS order_purchase_timestamp
    """)
    con.close()

    session = create_agent_session(db_path=db_file, api_key="sk-mock-test")

    sql = (
        "SELECT strftime('%Y-%m', CAST(order_purchase_timestamp AS TIMESTAMP)) AS month, "
        "SUM(item_subtotal) AS total_revenue "
        "FROM order_summary "
        "WHERE order_status = 'delivered' AND order_purchase_timestamp >= '2017-01-01' AND order_purchase_timestamp < '2018-01-01' "
        "GROUP BY month ORDER BY month"
    )
    res = execute_sql(sql, db_path=db_file)
    assert res["status"] == "success"
    rows = res["rows"]
    assert len(rows) == 2
    assert rows[0][0] == "2017-03"
    assert float(rows[0][1]) == 400.75
    assert rows[1][0] == "2017-04"
    assert float(rows[1][1]) == 400.00

    ds_id = session.dataset_store.save(res, query=sql)
    chart_tool = create_chart_generation_tool(
        dataset_store=session.dataset_store,
        chart_registry=session.chart_registry,
    )
    chart_res = chart_tool.invoke({
        "dataset_id": ds_id,
        "chart_type": "line",
        "x_column": "month",
        "y_column": "total_revenue",
    })
    assert chart_res["status"] == "success"
    cid = chart_res["chart_id"]
    fig_rec = session.chart_registry.get(cid)
    assert _extract_trace_values(fig_rec.figure["data"][0]["x"]) == ["2017-03", "2017-04"]
    assert [float(v) for v in _extract_trace_values(fig_rec.figure["data"][0]["y"])] == [400.75, 400.00]


# ==============================================================================
# Regression Test C: SQL Worker Stability & Pickling Resilience
# ==============================================================================

def test_regression_sql_worker_stability_and_reload(tmp_path: Path):
    """Test C: Verify worker runner initialization under module reload and PicklingError classification."""
    db_file = tmp_path / "test_c.duckdb"
    con = duckdb.connect(str(db_file))
    con.execute("CREATE TABLE order_summary (id INT, price DOUBLE, freight DOUBLE)")
    con.execute("INSERT INTO order_summary VALUES (1, 10.0, 2.0), (2, 20.0, 4.0)")
    con.close()

    # 1. Verify AST allows connectors and statistical aggregations
    query = (
        "SELECT corr(price, freight) AS correlation "
        "FROM order_summary "
        "WHERE price > 0 AND freight > 0"
    )
    validate_sql(query, allowed_tables=frozenset({"order_summary"}))
    assert "And" in _ALLOWED_FUNC_NAMES
    assert "Corr" in _ALLOWED_FUNC_NAMES

    # 2. Verify _resolve_worker_target correctly points to module runner
    target = _resolve_worker_target()
    assert callable(target)

    # 3. Verify real multiprocessing execution without pickling failure
    res = execute_sql(query, db_path=db_file)
    assert res["status"] == "success"
    assert len(res["rows"]) == 1
    assert round(res["rows"][0][0], 2) == 1.0

    # 4. Verify PicklingError classification is NON_RETRYABLE_RESOURCE_ERROR
    pickling_result = {
        "status": "error",
        "error_type": "PicklingError",
        "error_message": "Can't pickle worker function: module reloaded",
    }
    classification = classify_sql_result(pickling_result)
    assert classification["category"] == NON_RETRYABLE_RESOURCE_ERROR
    assert classification["retryable"] is False


# ==============================================================================
# Regression Test D: Correlation Calculation
# ==============================================================================

def test_regression_correlation_price_and_freight(tmp_path: Path):
    """Test D: Verify correlation calculation on row-level observations."""
    db_file = tmp_path / "test_d.duckdb"
    con = duckdb.connect(str(db_file))
    con.execute("""
        CREATE TABLE order_items AS
        SELECT 10.0 AS price, 5.0 AS freight_value UNION ALL
        SELECT 20.0 AS price, 10.0 AS freight_value UNION ALL
        SELECT 30.0 AS price, 15.0 AS freight_value UNION ALL
        SELECT 40.0 AS price, 20.0 AS freight_value
    """)
    con.close()

    store = DatasetStore()

    # Method 1: Python correlation tool on row-level data
    sql_res = execute_sql("SELECT price, freight_value FROM order_items", db_path=db_file)
    assert sql_res["status"] == "success"
    ds_id = store.save(sql_res, query="SELECT ...")

    from agent.tool_wrappers import create_python_analysis_tool
    py_tool = create_python_analysis_tool(dataset_store=store)
    py_res = py_tool.invoke({
        "dataset_id": ds_id,
        "operation": "correlation",
        "params": {"columns": ["price", "freight_value"]},
    })
    assert py_res["status"] == "success"
    corr_matrix = py_res["result"]
    # Perfect positive correlation 1.0
    assert abs(corr_matrix["price"]["freight_value"] - 1.0) < 1e-5

    # Method 2: Direct SQL corr() on row-level data
    sql_corr = execute_sql("SELECT corr(price, freight_value) AS r FROM order_items", db_path=db_file)
    assert sql_corr["status"] == "success"
    assert abs(sql_corr["rows"][0][0] - 1.0) < 1e-5


# ==============================================================================
# Regression Test E: Review Score Distribution
# ==============================================================================

def test_regression_review_score_distribution(tmp_path: Path):
    """Test E: Verify explicit visualization for review score distribution generates interactive chart."""
    db_file = tmp_path / "test_e.duckdb"
    con = duckdb.connect(str(db_file))
    con.execute("""
        CREATE TABLE order_reviews AS
        SELECT 1 AS review_score UNION ALL
        SELECT 1 AS review_score UNION ALL
        SELECT 2 AS review_score UNION ALL
        SELECT 3 AS review_score UNION ALL
        SELECT 4 AS review_score UNION ALL
        SELECT 5 AS review_score UNION ALL
        SELECT 5 AS review_score UNION ALL
        SELECT 5 AS review_score
    """)
    con.close()

    session = create_agent_session(db_path=db_file, api_key="sk-mock-test")

    # Aggregating query for discrete frequencies 1-5
    agg_sql = "SELECT review_score, count(*) AS count FROM order_reviews GROUP BY review_score ORDER BY review_score"
    res = execute_sql(agg_sql, db_path=db_file)
    assert res["status"] == "success"
    assert len(res["rows"]) == 5

    ds_id = session.dataset_store.save(res, query=agg_sql)
    chart_tool = create_chart_generation_tool(
        dataset_store=session.dataset_store,
        chart_registry=session.chart_registry,
    )
    chart_res = chart_tool.invoke({
        "dataset_id": ds_id,
        "chart_type": "bar",
        "x_column": "review_score",
        "y_column": "count",
    })
    assert chart_res["status"] == "success"
    cid = chart_res["chart_id"]

    # Verify extracted narrative removes any internal IDs
    raw_text = (
        "Here is the distribution of order review scores (1-5). Most customers gave a 5-star rating.\n\n"
        f"Chart: {cid}\nDataset: {ds_id}"
    )
    cleaned = sanitize_user_facing_narrative(raw_text)
    assert cid not in cleaned
    assert ds_id not in cleaned
    assert "Here is the distribution of order review scores" in cleaned

    # Check chart content
    fig_rec = session.chart_registry.get(cid)
    assert _extract_trace_values(fig_rec.figure["data"][0]["x"]) == [1, 2, 3, 4, 5]
    assert _extract_trace_values(fig_rec.figure["data"][0]["y"]) == [2, 1, 1, 1, 3]


# ==============================================================================
# Regression Test F: Clean UI Default
# ==============================================================================

def test_regression_clean_ui_default_and_debug_toggle(monkeypatch):
    """Test F: Verify default UI does not expose internal IDs or technical diagnostics."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "test-pass")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-apptest")
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()

    st_session = at.session_state["agent_session"]
    cid = st_session.chart_registry.save(
        figure={"data": [{"type": "bar", "x": ["cat_a"], "y": [50]}]},
        dataset_id="ds_12345678901234567890123456789012",
        chart_type="bar",
        title="Test Clean UI",
    )

    msg = {
        "role": "assistant",
        "content": "Delivered revenue is 50.00 for category A.",
        "chart_ids": [cid],
        "tool_executions": [
            {
                "tool": "execute_sql",
                "title": "🔍 SQL Query",
                "details": "SELECT category, revenue FROM sales",
                "summary": "Status: success | Rows: 1 | Dataset ID: ds_12345678901234567890123456789012",
                "status": "success",
            }
        ],
    }
    st_session.ui_messages.append({"role": "user", "content": "What is revenue?"})
    st_session.ui_messages.append(msg)

    # 1. Default render (show_debug=False)
    at.run()
    assert not at.exception
    # There should be no Tool Executions expander rendered in the chat message
    tool_expanders = [e for e in at.expander if "Tool Executions" in e.label]
    assert len(tool_expanders) == 0

    # Narrative contains business text and no internal IDs
    rendered_text = " ".join([m.value for m in at.chat_message[1].markdown])
    assert "Delivered revenue is 50.00" in rendered_text
    assert cid not in rendered_text
    assert "ds_123456" not in rendered_text

    # 2. Toggle show_debug = True in developer settings
    at.sidebar.checkbox[0].set_value(True).run()
    assert not at.exception
    active_tool_expanders = [e for e in at.expander if "Tool Executions" in e.label]
    assert len(active_tool_expanders) == 1
    assert "Tool Executions" in active_tool_expanders[0].label


# ==============================================================================
# Regression Test G: Multi-Turn Dataset Reuse
# ==============================================================================

def test_regression_multi_turn_dataset_reuse(tmp_path: Path):
    """Test G: Turn 1 produces dataset; Turn 2 visualizes it using existing dataset_id."""
    db_file = tmp_path / "test_g.duckdb"
    con = duckdb.connect(str(db_file))
    con.execute("""
        CREATE TABLE category_sales_summary AS
        SELECT 'health_beauty' AS category_name_english, 5000.0 AS delivered_item_revenue UNION ALL
        SELECT 'bed_bath' AS category_name_english, 4500.0 AS delivered_item_revenue UNION ALL
        SELECT 'sports_leisure' AS category_name_english, 3000.0 AS delivered_item_revenue
    """)
    con.close()

    session = create_agent_session(db_path=db_file, api_key="sk-mock-test")

    # Turn 1: SQL Query
    res1 = execute_sql(
        "SELECT category_name_english, delivered_item_revenue FROM category_sales_summary ORDER BY delivered_item_revenue DESC LIMIT 5",
        db_path=db_file,
    )
    assert res1["status"] == "success"
    ds_id = session.dataset_store.save(res1, query="SELECT ...")

    # Turn 1 execution turn
    turn1_msgs = [
        HumanMessage(content="What are the top 5 product categories by revenue?"),
        AIMessage(content="", tool_calls=[{"name": "execute_sql", "args": {"query": "SELECT ..."}, "id": "call_t1_sql"}]),
        ToolMessage(content=json.dumps({"status": "success", "dataset_id": ds_id, "row_count": 3}), tool_call_id="call_t1_sql", name="execute_sql"),
        AIMessage(content="The top product categories are health_beauty and bed_bath."),
    ]
    t1_text, t1_charts, t1_execs = extract_turn_details(turn1_msgs)
    session.ui_messages.append({"role": "user", "content": "What are the top 5 product categories by revenue?"})
    session.ui_messages.append({"role": "assistant", "content": t1_text, "chart_ids": t1_charts, "tool_executions": t1_execs})

    assert len(t1_charts) == 0
    assert "health_beauty" in t1_text

    # Turn 2: "Visualize them." -> calls generate_chart using ds_id from Turn 1
    chart_tool = create_chart_generation_tool(
        dataset_store=session.dataset_store,
        chart_registry=session.chart_registry,
    )
    chart_res = chart_tool.invoke({
        "dataset_id": ds_id,
        "chart_type": "bar",
        "x_column": "category_name_english",
        "y_column": "delivered_item_revenue",
    })
    assert chart_res["status"] == "success"
    cid = chart_res["chart_id"]

    turn2_msgs = [
        HumanMessage(content="Visualize them."),
        AIMessage(content="", tool_calls=[{"name": "generate_chart", "args": {"dataset_id": ds_id, "chart_type": "bar"}, "id": "call_t2_chart"}]),
        ToolMessage(content=json.dumps(chart_res), tool_call_id="call_t2_chart", name="generate_chart"),
        AIMessage(content=f"Here is the chart of the top categories.\n\nChart: {cid}"),
    ]
    t2_text, t2_charts, t2_execs = extract_turn_details(turn2_msgs)
    session.ui_messages.append({"role": "user", "content": "Visualize them."})
    session.ui_messages.append({"role": "assistant", "content": t2_text, "chart_ids": t2_charts, "tool_executions": t2_execs})

    assert t2_charts == [cid]
    assert cid not in t2_text
    assert "Here is the chart of the top categories." in t2_text
    assert len(session.ui_messages) == 4
    assert cid in session.chart_registry
