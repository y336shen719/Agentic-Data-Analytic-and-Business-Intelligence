"""Integration tests for LangChain Chart Tool & LangGraph Data Analyst workflow (Step 6.2).

Verifies:
A. Chart Tool Registration & Schemas
B. SQL -> Chart Workflow
C. SQL -> Python Analysis -> Derived Dataset -> Chart Workflow
D. Unsupported Analysis Handoff (detect_outliers)
E. Chart Artifact Registry Lifecycle, Retrieval & Session Isolation
F. Error Handling & Retry-State Isolation (SQL/Python budgets preserved)
G. Full LangGraph End-to-End Orchestration & Bound Execution
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import duckdb
import numpy as np
import plotly.graph_objects as go
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


def _extract_trace_values(val: Any) -> list:
    """Extract coordinate values from Plotly trace, supporting lists and binary bdata dicts."""
    if isinstance(val, dict) and "bdata" in val:
        raw_bytes = base64.b64decode(val["bdata"])
        dtype = np.dtype(val.get("dtype", "f8"))
        return np.frombuffer(raw_bytes, dtype=dtype).tolist()
    return list(val)

from agent.graph import (
    CHART_TOOL_NAME,
    NON_RETRYABLE_CHART_ERROR,
    PYTHON_TOOL_NAME,
    SQL_TOOL_NAME,
    classify_chart_result,
    create_agent_graph,
    inspect_tool_results,
)
from agent.python_error_classifier import (
    CATEGORY_INVALID_COLUMN,
    CATEGORY_MISSING_DATASET,
    RECOVERABLE_PYTHON_CATEGORIES,
)
from agent.sql_error_classifier import RECOVERABLE_SQL_ERROR
from agent.state import create_initial_state
from agent.tool_wrappers import (
    ToolSequence,
    create_chart_generation_tool,
    create_chart_generation_tools,
    create_python_analysis_tool,
    create_python_analysis_tools,
    create_sql_tools,
)
from tools.chart_generation import (
    SUPPORTED_CHART_TYPES,
    ChartGenerationEngine,
    InvalidChartTypeError,
    InvalidColumnError,
)
from tools.chart_registry import (
    ChartNotFoundError,
    ChartRegistry,
    ChartRegistryCapacityError,
)
from tools.dataset_store import DatasetCapacityError, DatasetNotFoundError, DatasetStore
from tools.python_analysis import PythonAnalysisEngine


# ==============================================================================
# Test Fixtures and ScriptedLLM
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
        return "scripted-chart-llm"


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
    args: Dict[str, Any] = {"dataset_id": dataset_id, "chart_type": chart_type}
    if x_column is not None:
        args["x_column"] = x_column
    if y_column is not None:
        args["y_column"] = y_column
    if params is not None:
        args["params"] = params
    return {"name": CHART_TOOL_NAME, "args": args, "id": call_id}


def _make_test_db(tmp_path: Path) -> Path:
    """Create a temporary DuckDB database populated with analytics tables."""
    db_file = tmp_path / "chart_analytics.duckdb"
    with duckdb.connect(str(db_file)) as con:
        con.execute(
            """
            CREATE TABLE monthly_sales (
                period VARCHAR,
                category VARCHAR,
                revenue DOUBLE,
                units INT
            );
            """
        )
        con.execute(
            """
            INSERT INTO monthly_sales VALUES
                ('2023-01', 'Electronics', 1000.0, 10),
                ('2023-02', 'Electronics', 1200.0, 12),
                ('2023-03', 'Electronics', 1500.0, 14),
                ('2023-01', 'Furniture', 500.0, 5),
                ('2023-02', 'Furniture', 600.0, 6),
                ('2023-03', 'Furniture', 650.0, 7);
            """
        )
    return db_file


# ==============================================================================
# A. Chart Tool Registration Tests
# ==============================================================================


def test_chart_tool_registration_schema():
    """Verify Chart Tool exposed schema, name, and allowed chart types."""
    store = DatasetStore()
    registry = ChartRegistry()
    chart_tool = create_chart_generation_tool(dataset_store=store, chart_registry=registry)

    assert chart_tool.name == CHART_TOOL_NAME
    assert "Plotly" in chart_tool.description
    assert chart_tool.args_schema is not None

    schema = chart_tool.args_schema.model_json_schema()
    props = schema.get("properties", {})
    assert "dataset_id" in props
    assert "chart_type" in props
    assert "x_column" in props
    assert "y_column" in props
    assert "params" in props

    # Allowed chart types defined in engine
    assert set(SUPPORTED_CHART_TYPES) == {"bar", "line", "scatter", "histogram", "box", "heatmap"}

    # Fix A: Ensure accurate parameter names and no misleading aliases
    params_desc = props["params"]["description"]
    assert "sort ('asc'|'desc'" in params_desc or "sort" in params_desc
    assert "sort_order" not in params_desc
    assert "color_column" in params_desc
    assert "color " not in params_desc and "color," not in params_desc
    assert "nbins" in params_desc
    assert "bins" not in params_desc.replace("nbins", "")

    # Docstring contract assertions
    doc = chart_tool.description
    assert "Requires x_column and y_column" in doc  # Bar chart
    assert "2D density" not in doc  # Heatmap does not claim generic 2D density


def test_chart_tool_sequence_concatenation(tmp_path):
    """Verify ToolSequence concatenation across SQL, Python, and Chart tool factories."""
    db_path = _make_test_db(tmp_path)
    store = DatasetStore()
    registry = ChartRegistry()

    sql_tools = create_sql_tools(db_path=db_path, dataset_store=store)
    py_tools = create_python_analysis_tools(dataset_store=store)
    chart_tools = create_chart_generation_tools(dataset_store=store, chart_registry=registry)

    all_tools = sql_tools + py_tools + chart_tools
    tool_names = [t.name for t in all_tools]
    assert tool_names == [
        "get_database_schema",
        "execute_sql",
        "execute_python_analysis",
        "generate_chart",
    ]


# ==============================================================================
# B. SQL -> Chart Workflow
# ==============================================================================


def test_sql_to_chart_execution(tmp_path):
    """Verify SQL execution -> dataset registration -> direct Line Chart generation."""
    db_path = _make_test_db(tmp_path)
    store = DatasetStore()
    registry = ChartRegistry()

    sql_tools = create_sql_tools(db_path=db_path, dataset_store=store)
    py_tools = create_python_analysis_tools(dataset_store=store)
    chart_tools = create_chart_generation_tools(dataset_store=store, chart_registry=registry)
    all_tools = sql_tools + py_tools + chart_tools

    # 1. SQL Query
    sql_tool = next(t for t in all_tools if t.name == SQL_TOOL_NAME)
    sql_res = sql_tool.invoke({"query": "SELECT period, SUM(revenue) AS total_revenue FROM monthly_sales GROUP BY period ORDER BY period"})
    assert sql_res["status"] == "success"
    dataset_id = sql_res["dataset_id"]

    # 2. Chart Generation
    chart_tool = next(t for t in all_tools if t.name == CHART_TOOL_NAME)
    chart_res = chart_tool.invoke({
        "dataset_id": dataset_id,
        "chart_type": "line",
        "x_column": "period",
        "y_column": "total_revenue",
        "params": {"title": "Revenue Trend"},
    })

    assert chart_res["status"] == "success"
    assert "chart_id" in chart_res
    chart_id = chart_res["chart_id"]
    assert chart_id.startswith("chart_")
    assert chart_res["dataset_id"] == dataset_id
    assert chart_res["chart_type"] == "line"
    assert chart_res["title"] == "Revenue Trend"
    # Ensure full figure JSON is NOT in the lightweight result
    assert "figure" not in chart_res

    # 3. Retrieve full figure from registry
    record = registry.get(chart_id)
    assert record.chart_id == chart_id
    assert record.dataset_id == dataset_id
    assert "data" in record.figure
    assert "layout" in record.figure
    assert record.metadata.get("rows_used") == 3


def test_sql_to_chart_langgraph_orchestration(tmp_path):
    """Verify LangGraph single-agent orchestration for SQL -> Chart with lightweight ToolMessage."""
    db_path = _make_test_db(tmp_path)
    store = DatasetStore()
    registry = ChartRegistry()

    sql_tools = create_sql_tools(db_path=db_path, dataset_store=store)
    py_tools = create_python_analysis_tools(dataset_store=store)
    chart_tools = create_chart_generation_tools(dataset_store=store, chart_registry=registry)
    all_tools = sql_tools + py_tools + chart_tools

    # Step 1: LLM executes SQL
    # Step 2: LLM inspects ToolMessage with dataset_id and invokes generate_chart
    # Step 3: LLM receives lightweight chart reference and generates final answer
    class DynamicScriptedLLM(BaseChatModel):
        script_step: int = 0

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if self.script_step == 0:
                self.script_step += 1
                return ChatResult(generations=[ChatGeneration(message=ai(
                    sql_call("SELECT category, SUM(revenue) as rev FROM monthly_sales GROUP BY category", "call_1")
                ))])
            elif self.script_step == 1:
                self.script_step += 1
                # Find the dataset_id from previous tool output
                last_tool = messages[-1]
                content = json.loads(last_tool.content)
                ds_id = content["dataset_id"]
                return ChatResult(generations=[ChatGeneration(message=ai(
                    chart_call(ds_id, "bar", x_column="category", y_column="rev", params={"title": "Category Sales"}, call_id="call_2")
                ))])
            else:
                last_tool = messages[-1]
                content = json.loads(last_tool.content)
                cid = content["chart_id"]
                return ChatResult(generations=[ChatGeneration(message=AIMessage(
                    content=f"Generated chart {cid} displaying category sales."
                ))])

        def bind_tools(self, tools, **kwargs):
            return self

        @property
        def _llm_type(self):
            return "dynamic-scripted"

    graph = create_agent_graph(llm=DynamicScriptedLLM(), tools=all_tools)
    initial_state = create_initial_state("Chart total sales by category.")
    final_state = graph.invoke(initial_state)

    messages = final_state["messages"]
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 2

    # Verify chart ToolMessage is compact
    chart_tool_msg = tool_msgs[1]
    chart_tool_data = json.loads(chart_tool_msg.content)
    assert chart_tool_data["status"] == "success"
    assert "chart_id" in chart_tool_data
    assert "figure" not in chart_tool_data

    # Verify registry has the full chart
    record = registry.get(chart_tool_data["chart_id"])
    assert record.chart_type == "bar"
    assert record.title == "Category Sales"
    assert "category sales" in final_state["messages"][-1].content.lower()

    # Verify reconstructed Plotly figure preserves aggregated values
    fig = go.Figure(record.figure)
    assert len(fig.data) == 1
    trace = fig.data[0]
    cats = _extract_trace_values(trace.x)
    vals = _extract_trace_values(trace.y)
    data_map = dict(zip(cats, vals))
    assert data_map["Electronics"] == pytest.approx(3700.0)
    assert data_map["Furniture"] == pytest.approx(1750.0)


# ==============================================================================
# C. SQL -> Python -> Chart Workflow (Genuine Analytical Results)
# ==============================================================================


def test_sql_to_python_growth_rate_to_chart(tmp_path):
    """Verify SQL -> growth_rate -> derived_dataset -> Line Chart preserves numerical values."""
    db_path = _make_test_db(tmp_path)
    store = DatasetStore()
    registry = ChartRegistry()

    sql_tools = create_sql_tools(db_path=db_path, dataset_store=store)
    py_tools = create_python_analysis_tools(dataset_store=store)
    chart_tools = create_chart_generation_tools(dataset_store=store, chart_registry=registry)
    all_tools = sql_tools + py_tools + chart_tools

    sql_tool = next(t for t in all_tools if t.name == SQL_TOOL_NAME)
    py_tool = next(t for t in all_tools if t.name == PYTHON_TOOL_NAME)
    chart_tool = next(t for t in all_tools if t.name == CHART_TOOL_NAME)

    # 1. SQL
    sql_res = sql_tool.invoke({
        "query": "SELECT period, SUM(revenue) as revenue FROM monthly_sales GROUP BY period ORDER BY period"
    })
    src_dataset_id = sql_res["dataset_id"]
    initial_source_record = store.get(src_dataset_id)
    initial_source_rows = list(initial_source_record.rows)

    # 2. Python Growth Rate
    py_res = py_tool.invoke({
        "dataset_id": src_dataset_id,
        "operation": "growth_rate",
        "params": {"period_column": "period", "value_column": "revenue"},
    })
    assert py_res["status"] == "success"
    assert py_res["derived_dataset_status"] == "registered"
    derived_id = py_res["derived_dataset_id"]
    assert derived_id is not None
    assert derived_id.startswith("ds_")
    assert derived_id != src_dataset_id

    # Verify source dataset was not mutated
    after_source_record = store.get(src_dataset_id)
    assert after_source_record.rows == initial_source_rows

    # Verify derived dataset content
    derived_record = store.get(derived_id)
    assert derived_record.columns == ["period", "value", "growth_rate"]
    # 2023-01: 1500; 2023-02: 1800 (growth: +0.20); 2023-03: 2150 (growth: ~0.1944)
    assert len(derived_record.rows) == 3
    assert derived_record.rows[0][2] is None
    assert round(derived_record.rows[1][2], 4) == 0.2000

    # 3. Chart Generation on Derived Dataset
    chart_res = chart_tool.invoke({
        "dataset_id": derived_id,
        "chart_type": "line",
        "x_column": "period",
        "y_column": "growth_rate",
        "params": {"title": "Revenue Growth Rate"},
    })
    assert chart_res["status"] == "success"
    chart_id = chart_res["chart_id"]

    record = registry.get(chart_id)
    assert record.chart_type == "line"
    assert record.title == "Revenue Growth Rate"


def test_sql_to_python_groupby_to_chart(tmp_path):
    """Verify SQL -> groupby -> derived_dataset -> Bar Chart."""
    db_path = _make_test_db(tmp_path)
    store = DatasetStore()
    registry = ChartRegistry()

    sql_tools = create_sql_tools(db_path=db_path, dataset_store=store)
    py_tools = create_python_analysis_tools(dataset_store=store)
    chart_tools = create_chart_generation_tools(dataset_store=store, chart_registry=registry)
    all_tools = sql_tools + py_tools + chart_tools

    sql_tool = next(t for t in all_tools if t.name == SQL_TOOL_NAME)
    py_tool = next(t for t in all_tools if t.name == PYTHON_TOOL_NAME)
    chart_tool = next(t for t in all_tools if t.name == CHART_TOOL_NAME)

    # 1. SQL: load raw monthly sales
    sql_res = sql_tool.invoke({"query": "SELECT category, revenue FROM monthly_sales"})
    src_id = sql_res["dataset_id"]

    # 2. Python Groupby
    py_res = py_tool.invoke({
        "dataset_id": src_id,
        "operation": "groupby",
        "params": {"by": "category", "agg_column": "revenue", "agg_func": "sum"},
    })
    assert py_res["status"] == "success"
    assert py_res["derived_dataset_status"] == "registered"
    derived_id = py_res["derived_dataset_id"]

    derived_record = store.get(derived_id)
    assert "category" in derived_record.columns
    assert "revenue_sum" in derived_record.columns
    assert len(derived_record.rows) == 2

    # 3. Chart Generation
    chart_res = chart_tool.invoke({
        "dataset_id": derived_id,
        "chart_type": "bar",
        "x_column": "category",
        "y_column": "revenue_sum",
        "params": {"title": "Category Revenue"},
    })
    assert chart_res["status"] == "success"
    record = registry.get(chart_res["chart_id"])
    assert record.chart_type == "bar"
    assert record.metadata["rows_used"] == 2


def test_sql_to_python_correlation_to_precomputed_heatmap(tmp_path):
    """Verify SQL -> correlation -> derived_dataset -> precomputed Heatmap."""
    db_path = _make_test_db(tmp_path)
    store = DatasetStore()
    registry = ChartRegistry()

    sql_tools = create_sql_tools(db_path=db_path, dataset_store=store)
    py_tools = create_python_analysis_tools(dataset_store=store)
    chart_tools = create_chart_generation_tools(dataset_store=store, chart_registry=registry)
    all_tools = sql_tools + py_tools + chart_tools

    sql_tool = next(t for t in all_tools if t.name == SQL_TOOL_NAME)
    py_tool = next(t for t in all_tools if t.name == PYTHON_TOOL_NAME)
    chart_tool = next(t for t in all_tools if t.name == CHART_TOOL_NAME)

    # 1. SQL
    sql_res = sql_tool.invoke({"query": "SELECT revenue, units FROM monthly_sales"})
    src_id = sql_res["dataset_id"]

    # 2. Python Correlation
    py_res = py_tool.invoke({
        "dataset_id": src_id,
        "operation": "correlation",
        "params": {"columns": ["revenue", "units"]},
    })
    assert py_res["status"] == "success"
    assert py_res["derived_dataset_status"] == "registered"
    derived_id = py_res["derived_dataset_id"]

    derived_record = store.get(derived_id)
    assert derived_record.columns == ["column_name", "revenue", "units"]

    # 3. Precomputed Heatmap Chart
    chart_res = chart_tool.invoke({
        "dataset_id": derived_id,
        "chart_type": "heatmap",
        "params": {"is_precomputed": True, "title": "Correlation Matrix"},
    })
    assert chart_res["status"] == "success"
    record = registry.get(chart_res["chart_id"])
    assert record.chart_type == "heatmap"
    assert record.title == "Correlation Matrix"


def test_sql_to_python_growth_rate_to_chart_langgraph_orchestration(tmp_path):
    """Verify full LangGraph orchestration: SQL -> Python growth_rate -> Derived Dataset -> Chart."""
    db_path = _make_test_db(tmp_path)
    store = DatasetStore()
    registry = ChartRegistry()

    sql_tools = create_sql_tools(db_path=db_path, dataset_store=store)
    py_tools = create_python_analysis_tools(dataset_store=store)
    chart_tools = create_chart_generation_tools(dataset_store=store, chart_registry=registry)
    all_tools = sql_tools + py_tools + chart_tools

    class GrowthFlowLLM(BaseChatModel):
        step: int = 0

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if self.step == 0:
                self.step += 1
                return ChatResult(generations=[ChatGeneration(message=ai(
                    sql_call("SELECT period, SUM(revenue) as revenue FROM monthly_sales GROUP BY period ORDER BY period", "c1")
                ))])
            elif self.step == 1:
                self.step += 1
                ds_id = json.loads(messages[-1].content)["dataset_id"]
                return ChatResult(generations=[ChatGeneration(message=ai(
                    py_call(ds_id, "growth_rate", {"period_column": "period", "value_column": "revenue"}, call_id="c2")
                ))])
            elif self.step == 2:
                self.step += 1
                py_data = json.loads(messages[-1].content)
                derived_id = py_data["derived_dataset_id"]
                return ChatResult(generations=[ChatGeneration(message=ai(
                    chart_call(derived_id, "line", x_column="period", y_column="growth_rate", params={"title": "Revenue Growth"}, call_id="c3")
                ))])
            else:
                cid = json.loads(messages[-1].content)["chart_id"]
                return ChatResult(generations=[ChatGeneration(message=AIMessage(
                    content=f"Workflow complete. Created chart {cid}."
                ))])

        def bind_tools(self, tools, **kwargs):
            return self

        @property
        def _llm_type(self):
            return "growth-flow-llm"

    graph = create_agent_graph(llm=GrowthFlowLLM(), tools=all_tools)
    final_state = graph.invoke(create_initial_state("Analyze and chart growth."))

    assert final_state.get("last_error") is None
    assert final_state.get("sql_retry_count", 0) == 0
    assert final_state.get("python_retry_count", 0) == 0
    assert final_state.get("pending_python_recovery") is False

    tool_msgs = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 3

    # Verify chart ToolMessage has no bulky figure JSON
    chart_msg_data = json.loads(tool_msgs[2].content)
    assert chart_msg_data["status"] == "success"
    assert "figure" not in chart_msg_data
    chart_id = chart_msg_data["chart_id"]

    # Verify figure in registry
    record = registry.get(chart_id)
    assert record.chart_type == "line"
    assert record.title == "Revenue Growth"
    assert "data" in record.figure

    # B1: Reconstruct Plotly Figure and verify coordinates
    fig = go.Figure(record.figure)
    assert len(fig.data) == 1
    trace = fig.data[0]
    actual_x = _extract_trace_values(trace.x)
    actual_y = _extract_trace_values(trace.y)

    expected_x = ["2023-02", "2023-03"]
    expected_y = [0.20, 350.0 / 1800.0]

    assert actual_x == expected_x
    assert actual_y == pytest.approx(expected_y, abs=1e-4)

    # Initial undefined growth rate is excluded
    assert "2023-01" not in actual_x

    # Row counts in metadata
    assert record.metadata.get("total_rows") == 3
    assert record.metadata.get("rows_used") == 2
    assert record.metadata.get("excluded_rows") == 1


def test_sql_to_python_correlation_to_heatmap_langgraph_orchestration(tmp_path):
    """Verify full LangGraph orchestration: SQL -> Python correlation -> Derived Dataset -> Heatmap."""
    db_path = _make_test_db(tmp_path)
    store = DatasetStore()
    registry = ChartRegistry()

    sql_tools = create_sql_tools(db_path=db_path, dataset_store=store)
    py_tools = create_python_analysis_tools(dataset_store=store)
    chart_tools = create_chart_generation_tools(dataset_store=store, chart_registry=registry)
    all_tools = sql_tools + py_tools + chart_tools

    class CorrelationHeatmapLLM(BaseChatModel):
        step: int = 0

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if self.step == 0:
                self.step += 1
                return ChatResult(generations=[ChatGeneration(message=ai(
                    sql_call("SELECT revenue, units FROM monthly_sales", "c1")
                ))])
            elif self.step == 1:
                self.step += 1
                ds_id = json.loads(messages[-1].content)["dataset_id"]
                return ChatResult(generations=[ChatGeneration(message=ai(
                    py_call(ds_id, "correlation", {"columns": ["revenue", "units"]}, call_id="c2")
                ))])
            elif self.step == 2:
                self.step += 1
                py_data = json.loads(messages[-1].content)
                derived_id = py_data["derived_dataset_id"]
                return ChatResult(generations=[ChatGeneration(message=ai(
                    chart_call(derived_id, "heatmap", params={"is_precomputed": True, "title": "Correlation Matrix"}, call_id="c3")
                ))])
            else:
                cid = json.loads(messages[-1].content)["chart_id"]
                return ChatResult(generations=[ChatGeneration(message=AIMessage(
                    content=f"Workflow complete. Created heatmap {cid}."
                ))])

        def bind_tools(self, tools, **kwargs):
            return self

        @property
        def _llm_type(self):
            return "corr-heatmap-llm"

    graph = create_agent_graph(llm=CorrelationHeatmapLLM(), tools=all_tools)
    final_state = graph.invoke(create_initial_state("Compute correlation heatmap."))

    assert final_state.get("last_error") is None
    assert final_state.get("sql_retry_count", 0) == 0
    assert final_state.get("python_retry_count", 0) == 0
    assert final_state.get("pending_python_recovery") is False

    tool_msgs = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 3

    chart_msg_data = json.loads(tool_msgs[2].content)
    assert chart_msg_data["status"] == "success"
    assert "figure" not in chart_msg_data
    chart_id = chart_msg_data["chart_id"]

    record = registry.get(chart_id)
    assert record.chart_type == "heatmap"
    assert record.title == "Correlation Matrix"
    assert "data" in record.figure
    assert record.metadata.get("is_precomputed") is True

    # B2: Reconstruct Plotly Figure and verify matrix values and symmetry
    fig = go.Figure(record.figure)
    assert len(fig.data) == 1
    trace = fig.data[0]

    actual_x = _extract_trace_values(trace.x)
    actual_y = _extract_trace_values(trace.y)
    assert actual_x == ["revenue", "units"]
    assert actual_y == ["revenue", "units"]

    raw_z = trace.z
    if isinstance(raw_z, dict) and "bdata" in raw_z:
        matrix = np.array(_extract_trace_values(raw_z)).reshape(2, 2).tolist()
    else:
        matrix = [list(row) for row in raw_z]

    assert len(matrix) == 2
    assert len(matrix[0]) == 2
    assert len(matrix[1]) == 2

    # Diagonal correlations ~ 1.0
    assert matrix[0][0] == pytest.approx(1.0)
    assert matrix[1][1] == pytest.approx(1.0)

    # Independent Pearson calculation from fixture data
    revs = [1000.0, 1200.0, 1500.0, 500.0, 600.0, 650.0]
    units = [10.0, 12.0, 14.0, 5.0, 6.0, 7.0]
    expected_corr = float(np.corrcoef(revs, units)[0, 1])

    # Off-diagonal values and symmetry
    assert matrix[0][1] == pytest.approx(expected_corr, abs=0.01)
    assert matrix[1][0] == pytest.approx(expected_corr, abs=0.01)
    assert matrix[0][1] == pytest.approx(matrix[1][0])


# ==============================================================================
# D. Unsupported Analysis Handoff (detect_outliers)
# ==============================================================================


def test_detect_outliers_does_not_fabricate_derived_dataset(tmp_path):
    """Verify detect_outliers returns original analysis and does NOT fabricate a derived dataset."""
    db_path = _make_test_db(tmp_path)
    store = DatasetStore()
    py_tool = create_python_analysis_tool(dataset_store=store)
    sql_tool = create_sql_tools(db_path=db_path, dataset_store=store)[1]

    sql_res = sql_tool.invoke({"query": "SELECT revenue FROM monthly_sales"})
    dataset_id = sql_res["dataset_id"]

    py_res = py_tool.invoke({
        "dataset_id": dataset_id,
        "operation": "detect_outliers",
        "params": {"column": "revenue"},
    })

    assert py_res["status"] == "success"
    assert py_res["operation"] == "detect_outliers"
    assert "result" in py_res
    # Genuine outliers output structure preserved
    assert "lower_bound" in py_res["result"]
    assert "upper_bound" in py_res["result"]
    assert "outlier_count" in py_res["result"]

    # Explicit unsupported handoff status
    assert py_res["derived_dataset_status"] == "unsupported"
    assert py_res["derived_dataset_id"] is None
    assert py_res["derived_dataset_error"] == "This operation does not produce a complete chart-ready dataset."


# ==============================================================================
# E. Chart Artifact Registry Lifecycle & Isolation
# ==============================================================================


def test_chart_registry_lifecycle_and_retrieval():
    """Verify ChartRegistry save, get, bounds, and safe missing ID errors."""
    registry = ChartRegistry(max_charts=2, max_memory_bytes=100000)

    fig = {"data": [{"type": "scatter", "x": [1, 2], "y": [3, 4]}], "layout": {"title": "Test"}}
    cid1 = registry.save(fig, "ds_1", "scatter", "Scatter 1", {"rows_used": 2})
    assert cid1 in registry
    assert len(registry) == 1

    rec1 = registry.get(cid1)
    assert rec1.chart_id == cid1
    assert rec1.dataset_id == "ds_1"
    assert rec1.figure == fig

    # Querying invalid / missing chart_id
    with pytest.raises(ChartNotFoundError):
        registry.get("chart_nonexistent")

    with pytest.raises(ChartNotFoundError):
        registry.get("")

    # Enforce capacity limits
    cid2 = registry.save(fig, "ds_2", "scatter", "Scatter 2")
    assert len(registry) == 2

    with pytest.raises(ChartRegistryCapacityError):
        registry.save(fig, "ds_3", "scatter", "Scatter 3")


def test_chart_registry_session_isolation():
    """Verify independent ChartRegistry instances never leak charts across sessions."""
    reg_session_a = ChartRegistry()
    reg_session_b = ChartRegistry()

    fig = {"data": [], "layout": {}}
    cid = reg_session_a.save(fig, "ds_a", "bar", "Bar A")

    assert cid in reg_session_a
    assert cid not in reg_session_b

    with pytest.raises(ChartNotFoundError):
        reg_session_b.get(cid)


# ==============================================================================
# F. Error Handling & State Machine Isolation
# ==============================================================================


def test_chart_error_invalid_chart_type(tmp_path):
    """Verify invalid chart type produces structured error without exception."""
    db_path = _make_test_db(tmp_path)
    store = DatasetStore()
    chart_tool = create_chart_generation_tool(dataset_store=store)
    sql_tool = create_sql_tools(db_path=db_path, dataset_store=store)[1]

    sql_res = sql_tool.invoke({"query": "SELECT revenue FROM monthly_sales"})
    ds_id = sql_res["dataset_id"]

    res = chart_tool.invoke({"dataset_id": ds_id, "chart_type": "pie", "x_column": "revenue"})
    assert res["status"] == "error"
    assert res["error_type"] == "InvalidChartTypeError"
    assert "Unsupported chart type 'pie'" in res["error_message"]


def test_chart_error_missing_dataset(tmp_path):
    """Verify invalid dataset reference produces structured DatasetNotFoundError."""
    store = DatasetStore()
    chart_tool = create_chart_generation_tool(dataset_store=store)

    res = chart_tool.invoke({"dataset_id": "ds_missing", "chart_type": "bar", "x_column": "cat"})
    assert res["status"] == "error"
    assert res["error_type"] == "DatasetNotFoundError"


def test_chart_error_missing_column(tmp_path):
    """Verify non-existent column produces structured InvalidColumnError."""
    db_path = _make_test_db(tmp_path)
    store = DatasetStore()
    chart_tool = create_chart_generation_tool(dataset_store=store)
    sql_tool = create_sql_tools(db_path=db_path, dataset_store=store)[1]

    sql_res = sql_tool.invoke({"query": "SELECT revenue FROM monthly_sales"})
    ds_id = sql_res["dataset_id"]

    res = chart_tool.invoke({"dataset_id": ds_id, "chart_type": "histogram", "x_column": "non_existent"})
    assert res["status"] == "error"
    assert res["error_type"] == "InvalidColumnError"


def test_chart_error_does_not_consume_sql_or_python_retry_budgets(tmp_path):
    """Verify chart tool failures do not increment sql_retry_count or python_retry_count and stop via recovery_stop."""
    db_path = _make_test_db(tmp_path)
    store = DatasetStore()
    registry = ChartRegistry()

    sql_tools = create_sql_tools(db_path=db_path, dataset_store=store)
    py_tools = create_python_analysis_tools(dataset_store=store)
    chart_tools = create_chart_generation_tools(dataset_store=store, chart_registry=registry)
    all_tools = sql_tools + py_tools + chart_tools

    class DynamicErrorLLM(BaseChatModel):
        invocations: int = 0
        extracted_dataset_id: Optional[str] = None

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.invocations += 1
            if self.invocations == 1:
                return ChatResult(generations=[ChatGeneration(message=ai(
                    sql_call("SELECT revenue FROM monthly_sales", "c1")
                ))])
            elif self.invocations == 2:
                sql_msg = messages[-1]
                data = json.loads(sql_msg.content)
                self.extracted_dataset_id = data["dataset_id"]
                return ChatResult(generations=[ChatGeneration(message=ai(
                    chart_call(self.extracted_dataset_id, "invalid_type", x_column="revenue", call_id="c2")
                ))])
            raise AssertionError("LLM invoked more times than expected (should stop at recovery_stop).")

        def bind_tools(self, tools, **kwargs):
            return self

        @property
        def _llm_type(self):
            return "dynamic-error-llm"

    llm = DynamicErrorLLM()
    graph = create_agent_graph(llm=llm, tools=all_tools, max_sql_retries=2, max_python_retries=2)
    state = graph.invoke(create_initial_state("Run SQL and generate invalid chart."))

    assert llm.invocations == 2
    assert llm.extracted_dataset_id is not None
    assert llm.extracted_dataset_id.startswith("ds_")

    assert state.get("sql_retry_count", 0) == 0
    assert state.get("python_retry_count", 0) == 0
    assert state.get("last_error_category") == NON_RETRYABLE_CHART_ERROR
    assert "InvalidChartTypeError" in state.get("last_error", "")
    last_msg = state["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    assert "Execution stopped: chart generation failed and will not be automatically retried." in last_msg.content
    assert "InvalidChartTypeError" in last_msg.content


def test_chart_success_does_not_clear_pending_python_recovery():
    """Verify that a successful chart call does not clear pending_python_recovery or reset retries."""
    state = {
        "messages": [
            HumanMessage(content="Analyze and chart"),
            ai(chart_call("ds_sales", "line", x_column="period", y_column="val", call_id="c_chart")),
            ToolMessage(
                content=json.dumps({"status": "success", "chart_id": "chart_123", "chart_type": "line"}),
                name=CHART_TOOL_NAME,
                tool_call_id="c_chart",
            ),
        ],
        "step_count": 2,
        "last_error": "DatasetNotFoundError: ds_missing not found",
        "last_error_category": CATEGORY_MISSING_DATASET,
        "pending_python_recovery": True,
        "sql_retry_count": 0,
        "python_retry_count": 1,
    }

    updates = inspect_tool_results(state)

    assert updates.get("pending_python_recovery") is True
    assert updates.get("python_retry_count") == 1
    assert updates.get("sql_retry_count") == 0
    assert updates.get("last_error") == "DatasetNotFoundError: ds_missing not found"
    assert updates.get("last_error_category") == CATEGORY_MISSING_DATASET


def test_chart_success_does_not_clear_pending_sql_recovery():
    """Verify that a successful chart call does not clear pending SQL recovery."""
    state = {
        "messages": [
            HumanMessage(content="SQL and chart"),
            ai(chart_call("ds_sales", "line", x_column="period", y_column="val", call_id="c_chart")),
            ToolMessage(
                content=json.dumps({"status": "success", "chart_id": "chart_123", "chart_type": "line"}),
                name=CHART_TOOL_NAME,
                tool_call_id="c_chart",
            ),
        ],
        "step_count": 2,
        "last_error": "ParserError: syntax error at or near 'FORM'",
        "last_error_category": RECOVERABLE_SQL_ERROR,
        "pending_python_recovery": False,
        "sql_retry_count": 1,
        "python_retry_count": 0,
    }

    updates = inspect_tool_results(state)

    assert updates.get("last_error") == "ParserError: syntax error at or near 'FORM'"
    assert updates.get("last_error_category") == RECOVERABLE_SQL_ERROR
    assert updates.get("sql_retry_count") == 1
    assert updates.get("python_retry_count") == 0
    assert updates.get("pending_python_recovery") is False


def test_chart_failure_during_pending_python_recovery():
    """Verify chart failure takes precedence over pending Python recovery and routes to recovery_stop."""
    state = {
        "messages": [
            HumanMessage(content="Chart after failed python"),
            ai(chart_call("ds_missing", "bar", x_column="col", call_id="c_chart")),
            ToolMessage(
                content=json.dumps({
                    "status": "error",
                    "error_type": "DatasetNotFoundError",
                    "error_message": "Dataset 'ds_missing' not found.",
                }),
                name=CHART_TOOL_NAME,
                tool_call_id="c_chart",
            ),
        ],
        "step_count": 2,
        "last_error": "InvalidColumnError: col_old does not exist",
        "last_error_category": CATEGORY_INVALID_COLUMN,
        "pending_python_recovery": True,
        "sql_retry_count": 0,
        "python_retry_count": 1,
    }

    updates = inspect_tool_results(state)

    assert updates.get("last_error_category") == NON_RETRYABLE_CHART_ERROR
    assert "DatasetNotFoundError" in updates.get("last_error", "")
    assert updates.get("python_retry_count") == 1
    assert updates.get("sql_retry_count") == 0


def test_chart_failure_during_pending_sql_recovery():
    """Verify chart failure takes precedence over pending SQL recovery and routes to recovery_stop."""
    state = {
        "messages": [
            HumanMessage(content="Chart after failed sql"),
            ai(chart_call("ds_1", "pie", x_column="col", call_id="c_chart")),
            ToolMessage(
                content=json.dumps({
                    "status": "error",
                    "error_type": "InvalidChartTypeError",
                    "error_message": "Unsupported chart type 'pie'.",
                }),
                name=CHART_TOOL_NAME,
                tool_call_id="c_chart",
            ),
        ],
        "step_count": 2,
        "last_error": "CatalogError: Table not found",
        "last_error_category": RECOVERABLE_SQL_ERROR,
        "pending_python_recovery": False,
        "sql_retry_count": 1,
        "python_retry_count": 0,
    }

    updates = inspect_tool_results(state)

    assert updates.get("last_error_category") == NON_RETRYABLE_CHART_ERROR
    assert "InvalidChartTypeError" in updates.get("last_error", "")
    assert updates.get("sql_retry_count") == 1
    assert updates.get("python_retry_count") == 0


def test_chart_error_stops_immediately_via_recovery_stop(tmp_path):
    """Verify graph terminates cleanly via recovery_stop on chart error without unbounded tool loops."""
    store = DatasetStore()
    chart_tool = create_chart_generation_tool(dataset_store=store)

    # Even if LLM scripts multiple attempts, graph stops immediately after first chart failure
    script = [
        ai(chart_call("ds_missing", "bar", x_column="col", call_id="c_0")),
        ai(chart_call("ds_missing", "bar", x_column="col", call_id="c_1")),
    ]
    llm = ScriptedLLM(script=script)

    graph = create_agent_graph(llm=llm, tools=[chart_tool], max_steps=10)
    final_state = graph.invoke(create_initial_state("Chart test"))

    assert final_state["step_count"] == 1
    assert len(llm.invocations) == 1
    assert final_state["last_error_category"] == NON_RETRYABLE_CHART_ERROR
    last_msg = final_state["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    assert "Execution stopped: chart generation failed and will not be automatically retried." in last_msg.content
    assert "DatasetNotFoundError" in last_msg.content


def test_chart_result_inspection_unit_scenarios():
    """Verify deterministic outcomes of classify_chart_result for all specified result types."""
    # 1. Valid success
    res_succ = classify_chart_result({"status": "success", "chart_id": "chart_abc123", "chart_type": "line"})
    assert res_succ["status"] == "success"
    assert res_succ["category"] is None
    assert res_succ["retryable"] is False
    assert res_succ["chart_id"] == "chart_abc123"

    # 2. Structured chart error
    res_err = classify_chart_result({
        "status": "error",
        "error_type": "InvalidColumnError",
        "error_message": "Column 'foo' does not exist.",
    })
    assert res_err["status"] == "error"
    assert res_err["category"] == NON_RETRYABLE_CHART_ERROR
    assert res_err["retryable"] is False
    assert res_err["error_type"] == "InvalidColumnError"
    assert "Column 'foo' does not exist." in res_err["error_message"]

    # 3. ChartRegistry capacity error
    res_cap = classify_chart_result({
        "status": "error",
        "error_type": "ChartRegistryCapacityError",
        "error_message": "ChartRegistry capacity reached.",
    })
    assert res_cap["status"] == "error"
    assert res_cap["category"] == NON_RETRYABLE_CHART_ERROR
    assert res_cap["retryable"] is False
    assert res_cap["error_type"] == "ChartRegistryCapacityError"

    # 4. Missing / None result
    res_none = classify_chart_result(None)
    assert res_none["status"] == "error"
    assert res_none["category"] == NON_RETRYABLE_CHART_ERROR
    assert res_none["retryable"] is False
    assert res_none["error_type"] == "MalformedChartResultError"

    # 5. Malformed result (not a dict)
    res_str = classify_chart_result("invalid string result")
    assert res_str["status"] == "error"
    assert res_str["category"] == NON_RETRYABLE_CHART_ERROR
    assert res_str["error_type"] == "MalformedChartResultError"

    # 6. ToolNode-level failure via inspect_tool_results (sanitized)
    state_tn_err = {
        "messages": [
            HumanMessage(content="chart"),
            ai(chart_call("ds_1", "line", x_column="x", call_id="c_err")),
            ToolMessage(
                content="Error: missing required argument 'y_column'",
                name=CHART_TOOL_NAME,
                tool_call_id="c_err",
                status="error",
            ),
        ],
        "step_count": 1,
    }
    updates_tn = inspect_tool_results(state_tn_err)
    assert updates_tn.get("last_error_category") == NON_RETRYABLE_CHART_ERROR
    assert "ChartExecutionError" in updates_tn.get("last_error", "")
    assert "Chart tool execution failed." in updates_tn.get("last_error", "")
    assert "missing required argument" not in updates_tn.get("last_error", "")

    # 7. Missing ToolMessage via inspect_tool_results
    state_missing_msg = {
        "messages": [
            HumanMessage(content="chart"),
            ai(chart_call("ds_1", "line", x_column="x", call_id="c_unreturned")),
        ],
        "step_count": 1,
    }
    updates_missing = inspect_tool_results(state_missing_msg)
    assert updates_missing.get("last_error_category") == NON_RETRYABLE_CHART_ERROR
    assert "MalformedChartResultError" in updates_missing.get("last_error", "")


@pytest.mark.parametrize("order", ["sql_first", "chart_first"])
def test_mixed_batch_sql_success_and_chart_success(order: str):
    """Verify SQL success + Chart success yields overall success regardless of order."""
    sql_tc = sql_call("SELECT 1", "c_sql")
    chart_tc = chart_call("ds_1", "bar", x_column="x", call_id="c_chart")
    sql_tm = ToolMessage(
        content=json.dumps({"status": "success", "columns": ["x"], "rows": [[1]], "row_count": 1, "dataset_id": "ds_1"}),
        name=SQL_TOOL_NAME,
        tool_call_id="c_sql",
    )
    chart_tm = ToolMessage(
        content=json.dumps({"status": "success", "chart_id": "chart_1", "chart_type": "bar"}),
        name=CHART_TOOL_NAME,
        tool_call_id="c_chart",
    )

    calls = [sql_tc, chart_tc] if order == "sql_first" else [chart_tc, sql_tc]
    messages = [HumanMessage(content="run"), ai(*calls), sql_tm, chart_tm]

    updates = inspect_tool_results({"messages": messages, "step_count": 1})
    assert updates.get("last_error") is None
    assert updates.get("last_error_category") is None


@pytest.mark.parametrize("order", ["py_first", "chart_first"])
def test_mixed_batch_python_success_and_chart_success(order: str):
    """Verify Python success + Chart success yields overall success regardless of order."""
    py_tc = py_call("ds_1", "describe", {}, call_id="c_py")
    chart_tc = chart_call("ds_1", "bar", x_column="x", call_id="c_chart")
    py_tm = ToolMessage(
        content=json.dumps({"status": "success", "operation": "describe", "result": []}),
        name=PYTHON_TOOL_NAME,
        tool_call_id="c_py",
    )
    chart_tm = ToolMessage(
        content=json.dumps({"status": "success", "chart_id": "chart_1", "chart_type": "bar"}),
        name=CHART_TOOL_NAME,
        tool_call_id="c_chart",
    )

    calls = [py_tc, chart_tc] if order == "py_first" else [chart_tc, py_tc]
    messages = [HumanMessage(content="run"), ai(*calls), py_tm, chart_tm]

    updates = inspect_tool_results({"messages": messages, "step_count": 1})
    assert updates.get("last_error") is None
    assert updates.get("last_error_category") is None


@pytest.mark.parametrize("order", ["sql_first", "chart_first"])
def test_mixed_batch_sql_success_and_chart_error(order: str):
    """Verify SQL success + Chart error yields Non-retryable Chart error."""
    sql_tc = sql_call("SELECT 1", "c_sql")
    chart_tc = chart_call("ds_1", "line", x_column="missing", call_id="c_chart")
    sql_tm = ToolMessage(
        content=json.dumps({"status": "success", "columns": ["x"], "rows": [[1]], "row_count": 1, "dataset_id": "ds_1"}),
        name=SQL_TOOL_NAME,
        tool_call_id="c_sql",
    )
    chart_tm = ToolMessage(
        content=json.dumps({"status": "error", "error_type": "InvalidColumnError", "error_message": "col missing"}),
        name=CHART_TOOL_NAME,
        tool_call_id="c_chart",
    )

    calls = [sql_tc, chart_tc] if order == "sql_first" else [chart_tc, sql_tc]
    messages = [HumanMessage(content="run"), ai(*calls), sql_tm, chart_tm]

    updates = inspect_tool_results({"messages": messages, "step_count": 1})
    assert updates.get("last_error_category") == NON_RETRYABLE_CHART_ERROR
    assert "InvalidColumnError" in updates.get("last_error", "")


@pytest.mark.parametrize("order", ["py_first", "chart_first"])
def test_mixed_batch_python_success_and_chart_error(order: str):
    """Verify Python success + Chart error yields Non-retryable Chart error."""
    py_tc = py_call("ds_1", "describe", {}, call_id="c_py")
    chart_tc = chart_call("ds_1", "line", x_column="missing", call_id="c_chart")
    py_tm = ToolMessage(
        content=json.dumps({"status": "success", "operation": "describe", "result": []}),
        name=PYTHON_TOOL_NAME,
        tool_call_id="c_py",
    )
    chart_tm = ToolMessage(
        content=json.dumps({"status": "error", "error_type": "InvalidColumnError", "error_message": "col missing"}),
        name=CHART_TOOL_NAME,
        tool_call_id="c_chart",
    )

    calls = [py_tc, chart_tc] if order == "py_first" else [chart_tc, py_tc]
    messages = [HumanMessage(content="run"), ai(*calls), py_tm, chart_tm]

    updates = inspect_tool_results({"messages": messages, "step_count": 1})
    assert updates.get("last_error_category") == NON_RETRYABLE_CHART_ERROR
    assert "InvalidColumnError" in updates.get("last_error", "")


@pytest.mark.parametrize("order", ["sql_first", "chart_first"])
def test_mixed_batch_sql_recoverable_error_and_chart_success(order: str):
    """Verify SQL recoverable error + Chart success preserves SQL recovery."""
    sql_tc = sql_call("SELECT FORM table", "c_sql")
    chart_tc = chart_call("ds_1", "bar", x_column="x", call_id="c_chart")
    sql_tm = ToolMessage(
        content=json.dumps({"status": "error", "error_type": "ParserException", "error_message": "syntax error"}),
        name=SQL_TOOL_NAME,
        tool_call_id="c_sql",
    )
    chart_tm = ToolMessage(
        content=json.dumps({"status": "success", "chart_id": "chart_1", "chart_type": "bar"}),
        name=CHART_TOOL_NAME,
        tool_call_id="c_chart",
    )

    calls = [sql_tc, chart_tc] if order == "sql_first" else [chart_tc, sql_tc]
    messages = [HumanMessage(content="run"), ai(*calls), sql_tm, chart_tm]

    updates = inspect_tool_results({"messages": messages, "step_count": 1})
    assert updates.get("last_error_category") == RECOVERABLE_SQL_ERROR
    assert "ParserException" in updates.get("last_error", "")


@pytest.mark.parametrize("order", ["py_first", "chart_first"])
def test_mixed_batch_python_recoverable_error_and_chart_success(order: str):
    """Verify Python recoverable error + Chart success preserves Python recovery."""
    py_tc = py_call("ds_1", "describe", {"columns": ["bad"]}, call_id="c_py")
    chart_tc = chart_call("ds_1", "bar", x_column="x", call_id="c_chart")
    py_tm = ToolMessage(
        content=json.dumps({"status": "error", "error_type": "InvalidColumnError", "error_message": "Column 'bad' missing"}),
        name=PYTHON_TOOL_NAME,
        tool_call_id="c_py",
    )
    chart_tm = ToolMessage(
        content=json.dumps({"status": "success", "chart_id": "chart_1", "chart_type": "bar"}),
        name=CHART_TOOL_NAME,
        tool_call_id="c_chart",
    )

    calls = [py_tc, chart_tc] if order == "py_first" else [chart_tc, py_tc]
    messages = [HumanMessage(content="run"), ai(*calls), py_tm, chart_tm]

    updates = inspect_tool_results({"messages": messages, "step_count": 1})
    assert updates.get("last_error_category") == CATEGORY_INVALID_COLUMN
    assert updates.get("pending_python_recovery") is True


@pytest.mark.parametrize("order", ["sql_first", "chart_first"])
def test_mixed_batch_sql_recoverable_error_and_chart_error(order: str):
    """Verify SQL recoverable error + Chart error gives precedence to Chart failure (non-retryable)."""
    sql_tc = sql_call("SELECT FORM table", "c_sql")
    chart_tc = chart_call("ds_1", "bar", x_column="bad", call_id="c_chart")
    sql_tm = ToolMessage(
        content=json.dumps({"status": "error", "error_type": "ParserException", "error_message": "syntax error"}),
        name=SQL_TOOL_NAME,
        tool_call_id="c_sql",
    )
    chart_tm = ToolMessage(
        content=json.dumps({"status": "error", "error_type": "InvalidColumnError", "error_message": "col bad"}),
        name=CHART_TOOL_NAME,
        tool_call_id="c_chart",
    )

    calls = [sql_tc, chart_tc] if order == "sql_first" else [chart_tc, sql_tc]
    messages = [HumanMessage(content="run"), ai(*calls), sql_tm, chart_tm]

    updates = inspect_tool_results({"messages": messages, "step_count": 1})
    assert updates.get("last_error_category") == NON_RETRYABLE_CHART_ERROR
    assert "InvalidColumnError" in updates.get("last_error", "")


@pytest.mark.parametrize("order", ["py_first", "chart_first"])
def test_mixed_batch_python_recoverable_error_and_chart_error(order: str):
    """Verify Python recoverable error + Chart error gives precedence to Chart failure (non-retryable)."""
    py_tc = py_call("ds_1", "describe", {"columns": ["bad"]}, call_id="c_py")
    chart_tc = chart_call("ds_1", "bar", x_column="bad", call_id="c_chart")
    py_tm = ToolMessage(
        content=json.dumps({"status": "error", "error_type": "InvalidColumnError", "error_message": "col bad py"}),
        name=PYTHON_TOOL_NAME,
        tool_call_id="c_py",
    )
    chart_tm = ToolMessage(
        content=json.dumps({"status": "error", "error_type": "InvalidChartTypeError", "error_message": "type bad"}),
        name=CHART_TOOL_NAME,
        tool_call_id="c_chart",
    )

    calls = [py_tc, chart_tc] if order == "py_first" else [chart_tc, py_tc]
    messages = [HumanMessage(content="run"), ai(*calls), py_tm, chart_tm]

    updates = inspect_tool_results({"messages": messages, "step_count": 1})
    assert updates.get("last_error_category") == NON_RETRYABLE_CHART_ERROR
    assert "InvalidChartTypeError" in updates.get("last_error", "")


@pytest.mark.parametrize("order", ["success_first", "error_first"])
def test_mixed_batch_chart_success_and_chart_error(order: str):
    """Verify Chart success + Chart error gives precedence to Chart failure."""
    c_succ = chart_call("ds_1", "bar", x_column="x", call_id="c_ok")
    c_err = chart_call("ds_1", "pie", x_column="x", call_id="c_err")
    tm_succ = ToolMessage(
        content=json.dumps({"status": "success", "chart_id": "chart_ok", "chart_type": "bar"}),
        name=CHART_TOOL_NAME,
        tool_call_id="c_ok",
    )
    tm_err = ToolMessage(
        content=json.dumps({"status": "error", "error_type": "InvalidChartTypeError", "error_message": "Unsupported"}),
        name=CHART_TOOL_NAME,
        tool_call_id="c_err",
    )

    calls = [c_succ, c_err] if order == "success_first" else [c_err, c_succ]
    messages = [HumanMessage(content="run"), ai(*calls), tm_succ, tm_err]

    updates = inspect_tool_results({"messages": messages, "step_count": 1})
    assert updates.get("last_error_category") == NON_RETRYABLE_CHART_ERROR
    assert "InvalidChartTypeError" in updates.get("last_error", "")


@pytest.mark.parametrize("order", ["malformed_first", "success_first"])
def test_mixed_batch_chart_malformed_and_sql_success(order: str):
    """Verify Chart malformed result + SQL success gives precedence to Non-retryable Chart error."""
    sql_tc = sql_call("SELECT 1", "c_sql")
    chart_tc = chart_call("ds_1", "bar", x_column="x", call_id="c_chart")
    sql_tm = ToolMessage(
        content=json.dumps({"status": "success", "columns": ["x"], "rows": [[1]], "row_count": 1, "dataset_id": "ds_1"}),
        name=SQL_TOOL_NAME,
        tool_call_id="c_sql",
    )
    chart_tm = ToolMessage(
        content="This is not valid JSON at all!",
        name=CHART_TOOL_NAME,
        tool_call_id="c_chart",
    )

    calls = [chart_tc, sql_tc] if order == "malformed_first" else [sql_tc, chart_tc]
    messages = [HumanMessage(content="run"), ai(*calls), sql_tm, chart_tm]

    updates = inspect_tool_results({"messages": messages, "step_count": 1})
    assert updates.get("last_error_category") == NON_RETRYABLE_CHART_ERROR
    assert "MalformedChartResultError" in updates.get("last_error", "")


# ==============================================================================
# G. Fix B: Engine & DatasetStore Identity Consistency Tests
# ==============================================================================


def test_python_engine_dataset_store_identity_validation():
    """Verify create_python_analysis_tool enforces identity when engine and store are supplied."""
    store_a = DatasetStore()
    store_b = DatasetStore()
    engine = PythonAnalysisEngine(dataset_store=store_a)

    # 1. Matching store succeeds
    tool_ok = create_python_analysis_tool(dataset_store=store_a, engine=engine)
    assert tool_ok is not None

    # 2. Mismatched store raises ValueError immediately
    with pytest.raises(ValueError, match="Mismatched DatasetStore instances"):
        create_python_analysis_tool(dataset_store=store_b, engine=engine)

    # 3. Only engine supplied succeeds
    tool_engine_only = create_python_analysis_tool(engine=engine)
    assert tool_engine_only is not None

    # 4. Only store supplied succeeds
    tool_store_only = create_python_analysis_tool(dataset_store=store_a)
    assert tool_store_only is not None


def test_chart_engine_dataset_store_identity_validation():
    """Verify create_chart_generation_tool enforces identity when engine and store are supplied."""
    store_a = DatasetStore()
    store_b = DatasetStore()
    engine = ChartGenerationEngine(dataset_store=store_a)

    # 1. Matching store succeeds
    tool_ok = create_chart_generation_tool(dataset_store=store_a, engine=engine)
    assert tool_ok is not None

    # 2. Mismatched store raises ValueError immediately
    with pytest.raises(ValueError, match="Mismatched DatasetStore instances"):
        create_chart_generation_tool(dataset_store=store_b, engine=engine)

    # 3. Only engine supplied succeeds
    tool_engine_only = create_chart_generation_tool(engine=engine)
    assert tool_engine_only is not None

    # 4. Only store supplied succeeds
    tool_store_only = create_chart_generation_tool(dataset_store=store_a)
    assert tool_store_only is not None


# ==============================================================================
# H. Fix D: Derived Dataset Registration Failure Reporting & Retry Independence
# ==============================================================================


def test_derived_dataset_failure_reporting_capacity_exceeded(tmp_path):
    """Verify safe failure reporting when DatasetStore capacity is exceeded."""
    db_path = _make_test_db(tmp_path)
    # Store that allows only 1 dataset (the SQL result)
    tiny_store = DatasetStore(max_datasets=1)
    sql_tool = create_sql_tools(db_path=db_path, dataset_store=tiny_store)[1]
    py_tool = create_python_analysis_tool(dataset_store=tiny_store)

    sql_res = sql_tool.invoke({"query": "SELECT period, SUM(revenue) as revenue FROM monthly_sales GROUP BY period ORDER BY period"})
    src_id = sql_res["dataset_id"]

    # Analysis succeeds, but storing derived dataset fails due to capacity
    py_res = py_tool.invoke({
        "dataset_id": src_id,
        "operation": "growth_rate",
        "params": {"period_column": "period", "value_column": "revenue"},
    })

    # Preserves overall success of analytical computation
    assert py_res["status"] == "success"
    assert len(py_res["result"]) == 3
    # Explicit safe failure status
    assert py_res["derived_dataset_id"] is None
    assert py_res["derived_dataset_status"] == "failed"
    assert "DatasetStore capacity was exceeded" in py_res["derived_dataset_error"]
    # No stack traces or secrets leaked
    assert "Traceback" not in py_res["derived_dataset_error"]


def test_derived_dataset_failure_does_not_increment_python_retry_counter(tmp_path):
    """Verify derived dataset registration failure does not increment python_retry_count."""
    db_path = _make_test_db(tmp_path)
    tiny_store = DatasetStore(max_datasets=1)
    sql_tool = create_sql_tools(db_path=db_path, dataset_store=tiny_store)[1]
    py_tool = create_python_analysis_tool(dataset_store=tiny_store)

    # 1. SQL
    sql_res = sql_tool.invoke({"query": "SELECT period, SUM(revenue) as revenue FROM monthly_sales GROUP BY period ORDER BY period"})
    src_id = sql_res["dataset_id"]

    # 2. Python tool returns success with derived_dataset_status='failed'
    state = {
        "messages": [
            HumanMessage(content="Calculate growth"),
            ai(py_call(src_id, "growth_rate", {"period_column": "period", "value_column": "revenue"}, call_id="c_py")),
            ToolMessage(
                content=json.dumps({
                    "status": "success",
                    "dataset_id": src_id,
                    "operation": "growth_rate",
                    "result": [{"period": "2023-01", "value": 1500.0, "growth_rate": None}],
                    "derived_dataset_id": None,
                    "derived_dataset_status": "failed",
                    "derived_dataset_error": "The analysis succeeded, but its derived dataset could not be registered.",
                }),
                name=PYTHON_TOOL_NAME,
                tool_call_id="c_py",
            ),
        ],
        "step_count": 1,
        "last_error": None,
        "last_error_category": None,
        "pending_python_recovery": False,
        "sql_retry_count": 0,
        "python_retry_count": 0,
    }

    updates = inspect_tool_results(state)
    assert updates.get("python_retry_count", 0) == 0
    assert updates.get("last_error") is None


# ==============================================================================
# I. Fix C: Shared Dependency Lifecycle & Cross-Session Isolation
# ==============================================================================


def test_cross_session_artifact_state_isolation(tmp_path):
    """Verify separate agent sessions do not share DatasetStore or ChartRegistry artifacts."""
    db_path = _make_test_db(tmp_path)

    # Session 1
    store_1 = DatasetStore()
    registry_1 = ChartRegistry()
    sql_1 = create_sql_tools(db_path=db_path, dataset_store=store_1)[1]
    chart_1 = create_chart_generation_tool(dataset_store=store_1, chart_registry=registry_1)

    res_1 = sql_1.invoke({"query": "SELECT period, revenue FROM monthly_sales LIMIT 2"})
    ds_id_1 = res_1["dataset_id"]
    cres_1 = chart_1.invoke({"dataset_id": ds_id_1, "chart_type": "line", "x_column": "period", "y_column": "revenue"})
    chart_id_1 = cres_1["chart_id"]

    # Session 2 (Independent)
    store_2 = DatasetStore()
    registry_2 = ChartRegistry()
    chart_2 = create_chart_generation_tool(dataset_store=store_2, chart_registry=registry_2)

    # Session 2 cannot access Session 1's dataset
    res_cross_ds = chart_2.invoke({"dataset_id": ds_id_1, "chart_type": "line", "x_column": "period", "y_column": "revenue"})
    assert res_cross_ds["status"] == "error"
    assert res_cross_ds["error_type"] == "DatasetNotFoundError"

    # Session 2 registry does not contain Session 1's chart
    assert chart_id_1 in registry_1
    assert chart_id_1 not in registry_2
    with pytest.raises(ChartNotFoundError):
        registry_2.get(chart_id_1)


# ==============================================================================
# J. Step 6.2 Targeted Fixes: Strict Success Validation & Error Sanitization
# ==============================================================================


def test_chart_success_validation_valid():
    """Verify valid opaque chart_id string results in successful classification."""
    res = classify_chart_result({"status": "success", "chart_id": "chart_valid_12345"})
    assert res["status"] == "success"
    assert res["category"] is None
    assert res["retryable"] is False
    assert res["chart_id"] == "chart_valid_12345"


@pytest.mark.parametrize(
    "invalid_payload",
    [
        {"status": "success"},
        {"status": "success", "chart_id": None},
        {"status": "success", "chart_id": ""},
        {"status": "success", "chart_id": "   "},
        {"status": "success", "chart_id": 123},
        {"status": "success", "chart_id": []},
    ],
)
def test_chart_success_validation_invalid_parametrized(invalid_payload):
    """Verify invalid chart_id formats are classified as MalformedChartResultError."""
    res = classify_chart_result(invalid_payload)
    assert res["status"] == "error"
    assert res["category"] == NON_RETRYABLE_CHART_ERROR
    assert res["retryable"] is False
    assert res["error_type"] == "MalformedChartResultError"
    assert res["error_message"] == "Chart tool reported success without a valid chart_id."


@pytest.mark.parametrize(
    "invalid_chart_id",
    [None, "", "   ", 123, []],
)
def test_chart_success_without_valid_chart_id_stops_execution(invalid_chart_id):
    """Verify that a chart tool returning success without valid chart_id routes to recovery_stop."""
    state = {
        "messages": [
            HumanMessage(content="Make chart"),
            ai(chart_call("ds_1", "line", x_column="x", call_id="c_bad_succ")),
            ToolMessage(
                content=json.dumps({"status": "success", "chart_id": invalid_chart_id}),
                name=CHART_TOOL_NAME,
                tool_call_id="c_bad_succ",
            ),
        ],
        "step_count": 1,
        "sql_retry_count": 0,
        "python_retry_count": 0,
    }

    updates = inspect_tool_results(state)
    assert updates.get("last_error_category") == NON_RETRYABLE_CHART_ERROR
    assert "MalformedChartResultError" in updates.get("last_error", "")
    assert "Chart tool reported success without a valid chart_id." in updates.get("last_error", "")
    assert updates.get("sql_retry_count", 0) == 0
    assert updates.get("python_retry_count", 0) == 0


def test_toolnode_error_sanitization_leak_prevention():
    """Verify ToolNode-level Chart errors do not leak internal paths, credentials or stack traces."""
    sensitive_raw_content = (
        "Traceback (most recent call last):\n"
        "  File '/Users/secret/app/db.py', line 42, in connect\n"
        "  AWS_SECRET_KEY='AKIAIOSFODNN7EXAMPLE'\n"
        "InternalDBException: Connection refused to 10.0.0.1:5432"
    )

    state = {
        "messages": [
            HumanMessage(content="Plot"),
            ai(chart_call("ds_1", "bar", x_column="x", call_id="call_secret_leak")),
            ToolMessage(
                content=sensitive_raw_content,
                name=CHART_TOOL_NAME,
                tool_call_id="call_secret_leak",
                status="error",
            ),
        ],
        "step_count": 1,
        "sql_retry_count": 0,
        "python_retry_count": 0,
    }

    updates = inspect_tool_results(state)
    assert updates.get("last_error_category") == NON_RETRYABLE_CHART_ERROR
    assert updates.get("sql_retry_count") == 0
    assert updates.get("python_retry_count") == 0

    last_error = updates.get("last_error", "")
    assert "ChartExecutionError: Chart tool execution failed." in last_error
    # Ensure no sensitive content leaked into last_error
    assert "Traceback" not in last_error
    assert "secret" not in last_error
    assert "AKIAIOSFODNN7EXAMPLE" not in last_error
    assert "10.0.0.1" not in last_error

    # Original ToolMessage remains intact with original tool_call_id
    tool_msg = state["messages"][2]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.tool_call_id == "call_secret_leak"


def test_toolnode_error_graph_recovery_stop_sanitization():
    """Verify full graph execution with ToolNode-level failure sanitizes recovery_stop message."""
    sensitive_msg = "Critical failure in /etc/passwd with token secret_token_xyz123"

    class FailingToolNodeLLM(BaseChatModel):
        invocations: int = 0

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.invocations += 1
            if self.invocations == 1:
                return ChatResult(generations=[ChatGeneration(message=ai(
                    chart_call("ds_1", "bar", x_column="x", call_id="c_fail")
                ))])
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Should not be reached"))])

        def bind_tools(self, tools, **kwargs):
            return self

        @property
        def _llm_type(self):
            return "failing-toolnode-llm"

    from langchain_core.tools import tool

    @tool(CHART_TOOL_NAME)
    def failing_chart_tool(dataset_id: str, chart_type: str, x_column: Optional[str] = None) -> str:
        """Mock chart tool that crashes."""
        raise RuntimeError(sensitive_msg)

    llm = FailingToolNodeLLM()
    graph = create_agent_graph(llm=llm, tools=[failing_chart_tool], max_steps=5)
    final_state = graph.invoke(create_initial_state("Chart data"))

    assert final_state["step_count"] == 1
    assert llm.invocations == 1
    assert final_state["last_error_category"] == NON_RETRYABLE_CHART_ERROR
    assert final_state["sql_retry_count"] == 0
    assert final_state["python_retry_count"] == 0

    last_msg = final_state["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    assert "Execution stopped: chart generation failed and will not be automatically retried." in last_msg.content
    assert "ChartExecutionError: Chart tool execution failed." in last_msg.content

    # Strict check: secret message not leaked into recovery_stop AIMessage
    assert "secret_token_xyz123" not in last_msg.content
    assert "/etc/passwd" not in last_msg.content


def test_structured_business_error_remains_actionable():
    """Verify structured ChartGenerationEngine errors (e.g. InvalidColumnError) preserve actionable details."""
    state = {
        "messages": [
            HumanMessage(content="Plot"),
            ai(chart_call("ds_1", "bar", x_column="bad_col", call_id="c_col_err")),
            ToolMessage(
                content=json.dumps({
                    "status": "error",
                    "error_type": "InvalidColumnError",
                    "error_message": "Column 'bad_col' does not exist in dataset.",
                    "dataset_id": "ds_1",
                    "chart_type": "bar",
                }),
                name=CHART_TOOL_NAME,
                tool_call_id="c_col_err",
            ),
        ],
        "step_count": 1,
    }

    updates = inspect_tool_results(state)
    assert updates.get("last_error_category") == NON_RETRYABLE_CHART_ERROR
    last_err = updates.get("last_error", "")
    assert "InvalidColumnError" in last_err
    assert "Column 'bad_col' does not exist in dataset." in last_err


