"""Comprehensive offline integration tests for Python Analysis Tool with LangChain and LangGraph (Step 5.3).

Coverage:
A. Tool creation and schema: name, description, arguments, LangChain compatibility.
B. Shared DatasetStore: SQL tool writes dataset, Python tool reads and analyzes it.
C. Different DatasetStore instances: store isolation and DatasetNotFoundError.
D. All five operations: describe, correlation, growth_rate, detect_outliers, groupby via tool invocation.
E. Structured errors: invalid dataset_id, unsupported operation, invalid column, invalid parameters.
F. LangGraph tool registration: tool binding, ToolNode execution, graph compilation.
G. SQL -> Python multi-step agent workflow: dynamic multi-turn data handoff inside LangGraph.
H. Truncation and result boundaries: metadata retention, preview bounding, warning propagation.
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
from langchain_core.tools import BaseTool

from agent.graph import create_agent_graph
from agent.state import create_initial_state
from agent.tool_wrappers import (
    ToolSequence,
    create_python_analysis_tool,
    create_python_analysis_tools,
    create_sql_tools,
    default_python_tool,
)
from tools.dataset_store import DatasetStore


# ==============================================================================
# Helper / Fixtures
# ==============================================================================

def _make_test_db(tmp_path: Path, name: str = "test_analytics.duckdb") -> Path:
    """Create a temporary DuckDB database for offline testing."""
    db_file = tmp_path / name
    with duckdb.connect(str(db_file)) as con:
        con.execute(
            """
            CREATE TABLE sales_data (
                order_id INT,
                period VARCHAR,
                category VARCHAR,
                revenue DOUBLE,
                units INT
            );
            """
        )
        con.execute(
            """
            INSERT INTO sales_data VALUES
                (1, '2023-01', 'Electronics', 100.0, 2),
                (2, '2023-02', 'Electronics', 120.0, 3),
                (3, '2023-03', 'Electronics', 150.0, 3),
                (4, '2023-01', 'Books', 50.0, 5),
                (5, '2023-02', 'Books', 45.0, 4),
                (6, '2023-03', 'Books', 60.0, 6);
            """
        )
    return db_file


class DynamicMultiStepAnalystModel(BaseChatModel):
    """Dynamic fake chat model that implements a multi-step SQL -> Python analysis workflow."""

    def __init__(self, **data: Any):
        super().__init__(**data)
        shared = data.get("_shared_calls")
        if shared is None:
            shared = []
        object.__setattr__(self, "_shared_calls", shared)

    @property
    def calls(self) -> List[List[BaseMessage]]:
        return getattr(self, "_shared_calls", [])

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        last_msg = messages[-1]

        # Step 1: Initial query -> call execute_sql
        if isinstance(last_msg, HumanMessage):
            ai_msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute_sql",
                        "args": {"query": "SELECT period, revenue FROM sales_data WHERE category = 'Electronics' ORDER BY period"},
                        "id": "sql_call_1",
                    }
                ],
                id=str(uuid.uuid4()),
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        # Step 2: SQL returned ToolMessage -> call execute_python_analysis
        if isinstance(last_msg, ToolMessage) and last_msg.name == "execute_sql":
            res = json.loads(last_msg.content)
            dataset_id = res["dataset_id"]
            ai_msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute_python_analysis",
                        "args": {
                            "dataset_id": dataset_id,
                            "operation": "growth_rate",
                            "params": {
                                "period_column": "period",
                                "value_column": "revenue",
                            },
                        },
                        "id": "py_call_1",
                    }
                ],
                id=str(uuid.uuid4()),
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        # Step 3: Python analysis returned ToolMessage -> final conclusion
        if isinstance(last_msg, ToolMessage) and last_msg.name == "execute_python_analysis":
            res = json.loads(last_msg.content)
            growth_results = res.get("result", [])
            latest_growth = growth_results[-1].get("growth_rate")
            ai_msg = AIMessage(
                content=f"Electronics revenue grew with final period growth rate of {latest_growth:.2f}.",
                id=str(uuid.uuid4()),
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        # Fallback
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Workflow ended."))])

    def bind_tools(self, tools: Any, **kwargs: Any) -> DynamicMultiStepAnalystModel:
        return DynamicMultiStepAnalystModel(_shared_calls=self.calls)

    @property
    def _llm_type(self) -> str:
        return "dynamic-multi-step-analyst-model"


# ==============================================================================
# Category A: Tool Creation & Schema
# ==============================================================================

def test_python_tool_creation_and_metadata():
    store = DatasetStore()
    py_tool = create_python_analysis_tool(dataset_store=store)

    assert isinstance(py_tool, BaseTool)
    assert py_tool.name == "execute_python_analysis"
    assert "Execute controlled analytical Python computations" in py_tool.description
    assert "describe" in py_tool.description
    assert "correlation" in py_tool.description
    assert "Required: at least two numeric columns" in py_tool.description
    assert "growth_rate" in py_tool.description
    assert "detect_outliers" in py_tool.description
    assert "groupby" in py_tool.description

    # Argument schema verification
    schema = py_tool.args_schema.model_json_schema() if py_tool.args_schema else py_tool.args
    assert "dataset_id" in schema["properties"]
    assert "operation" in schema["properties"]
    assert "params" in schema["properties"]


def test_standalone_default_tools_have_independent_stores():
    """Verify standalone default tool instances exist for backward compatibility but do not share a store."""
    from agent.tool_wrappers import default_schema_tool, default_sql_tool, default_python_tool

    assert default_schema_tool is not None
    assert default_sql_tool is not None
    assert default_python_tool is not None

    # Standalone default_python_tool cannot find an arbitrary dataset ID because it owns an independent empty store
    res = default_python_tool.invoke({"dataset_id": "ds_any_unshared", "operation": "describe"})
    assert res["status"] == "error"
    assert res["error_type"] == "DatasetNotFoundError"


def test_python_tools_sequence_concatenation():
    store = DatasetStore()
    sql_tools = create_sql_tools(dataset_store=store)
    py_tools = create_python_analysis_tools(dataset_store=store)

    # Tuple + ToolSequence concatenation
    all_tools = sql_tools + py_tools
    assert len(all_tools) == 3
    tool_names = [t.name for t in all_tools]
    assert tool_names == ["get_database_schema", "execute_sql", "execute_python_analysis"]

    # List + ToolSequence concatenation
    all_list = list(sql_tools) + py_tools
    assert len(all_list) == 3
    assert all_list[2].name == "execute_python_analysis"


# ==============================================================================
# Category B: Shared DatasetStore Workflow
# ==============================================================================

def test_shared_dataset_store_sql_to_python_handoff(tmp_path: Path):
    db_file = _make_test_db(tmp_path)
    shared_store = DatasetStore()

    _, sql_tool = create_sql_tools(db_path=db_file, dataset_store=shared_store)
    py_tool = create_python_analysis_tool(dataset_store=shared_store)

    # Step 1: SQL execution saves dataset
    sql_res = sql_tool.invoke({"query": "SELECT revenue FROM sales_data WHERE category = 'Books'"})
    assert sql_res["status"] == "success"
    dataset_id = sql_res["dataset_id"]
    assert shared_store.get(dataset_id) is not None

    # Step 2: Python tool analyzes the exact dataset_id
    py_res = py_tool.invoke(
        {
            "dataset_id": dataset_id,
            "operation": "describe",
            "params": {"columns": ["revenue"]},
        }
    )
    assert py_res["status"] == "success"
    assert py_res["dataset_id"] == dataset_id
    assert py_res["operation"] == "describe"
    stats = py_res["result"]["revenue"]
    assert stats["count"] == 3
    # 50 + 45 + 60 = 155 / 3 = 51.666...
    assert round(stats["mean"], 2) == 51.67
    assert stats["min"] == 45.0
    assert stats["max"] == 60.0


# ==============================================================================
# Category C: Different DatasetStore Instances (Isolation)
# ==============================================================================

def test_different_dataset_store_isolation(tmp_path: Path):
    db_file = _make_test_db(tmp_path)
    store_a = DatasetStore()
    store_b = DatasetStore()

    _, sql_tool_a = create_sql_tools(db_path=db_file, dataset_store=store_a)
    py_tool_b = create_python_analysis_tool(dataset_store=store_b)

    # Save to Store A
    sql_res = sql_tool_a.invoke({"query": "SELECT * FROM sales_data"})
    dataset_id = sql_res["dataset_id"]

    # Attempt to analyze via Tool B connected to Store B
    py_res = py_tool_b.invoke(
        {
            "dataset_id": dataset_id,
            "operation": "describe",
            "params": {},
        }
    )
    assert py_res["status"] == "error"
    assert py_res["error_type"] == "DatasetNotFoundError"
    assert f"Dataset '{dataset_id}' not found" in py_res["error_message"]


# ==============================================================================
# Category D: All Five Operations via Tool Interface
# ==============================================================================

@pytest.fixture
def populated_tool():
    store = DatasetStore()
    py_tool = create_python_analysis_tool(dataset_store=store)
    d_id = store.save(
        {
            "status": "success",
            "columns": ["period", "dept", "x", "y"],
            "rows": [
                ["2023-01", "eng", 10.0, 100.0],
                ["2023-02", "eng", 20.0, 200.0],
                ["2023-03", "sales", 30.0, 300.0],
                ["2023-04", "sales", 40.0, 4000.0],  # Outlier in y
            ],
            "row_count": 4,
        }
    )
    return py_tool, d_id


def test_tool_op_describe(populated_tool):
    tool, d_id = populated_tool
    res = tool.invoke({"dataset_id": d_id, "operation": "describe", "params": {"columns": ["x"]}})
    assert res["status"] == "success"
    assert res["result"]["x"]["count"] == 4
    assert res["result"]["x"]["mean"] == 25.0


def test_tool_op_correlation(populated_tool):
    tool, d_id = populated_tool
    res = tool.invoke({"dataset_id": d_id, "operation": "correlation", "params": {"columns": ["x", "y"]}})
    assert res["status"] == "success"
    assert res["result"]["x"]["x"] == 1.0
    assert isinstance(res["result"]["x"]["y"], float)


def test_tool_op_growth_rate(populated_tool):
    tool, d_id = populated_tool
    res = tool.invoke(
        {
            "dataset_id": d_id,
            "operation": "growth_rate",
            "params": {"period_column": "period", "value_column": "x"},
        }
    )
    assert res["status"] == "success"
    rows = res["result"]
    assert len(rows) == 4
    assert rows[0]["growth_rate"] is None
    assert rows[1]["growth_rate"] == 1.0  # (20 - 10) / 10 = 1.0


def test_tool_op_detect_outliers(populated_tool):
    tool, d_id = populated_tool
    res = tool.invoke(
        {
            "dataset_id": d_id,
            "operation": "detect_outliers",
            "params": {"column": "y", "iqr_multiplier": 1.5},
        }
    )
    assert res["status"] == "success"
    assert res["result"]["outlier_count"] == 1
    assert res["result"]["outliers_preview"][0]["value"] == 4000.0


def test_tool_op_groupby(populated_tool):
    tool, d_id = populated_tool
    res = tool.invoke(
        {
            "dataset_id": d_id,
            "operation": "groupby",
            "params": {"by": "dept", "agg_column": "x", "agg_func": "sum"},
        }
    )
    assert res["status"] == "success"
    dept_map = {r["dept"]: r["x_sum"] for r in res["result"]}
    assert dept_map["eng"] == 30.0
    assert dept_map["sales"] == 70.0


# ==============================================================================
# Category E: Structured Error Returns
# ==============================================================================

def test_structured_error_invalid_dataset_id(populated_tool):
    tool, _ = populated_tool
    res = tool.invoke({"dataset_id": "nonexistent_id", "operation": "describe"})
    assert res["status"] == "error"
    assert res["error_type"] == "DatasetNotFoundError"


def test_structured_error_unsupported_operation(populated_tool):
    tool, d_id = populated_tool
    res = tool.invoke({"dataset_id": d_id, "operation": "unsupported_kmeans"})
    assert res["status"] == "error"
    assert res["error_type"] == "InvalidOperationError"


def test_structured_error_missing_column(populated_tool):
    tool, d_id = populated_tool
    res = tool.invoke({"dataset_id": d_id, "operation": "describe", "params": {"columns": ["no_such_col"]}})
    assert res["status"] == "error"
    assert res["error_type"] == "InvalidColumnError"


def test_structured_error_invalid_params(populated_tool):
    tool, d_id = populated_tool
    res = tool.invoke({"dataset_id": d_id, "operation": "detect_outliers", "params": {"column": "x", "iqr_multiplier": -2.0}})
    assert res["status"] == "error"
    assert res["error_type"] == "InvalidParameterError"


# ==============================================================================
# Category F: LangGraph Tool Registration
# ==============================================================================

def test_langgraph_registration_with_all_tools():
    store = DatasetStore()
    all_tools = create_sql_tools(dataset_store=store) + create_python_analysis_tools(dataset_store=store)

    graph = create_agent_graph(llm=DynamicMultiStepAnalystModel(), tools=all_tools)
    assert graph is not None

    # Inspect nodes in compiled state graph
    node_names = set(graph.nodes.keys())
    assert {"llm", "tools", "inspect_tool_results", "recovery_stop", "limit_exceeded"}.issubset(node_names)


# ==============================================================================
# Category G: Multi-Step SQL -> Python Agent Workflow
# ==============================================================================

def test_langgraph_sql_to_python_workflow_end_to_end(tmp_path: Path):
    db_file = _make_test_db(tmp_path)
    store = DatasetStore()

    all_tools = create_sql_tools(db_path=db_file, dataset_store=store) + create_python_analysis_tools(dataset_store=store)
    fake_llm = DynamicMultiStepAnalystModel()

    graph = create_agent_graph(llm=fake_llm, tools=all_tools, max_steps=10)
    initial_state = create_initial_state(messages=[HumanMessage(content="Analyze Electronics revenue growth rate.")])

    final_state = graph.invoke(initial_state)

    messages = final_state["messages"]
    # Verify sequence: Human -> AI (sql) -> Tool (sql) -> AI (py) -> Tool (py) -> AI (final)
    assert len(messages) >= 6

    # Verify SQL tool message produced a valid dataset_id
    sql_tool_msgs = [m for m in messages if isinstance(m, ToolMessage) and m.name == "execute_sql"]
    assert len(sql_tool_msgs) == 1
    sql_output = json.loads(sql_tool_msgs[0].content)
    assert sql_output["status"] == "success"
    ds_id = sql_output["dataset_id"]
    assert ds_id.startswith("ds_")

    # Verify Python analysis tool message consumed the exact same dataset_id
    py_tool_msgs = [m for m in messages if isinstance(m, ToolMessage) and m.name == "execute_python_analysis"]
    assert len(py_tool_msgs) == 1
    py_output = json.loads(py_tool_msgs[0].content)
    assert py_output["status"] == "success"
    assert py_output["dataset_id"] == ds_id
    assert py_output["operation"] == "growth_rate"

    # Verify final response
    last_msg = messages[-1]
    assert isinstance(last_msg, AIMessage)
    assert "Electronics revenue grew with final period growth rate of 0.25." in last_msg.content
    assert final_state["step_count"] == 3
    assert final_state.get("last_error") is None


# ==============================================================================
# Category H: Truncation and Output Limits
# ==============================================================================

def test_truncation_warning_propagated_to_python_tool(tmp_path: Path):
    db_file = _make_test_db(tmp_path)
    store = DatasetStore()

    # Query with max_rows=2 so DuckDB results are truncated
    _, sql_tool = create_sql_tools(db_path=db_file, max_rows=2, dataset_store=store)
    py_tool = create_python_analysis_tool(dataset_store=store)

    sql_res = sql_tool.invoke({"query": "SELECT revenue FROM sales_data"})
    assert sql_res["status"] == "success"
    assert sql_res["truncated"] is True
    ds_id = sql_res["dataset_id"]

    # Verify Python tool preserves truncation flag and issues warning
    py_res = py_tool.invoke({"dataset_id": ds_id, "operation": "describe"})
    assert py_res["status"] == "success"
    assert py_res["truncated"] is True
    assert any("truncated by sql execution limits" in w.lower() for w in py_res["warnings"])
