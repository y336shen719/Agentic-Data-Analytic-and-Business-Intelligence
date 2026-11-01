"""Comprehensive tests for DatasetStore and SQL-to-Python data handoff.

Tests:
1. Save and retrieve a valid SQL dataset.
2. Store an empty successful dataset.
3. Save multiple independent datasets.
4. Verify unique, opaque dataset IDs.
5. Verify row/column metadata.
6. Verify saved result cannot be unintentionally mutated through retrieval.
7. Reject failed SQL results.
8. Reject malformed data.
9. Handle unknown and deleted IDs.
10. Clear stored datasets.
11. Enforce storage count limit.
12. Enforce approximate size limit.
13. Verify independent DatasetStore instances are isolated.
14. Successful SQL Tool Wrapper returns a valid dataset reference.
15. Failed SQL execution does not create a dataset.
16. ToolMessage output remains compatible with SQL Result Inspection.
17. Existing SQL Recovery behavior remains unchanged.
18. Dataset preview size is bounded.
19. Truncated SQL results retain accurate truncation metadata.
20. No additional SQL query is executed merely to store a result.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent.graph import create_agent_graph, inspect_tool_results
from agent.sql_error_classifier import SUCCESS, classify_sql_result
from agent.state import create_initial_state
from agent.tool_wrappers import create_sql_tools
from tools.dataset_store import (
    DatasetCapacityError,
    DatasetNotFoundError,
    DatasetRecord,
    DatasetStore,
    InvalidDatasetError,
)


# --- 1. Save and retrieve a valid SQL dataset ---
def test_save_and_retrieve_valid_dataset():
    store = DatasetStore()
    sql_res = {
        "status": "success",
        "columns": ["id", "val"],
        "rows": [[1, "apple"], [2, "banana"]],
        "row_count": 2,
        "truncated": False,
        "execution_time_ms": 1.5,
    }
    d_id = store.save(sql_res, query="SELECT * FROM fruits")
    assert isinstance(d_id, str) and d_id.startswith("ds_")

    rec = store.get(d_id)
    assert isinstance(rec, DatasetRecord)
    assert rec.dataset_id == d_id
    assert rec.columns == ["id", "val"]
    assert rec.rows == [[1, "apple"], [2, "banana"]]
    assert rec.row_count == 2
    assert rec.source_sql == "SELECT * FROM fruits"
    assert rec.truncated is False
    assert rec.size_bytes > 0


# --- 2. Store an empty successful dataset ---
def test_store_empty_successful_dataset():
    store = DatasetStore()
    sql_res = {
        "status": "success",
        "columns": ["id", "val"],
        "rows": [],
        "row_count": 0,
        "truncated": False,
    }
    d_id = store.save(sql_res)
    rec = store.get(d_id)
    assert rec.columns == ["id", "val"]
    assert rec.rows == []
    assert rec.row_count == 0


# --- 3. Save multiple independent datasets ---
def test_save_multiple_independent_datasets():
    store = DatasetStore()
    d1 = store.save({"status": "success", "columns": ["a"], "rows": [[1]]})
    d2 = store.save({"status": "success", "columns": ["b"], "rows": [[2]]})
    assert d1 != d2
    assert len(store) == 2
    assert store.get(d1).columns == ["a"]
    assert store.get(d2).columns == ["b"]


# --- 4. Verify unique, opaque dataset IDs ---
def test_unique_opaque_dataset_ids():
    store = DatasetStore()
    ids = set()
    for i in range(10):
        d_id = store.save({"status": "success", "columns": ["x"], "rows": [[i]]})
        assert d_id.startswith("ds_")
        # Opaque: no file path or DB details
        assert "/" not in d_id
        assert "\\" not in d_id
        assert ".duckdb" not in d_id
        ids.add(d_id)
    assert len(ids) == 10


# --- 5. Verify row/column metadata ---
def test_describe_metadata():
    store = DatasetStore()
    sql_res = {
        "status": "success",
        "columns": ["metric", "value"],
        "rows": [["revenue", 1000], ["cost", 400]],
        "row_count": 2,
        "truncated": False,
    }
    d_id = store.save(sql_res, query="SELECT metric, value FROM t")
    meta = store.describe(d_id)
    assert meta["dataset_id"] == d_id
    assert meta["columns"] == ["metric", "value"]
    assert meta["row_count"] == 2
    assert meta["source_sql"] == "SELECT metric, value FROM t"
    assert meta["truncated"] is False
    assert meta["size_bytes"] > 0
    assert "rows" not in meta  # describe is lightweight, no full rows


# --- 6. Verify saved result cannot be mutated through retrieval ---
def test_mutation_protection():
    store = DatasetStore()
    original_rows = [[1, "orig"]]
    sql_res = {
        "status": "success",
        "columns": ["col"],
        "rows": original_rows,
        "row_count": 1,
    }
    d_id = store.save(sql_res)

    # Mutate the input dictionary afterwards
    original_rows[0][1] = "mutated_input"
    rec1 = store.get(d_id)
    assert rec1.rows[0][1] == "orig"

    # Mutate the returned DatasetRecord
    rec1.rows[0][1] = "mutated_after_get"
    rec2 = store.get(d_id)
    assert rec2.rows[0][1] == "orig"


# --- 7. Reject failed SQL results ---
def test_reject_failed_sql_results():
    store = DatasetStore()
    error_res = {
        "status": "error",
        "error_type": "BinderException",
        "error_message": "Table not found",
    }
    with pytest.raises(InvalidDatasetError, match="Cannot save non-successful SQL result"):
        store.save(error_res)
    assert len(store) == 0


# --- 8. Reject malformed data ---
def test_reject_malformed_data():
    store = DatasetStore()
    with pytest.raises(InvalidDatasetError, match="sql_result must be a dict"):
        store.save("not a dict")  # type: ignore

    with pytest.raises(InvalidDatasetError, match="missing a valid 'columns' list"):
        store.save({"status": "success", "rows": []})

    with pytest.raises(InvalidDatasetError, match="missing a valid 'rows' list"):
        store.save({"status": "success", "columns": ["a"]})

    with pytest.raises(InvalidDatasetError, match="Column name at index 0 must be a str"):
        store.save({"status": "success", "columns": [123], "rows": []})


# --- 9. Handle unknown and deleted IDs ---
def test_handle_unknown_and_deleted_ids():
    store = DatasetStore()
    with pytest.raises(DatasetNotFoundError):
        store.get("ds_nonexistent")

    with pytest.raises(DatasetNotFoundError):
        store.describe("ds_nonexistent")

    assert store.delete("ds_nonexistent") is False

    d_id = store.save({"status": "success", "columns": ["a"], "rows": [[1]]})
    assert store.delete(d_id) is True
    assert store.delete(d_id) is False

    with pytest.raises(DatasetNotFoundError):
        store.get(d_id)


# --- 10. Clear stored datasets ---
def test_clear_stored_datasets():
    store = DatasetStore()
    store.save({"status": "success", "columns": ["a"], "rows": [[1]]})
    store.save({"status": "success", "columns": ["b"], "rows": [[2]]})
    assert len(store) == 2
    assert store.total_memory_bytes > 0

    store.clear()
    assert len(store) == 0
    assert store.total_memory_bytes == 0
    assert store.list_datasets() == []


# --- 11. Enforce storage count limit ---
def test_enforce_storage_count_limit():
    store = DatasetStore(max_datasets=2)
    store.save({"status": "success", "columns": ["a"], "rows": [[1]]})
    store.save({"status": "success", "columns": ["b"], "rows": [[2]]})

    with pytest.raises(DatasetCapacityError, match="maximum dataset count limit"):
        store.save({"status": "success", "columns": ["c"], "rows": [[3]]})


# --- 12. Enforce approximate size limit ---
def test_enforce_approximate_size_limit():
    # Very small memory capacity: 500 bytes
    store = DatasetStore(max_memory_bytes=500)
    store.save({"status": "success", "columns": ["a"], "rows": [[1]]})

    # Massive payload exceeding 500 bytes
    big_rows = [[i, "x" * 200] for i in range(10)]
    with pytest.raises(DatasetCapacityError, match="memory limit exceeded"):
        store.save({"status": "success", "columns": ["id", "blob"], "rows": big_rows})


# --- 13. Verify independent DatasetStore instances are isolated ---
def test_independent_instances_are_isolated():
    store1 = DatasetStore()
    store2 = DatasetStore()

    d1 = store1.save({"status": "success", "columns": ["a"], "rows": [[1]]})
    assert len(store1) == 1
    assert len(store2) == 0

    with pytest.raises(DatasetNotFoundError):
        store2.get(d1)


# --- 14. Successful SQL Tool Wrapper returns a valid dataset reference ---
def test_sql_wrapper_returns_dataset_reference():
    mock_store = DatasetStore()
    fake_sql_result = {
        "status": "success",
        "columns": ["col1", "col2"],
        "rows": [[1, "a"], [2, "b"], [3, "c"]],
        "row_count": 3,
        "truncated": False,
        "execution_time_ms": 2.0,
    }

    with patch("agent.tool_wrappers.raw_execute_sql", return_value=fake_sql_result):
        _, sql_tool = create_sql_tools(dataset_store=mock_store, preview_rows_limit=2)
        tool_output = sql_tool.invoke({"query": "SELECT col1, col2 FROM fake"})

    assert tool_output["status"] == "success"
    assert "dataset_id" in tool_output
    assert tool_output["dataset_id"].startswith("ds_")
    assert tool_output["columns"] == ["col1", "col2"]
    assert tool_output["row_count"] == 3
    assert tool_output["preview_row_count"] == 2
    assert tool_output["rows"] == [[1, "a"], [2, "b"]]
    assert tool_output["truncated"] is False

    # Stored record holds the full rows
    stored = mock_store.get(tool_output["dataset_id"])
    assert stored.rows == [[1, "a"], [2, "b"], [3, "c"]]
    assert stored.row_count == 3


# --- 15. Failed SQL execution does not create a dataset ---
def test_failed_sql_execution_does_not_create_dataset():
    mock_store = DatasetStore()
    error_result = {
        "status": "error",
        "error_type": "CatalogException",
        "error_message": "Table unknown not found",
        "execution_time_ms": 1.0,
    }

    with patch("agent.tool_wrappers.raw_execute_sql", return_value=error_result):
        _, sql_tool = create_sql_tools(dataset_store=mock_store)
        tool_output = sql_tool.invoke({"query": "SELECT * FROM unknown"})

    assert tool_output["status"] == "error"
    assert "dataset_id" not in tool_output
    assert len(mock_store) == 0


# --- 16. ToolMessage output remains compatible with SQL Result Inspection ---
def test_toolmessage_compatible_with_sql_inspection():
    mock_store = DatasetStore()
    tool_output = {
        "status": "success",
        "dataset_id": "ds_1234567890abcdef",
        "columns": ["x", "y"],
        "row_count": 100,
        "preview_row_count": 2,
        "rows": [[1, 2], [3, 4]],
        "truncated": False,
        "execution_time_ms": 3.0,
    }

    # Verify classifier handles this payload cleanly
    classification = classify_sql_result(tool_output)
    assert classification["category"] == SUCCESS
    assert classification["retryable"] is False

    # Verify inspect_tool_results in LangGraph state
    call_id = "call_abc"
    ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "execute_sql", "args": {"query": "SELECT x, y FROM t"}, "id": call_id}],
    )
    t_msg = ToolMessage(
        content=json.dumps(tool_output),
        name="execute_sql",
        tool_call_id=call_id,
    )
    state = create_initial_state([HumanMessage(content="run query"), ai_msg, t_msg])
    state["last_error"] = "Previous error"

    update = inspect_tool_results(state)
    assert update["last_error"] is None
    assert update["last_error_category"] is None


# --- 17. Existing SQL Recovery behavior remains unchanged ---
def test_sql_recovery_behavior_remains_unchanged():
    # Verify compiled graph with dataset_store-equipped tools
    mock_store = DatasetStore()
    schema_tool, sql_tool = create_sql_tools(dataset_store=mock_store)

    class FakeChatModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "fake-chat-model"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            msg = AIMessage(content="Analysis complete.")
            return ChatResult(generations=[ChatGeneration(message=msg)])

        def bind_tools(self, tools, **kwargs):
            return self

    graph = create_agent_graph(llm=FakeChatModel(), tools=[schema_tool, sql_tool])
    result = graph.invoke(create_initial_state("Explain revenue"))
    assert result["step_count"] == 1
    assert result["messages"][-1].content == "Analysis complete."


# --- 18. Dataset preview size is bounded ---
def test_dataset_preview_size_is_bounded():
    mock_store = DatasetStore()
    large_rows = [[i, f"item_{i}"] for i in range(100)]
    sql_res = {
        "status": "success",
        "columns": ["id", "name"],
        "rows": large_rows,
        "row_count": 100,
        "truncated": False,
    }
    with patch("agent.tool_wrappers.raw_execute_sql", return_value=sql_res):
        _, sql_tool = create_sql_tools(dataset_store=mock_store, preview_rows_limit=5)
        res = sql_tool.invoke({"query": "SELECT * FROM large_table"})

    assert len(res["rows"]) == 5
    assert res["preview_row_count"] == 5
    assert res["row_count"] == 100

    # Underlying stored dataset has all 100 rows
    stored = mock_store.get(res["dataset_id"])
    assert len(stored.rows) == 100


# --- 19. Truncated SQL results retain accurate truncation metadata ---
def test_truncated_sql_results_retain_truncation_metadata():
    mock_store = DatasetStore()
    sql_res = {
        "status": "success",
        "columns": ["id"],
        "rows": [[i] for i in range(1000)],
        "row_count": 1000,
        "truncated": True,
        "execution_time_ms": 15.0,
    }
    with patch("agent.tool_wrappers.raw_execute_sql", return_value=sql_res):
        _, sql_tool = create_sql_tools(dataset_store=mock_store)
        res = sql_tool.invoke({"query": "SELECT id FROM million_rows"})

    assert res["truncated"] is True
    assert res["dataset_id"] in mock_store._datasets
    stored = mock_store.get(res["dataset_id"])
    assert stored.truncated is True


# --- 20. No additional SQL query is executed merely to store a result ---
def test_no_additional_sql_executed_to_store_result():
    mock_store = DatasetStore()
    sql_res = {
        "status": "success",
        "columns": ["val"],
        "rows": [[42]],
        "row_count": 1,
        "truncated": False,
    }
    with patch("agent.tool_wrappers.raw_execute_sql", return_value=sql_res) as mock_raw:
        _, sql_tool = create_sql_tools(dataset_store=mock_store)
        sql_tool.invoke({"query": "SELECT 42"})
        assert mock_raw.call_count == 1
