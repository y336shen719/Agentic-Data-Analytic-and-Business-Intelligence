"""Unit and integration tests for LangChain SQL tool wrappers (Step 4.3).

These tests run completely offline and make zero external API calls.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict
from unittest.mock import MagicMock

import duckdb
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.tool_wrappers import (
    create_sql_tools,
    default_schema_tool,
    default_sql_tool,
)
from data.prepare_data import DEFAULT_DB_PATH


# ==============================================================================
# Helper
# ==============================================================================

def _make_sample_db(tmp_path: Path, name: str = "test.duckdb") -> Path:
    db_file = tmp_path / name
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE products (product_id INT, name VARCHAR, price DECIMAL(10,2));")
        con.execute("INSERT INTO products VALUES (1, 'A', 10.50), (2, 'B', 20.00), (3, 'C', 35.75);")
        con.execute("CREATE VIEW expensive_products AS SELECT * FROM products WHERE price > 15.00;")
    return db_file


# ==============================================================================
# Tool Schema & Metadata Tests
# ==============================================================================

@pytest.mark.unit
def test_tool_names_and_descriptions() -> None:
    schema_tool, sql_tool = create_sql_tools()

    assert schema_tool.name == "get_database_schema"
    assert "Discover database schema" in schema_tool.description

    assert sql_tool.name == "execute_sql"
    assert "Safely execute a read-only analytical SQL query" in sql_tool.description


@pytest.mark.unit
def test_tool_arguments_schema() -> None:
    schema_tool, sql_tool = create_sql_tools()

    # Schema tool takes 0 arguments from the LLM
    assert schema_tool.args == {}

    # SQL tool exposes ONLY 'query: str' to the LLM
    assert set(sql_tool.args.keys()) == {"query"}
    assert sql_tool.args["query"]["type"] == "string"

    # Verify security-sensitive arguments are NOT exposed to the LLM
    forbidden_args = {"db_path", "max_rows", "timeout_seconds", "allowed_tables"}
    assert not forbidden_args.intersection(set(sql_tool.args.keys()))


# ==============================================================================
# Delegation & Injection Tests
# ==============================================================================

@pytest.mark.unit
def test_schema_wrapper_delegation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom_path = tmp_path / "custom.duckdb"
    mock_raw_schema = MagicMock(return_value={"tables": [{"name": "mock_tbl"}], "views": []})
    monkeypatch.setattr("agent.tool_wrappers.raw_get_database_schema", mock_raw_schema)

    schema_tool, _ = create_sql_tools(db_path=custom_path)
    res = schema_tool.invoke({})

    assert res == {"tables": [{"name": "mock_tbl"}], "views": []}
    mock_raw_schema.assert_called_once_with(db_path=custom_path)


@pytest.mark.unit
def test_sql_wrapper_delegation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom_path = tmp_path / "custom.duckdb"
    mock_raw_exec = MagicMock(return_value={
        "status": "success",
        "columns": ["cnt"],
        "rows": [[42]],
        "row_count": 1,
        "truncated": False,
        "execution_time_ms": 1.2,
    })
    monkeypatch.setattr("agent.tool_wrappers.raw_execute_sql", mock_raw_exec)

    _, sql_tool = create_sql_tools(
        db_path=custom_path,
        max_rows=50,
        timeout_seconds=3.5,
        allowed_tables=["orders"],
    )

    res = sql_tool.invoke({"query": "SELECT count(*) FROM orders"})

    assert res["status"] == "success"
    assert res["rows"] == [[42]]
    mock_raw_exec.assert_called_once_with(
        query="SELECT count(*) FROM orders",
        db_path=custom_path,
        max_rows=50,
        timeout_seconds=3.5,
        allowed_tables=["orders"],
    )


# ==============================================================================
# Security & Error Handling Tests
# ==============================================================================

@pytest.mark.unit
@pytest.mark.parametrize("empty_query", ["", "   ", "\n\t"])
def test_sql_wrapper_rejects_empty_query(empty_query: str) -> None:
    _, sql_tool = create_sql_tools()

    res = sql_tool.invoke({"query": empty_query})

    assert isinstance(res, dict)
    assert res["status"] == "error"
    assert res["error_type"] == "ValueError"
    assert "SQL query must be a non-empty string" in res["error_message"]


@pytest.mark.unit
@pytest.mark.parametrize("invalid_arg", [None, 123, ["SELECT 1"], {"query": "SELECT 1"}])
def test_sql_wrapper_pydantic_rejects_non_string_query(invalid_arg: Any) -> None:
    from pydantic import ValidationError

    _, sql_tool = create_sql_tools()

    with pytest.raises(ValidationError):
        sql_tool.invoke({"query": invalid_arg})


@pytest.mark.unit
def test_sql_wrapper_preserves_underlying_validation_errors(tmp_path: Path) -> None:
    db_file = _make_sample_db(tmp_path)
    _, sql_tool = create_sql_tools(db_path=db_file)

    # Attempt forbidden DDL through the tool wrapper
    res = sql_tool.invoke({"query": "DROP TABLE products"})

    assert isinstance(res, dict)
    assert res["status"] == "error"
    assert res["error_type"] == "SQLValidationError"
    assert "Disallowed DDL/DML" in res["error_message"] or "Drop" in res["error_message"]


@pytest.mark.unit
def test_sql_wrapper_preserves_unauthorized_table_error(tmp_path: Path) -> None:
    db_file = _make_sample_db(tmp_path)
    _, sql_tool = create_sql_tools(db_path=db_file, allowed_tables=["products"])

    # Attempt access to a table not in allowed_tables
    res = sql_tool.invoke({"query": "SELECT * FROM secret_table"})

    assert isinstance(res, dict)
    assert res["status"] == "error"
    assert res["error_type"] == "SQLValidationError"
    assert "not in the allowed tables list" in res["error_message"]


# ==============================================================================
# Real DuckDB Integration Tests
# ==============================================================================

@pytest.mark.unit
def test_tools_end_to_end_on_isolated_duckdb(tmp_path: Path) -> None:
    """Verify both tools on a freshly created synthetic DuckDB database."""
    db_file = _make_sample_db(tmp_path)
    schema_tool, sql_tool = create_sql_tools(db_path=db_file)

    # 1. Schema discovery
    schema_info = schema_tool.invoke({})
    table_names = [t["name"] for t in schema_info["tables"]]
    view_names = [v["name"] for v in schema_info["views"]]
    assert "products" in table_names
    assert "expensive_products" in view_names

    # 2. Safe analytical execution
    exec_res = sql_tool.invoke({"query": "SELECT count(*) AS total FROM products"})
    assert exec_res["status"] == "success"
    assert exec_res["columns"] == ["total"]
    assert exec_res["rows"] == [[3]]
    assert exec_res["row_count"] == 1
    assert exec_res["truncated"] is False


@pytest.mark.integration
def test_tools_end_to_end_on_production_database() -> None:
    """Verify tools preserve business metadata and query against production Olist database."""
    if not DEFAULT_DB_PATH.exists():
        pytest.skip(f"Production database not found at {DEFAULT_DB_PATH}.")

    schema_tool, sql_tool = create_sql_tools(db_path=DEFAULT_DB_PATH)

    # 1. Schema discovery on prod DB
    schema_info = schema_tool.invoke({})
    views_by_name = {v["name"]: v for v in schema_info["views"]}
    assert "order_summary" in views_by_name

    # Verify business metadata is preserved directly on the view item and columns
    view = views_by_name["order_summary"]
    assert view["grain"] == "One row per order."
    assert "payments, revenue, and delivery metrics" in view["purpose"]
    assert any(col["name"] == "order_id" and "description" in col for col in view["columns"])

    # 2. Safe analytical SQL query on order_summary
    query = "SELECT count(*) AS total_orders FROM order_summary"
    exec_res = sql_tool.invoke({"query": query})
    assert exec_res["status"] == "success"
    assert exec_res["columns"] == ["total_orders"]
    assert exec_res["row_count"] == 1
    assert exec_res["rows"][0][0] > 0

