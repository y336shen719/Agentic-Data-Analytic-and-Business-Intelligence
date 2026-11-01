"""LangChain-compatible tool wrappers for database schema discovery and safe SQL execution.

These wrappers bridge raw analytical database capabilities in ``tools/sql_tool``
to the LangChain tool interface (``BaseTool``). The LLM is restricted to supplying
only the SQL query string, while database path, timeouts, row limits, and table
allowlists remain strictly controlled by the application environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Tuple, Union

from langchain_core.tools import BaseTool, tool

from data.prepare_data import DEFAULT_DB_PATH
from tools.dataset_store import DatasetStore
from tools.python_analysis import (
    DEFAULT_MAX_CORR_COLUMNS,
    DEFAULT_MAX_PREVIEW_ROWS,
    DEFAULT_MAX_TABLE_ROWS,
    PythonAnalysisEngine,
)
from tools.sql_tool import (
    execute_sql as raw_execute_sql,
    get_database_schema as raw_get_database_schema,
)


class ToolSequence(tuple):
    """Tuple subclass of tools supporting clean concatenation with both tuples and lists."""

    def __add__(self, other: Any) -> Any:
        if isinstance(other, list):
            return list(self) + other
        if isinstance(other, tuple):
            return ToolSequence(super().__add__(other))
        return super().__add__(other)

    def __radd__(self, other: Any) -> Any:
        if isinstance(other, list):
            return other + list(self)
        if isinstance(other, tuple):
            return other + tuple(self)
        return NotImplemented


def create_sql_tools(
    db_path: Optional[Union[str, Path]] = None,
    max_rows: int = 1000,
    timeout_seconds: float = 10.0,
    allowed_tables: Optional[Collection[str]] = None,
    dataset_store: Optional[DatasetStore] = None,
    preview_rows_limit: int = 10,
) -> Tuple[BaseTool, BaseTool]:
    """Factory creating configured LangChain tools for schema discovery and SQL execution.

    Args:
        db_path: Path to target DuckDB file. Defaults to DEFAULT_DB_PATH.
        max_rows: Upper bound on rows returned by SQL queries (application-controlled).
        timeout_seconds: Process execution timeout in seconds (application-controlled).
        allowed_tables: Optional override for permitted table names (application-controlled).
        dataset_store: Optional DatasetStore instance to persist query results and emit dataset references.
        preview_rows_limit: Maximum preview rows returned in the LLM-facing ToolMessage when a dataset is saved.

    Returns:
        Tuple of (schema_tool, sql_tool) typed as BaseTool.
    """
    target_path = Path(db_path) if db_path is not None else Path(DEFAULT_DB_PATH)

    @tool
    def get_database_schema() -> Dict[str, Any]:
        """Discover database schema including all tables, views, columns, types, and business metadata for analysis.
        
        Call this tool first to understand available tables, columns, metrics, and relationships
        before constructing analytical SQL queries. Takes no arguments.
        """
        return raw_get_database_schema(db_path=target_path)

    @tool
    def execute_sql(query: str) -> Dict[str, Any]:
        """Safely execute a read-only analytical SQL query against the database.
        
        Only read-only SELECT queries are permitted (including CTEs, JOINs, aggregations,
        and window functions). Modifying operations (INSERT, UPDATE, DELETE, DROP, ALTER, etc.)
        and direct filesystem readers are strictly forbidden and will be rejected.
        
        Args:
            query: The analytical SQL SELECT query string to execute.
        """
        if not isinstance(query, str) or not query.strip():
            return {
                "status": "error",
                "error_type": "ValueError",
                "error_message": "SQL query must be a non-empty string.",
                "execution_time_ms": 0.0,
            }

        res = raw_execute_sql(
            query=query,
            db_path=target_path,
            max_rows=max_rows,
            timeout_seconds=timeout_seconds,
            allowed_tables=allowed_tables,
        )

        if res.get("status") == "success" and dataset_store is not None:
            dataset_id = dataset_store.save(res, query=query)
            full_rows = res.get("rows", [])
            preview = full_rows[:preview_rows_limit]
            return {
                "status": "success",
                "dataset_id": dataset_id,
                "columns": res.get("columns", []),
                "row_count": res.get("row_count", len(full_rows)),
                "preview_row_count": len(preview),
                "rows": preview,
                "truncated": res.get("truncated", False),
                "execution_time_ms": res.get("execution_time_ms", 0.0),
            }

        return res

    return get_database_schema, execute_sql


def create_python_analysis_tool(
    dataset_store: Optional[DatasetStore] = None,
    engine: Optional[PythonAnalysisEngine] = None,
    max_preview_rows: int = DEFAULT_MAX_PREVIEW_ROWS,
    max_table_rows: int = DEFAULT_MAX_TABLE_ROWS,
    max_corr_columns: int = DEFAULT_MAX_CORR_COLUMNS,
) -> BaseTool:
    """Factory creating a LangChain tool for controlled Python data analysis.

    Delegates analytical computations on stored datasets (identified by dataset_id)
    to a PythonAnalysisEngine instance without arbitrary code execution.

    Args:
        dataset_store: Shared DatasetStore instance. If engine is None, this store is used.
        engine: Optional pre-configured PythonAnalysisEngine.
        max_preview_rows: Max rows in outlier observation previews.
        max_table_rows: Max rows in tabular outputs (growth, groupby).
        max_corr_columns: Max columns in correlation matrix.

    Returns:
        BaseTool instance named 'execute_python_analysis'.
    """
    if engine is not None:
        effective_engine = engine
    else:
        store = dataset_store if dataset_store is not None else DatasetStore()
        effective_engine = PythonAnalysisEngine(
            dataset_store=store,
            max_preview_rows=max_preview_rows,
            max_table_rows=max_table_rows,
            max_corr_columns=max_corr_columns,
        )

    @tool
    def execute_python_analysis(
        dataset_id: str,
        operation: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute controlled analytical Python computations on a dataset stored in DatasetStore.

        Call this tool to perform advanced statistical or mathematical analysis on datasets
        previously retrieved via execute_sql. The dataset must be referenced by its opaque dataset_id.
        Arbitrary Python code execution is strictly disallowed.

        Supported operations and parameter requirements:
        1. 'describe': Summary statistics (count, mean, std, min, 25%, 50%, 75%, max) for numeric columns.
           params: {"columns": ["col1", "col2"]} (optional, defaults to all numeric columns).
        2. 'correlation': Pearson correlation matrix.
           params: {"columns": ["col1", "col2", ...]}
           Required: at least two numeric columns.
        3. 'growth_rate': Chronological period-over-period percentage growth.
           params: {"period_column": "date_col", "value_column": "metric_col", "group_column": "optional_grp"}.
        4. 'detect_outliers': Interquartile Range (IQR) outlier detection with bounds and preview.
           params: {"column": "metric_col", "iqr_multiplier": 1.5} (multiplier defaults to 1.5).
        5. 'groupby': Secondary grouping and aggregation on stored dataset observations.
           params: {"by": "group_col" or ["col1", "col2"], "agg_column": "metric_col", "agg_func": "sum"|"mean"|"count"|"min"|"max"}.

        Args:
            dataset_id: Opaque dataset reference string returned by execute_sql (e.g. 'ds_...').
            operation: One of the five allowlisted operations listed above.
            params: Optional dictionary containing operation-specific parameter arguments.

        Returns:
            Structured JSON-compatible dictionary with 'status', 'dataset_id', 'operation',
            'result' (or 'error_type'/'error_message'), 'truncated', and 'warnings'.
        """
        return effective_engine.analyze(
            dataset_id=dataset_id,
            operation=operation,
            params=params,
        )

    return execute_python_analysis


def create_python_analysis_tools(
    dataset_store: Optional[DatasetStore] = None,
    engine: Optional[PythonAnalysisEngine] = None,
    max_preview_rows: int = DEFAULT_MAX_PREVIEW_ROWS,
    max_table_rows: int = DEFAULT_MAX_TABLE_ROWS,
    max_corr_columns: int = DEFAULT_MAX_CORR_COLUMNS,
) -> ToolSequence:
    """Factory creating a sequence of Python analysis tools for LangChain/LangGraph.

    Returns a ToolSequence containing the 'execute_python_analysis' tool. Supports
    concatenation with both tuples (e.g. sql_tools + python_tools) and lists.
    """
    py_tool = create_python_analysis_tool(
        dataset_store=dataset_store,
        engine=engine,
        max_preview_rows=max_preview_rows,
        max_table_rows=max_table_rows,
        max_corr_columns=max_corr_columns,
    )
    return ToolSequence((py_tool,))


# Standalone default tool instances (preserved strictly for backward compatibility with older tests/imports).
# WARNING: These standalone default instances do NOT share a DatasetStore with each other.
# For integrated SQL -> DatasetStore -> Python Analytical workflows, callers must instantiate
# a shared DatasetStore and inject it explicitly into create_sql_tools(dataset_store=store) and
# create_python_analysis_tools(dataset_store=store).
default_schema_tool, default_sql_tool = create_sql_tools()
default_python_tool = create_python_analysis_tool()


