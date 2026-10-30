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
from tools.sql_tool import (
    execute_sql as raw_execute_sql,
    get_database_schema as raw_get_database_schema,
)


def create_sql_tools(
    db_path: Optional[Union[str, Path]] = None,
    max_rows: int = 1000,
    timeout_seconds: float = 10.0,
    allowed_tables: Optional[Collection[str]] = None,
) -> Tuple[BaseTool, BaseTool]:
    """Factory creating configured LangChain tools for schema discovery and SQL execution.

    Args:
        db_path: Path to target DuckDB file. Defaults to DEFAULT_DB_PATH.
        max_rows: Upper bound on rows returned by SQL queries (application-controlled).
        timeout_seconds: Process execution timeout in seconds (application-controlled).
        allowed_tables: Optional override for permitted table names (application-controlled).

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

        return raw_execute_sql(
            query=query,
            db_path=target_path,
            max_rows=max_rows,
            timeout_seconds=timeout_seconds,
            allowed_tables=allowed_tables,
        )

    return get_database_schema, execute_sql


# Default application tool instances using standard production database settings
default_schema_tool, default_sql_tool = create_sql_tools()
