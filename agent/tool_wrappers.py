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
from pydantic import BaseModel, Field

from data.prepare_data import DEFAULT_DB_PATH
from tools.chart_generation import (
    DEFAULT_MAX_BINS,
    DEFAULT_MAX_CATEGORIES,
    DEFAULT_MAX_CHART_ROWS,
    DEFAULT_MAX_FIGURE_BYTES,
    DEFAULT_MAX_HEATMAP_COLUMNS,
    DEFAULT_MAX_TITLE_LENGTH,
    SUPPORTED_CHART_TYPES,
    ChartGenerationEngine,
    analysis_result_to_dataset_payload,
)
from tools.chart_registry import (
    ChartRecord,
    ChartRegistry,
    ChartRegistryCapacityError,
)
from tools.dataset_store import DatasetCapacityError, DatasetStore
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
        if dataset_store is not None and engine.dataset_store is not dataset_store:
            raise ValueError(
                "Mismatched DatasetStore instances: the supplied engine.dataset_store "
                "is not identical to the supplied dataset_store."
            )
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
        res = effective_engine.analyze(
            dataset_id=dataset_id,
            operation=operation,
            params=params,
        )

        # Automatically register derived analysis results into the shared DatasetStore for supported operations
        if res.get("status") == "success" and effective_engine.dataset_store is not None:
            if operation == "detect_outliers":
                res["derived_dataset_id"] = None
                res["derived_dataset_status"] = "unsupported"
                res["derived_dataset_error"] = "This operation does not produce a complete chart-ready dataset."
            else:
                try:
                    payload = analysis_result_to_dataset_payload(res)
                    derived_id = effective_engine.dataset_store.save(
                        payload, query=f"derived_{operation}_{dataset_id}"
                    )
                    res["derived_dataset_id"] = derived_id
                    res["derived_dataset_status"] = "registered"
                except DatasetCapacityError:
                    res["derived_dataset_id"] = None
                    res["derived_dataset_status"] = "failed"
                    res["derived_dataset_error"] = (
                        "The analysis succeeded, but its derived dataset could not be registered "
                        "because DatasetStore capacity was exceeded."
                    )
                except Exception:
                    res["derived_dataset_id"] = None
                    res["derived_dataset_status"] = "failed"
                    res["derived_dataset_error"] = (
                        "The analysis succeeded, but its derived dataset could not be registered."
                    )

        return res

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


class ChartGenerationInput(BaseModel):
    """Input schema for the generate_chart tool."""

    dataset_id: str = Field(
        ...,
        description="Opaque dataset reference ID in the DatasetStore (e.g. 'ds_...').",
    )
    chart_type: str = Field(
        ...,
        description="Type of chart to generate: exactly one of 'bar', 'line', 'scatter', 'histogram', 'box', 'heatmap'.",
    )
    x_column: Optional[str] = Field(
        default=None,
        description="Column name for horizontal axis. Required for 'bar', 'line', 'scatter', 'histogram'; optional for 'box' (grouping); not used for 'heatmap'.",
    )
    y_column: Optional[str] = Field(
        default=None,
        description="Column name for vertical axis. Required for 'bar', 'line', 'scatter', 'box'; not used for 'histogram' or 'heatmap'.",
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional chart-specific parameters. Supported options: title (str), orientation ('v'|'h' for bar), sort ('asc'|'desc' for bar), top_n (int for bar), color_column (str for bar, line, scatter, histogram, box), sort_x (bool for line), markers (bool for line), outlier_column (str for scatter), nbins (int for histogram), points ('outliers'|'all'|False for box), columns (list[str] for heatmap), correlation_method ('pearson'|'spearman' for heatmap), annotated (bool for heatmap), is_precomputed (bool for heatmap).",
    )


def create_chart_generation_tool(
    dataset_store: Optional[DatasetStore] = None,
    chart_registry: Optional[ChartRegistry] = None,
    engine: Optional[ChartGenerationEngine] = None,
    max_chart_rows: int = DEFAULT_MAX_CHART_ROWS,
    max_categories: int = DEFAULT_MAX_CATEGORIES,
    max_heatmap_columns: int = DEFAULT_MAX_HEATMAP_COLUMNS,
    max_figure_bytes: int = DEFAULT_MAX_FIGURE_BYTES,
    max_title_length: int = DEFAULT_MAX_TITLE_LENGTH,
    max_bins: int = DEFAULT_MAX_BINS,
) -> BaseTool:
    """Factory creating a LangChain tool for generating Plotly charts.

    Consumes datasets from an injected DatasetStore and stores generated Plotly Figure
    specifications in an in-memory ChartRegistry. The LLM receives a lightweight reference
    containing the opaque chart_id, dataset_id, chart_type, title, and metadata.

    Args:
        dataset_store: Injected DatasetStore. Required if engine is None.
        chart_registry: In-memory registry to store full Plotly figures. If None, a new instance is created.
        engine: Optional pre-configured ChartGenerationEngine.
        max_chart_rows: Maximum rows allowed in dataset to prevent oversized figures.
        max_categories: Maximum unique categories displayed on bar charts.
        max_heatmap_columns: Maximum columns in correlation heatmaps.
        max_figure_bytes: Maximum size in bytes of the serialized Plotly JSON figure.
        max_title_length: Maximum allowed character length for chart title.
        max_bins: Maximum allowed histogram bins.

    Returns:
        BaseTool instance named 'generate_chart'.
    """
    if engine is not None:
        if dataset_store is not None and engine.dataset_store is not dataset_store:
            raise ValueError(
                "Mismatched DatasetStore instances: the supplied engine.dataset_store "
                "is not identical to the supplied dataset_store."
            )
        effective_engine = engine
    else:
        store = dataset_store if dataset_store is not None else DatasetStore()
        effective_engine = ChartGenerationEngine(
            dataset_store=store,
            max_chart_rows=max_chart_rows,
            max_categories=max_categories,
            max_heatmap_columns=max_heatmap_columns,
            max_figure_bytes=max_figure_bytes,
            max_title_length=max_title_length,
            max_bins=max_bins,
        )

    effective_registry = chart_registry if chart_registry is not None else ChartRegistry()

    @tool(args_schema=ChartGenerationInput)
    def generate_chart(
        dataset_id: str,
        chart_type: str,
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a deterministic Plotly visualization from a dataset stored in DatasetStore.

        Call this tool to generate charts from datasets previously created via execute_sql or derived
        from execute_python_analysis. The LLM receives a compact reference (chart_id, metadata); the
        full Plotly figure JSON is preserved in application memory for downstream rendering.

        Supported chart types and parameter contracts:
        1. 'bar': Categorical ranking and comparison. Requires x_column and y_column. Does not automatically
           aggregate duplicate categories. Top-N selects input observations after sorting. Supports orientation
           ('v'|'h'), sort ('asc'|'desc'), top_n (int), and color_column (str).
        2. 'line': Chronological or continuous sequence trends. Requires x_column and y_column. Supports
           sort_x (bool), markers (bool), and color_column (str).
        3. 'scatter': Relationship and distribution between two continuous numeric variables. Requires numeric
           x_column and y_column. Supports color_column (str) and an existing, validated outlier_column (str).
        4. 'histogram': Univariate frequency distribution across continuous numeric values. Requires numeric
           x_column. Supports nbins (int) and color_column (str).
        5. 'box': Statistical dispersion and quartiles across categories. Requires numeric y_column,
           optionally supports categorical x_column. Supports points ('outliers'|'all'|False) and color_column (str).
        6. 'heatmap': Correlation matrices across numeric columns. Supports raw-data correlation matrices via
           columns (list[str]), correlation_method ('pearson'|'spearman'), and annotated (bool). Supports precomputed
           correlation matrices from Python analysis via params: {"is_precomputed": true}. Only correlation heatmaps are supported.

        Args:
            dataset_id: Opaque dataset reference ID returned by execute_sql or derived_dataset_id from execute_python_analysis.
            chart_type: Exactly one of 'bar', 'line', 'scatter', 'histogram', 'box', 'heatmap'.
            x_column: Optional column name for horizontal axis (required for bar, line, scatter, histogram).
            y_column: Optional column name for vertical axis (required for bar, line, scatter, box).
            params: Optional dictionary of chart configuration options (title, orientation, sort, top_n, color_column, sort_x, markers, outlier_column, nbins, points, columns, correlation_method, annotated, is_precomputed).

        Returns:
            Structured JSON-compatible dictionary with 'status', 'chart_id', 'dataset_id',
            'chart_type', 'title', and 'metadata' (or error details on failure).
        """
        res = effective_engine.generate(
            dataset_id=dataset_id,
            chart_type=chart_type,
            x_column=x_column,
            y_column=y_column,
            params=params,
        )

        if res.get("status") == "success":
            figure = res.get("figure", {})
            try:
                chart_id = effective_registry.save(
                    figure=figure,
                    dataset_id=res["dataset_id"],
                    chart_type=res["chart_type"],
                    title=res.get("title", ""),
                    metadata=res.get("metadata", {}),
                )
                return {
                    "status": "success",
                    "chart_id": chart_id,
                    "dataset_id": res["dataset_id"],
                    "chart_type": res["chart_type"],
                    "title": res.get("title", ""),
                    "metadata": res.get("metadata", {}),
                }
            except ChartRegistryCapacityError as exc:
                return {
                    "status": "error",
                    "dataset_id": res.get("dataset_id", dataset_id),
                    "chart_type": res.get("chart_type", chart_type),
                    "error_type": "ChartRegistryCapacityError",
                    "error_message": str(exc),
                }

        return res

    return generate_chart


def create_chart_generation_tools(
    dataset_store: Optional[DatasetStore] = None,
    chart_registry: Optional[ChartRegistry] = None,
    engine: Optional[ChartGenerationEngine] = None,
    max_chart_rows: int = DEFAULT_MAX_CHART_ROWS,
    max_categories: int = DEFAULT_MAX_CATEGORIES,
    max_heatmap_columns: int = DEFAULT_MAX_HEATMAP_COLUMNS,
    max_figure_bytes: int = DEFAULT_MAX_FIGURE_BYTES,
    max_title_length: int = DEFAULT_MAX_TITLE_LENGTH,
    max_bins: int = DEFAULT_MAX_BINS,
) -> ToolSequence:
    """Factory creating a sequence containing the 'generate_chart' tool."""
    c_tool = create_chart_generation_tool(
        dataset_store=dataset_store,
        chart_registry=chart_registry,
        engine=engine,
        max_chart_rows=max_chart_rows,
        max_categories=max_categories,
        max_heatmap_columns=max_heatmap_columns,
        max_figure_bytes=max_figure_bytes,
        max_title_length=max_title_length,
        max_bins=max_bins,
    )
    return ToolSequence((c_tool,))


# Standalone default tool instances (preserved strictly for backward compatibility with older tests/imports).
# WARNING: These standalone default instances do NOT share a DatasetStore with each other.
# For integrated SQL -> DatasetStore -> Python Analytical -> Chart workflows, callers must instantiate
# a shared DatasetStore and inject it explicitly into create_sql_tools(dataset_store=store),
# create_python_analysis_tools(dataset_store=store), and create_chart_generation_tools(dataset_store=store).
default_schema_tool, default_sql_tool = create_sql_tools()
default_python_tool = create_python_analysis_tool()
default_chart_tool = create_chart_generation_tool()



