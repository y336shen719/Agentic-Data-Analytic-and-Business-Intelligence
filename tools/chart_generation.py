"""Production-ready Chart Generation Engine for Agentic Data Analyst.

Generates six allowlisted, deterministic Plotly chart types:
1. Bar Chart
2. Line Chart
3. Scatter Plot
4. Histogram
5. Box Plot
6. Heatmap

Architecture & Safety:
- Consumes datasets exclusively from an injected DatasetStore.
- Purely deterministic dispatch; arbitrary code execution (eval, exec) is prohibited.
- Validates all input dimensions, column types, and configuration parameters.
- Produces valid, bounded, JSON-serializable Plotly Figure specifications.
- Protects stored datasets from mutation.
"""

from __future__ import annotations

import copy
from datetime import date, datetime
from decimal import Decimal
import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from tools.dataset_store import DatasetNotFoundError, DatasetRecord, DatasetStore

SUPPORTED_CHART_TYPES = frozenset(
    {
        "bar",
        "line",
        "scatter",
        "histogram",
        "box",
        "heatmap",
    }
)

DEFAULT_MAX_CHART_ROWS: int = 5000
DEFAULT_MAX_CATEGORIES: int = 50
DEFAULT_MAX_HEATMAP_COLUMNS: int = 20
DEFAULT_MAX_FIGURE_BYTES: int = 2 * 1024 * 1024  # 2MB
DEFAULT_MAX_TITLE_LENGTH: int = 200
DEFAULT_MAX_BINS: int = 100


# ==============================================================================
# Exceptions
# ==============================================================================


class ChartEngineError(Exception):
    """Base exception for chart generation engine errors."""


class InvalidChartTypeError(ChartEngineError, ValueError):
    """Raised when an unsupported chart type is requested."""


class InvalidColumnError(ChartEngineError, KeyError):
    """Raised when a requested column does not exist or has an incompatible data type."""


class InvalidParameterError(ChartEngineError, ValueError):
    """Raised when chart parameters are invalid or outside allowable ranges."""


class EmptyDatasetError(ChartEngineError, ValueError):
    """Raised when a chart cannot be generated from an empty dataset."""


class DataLimitExceededError(ChartEngineError, ValueError):
    """Raised when dataset dimensions or figure serialization exceeds configured bounds."""


# ==============================================================================
# Data Cleaning & Conversion Helpers
# ==============================================================================


def _clean_json_value(val: Any) -> Any:
    """Recursively convert NumPy scalars, NaNs, infinities, Decimals, and timestamps to JSON-compatible Python primitives."""
    if isinstance(val, (list, tuple, set, pd.Series, np.ndarray)):
        return [_clean_json_value(item) for item in val]
    if isinstance(val, dict):
        return {str(k): _clean_json_value(v) for k, v in val.items()}

    if val is None or val is pd.NA or val is pd.NaT:
        return None

    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)

    if isinstance(val, Decimal):
        if val.is_nan() or val.is_infinite():
            return None
        return float(val)

    if isinstance(val, (bool, np.bool_)):
        return bool(val)

    if isinstance(val, (int, np.integer)):
        return int(val)

    if isinstance(val, (pd.Timestamp, datetime, date)):
        if pd.isna(val):
            return None
        return val.isoformat()

    if isinstance(val, str):
        return val

    try:
        if pd.isna(val):
            return None
    except Exception:
        pass

    return val


def _is_missing_value(val: Any) -> bool:
    """Return True if val represents a missing/null value."""
    if val is None or val is pd.NA or val is pd.NaT:
        return True
    if isinstance(val, (float, np.floating)) and math.isnan(val):
        return True
    if isinstance(val, Decimal) and val.is_nan():
        return True
    try:
        if pd.isna(val):
            return True
    except Exception:
        pass
    return False


def _coerce_numeric_series(series: pd.Series, col_name: str) -> pd.Series:
    """Validate and convert a Series to float numeric values.

    Strict Policies:
    - Genuine NULL/missing values (None, NaN, pd.NA, pd.NaT, Decimal('NaN')) are allowed and preserved as np.nan.
    - If the entire series has only missing/null values, raises InvalidColumnError.
    - Non-finite numeric values (inf, -inf, Decimal('Infinity'), Decimal('-Infinity')) are rejected with InvalidColumnError.
    - Any non-null value that cannot be converted to numeric (e.g. 'unknown', 'invalid', arbitrary text) is rejected with InvalidColumnError.
    - Valid numbers and convertible numeric strings are converted to float.
    """
    if len(series) == 0:
        raise InvalidColumnError(f"Column '{col_name}' contains no rows.")

    # Fast-path for purely numeric dtypes without object overhead
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        if series.isna().all():
            raise InvalidColumnError(f"Column '{col_name}' contains only NULL or missing values.")
        if np.isinf(series).any():
            raise InvalidColumnError(
                f"Column '{col_name}' contains non-finite numeric value (infinity). Infinite values cannot be plotted."
            )
        return series.astype(float)

    # General/Object path: element-by-element inspection to catch specific invalid values
    converted = []
    has_valid_non_null = False

    for idx, val in series.items():
        if _is_missing_value(val):
            converted.append(np.nan)
            continue

        if isinstance(val, (bool, np.bool_)):
            raise InvalidColumnError(
                f"Column '{col_name}' contains boolean value '{val}' (type: bool) which cannot be used as a numeric axis."
            )

        if isinstance(val, Decimal):
            if val.is_infinite():
                raise InvalidColumnError(
                    f"Column '{col_name}' contains non-finite value (Decimal('{val}')). Infinite values cannot be plotted."
                )
            if val.is_nan():
                converted.append(np.nan)
                continue
            converted.append(float(val))
            has_valid_non_null = True
            continue

        if isinstance(val, (float, np.floating)):
            if math.isinf(val):
                raise InvalidColumnError(
                    f"Column '{col_name}' contains non-finite numeric value ({val}). Infinite values cannot be plotted."
                )
            if math.isnan(val):
                converted.append(np.nan)
                continue
            converted.append(float(val))
            has_valid_non_null = True
            continue

        if isinstance(val, (int, np.integer)):
            converted.append(float(val))
            has_valid_non_null = True
            continue

        if isinstance(val, str):
            clean_str = val.strip()
            if clean_str.lower() in ("inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"):
                raise InvalidColumnError(
                    f"Column '{col_name}' contains non-finite value '{val}'. Infinite values cannot be plotted."
                )
            try:
                num = float(clean_str)
                if math.isinf(num):
                    raise InvalidColumnError(
                        f"Column '{col_name}' contains non-finite value '{val}'. Infinite values cannot be plotted."
                    )
                if math.isnan(num):
                    converted.append(np.nan)
                    continue
                converted.append(num)
                has_valid_non_null = True
            except ValueError:
                raise InvalidColumnError(
                    f"Column '{col_name}' contains non-numeric value '{val}' (type: str)."
                )
            continue

        try:
            num = float(val)
            if math.isinf(num):
                raise InvalidColumnError(
                    f"Column '{col_name}' contains non-finite value ({val}). Infinite values cannot be plotted."
                )
            if math.isnan(num):
                converted.append(np.nan)
                continue
            converted.append(num)
            has_valid_non_null = True
        except (TypeError, ValueError):
            raise InvalidColumnError(
                f"Column '{col_name}' contains non-numeric value '{val}' (type: {type(val).__name__})."
            )

    if not has_valid_non_null:
        raise InvalidColumnError(f"Column '{col_name}' contains only NULL or missing values.")

    return pd.Series(converted, index=series.index, dtype=float)


def _validate_bool_param(params: Dict[str, Any], param_name: str, default: bool) -> bool:
    """Validate a boolean parameter strictly, rejecting strings, numbers, or None."""
    if param_name not in params:
        return default
    val = params[param_name]
    if type(val) is not bool:
        raise InvalidParameterError(
            f"Parameter '{param_name}' must be a boolean (True or False), got {val!r} (type: {type(val).__name__})."
        )
    return val


def _validate_str_param(
    params: Dict[str, Any], param_name: str, allowed: Optional[Sequence[str]] = None
) -> Optional[str]:
    """Validate an optional string parameter, rejecting non-strings."""
    if param_name not in params:
        return None
    val = params[param_name]
    if val is None:
        return None
    if not isinstance(val, str):
        raise InvalidParameterError(
            f"Parameter '{param_name}' must be a string, got {val!r} (type: {type(val).__name__})."
        )
    clean_val = val.strip()
    if not clean_val:
        raise InvalidParameterError(f"Parameter '{param_name}' cannot be an empty string.")
    if allowed is not None and clean_val not in allowed:
        raise InvalidParameterError(
            f"Parameter '{param_name}' must be one of {list(allowed)}, got {clean_val!r}."
        )
    return clean_val


def analysis_result_to_dataset_payload(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """Helper adapter to convert structured PythonAnalysisEngine results into DatasetStore-compatible payloads.

    Supported operations:
    - growth_rate: Tabular periods, baseline values, and computed growth rates.
    - groupby: Tabular group keys and aggregated values.
    - describe: Tabular summary statistics (statistic, metric columns).
    - correlation: Tabular correlation matrix with preserved row labels (column_name, metric columns).

    Unsupported operations:
    - detect_outliers: Rejects conversion because it returns statistical bounds and sample previews,
      not a full row-level dataset with outlier indicator flags.
    """
    if not isinstance(analysis_result, dict) or analysis_result.get("status") != "success":
        raise ValueError("Invalid or unsuccessful analytical result.")

    op = analysis_result.get("operation")
    res = analysis_result.get("result")
    if res is None:
        raise ValueError(f"Analytical result for operation '{op}' has no 'result' field.")

    if op == "growth_rate":
        periods = res if isinstance(res, list) else res.get("periods", [])
        if not periods or not isinstance(periods, list):
            raise ValueError("No period records in growth_rate result.")

        first = periods[0]
        has_group = "group" in first
        if has_group:
            columns = ["period", "value", "growth_rate", "group"]
            rows = [[p.get("period"), p.get("value"), p.get("growth_rate"), p.get("group")] for p in periods]
        else:
            columns = ["period", "value", "growth_rate"]
            rows = [[p.get("period"), p.get("value"), p.get("growth_rate")] for p in periods]

        return {
            "status": "success",
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "source_operation": op,
        }

    if op == "groupby":
        groups = res if isinstance(res, list) else res.get("groups", [])
        if not groups or not isinstance(groups, list):
            raise ValueError("No group records in groupby result.")

        columns = list(groups[0].keys())
        rows = [[g.get(c) for c in columns] for g in groups]
        return {
            "status": "success",
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "source_operation": op,
        }

    if op == "describe":
        summary = res.get("summary", res) if isinstance(res, dict) else {}
        if not summary or not isinstance(summary, dict):
            raise ValueError("No summary statistics in describe result.")

        numeric_cols = [c for c, v in summary.items() if isinstance(v, dict)]
        if not numeric_cols:
            raise ValueError("No numeric columns found in describe summary.")

        stat_names = ["count", "mean", "std", "min", "25%", "median", "75%", "max"]
        columns = ["statistic"] + numeric_cols
        rows = [[stat] + [summary[c].get(stat) for c in numeric_cols] for stat in stat_names]
        return {
            "status": "success",
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "source_operation": op,
        }

    if op == "correlation":
        matrix = res.get("matrix", res) if isinstance(res, dict) else {}
        if not matrix or not isinstance(matrix, dict):
            raise ValueError("No correlation matrix data in correlation result.")

        cols = [c for c, v in matrix.items() if isinstance(v, dict)]
        if len(cols) < 2:
            raise ValueError("Correlation matrix requires at least 2 columns.")

        columns = ["column_name"] + cols
        rows = [[row_col] + [matrix.get(row_col, {}).get(c) for c in cols] for row_col in cols]
        return {
            "status": "success",
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "source_operation": op,
        }

    if op == "detect_outliers":
        raise ValueError(
            "Operation 'detect_outliers' returns statistical bounds and sample previews, "
            "not a full row-level dataset with outlier indicator flags. "
            "Visualizing outliers on a scatter plot requires an explicit outlier indicator column in the stored dataset."
        )

    raise ValueError(f"Operation '{op}' does not provide a standard tabular dataset payload.")


# ==============================================================================
# Chart Generation Engine
# ==============================================================================


class ChartGenerationEngine:
    """Production-ready Chart Generation Engine for visualizing DatasetStore datasets.

    Produces six Plotly chart types with deterministic parameters and resource bounds.
    """

    def __init__(
        self,
        dataset_store: DatasetStore,
        max_chart_rows: int = DEFAULT_MAX_CHART_ROWS,
        max_categories: int = DEFAULT_MAX_CATEGORIES,
        max_heatmap_columns: int = DEFAULT_MAX_HEATMAP_COLUMNS,
        max_figure_bytes: int = DEFAULT_MAX_FIGURE_BYTES,
        max_title_length: int = DEFAULT_MAX_TITLE_LENGTH,
        max_bins: int = DEFAULT_MAX_BINS,
    ) -> None:
        """Initialize the ChartGenerationEngine.

        Args:
            dataset_store: Injected DatasetStore holding query results.
            max_chart_rows: Maximum rows allowed in dataset to prevent oversized figures.
            max_categories: Maximum unique categories displayed on bar charts.
            max_heatmap_columns: Maximum columns in correlation heatmaps.
            max_figure_bytes: Maximum size in bytes of the serialized Plotly JSON figure.
            max_title_length: Maximum allowed character length for chart title.
            max_bins: Maximum allowed histogram bins.
        """
        if not isinstance(dataset_store, DatasetStore):
            raise TypeError(
                f"dataset_store must be a DatasetStore instance, got {type(dataset_store).__name__}."
            )

        for name, val in [
            ("max_chart_rows", max_chart_rows),
            ("max_categories", max_categories),
            ("max_heatmap_columns", max_heatmap_columns),
            ("max_figure_bytes", max_figure_bytes),
            ("max_title_length", max_title_length),
            ("max_bins", max_bins),
        ]:
            if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
                raise ValueError(
                    f"{name} must be a strictly positive integer, got {val!r} (type: {type(val).__name__})."
                )

        self.dataset_store = dataset_store
        self.max_chart_rows = max_chart_rows
        self.max_categories = max_categories
        self.max_heatmap_columns = max_heatmap_columns
        self.max_figure_bytes = max_figure_bytes
        self.max_title_length = max_title_length
        self.max_bins = max_bins

    def generate(
        self,
        dataset_id: str,
        chart_type: str,
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a deterministic Plotly Figure specification from a stored dataset.

        Args:
            dataset_id: Opaque dataset reference ID in the DatasetStore.
            chart_type: One of 'bar', 'line', 'scatter', 'histogram', 'box', 'heatmap'.
            x_column: Optional column name for horizontal axis.
            y_column: Optional column name for vertical axis.
            params: Optional dictionary of chart-specific configuration options.

        Returns:
            Standardized dictionary with `status`, `dataset_id`, `chart_type`, `title`,
            `figure`, and `metadata`. On failure, returns `status: error` with details.
        """
        # 1. Basic argument validation
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            return self._error_result(
                dataset_id,
                chart_type,
                "InvalidParameterError",
                "dataset_id must be a non-empty string.",
            )

        clean_chart_type = chart_type.lower().strip() if isinstance(chart_type, str) else ""
        if clean_chart_type not in SUPPORTED_CHART_TYPES:
            return self._error_result(
                dataset_id,
                chart_type,
                "InvalidChartTypeError",
                f"Unsupported chart type '{chart_type}'. Supported types: {sorted(SUPPORTED_CHART_TYPES)}",
            )

        if params is not None and not isinstance(params, dict):
            return self._error_result(
                dataset_id,
                clean_chart_type,
                "InvalidParameterError",
                f"params must be a dictionary, got {type(params).__name__}.",
            )

        chart_params: Dict[str, Any] = params or {}

        # Validate title if supplied
        title = chart_params.get("title")
        if title is not None:
            if not isinstance(title, str):
                return self._error_result(
                    dataset_id,
                    clean_chart_type,
                    "InvalidParameterError",
                    f"title must be a string, got {type(title).__name__}.",
                )
            if len(title) > self.max_title_length:
                return self._error_result(
                    dataset_id,
                    clean_chart_type,
                    "InvalidParameterError",
                    f"title length ({len(title)}) exceeds maximum allowed ({self.max_title_length}).",
                )

        # 2. Retrieve dataset record
        try:
            record: DatasetRecord = self.dataset_store.get(dataset_id)
        except DatasetNotFoundError as exc:
            return self._error_result(
                dataset_id,
                clean_chart_type,
                "DatasetNotFoundError",
                str(exc),
            )

        # 3. Construct DataFrame safely without external mutation
        try:
            df = pd.DataFrame(record.rows, columns=record.columns)
        except Exception as exc:
            return self._error_result(
                dataset_id,
                clean_chart_type,
                "ChartEngineError",
                f"Failed to construct DataFrame from dataset: {exc}",
            )

        if len(df) == 0:
            return self._error_result(
                dataset_id,
                clean_chart_type,
                "EmptyDatasetError",
                "Dataset is empty. Cannot generate a chart from zero rows.",
            )

        if len(df) > self.max_chart_rows:
            return self._error_result(
                dataset_id,
                clean_chart_type,
                "DataLimitExceededError",
                f"Dataset row count ({len(df)}) exceeds max_chart_rows limit ({self.max_chart_rows}). "
                "Please aggregate or filter data via SQL before visualizing.",
            )

        # 4. Dispatch to chart type generator
        try:
            if clean_chart_type == "bar":
                fig, metadata = self._generate_bar(df, dataset_id, x_column, y_column, chart_params)
            elif clean_chart_type == "line":
                fig, metadata = self._generate_line(df, dataset_id, x_column, y_column, chart_params)
            elif clean_chart_type == "scatter":
                fig, metadata = self._generate_scatter(df, dataset_id, x_column, y_column, chart_params)
            elif clean_chart_type == "histogram":
                fig, metadata = self._generate_histogram(df, dataset_id, x_column, chart_params)
            elif clean_chart_type == "box":
                fig, metadata = self._generate_box(df, dataset_id, x_column, y_column, chart_params)
            elif clean_chart_type == "heatmap":
                fig, metadata = self._generate_heatmap(df, dataset_id, chart_params)
            else:
                return self._error_result(
                    dataset_id,
                    clean_chart_type,
                    "InvalidChartTypeError",
                    f"Unhandled chart type: {clean_chart_type}",
                )

            # 5. Apply standard styling and serialize figure safely
            fig.update_layout(template="plotly_white")
            fig_json_str = pio.to_json(fig)
            fig_bytes = len(fig_json_str.encode("utf-8"))

            if fig_bytes > self.max_figure_bytes:
                return self._error_result(
                    dataset_id,
                    clean_chart_type,
                    "DataLimitExceededError",
                    f"Generated chart specification ({fig_bytes} bytes) exceeds max_figure_bytes ({self.max_figure_bytes}). "
                    "Please reduce categories or rows.",
                )

            figure_dict = json.loads(fig_json_str)
            effective_title = metadata.get("title") or (
                title if title is not None else f"{clean_chart_type.title()} Chart"
            )

            return {
                "status": "success",
                "dataset_id": dataset_id,
                "chart_type": clean_chart_type,
                "title": effective_title,
                "figure": figure_dict,
                "metadata": metadata,
            }

        except ChartEngineError as exc:
            return self._error_result(
                dataset_id,
                clean_chart_type,
                type(exc).__name__,
                str(exc),
            )
        except Exception as exc:
            return self._error_result(
                dataset_id,
                clean_chart_type,
                "ChartEngineError",
                f"Unexpected error during chart generation: {exc}",
            )

    # ==========================================================================
    # Chart Generators
    # ==========================================================================

    def _generate_bar(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        x_column: Optional[str],
        y_column: Optional[str],
        params: Dict[str, Any],
    ) -> Tuple[go.Figure, Dict[str, Any]]:
        orientation = _validate_str_param(params, "orientation", allowed=("v", "h")) or "v"
        sort = _validate_str_param(params, "sort", allowed=("asc", "desc"))

        top_n = params.get("top_n")
        if top_n is not None:
            if type(top_n) is not int or top_n <= 0:
                raise InvalidParameterError(f"top_n must be a positive integer, got {top_n!r}.")
            if top_n > self.max_categories:
                raise InvalidParameterError(
                    f"top_n ({top_n}) exceeds max_categories limit ({self.max_categories})."
                )

        color_column = _validate_str_param(params, "color_column")
        if color_column is not None and color_column not in df.columns:
            raise InvalidColumnError(f"color_column '{color_column}' does not exist in dataset.")

        work_df = df.copy()

        if orientation == "v":
            if not x_column or not str(x_column).strip():
                raise InvalidParameterError("x_column is required for vertical bar chart.")
            if not y_column or not str(y_column).strip():
                raise InvalidParameterError("y_column is required for vertical bar chart.")
            if x_column not in work_df.columns:
                raise InvalidColumnError(f"x_column '{x_column}' does not exist in dataset.")
            if y_column not in work_df.columns:
                raise InvalidColumnError(f"y_column '{y_column}' does not exist in dataset.")

            work_df[y_column] = _coerce_numeric_series(work_df[y_column], y_column)
            cat_col, num_col = x_column, y_column
        else:
            if not x_column or not str(x_column).strip():
                raise InvalidParameterError("x_column (numeric measure) is required for horizontal bar chart.")
            if not y_column or not str(y_column).strip():
                raise InvalidParameterError("y_column (category) is required for horizontal bar chart.")
            if x_column not in work_df.columns:
                raise InvalidColumnError(f"x_column '{x_column}' does not exist in dataset.")
            if y_column not in work_df.columns:
                raise InvalidColumnError(f"y_column '{y_column}' does not exist in dataset.")

            work_df[x_column] = _coerce_numeric_series(work_df[x_column], x_column)
            cat_col, num_col = y_column, x_column

        # Missing value filtering: drop rows where cat_col or num_col is missing
        total_rows = len(df)
        valid_mask = work_df[cat_col].notna() & work_df[num_col].notna()
        work_df = work_df[valid_mask]
        rows_used = len(work_df)
        excluded_rows = total_rows - rows_used

        if rows_used == 0:
            raise EmptyDatasetError("No valid observations remaining for bar chart after excluding missing values.")

        # Category cardinality check on main categorical axis
        num_categories = work_df[cat_col].nunique(dropna=False)
        if num_categories > self.max_categories and top_n is None:
            raise DataLimitExceededError(
                f"Category column '{cat_col}' has {num_categories} distinct values, exceeding "
                f"max_categories limit ({self.max_categories}). Please specify 'top_n' in params "
                "or aggregate with SQL."
            )

        # Cardinality check on color_column
        if color_column is not None:
            num_colors = work_df[color_column].nunique(dropna=False)
            if num_colors > self.max_categories:
                raise DataLimitExceededError(
                    f"Color grouping column '{color_column}' has {num_colors} distinct categories, "
                    f"exceeding max_categories limit ({self.max_categories})."
                )

        # Apply sorting and top_n
        selection_method = "full"
        if sort is not None or top_n is not None:
            ascending = (sort == "asc")
            work_df = work_df.sort_values(by=num_col, ascending=ascending)

        if top_n is not None and len(work_df) > top_n:
            work_df = work_df.head(top_n)
            selection_method = f"top_{top_n}"

        rows_used = len(work_df)
        excluded_rows = total_rows - rows_used

        title = params.get("title") or f"{num_col} by {cat_col}"
        x_label = _validate_str_param(params, "x_label")
        y_label = _validate_str_param(params, "y_label")
        labels = {}
        if x_label:
            labels[x_column] = x_label
        if y_label:
            labels[y_column] = y_label

        fig = px.bar(
            work_df,
            x=x_column,
            y=y_column,
            color=color_column,
            orientation=orientation,
            title=title,
            labels=labels if labels else None,
        )

        metadata = {
            "x_column": x_column,
            "y_column": y_column,
            "orientation": orientation,
            "rows_used": rows_used,
            "total_rows": total_rows,
            "excluded_rows": excluded_rows,
            "categories_count": work_df[cat_col].nunique(dropna=False),
            "selection_method": selection_method,
            "title": title,
        }
        if color_column:
            metadata["color_column"] = color_column

        return fig, metadata

    def _generate_line(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        x_column: Optional[str],
        y_column: Optional[str],
        params: Dict[str, Any],
    ) -> Tuple[go.Figure, Dict[str, Any]]:
        if not x_column or not str(x_column).strip():
            raise InvalidParameterError("x_column is required for line chart.")
        if not y_column or not str(y_column).strip():
            raise InvalidParameterError("y_column is required for line chart.")
        if x_column not in df.columns:
            raise InvalidColumnError(f"x_column '{x_column}' does not exist in dataset.")
        if y_column not in df.columns:
            raise InvalidColumnError(f"y_column '{y_column}' does not exist in dataset.")

        color_column = _validate_str_param(params, "color_column")
        if color_column is not None and color_column not in df.columns:
            raise InvalidColumnError(f"color_column '{color_column}' does not exist in dataset.")

        sort_x = _validate_bool_param(params, "sort_x", True)
        markers = _validate_bool_param(params, "markers", True)

        work_df = df.copy()
        work_df[y_column] = _coerce_numeric_series(work_df[y_column], y_column)

        total_rows = len(df)
        valid_mask = work_df[x_column].notna() & work_df[y_column].notna()
        work_df = work_df[valid_mask]
        rows_used = len(work_df)
        excluded_rows = total_rows - rows_used

        if rows_used == 0:
            raise EmptyDatasetError("No valid observations remaining for line chart after excluding missing values.")

        if color_column is not None:
            num_colors = work_df[color_column].nunique(dropna=False)
            if num_colors > self.max_categories:
                raise DataLimitExceededError(
                    f"Color grouping column '{color_column}' has {num_colors} distinct categories, "
                    f"exceeding max_categories limit ({self.max_categories})."
                )

        if sort_x:
            try:
                dt_series = pd.to_datetime(work_df[x_column], errors="coerce")
                if dt_series.notna().sum() == len(work_df):
                    work_df["_sort_key"] = dt_series
                    work_df = work_df.sort_values(by="_sort_key").drop(columns=["_sort_key"])
                else:
                    work_df = work_df.sort_values(by=x_column)
            except Exception:
                work_df = work_df.sort_values(by=x_column)

        title = params.get("title") or f"{y_column} over {x_column}"
        x_label = _validate_str_param(params, "x_label")
        y_label = _validate_str_param(params, "y_label")
        labels = {}
        if x_label:
            labels[x_column] = x_label
        if y_label:
            labels[y_column] = y_label

        fig = px.line(
            work_df,
            x=x_column,
            y=y_column,
            color=color_column,
            markers=markers,
            title=title,
            labels=labels if labels else None,
        )

        metadata = {
            "x_column": x_column,
            "y_column": y_column,
            "rows_used": rows_used,
            "total_rows": total_rows,
            "excluded_rows": excluded_rows,
            "sort_x": sort_x,
            "markers": markers,
            "title": title,
        }
        if color_column:
            metadata["color_column"] = color_column

        return fig, metadata

    def _generate_scatter(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        x_column: Optional[str],
        y_column: Optional[str],
        params: Dict[str, Any],
    ) -> Tuple[go.Figure, Dict[str, Any]]:
        if not x_column or not str(x_column).strip():
            raise InvalidParameterError("x_column is required for scatter plot.")
        if not y_column or not str(y_column).strip():
            raise InvalidParameterError("y_column is required for scatter plot.")
        if x_column not in df.columns:
            raise InvalidColumnError(f"x_column '{x_column}' does not exist in dataset.")
        if y_column not in df.columns:
            raise InvalidColumnError(f"y_column '{y_column}' does not exist in dataset.")

        color_column = _validate_str_param(params, "color_column")
        if color_column is not None and color_column not in df.columns:
            raise InvalidColumnError(f"color_column '{color_column}' does not exist in dataset.")

        outlier_column = _validate_str_param(params, "outlier_column")
        if outlier_column is not None and outlier_column not in df.columns:
            raise InvalidColumnError(
                f"outlier_column '{outlier_column}' does not exist in dataset. "
                "Only explicitly supplied, verified outlier indicator columns are supported."
            )

        work_df = df.copy()
        work_df[x_column] = _coerce_numeric_series(work_df[x_column], x_column)
        work_df[y_column] = _coerce_numeric_series(work_df[y_column], y_column)

        total_rows = len(df)
        valid_mask = work_df[x_column].notna() & work_df[y_column].notna()
        work_df = work_df[valid_mask]
        rows_used = len(work_df)
        excluded_rows = total_rows - rows_used

        if rows_used == 0:
            raise EmptyDatasetError("No valid observations remaining for scatter plot after excluding missing values.")

        if color_column is not None:
            num_colors = work_df[color_column].nunique(dropna=False)
            if num_colors > self.max_categories:
                raise DataLimitExceededError(
                    f"Color grouping column '{color_column}' has {num_colors} distinct categories, "
                    f"exceeding max_categories limit ({self.max_categories})."
                )

        if outlier_column is not None:
            num_outliers = work_df[outlier_column].nunique(dropna=False)
            if num_outliers > self.max_categories:
                raise DataLimitExceededError(
                    f"Outlier indicator column '{outlier_column}' has {num_outliers} distinct values, "
                    f"exceeding max_categories limit ({self.max_categories}). "
                    "Outlier indicator must be a low-cardinality boolean or categorical column."
                )

        title = params.get("title") or f"{y_column} vs {x_column}"
        x_label = _validate_str_param(params, "x_label")
        y_label = _validate_str_param(params, "y_label")
        labels = {}
        if x_label:
            labels[x_column] = x_label
        if y_label:
            labels[y_column] = y_label

        effective_color = outlier_column if outlier_column is not None else color_column
        effective_symbol = outlier_column if outlier_column is not None else None

        fig = px.scatter(
            work_df,
            x=x_column,
            y=y_column,
            color=effective_color,
            symbol=effective_symbol,
            title=title,
            labels=labels if labels else None,
        )

        metadata = {
            "x_column": x_column,
            "y_column": y_column,
            "rows_used": rows_used,
            "total_rows": total_rows,
            "excluded_rows": excluded_rows,
            "title": title,
        }
        if color_column:
            metadata["color_column"] = color_column
        if outlier_column:
            metadata["outlier_column"] = outlier_column

        return fig, metadata

    def _generate_histogram(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        x_column: Optional[str],
        params: Dict[str, Any],
    ) -> Tuple[go.Figure, Dict[str, Any]]:
        if not x_column or not str(x_column).strip():
            raise InvalidParameterError("x_column is required for histogram.")
        if x_column not in df.columns:
            raise InvalidColumnError(f"x_column '{x_column}' does not exist in dataset.")

        color_column = _validate_str_param(params, "color_column")
        if color_column is not None and color_column not in df.columns:
            raise InvalidColumnError(f"color_column '{color_column}' does not exist in dataset.")

        nbins = params.get("nbins")
        if nbins is not None:
            if type(nbins) is not int or nbins <= 0:
                raise InvalidParameterError(f"nbins must be a positive integer, got {nbins!r}.")
            if nbins > self.max_bins:
                raise InvalidParameterError(
                    f"nbins ({nbins}) exceeds max_bins limit ({self.max_bins})."
                )

        work_df = df.copy()
        work_df[x_column] = _coerce_numeric_series(work_df[x_column], x_column)

        total_rows = len(df)
        valid_mask = work_df[x_column].notna()
        work_df = work_df[valid_mask]
        rows_used = len(work_df)
        excluded_rows = total_rows - rows_used

        if rows_used == 0:
            raise EmptyDatasetError("No valid observations remaining for histogram after excluding missing values.")

        if color_column is not None:
            num_colors = work_df[color_column].nunique(dropna=False)
            if num_colors > self.max_categories:
                raise DataLimitExceededError(
                    f"Color grouping column '{color_column}' has {num_colors} distinct categories, "
                    f"exceeding max_categories limit ({self.max_categories})."
                )

        title = params.get("title") or f"Distribution of {x_column}"
        x_label = _validate_str_param(params, "x_label")
        labels = {}
        if x_label:
            labels[x_column] = x_label

        fig = px.histogram(
            work_df,
            x=x_column,
            nbins=nbins,
            color=color_column,
            title=title,
            labels=labels if labels else None,
        )

        metadata = {
            "x_column": x_column,
            "rows_used": rows_used,
            "total_rows": total_rows,
            "excluded_rows": excluded_rows,
            "nbins": nbins,
            "title": title,
        }
        if color_column:
            metadata["color_column"] = color_column

        return fig, metadata

    def _generate_box(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        x_column: Optional[str],
        y_column: Optional[str],
        params: Dict[str, Any],
    ) -> Tuple[go.Figure, Dict[str, Any]]:
        if not y_column or not str(y_column).strip():
            raise InvalidParameterError("y_column is required for box plot.")
        if y_column not in df.columns:
            raise InvalidColumnError(f"y_column '{y_column}' does not exist in dataset.")

        if x_column is not None:
            if not str(x_column).strip() or x_column not in df.columns:
                raise InvalidColumnError(f"x_column '{x_column}' does not exist in dataset.")

        color_column = _validate_str_param(params, "color_column")
        if color_column is not None and color_column not in df.columns:
            raise InvalidColumnError(f"color_column '{color_column}' does not exist in dataset.")

        points = params.get("points", "outliers")
        if points is not False and points not in ("outliers", "all"):
            raise InvalidParameterError(
                f"points must be 'outliers', 'all', or False, got {points!r}."
            )

        work_df = df.copy()
        work_df[y_column] = _coerce_numeric_series(work_df[y_column], y_column)

        total_rows = len(df)
        if x_column is not None:
            valid_mask = work_df[y_column].notna() & work_df[x_column].notna()
        else:
            valid_mask = work_df[y_column].notna()

        work_df = work_df[valid_mask]
        rows_used = len(work_df)
        excluded_rows = total_rows - rows_used

        if rows_used == 0:
            raise EmptyDatasetError("No valid observations remaining for box plot after excluding missing values.")

        if x_column is not None:
            num_x_cats = work_df[x_column].nunique(dropna=False)
            if num_x_cats > self.max_categories:
                raise DataLimitExceededError(
                    f"Grouping column '{x_column}' has {num_x_cats} distinct categories, "
                    f"exceeding max_categories limit ({self.max_categories})."
                )

        if color_column is not None:
            num_colors = work_df[color_column].nunique(dropna=False)
            if num_colors > self.max_categories:
                raise DataLimitExceededError(
                    f"Color grouping column '{color_column}' has {num_colors} distinct categories, "
                    f"exceeding max_categories limit ({self.max_categories})."
                )

        title = params.get("title") or (
            f"Box Plot of {y_column} by {x_column}" if x_column else f"Box Plot of {y_column}"
        )
        x_label = _validate_str_param(params, "x_label")
        y_label = _validate_str_param(params, "y_label")
        labels = {}
        if x_column and x_label:
            labels[x_column] = x_label
        if y_label:
            labels[y_column] = y_label

        fig = px.box(
            work_df,
            x=x_column,
            y=y_column,
            color=color_column,
            points=points,
            title=title,
            labels=labels if labels else None,
        )

        metadata = {
            "x_column": x_column,
            "y_column": y_column,
            "rows_used": rows_used,
            "total_rows": total_rows,
            "excluded_rows": excluded_rows,
            "points": points,
            "title": title,
            "note": "Box plot displays median, IQR quartiles, and Tukey visual outliers (1.5 * IQR).",
        }
        if color_column:
            metadata["color_column"] = color_column

        return fig, metadata

    def _generate_heatmap(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        params: Dict[str, Any],
    ) -> Tuple[go.Figure, Dict[str, Any]]:
        columns = params.get("columns")
        method = _validate_str_param(params, "correlation_method", allowed=("pearson", "spearman")) or "pearson"
        annotated = _validate_bool_param(params, "annotated", True)
        is_precomputed = _validate_bool_param(params, "is_precomputed", False)

        work_df = df.copy()

        # Determine target numeric columns
        if columns is not None:
            if not isinstance(columns, (list, tuple)) or len(columns) < 2:
                raise InvalidParameterError("columns for heatmap must be a list of at least 2 column names.")
            if len(columns) > self.max_heatmap_columns:
                raise DataLimitExceededError(
                    f"Selected heatmap columns ({len(columns)}) exceeds max_heatmap_columns ({self.max_heatmap_columns})."
                )
            for c in columns:
                if not isinstance(c, str) or c not in work_df.columns:
                    raise InvalidColumnError(f"Column '{c}' does not exist in dataset.")
                work_df[c] = _coerce_numeric_series(work_df[c], c)
            target_cols = list(columns)
        else:
            # Auto-detect numeric columns
            potential_cols = []
            for c in work_df.columns:
                try:
                    work_df[c] = _coerce_numeric_series(work_df[c], c)
                    potential_cols.append(c)
                except InvalidColumnError:
                    continue

            if len(potential_cols) < 2:
                raise InvalidColumnError(
                    f"Heatmap requires at least 2 numeric columns; found {len(potential_cols)}."
                )
            if len(potential_cols) > self.max_heatmap_columns:
                raise DataLimitExceededError(
                    f"Dataset contains {len(potential_cols)} numeric columns, exceeding "
                    f"max_heatmap_columns ({self.max_heatmap_columns}). Please specify 'columns' in params."
                )
            target_cols = potential_cols

        total_rows = len(df)

        if is_precomputed:
            if len(work_df) != len(target_cols):
                raise InvalidParameterError(
                    f"Precomputed matrix requires square dimensions ({len(target_cols)}x{len(target_cols)}), "
                    f"but dataset has {len(work_df)} rows."
                )
            corr_display = work_df[target_cols].round(2)
            row_labels = list(corr_display.index)
            for col_candidate in work_df.columns:
                if col_candidate not in target_cols:
                    str_labels = [str(x) for x in work_df[col_candidate].tolist()]
                    if len(str_labels) == len(target_cols):
                        row_labels = str_labels
                        break
            x_cols = target_cols
            y_cols = row_labels
        else:
            corr_matrix = work_df[target_cols].corr(method=method)
            corr_display = corr_matrix.round(2)
            x_cols = list(corr_display.columns)
            y_cols = list(corr_display.index)

        # Build z_values and text_values, strictly preserving undefined values as None (JSON null)
        z_values = []
        text_values = []
        undefined_count = 0

        for row in corr_display.values.tolist():
            z_row = []
            text_row = []
            for val in row:
                if val is None or pd.isna(val) or math.isnan(val):
                    z_row.append(None)
                    text_row.append("N/A")
                    undefined_count += 1
                else:
                    float_val = float(val)
                    z_row.append(float_val)
                    text_row.append(f"{float_val:.2f}")
            z_values.append(z_row)
            text_values.append(text_row)

        title = params.get("title") or (
            f"Precomputed Correlation Matrix" if is_precomputed else f"{method.title()} Correlation Matrix"
        )

        trace = go.Heatmap(
            z=z_values,
            x=x_cols,
            y=y_cols,
            zmin=-1.0,
            zmax=1.0,
            colorscale="RdBu_r",
            text=text_values if annotated else None,
            texttemplate="%{text}" if annotated else None,
            colorbar=dict(title="Correlation"),
        )
        fig = go.Figure(data=[trace])
        fig.update_layout(title=title)

        metadata = {
            "columns": target_cols,
            "correlation_method": "precomputed" if is_precomputed else method,
            "rows_used": len(work_df),
            "total_rows": total_rows,
            "annotated": annotated,
            "is_precomputed": is_precomputed,
            "title": title,
            "pairwise_complete": not is_precomputed,
        }
        if undefined_count > 0:
            metadata["undefined_correlations_count"] = undefined_count
            metadata["warning"] = (
                f"{undefined_count} correlation value(s) are undefined (e.g. constant column with zero variance); "
                "preserved as null/N/A rather than zero."
            )

        return fig, metadata

    def _error_result(
        self,
        dataset_id: str,
        chart_type: str,
        error_type: str,
        error_message: str,
    ) -> Dict[str, Any]:
        """Construct standard structured error result."""
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "chart_type": chart_type,
            "error_type": error_type,
            "error_message": error_message,
        }
