"""Controlled Python analytical computation engine for DatasetStore query results.

This engine executes specific, allowlisted analytical operations (descriptive statistics,
correlation matrices, period-over-period growth rates, IQR outlier detection, and secondary
groupby aggregations) over in-memory datasets stored by the SQL tool.

Security & Architecture:
- This is a deterministic, controlled-dispatch engine, NOT a general-purpose code execution sandbox.
- Arbitrary code execution (`eval`, `exec`, dynamic imports) is strictly prohibited.
- Datasets are retrieved from an injected `DatasetStore` instance; no SQL queries are executed here.
"""

from __future__ import annotations

import copy
from datetime import date, datetime
from decimal import Decimal
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Union
import warnings

import numpy as np
import pandas as pd

from tools.dataset_store import DatasetNotFoundError, DatasetRecord, DatasetStore

SUPPORTED_OPERATIONS = frozenset(
    {
        "describe",
        "correlation",
        "growth_rate",
        "detect_outliers",
        "groupby",
    }
)

DEFAULT_MAX_PREVIEW_ROWS: int = 50
DEFAULT_MAX_TABLE_ROWS: int = 200
DEFAULT_MAX_CORR_COLUMNS: int = 50


class AnalysisEngineError(Exception):
    """Base exception for analysis engine errors."""


class InvalidOperationError(AnalysisEngineError, ValueError):
    """Raised when an unsupported operation is requested."""


class InvalidParameterError(AnalysisEngineError, ValueError):
    """Raised when operation parameters are invalid or missing."""


class InvalidColumnError(AnalysisEngineError, KeyError):
    """Raised when a requested column does not exist or has an incompatible data type."""


class EmptyDatasetError(AnalysisEngineError, ValueError):
    """Raised when an operation cannot be performed on an empty dataset."""


class DuplicatePeriodError(AnalysisEngineError, ValueError):
    """Raised when duplicate periods are detected in growth rate analysis without aggregation."""


def _clean_json_value(val: Any) -> Any:
    """Recursively convert NumPy scalars, NaNs, infinities, pd.NA/NaT, and timestamps to JSON-compatible Python primitives."""
    # Collections first to avoid ambiguous scalar comparisons
    if isinstance(val, (list, tuple, set, pd.Series, np.ndarray)):
        return [_clean_json_value(item) for item in val]
    if isinstance(val, dict):
        return {str(k): _clean_json_value(v) for k, v in val.items()}

    # Null / Missing scalars
    if val is None or val is pd.NA or val is pd.NaT:
        return None

    # Floats (standard + numpy float)
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)

    # Decimal values
    if isinstance(val, Decimal):
        if val.is_nan() or val.is_infinite():
            return None
        return float(val)

    # Booleans (check before integer since bool is a subclass of int in Python)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)

    # Integers (standard + numpy int)
    if isinstance(val, (int, np.integer)):
        return int(val)

    # Date / Time
    if isinstance(val, (pd.Timestamp, datetime, date)):
        if pd.isna(val):
            return None
        return val.isoformat()

    # Strings
    if isinstance(val, str):
        return val

    # Fallback missing check for pandas extension types
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass

    return val


def _check_numeric_series(series: pd.Series, col_name: str) -> pd.Series:
    """Validate that a Series is numeric, attempting numeric coercion if needed without silent failure.

    Raises:
        InvalidColumnError: If the series contains non-numeric data types.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series

    # If dtype is object, check if all non-null values are numbers
    non_nulls = series.dropna()
    if len(non_nulls) == 0:
        return pd.to_numeric(series, errors="coerce")

    converted = pd.to_numeric(series, errors="coerce")
    # If any non-null was coerced to NaN, it was an invalid string/object
    coerced_nans = converted.isna() & series.notna()
    if coerced_nans.any():
        sample_invalid = series[coerced_nans].iloc[0]
        raise InvalidColumnError(
            f"Column '{col_name}' contains non-numeric value '{sample_invalid}' (type: {type(sample_invalid).__name__})."
        )
    return converted


class PythonAnalysisEngine:
    """Executes structured analytical operations on datasets stored in a DatasetStore."""

    def __init__(
        self,
        dataset_store: DatasetStore,
        max_preview_rows: int = DEFAULT_MAX_PREVIEW_ROWS,
        max_table_rows: int = DEFAULT_MAX_TABLE_ROWS,
        max_corr_columns: int = DEFAULT_MAX_CORR_COLUMNS,
    ) -> None:
        """Initialize the analysis engine.

        Args:
            dataset_store: Injected DatasetStore containing query datasets.
            max_preview_rows: Upper bound on outlier observation previews.
            max_table_rows: Upper bound on rows returned by tabular outputs (growth, groupby).
            max_corr_columns: Upper bound on columns evaluated in correlation matrix.
        """
        if not isinstance(dataset_store, DatasetStore):
            raise TypeError(
                f"dataset_store must be a DatasetStore instance, got {type(dataset_store).__name__}."
            )

        for param_name, param_val in [
            ("max_preview_rows", max_preview_rows),
            ("max_table_rows", max_table_rows),
            ("max_corr_columns", max_corr_columns),
        ]:
            if isinstance(param_val, bool) or not isinstance(param_val, int) or param_val <= 0:
                raise ValueError(
                    f"{param_name} must be a strictly positive integer, got {param_val!r} "
                    f"(type: {type(param_val).__name__})."
                )

        self.dataset_store = dataset_store
        self.max_preview_rows = max_preview_rows
        self.max_table_rows = max_table_rows
        self.max_corr_columns = max_corr_columns

    def analyze(
        self,
        dataset_id: str,
        operation: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Dispatch an approved analytical operation on the designated dataset.

        Args:
            dataset_id: Unique dataset identifier in the DatasetStore.
            operation: Name of analytical operation (must be in SUPPORTED_OPERATIONS).
            params: Optional dictionary of operation-specific parameters.

        Returns:
            Standardized dictionary with `status`, `dataset_id`, `operation`, `result`/`error`,
            `truncated`, and `warnings`.
        """
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            return {
                "status": "error",
                "dataset_id": dataset_id,
                "operation": operation,
                "error_type": "InvalidParameterError",
                "error_message": "dataset_id must be a non-empty string.",
            }

        if not isinstance(operation, str) or operation not in SUPPORTED_OPERATIONS:
            return {
                "status": "error",
                "dataset_id": dataset_id,
                "operation": operation,
                "error_type": "InvalidOperationError",
                "error_message": (
                    f"Unsupported operation '{operation}'. Supported operations: "
                    f"{sorted(SUPPORTED_OPERATIONS)}"
                ),
            }

        if params is not None and not isinstance(params, dict):
            return {
                "status": "error",
                "dataset_id": dataset_id,
                "operation": operation,
                "error_type": "InvalidParameterError",
                "error_message": f"params must be a dictionary, got {type(params).__name__}.",
            }

        operation_params: Dict[str, Any] = params or {}

        # 1. Retrieve dataset
        try:
            record: DatasetRecord = self.dataset_store.get(dataset_id)
        except DatasetNotFoundError as exc:
            return {
                "status": "error",
                "dataset_id": dataset_id,
                "operation": operation,
                "error_type": "DatasetNotFoundError",
                "error_message": str(exc),
            }

        # 2. Build DataFrame
        try:
            df = pd.DataFrame(record.rows, columns=record.columns)
        except Exception as exc:
            return {
                "status": "error",
                "dataset_id": dataset_id,
                "operation": operation,
                "error_type": "DataFrameConstructionError",
                "error_message": f"Failed to construct DataFrame: {exc}",
            }

        # 3. Track truncation warnings
        warnings: List[str] = []
        is_truncated = bool(record.truncated)
        if is_truncated:
            warnings.append(
                "Dataset was truncated by SQL execution limits; analytical results reflect only the materialized sample."
            )

        # 4. Dispatch operation
        try:
            if operation == "describe":
                res = self._op_describe(df, operation_params)
            elif operation == "correlation":
                res = self._op_correlation(df, operation_params)
            elif operation == "growth_rate":
                res, extra_warnings = self._op_growth_rate(df, operation_params)
                warnings.extend(extra_warnings)
            elif operation == "detect_outliers":
                res = self._op_detect_outliers(df, operation_params)
            elif operation == "groupby":
                res, extra_warnings = self._op_groupby(df, operation_params)
                warnings.extend(extra_warnings)
            else:
                raise InvalidOperationError(f"Unhandled operation '{operation}'.")

            clean_res = _clean_json_value(res)
            return {
                "status": "success",
                "dataset_id": dataset_id,
                "operation": operation,
                "result": clean_res,
                "truncated": is_truncated,
                "warnings": warnings,
            }

        except AnalysisEngineError as err:
            return {
                "status": "error",
                "dataset_id": dataset_id,
                "operation": operation,
                "error_type": type(err).__name__,
                "error_message": str(err),
            }
        except Exception as err:
            return {
                "status": "error",
                "dataset_id": dataset_id,
                "operation": operation,
                "error_type": "InternalExecutionError",
                "error_message": f"Unexpected error during '{operation}': {err}",
            }

    # -------------------------------------------------------------------------
    # Operation A: Descriptive Statistics
    # -------------------------------------------------------------------------
    def _op_describe(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compute count, mean, sample std, min, 25%, median, 75%, max for numeric columns."""
        requested_cols = params.get("columns")

        if len(df) == 0 and requested_cols is None:
            return {}

        if requested_cols is not None:
            if not isinstance(requested_cols, (list, tuple)) or not requested_cols:
                raise InvalidParameterError("'columns' parameter in describe must be a non-empty list of strings.")
            for c in requested_cols:
                if not isinstance(c, str):
                    raise InvalidParameterError(f"Column name must be a string, got {type(c).__name__}.")
                if c not in df.columns:
                    raise InvalidColumnError(f"Column '{c}' does not exist in dataset.")
            target_cols = list(requested_cols)
        else:
            # Auto-detect only genuine numeric dtypes (do NOT treat numeric strings as numeric columns)
            target_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

        if not target_cols:
            if requested_cols is not None:
                raise InvalidColumnError("None of the specified columns are valid numeric columns.")
            return {}

        results: Dict[str, Any] = {}
        for col in target_cols:
            series = _check_numeric_series(df[col], col).dropna()
            count = len(series)

            if count == 0:
                results[col] = {
                    "count": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "25%": None,
                    "median": None,
                    "75%": None,
                    "max": None,
                }
                continue

            results[col] = {
                "count": count,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)) if count > 1 else None,
                "min": float(series.min()),
                "25%": float(series.quantile(0.25)),
                "median": float(series.median()),
                "75%": float(series.quantile(0.75)),
                "max": float(series.max()),
            }

        return results

    # -------------------------------------------------------------------------
    # Operation B: Correlation Analysis
    # -------------------------------------------------------------------------
    def _op_correlation(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compute Pearson correlation matrix across two or more numerical columns."""
        cols = params.get("columns")
        if not isinstance(cols, (list, tuple)) or len(cols) < 2:
            raise InvalidParameterError(
                "Correlation requires a 'columns' list containing at least 2 column names."
            )

        if len(cols) > self.max_corr_columns:
            raise InvalidParameterError(
                f"Requested {len(cols)} columns exceeds maximum allowed correlation limit ({self.max_corr_columns})."
            )

        for c in cols:
            if not isinstance(c, str):
                raise InvalidParameterError(f"Column name must be a string, got {type(c).__name__}.")
            if c not in df.columns:
                raise InvalidColumnError(f"Column '{c}' does not exist in dataset.")

        numeric_df = pd.DataFrame()
        for c in cols:
            numeric_df[c] = _check_numeric_series(df[c], c)

        if len(numeric_df) < 2:
            # Insufficient observations
            return {c1: {c2: None for c2 in cols} for c1 in cols}

        # Compute pairwise Pearson correlation
        corr_matrix = numeric_df.corr(method="pearson")
        output: Dict[str, Dict[str, Optional[float]]] = {}
        for c1 in cols:
            output[c1] = {}
            for c2 in cols:
                val = corr_matrix.loc[c1, c2]
                if pd.isna(val) or math.isnan(val) or math.isinf(val):
                    output[c1][c2] = None
                else:
                    output[c1][c2] = round(float(val), 6)

        return output

    # -------------------------------------------------------------------------
    # Operation C: Growth Rate Analysis
    # -------------------------------------------------------------------------
    def _op_growth_rate(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """Calculate period-over-period growth rates with explicit date validation, sorting, and standardized formula."""
        period_col = params.get("period_column")
        value_col = params.get("value_column")
        group_col = params.get("group_column")

        if not isinstance(period_col, str) or not period_col:
            raise InvalidParameterError("'period_column' parameter is required and must be a string.")
        if not isinstance(value_col, str) or not value_col:
            raise InvalidParameterError("'value_column' parameter is required and must be a string.")
        if period_col not in df.columns:
            raise InvalidColumnError(f"period_column '{period_col}' does not exist in dataset.")
        if value_col not in df.columns:
            raise InvalidColumnError(f"value_column '{value_col}' does not exist in dataset.")
        if group_col is not None:
            if not isinstance(group_col, str) or not group_col:
                raise InvalidParameterError("'group_column' must be a non-empty string when provided.")
            if group_col not in df.columns:
                raise InvalidColumnError(f"group_column '{group_col}' does not exist in dataset.")

        if len(df) == 0:
            return [], []

        # Validate period values: no missing or null values allowed
        period_series = df[period_col]
        if period_series.isna().any():
            raise InvalidParameterError(
                f"Missing or null period value found in period_column '{period_col}'. Period values must be non-null."
            )

        # Parse periods explicitly using pandas to_datetime
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            try:
                parsed_periods = pd.to_datetime(period_series, format="ISO8601", errors="coerce")
            except Exception:
                parsed_periods = pd.to_datetime(period_series, errors="coerce")

            if parsed_periods.isna().any():
                fallback = pd.to_datetime(period_series, errors="coerce")
                parsed_periods = parsed_periods.combine_first(fallback)

        if parsed_periods.isna().any():
            invalid_mask = parsed_periods.isna()
            invalid_sample = period_series[invalid_mask].iloc[0]
            raise InvalidParameterError(
                f"Invalid date/period value '{invalid_sample}' in period_column '{period_col}'. "
                "All periods must be valid dates, timestamps, or YYYY-MM period strings."
            )

        # Check for incompatible mixed temporal granularity (e.g. mixing YYYY-MM with YYYY-MM-DD)
        def _get_period_granularity(val: Any) -> str:
            s = str(val).strip()
            if re.match(r"^\d{4}[-/]\d{2}$", s):
                return "month"
            return "date_or_time"

        granularity_set = {_get_period_granularity(p) for p in period_series}
        if len(granularity_set) > 1:
            raise InvalidParameterError(
                f"Incompatible mixed temporal granularity detected in period_column '{period_col}'. "
                "Mixing monthly periods (e.g. 'YYYY-MM') with daily or timestamp dates is ambiguous for period-over-period growth rate."
            )

        # Validate numeric values
        numeric_values = _check_numeric_series(df[value_col], value_col)

        # Duplicate checking using normalized periods to prevent equivalent representations from bypassing detection
        if group_col is not None:
            dup_keys_df = pd.DataFrame({"_g": df[group_col], "_p": parsed_periods})
            duplicates = dup_keys_df.duplicated(subset=["_g", "_p"], keep=False)
        else:
            duplicates = parsed_periods.duplicated(keep=False)

        if duplicates.any():
            dup_idx = duplicates.idxmax()
            dup_period_str = str(period_series.loc[dup_idx])
            grp_info = f", group: '{df.loc[dup_idx, group_col]}'" if group_col else ""
            raise DuplicatePeriodError(
                f"Ambiguous duplicate periods found in period_column '{period_col}' "
                f"(example duplicate: period '{dup_period_str}'{grp_info}). "
                "Group or aggregate your data in SQL before computing growth rate."
            )

        # Temporary calculation DataFrame isolated from input df columns
        calc_df = pd.DataFrame(
            {
                "orig_idx": np.arange(len(df)),
                "norm_time": parsed_periods,
                "clean_val": numeric_values,
                "orig_period": period_series,
            }
        )
        if group_col is not None:
            calc_df["group_val"] = df[group_col]

        warn_list: List[str] = []
        total_records = len(calc_df)
        if total_records > self.max_table_rows:
            warn_list.append(
                f"Growth rate output was bounded to the first {self.max_table_rows} periods (out of {total_records} total)."
            )

        results: List[Dict[str, Any]] = []

        def _process_group_rows(sub: pd.DataFrame, g_val: Any = None) -> None:
            nonlocal results
            sorted_sub = sub.sort_values(by="norm_time")
            prev_val = None

            for _, row in sorted_sub.iterrows():
                if len(results) >= self.max_table_rows:
                    return

                cur_val = row["clean_val"]
                growth = None
                if prev_val is not None and not pd.isna(prev_val) and not pd.isna(cur_val):
                    # Zero denominator or non-finite baseline produces None
                    if prev_val == 0 or math.isnan(prev_val) or math.isinf(prev_val) or math.isnan(cur_val) or math.isinf(cur_val):
                        growth = None
                    else:
                        # Standard percentage change formula: (cur - prev) / prev
                        growth = (cur_val - prev_val) / prev_val

                item: Dict[str, Any] = {
                    "period": row["orig_period"],
                    "value": None if pd.isna(cur_val) else cur_val,
                    "growth_rate": None if growth is None or math.isnan(growth) or math.isinf(growth) else round(float(growth), 6),
                }
                if group_col is not None:
                    item["group"] = g_val
                results.append(item)
                prev_val = cur_val

        if group_col is not None:
            for g_val, grp in calc_df.groupby("group_val", sort=True, dropna=False):
                if len(results) >= self.max_table_rows:
                    break
                clean_g_val = None if (g_val is None or pd.isna(g_val)) else g_val
                _process_group_rows(grp, g_val=clean_g_val)
        else:
            _process_group_rows(calc_df)

        return results, warn_list

    # -------------------------------------------------------------------------
    # Operation D: Outlier Detection (IQR)
    # -------------------------------------------------------------------------
    def _op_detect_outliers(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect outliers using the Interquartile Range (IQR) method."""
        col = params.get("column")
        if not isinstance(col, str) or not col:
            raise InvalidParameterError("'column' parameter is required for detect_outliers.")
        if col not in df.columns:
            raise InvalidColumnError(f"Column '{col}' does not exist in dataset.")

        multiplier = params.get("iqr_multiplier", 1.5)
        if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)):
            raise InvalidParameterError(f"'iqr_multiplier' must be a positive number, got {multiplier!r}.")
        if math.isnan(multiplier) or math.isinf(multiplier) or multiplier <= 0:
            raise InvalidParameterError(f"'iqr_multiplier' must be positive and finite, got {multiplier!r}.")

        series = _check_numeric_series(df[col], col).dropna()
        count = len(series)

        if count < 4:
            # Insufficient data to construct meaningful IQR quartiles
            return {
                "column": col,
                "q1": None,
                "q3": None,
                "iqr": None,
                "lower_bound": None,
                "upper_bound": None,
                "iqr_multiplier": float(multiplier),
                "outlier_count": 0,
                "outliers_preview": [],
            }

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = float(q3 - q1)
        lower_bound = q1 - (multiplier * iqr)
        upper_bound = q3 + (multiplier * iqr)

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_series = series[outlier_mask]
        outlier_count = int(len(outlier_series))

        # Preview of outlier observations
        outliers_preview = []
        for idx, val in outlier_series.head(self.max_preview_rows).items():
            row_dict = df.loc[idx].to_dict()
            outliers_preview.append(
                {
                    "row_index": int(idx),
                    "value": float(val),
                    col: float(val),
                    "context": {k: row_dict[k] for k in row_dict if k != col},
                }
            )

        return {
            "column": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "iqr_multiplier": float(multiplier),
            "outlier_count": outlier_count,
            "outliers_preview": outliers_preview,
        }

    # -------------------------------------------------------------------------
    # Operation E: GroupBy Aggregation
    # -------------------------------------------------------------------------
    def _op_groupby(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """Perform secondary groupby aggregation over already-materialized dataset rows."""
        by = params.get("by")
        agg_col = params.get("agg_column")
        agg_func = params.get("agg_func")

        if isinstance(by, str):
            by_cols = [by]
        elif isinstance(by, (list, tuple)) and by:
            by_cols = list(by)
        else:
            raise InvalidParameterError("'by' must be a column name or a list of column names.")

        for c in by_cols:
            if not isinstance(c, str):
                raise InvalidParameterError(f"Group column must be a string, got {type(c).__name__}.")
            if c not in df.columns:
                raise InvalidColumnError(f"Grouping column '{c}' does not exist in dataset.")

        if not isinstance(agg_col, str) or not agg_col:
            raise InvalidParameterError("'agg_column' is required and must be a string.")
        if agg_col not in df.columns:
            raise InvalidColumnError(f"agg_column '{agg_col}' does not exist in dataset.")

        valid_funcs = {"sum", "mean", "count", "min", "max"}
        if not isinstance(agg_func, str) or agg_func.lower() not in valid_funcs:
            raise InvalidParameterError(
                f"agg_func '{agg_func}' is invalid. Supported aggregation functions: {sorted(valid_funcs)}."
            )
        func_name = agg_func.lower()

        if len(df) == 0:
            return [], []

        # Validate numeric aggregation if required
        if func_name in {"sum", "mean"}:
            numeric_agg = _check_numeric_series(df[agg_col], agg_col)
            work_df = df.copy()
            work_df[agg_col] = numeric_agg
        else:
            work_df = df.copy()

        # Group and aggregate, preserving null semantics
        grouped = work_df.groupby(by_cols, dropna=False)[agg_col].agg(func_name).reset_index()
        total_groups = len(grouped)

        warn_list: List[str] = []
        if total_groups > self.max_table_rows:
            warn_list.append(
                f"GroupBy output bounded to first {self.max_table_rows} groups (out of {total_groups} total)."
            )
            bounded_grouped = grouped.head(self.max_table_rows)
        else:
            bounded_grouped = grouped

        output_rows = []
        for _, row in bounded_grouped.iterrows():
            rec = {}
            for c in by_cols:
                rec[c] = row[c]
            rec[f"{agg_col}_{func_name}"] = row[agg_col]
            output_rows.append(rec)

        return output_rows, warn_list

