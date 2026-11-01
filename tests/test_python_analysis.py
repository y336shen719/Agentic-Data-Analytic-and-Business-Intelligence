"""Comprehensive tests for PythonAnalysisEngine including edge-case categories.

Coverage:
- Core 1..24 tests: Initialization, describe, correlation, growth_rate, detect_outliers, groupby, etc.
- Category A: GroupBy aggregation functions (sum, mean, count, min, max) and NULL semantics.
- Category B: Growth rate date granularity validation (rejection of mixed YYYY-MM vs daily).
- Category C: NULL group key handling in growth rate (dropna=False).
- Category D: Decimal, NumPy scalars, pd.NA/NaT, and nonstandard JSON serialization.
- Category E: Independent output limits (groupby, outliers preview, correlation dimension limit, config checks).
- Category F: Empty, all-NULL, single-observation, and nonfinite numerical data.
"""

from __future__ import annotations

from decimal import Decimal
import json
import math
import numpy as np
import pandas as pd
import pytest

from tools.dataset_store import DatasetStore
from tools.python_analysis import (
    PythonAnalysisEngine,
    SUPPORTED_OPERATIONS,
    _clean_json_value,
)


@pytest.fixture
def store() -> DatasetStore:
    return DatasetStore()


@pytest.fixture
def engine(store: DatasetStore) -> PythonAnalysisEngine:
    return PythonAnalysisEngine(dataset_store=store)


# =============================================================================
# Section 1: Core Engine Tests
# =============================================================================

def test_engine_initialization(store: DatasetStore):
    engine = PythonAnalysisEngine(store, max_preview_rows=10, max_table_rows=50)
    assert engine.dataset_store is store
    assert engine.max_preview_rows == 10
    assert engine.max_table_rows == 50

    with pytest.raises(TypeError, match="must be a DatasetStore instance"):
        PythonAnalysisEngine(dataset_store="not_a_store")  # type: ignore


def test_successful_dataset_retrieval(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save(
        {
            "status": "success",
            "columns": ["x", "y"],
            "rows": [[1, 2], [3, 4]],
            "row_count": 2,
        }
    )
    res = engine.analyze(d_id, "describe")
    assert res["status"] == "success"
    assert res["dataset_id"] == d_id
    assert res["operation"] == "describe"
    assert "x" in res["result"]
    assert "y" in res["result"]


def test_descriptive_statistics(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save(
        {
            "status": "success",
            "columns": ["val"],
            "rows": [[10.0], [20.0], [30.0], [40.0], [50.0]],
            "row_count": 5,
        }
    )
    res = engine.analyze(d_id, "describe", {"columns": ["val"]})
    assert res["status"] == "success"
    stats = res["result"]["val"]
    assert stats["count"] == 5
    assert stats["mean"] == 30.0
    assert stats["median"] == 30.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["25%"] == 20.0
    assert stats["75%"] == 40.0
    assert round(stats["std"], 2) == 15.81


def test_correlation_matrix_correctness(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save(
        {
            "status": "success",
            "columns": ["x", "y"],
            "rows": [[1, 2], [2, 4], [3, 6], [4, 8]],
            "row_count": 4,
        }
    )
    res = engine.analyze(d_id, "correlation", {"columns": ["x", "y"]})
    assert res["status"] == "success"
    matrix = res["result"]
    assert matrix["x"]["x"] == 1.0
    assert matrix["y"]["y"] == 1.0
    assert matrix["x"]["y"] == 1.0
    assert matrix["y"]["x"] == 1.0


def test_constant_column_correlation_behavior(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save(
        {
            "status": "success",
            "columns": ["x", "y"],
            "rows": [[1, 5], [2, 5], [3, 5]],
            "row_count": 3,
        }
    )
    res = engine.analyze(d_id, "correlation", {"columns": ["x", "y"]})
    assert res["status"] == "success"
    assert res["result"]["x"]["y"] is None
    assert res["result"]["y"]["x"] is None


def test_growth_rate_sorted_and_unsorted(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save(
        {
            "status": "success",
            "columns": ["period", "rev"],
            "rows": [["2023-03", 150], ["2023-01", 100], ["2023-02", 120]],
            "row_count": 3,
        }
    )
    res = engine.analyze(
        d_id,
        "growth_rate",
        {"period_column": "period", "value_column": "rev"},
    )
    assert res["status"] == "success"
    rows = res["result"]
    assert len(rows) == 3
    assert rows[0]["period"] == "2023-01" and rows[0]["growth_rate"] is None
    assert rows[1]["period"] == "2023-02" and rows[1]["growth_rate"] == 0.2
    assert rows[2]["period"] == "2023-03" and rows[2]["growth_rate"] == 0.25


def test_group_specific_growth_rates(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save(
        {
            "status": "success",
            "columns": ["cat", "period", "val"],
            "rows": [
                ["A", "2023-01", 100],
                ["A", "2023-02", 150],
                ["B", "2023-01", 200],
                ["B", "2023-02", 180],
            ],
            "row_count": 4,
        }
    )
    res = engine.analyze(
        d_id,
        "growth_rate",
        {"period_column": "period", "value_column": "val", "group_column": "cat"},
    )
    assert res["status"] == "success"
    results = res["result"]
    assert results[0]["group"] == "A" and results[0]["growth_rate"] is None
    assert results[1]["group"] == "A" and results[1]["growth_rate"] == 0.5
    assert results[2]["group"] == "B" and results[2]["growth_rate"] is None
    assert results[3]["group"] == "B" and results[3]["growth_rate"] == -0.1


def test_zero_denominator_growth_rate(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save(
        {
            "status": "success",
            "columns": ["period", "val"],
            "rows": [["2023-01", 0], ["2023-02", 100]],
            "row_count": 2,
        }
    )
    res = engine.analyze(
        d_id,
        "growth_rate",
        {"period_column": "period", "value_column": "val"},
    )
    assert res["status"] == "success"
    assert res["result"][1]["growth_rate"] is None


def test_duplicate_period_handling(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save(
        {
            "status": "success",
            "columns": ["period", "val"],
            "rows": [["2023-01", 100], ["2023-01", 120]],
            "row_count": 2,
        }
    )
    res = engine.analyze(
        d_id,
        "growth_rate",
        {"period_column": "period", "value_column": "val"},
    )
    assert res["status"] == "error"
    assert res["error_type"] == "DuplicatePeriodError"


def test_iqr_outlier_detection(store: DatasetStore, engine: PythonAnalysisEngine):
    rows = [[10.0], [10.2], [10.5], [10.8], [11.0], [11.2], [11.5], [11.8], [12.0], [1000.0]]
    d_id = store.save(
        {
            "status": "success",
            "columns": ["metric"],
            "rows": rows,
            "row_count": 10,
        }
    )
    res = engine.analyze(
        d_id,
        "detect_outliers",
        {"column": "metric", "iqr_multiplier": 1.5},
    )
    assert res["status"] == "success"
    outlier_info = res["result"]
    assert outlier_info["outlier_count"] == 1
    assert len(outlier_info["outliers_preview"]) == 1
    assert outlier_info["outliers_preview"][0]["value"] == 1000.0


def test_empty_dataset(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save(
        {
            "status": "success",
            "columns": ["a", "b"],
            "rows": [],
            "row_count": 0,
        }
    )
    res_desc = engine.analyze(d_id, "describe")
    assert res_desc["status"] == "success"
    assert res_desc["result"] == {}


def test_missing_or_invalid_dataset_ids(engine: PythonAnalysisEngine):
    res_unknown = engine.analyze("ds_nonexistent", "describe")
    assert res_unknown["status"] == "error"
    assert res_unknown["error_type"] == "DatasetNotFoundError"

    res_empty_id = engine.analyze("", "describe")
    assert res_empty_id["status"] == "error"
    assert res_empty_id["error_type"] == "InvalidParameterError"


def test_unknown_operation(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save({"status": "success", "columns": ["x"], "rows": [[1]]})
    res = engine.analyze(d_id, "unsupported_ml_cluster")
    assert res["status"] == "error"
    assert res["error_type"] == "InvalidOperationError"


def test_missing_column(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save({"status": "success", "columns": ["x"], "rows": [[1]]})
    res = engine.analyze(d_id, "detect_outliers", {"column": "nonexistent"})
    assert res["status"] == "error"
    assert res["error_type"] == "InvalidColumnError"


def test_nonnumeric_column_rejection(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save({"status": "success", "columns": ["txt"], "rows": [["hello"], ["world"]]})
    res = engine.analyze(d_id, "detect_outliers", {"column": "txt"})
    assert res["status"] == "error"
    assert res["error_type"] == "InvalidColumnError"


def test_missing_values_handling(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save(
        {
            "status": "success",
            "columns": ["x"],
            "rows": [[10.0], [None], [20.0]],
            "row_count": 3,
        }
    )
    res = engine.analyze(d_id, "describe", {"columns": ["x"]})
    assert res["status"] == "success"
    assert res["result"]["x"]["count"] == 2
    assert res["result"]["x"]["mean"] == 15.0


def test_invalid_parameters(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save({"status": "success", "columns": ["x"], "rows": [[1]]})
    res = engine.analyze(d_id, "describe", params="not-a-dict")  # type: ignore
    assert res["status"] == "error"
    assert res["error_type"] == "InvalidParameterError"

    res2 = engine.analyze(d_id, "detect_outliers", {"column": "x", "iqr_multiplier": -1.0})
    assert res2["status"] == "error"
    assert res2["error_type"] == "InvalidParameterError"


def test_truncated_dataset_warnings(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save(
        {
            "status": "success",
            "columns": ["x"],
            "rows": [[i] for i in range(1000)],
            "truncated": True,
            "row_count": 1000,
        }
    )
    res = engine.analyze(d_id, "describe")
    assert res["status"] == "success"
    assert res["truncated"] is True
    assert len(res["warnings"]) >= 1
    assert "truncated" in res["warnings"][0].lower()


def test_dataset_store_immutability(store: DatasetStore, engine: PythonAnalysisEngine):
    original_rows = [[1, 10], [2, 20]]
    d_id = store.save(
        {
            "status": "success",
            "columns": ["id", "val"],
            "rows": original_rows,
            "row_count": 2,
        }
    )
    engine.analyze(d_id, "describe")
    rec = store.get(d_id)
    assert rec.rows == [[1, 10], [2, 20]]


def test_isolation_between_independent_stores():
    store_a = DatasetStore()
    store_b = DatasetStore()
    engine_a = PythonAnalysisEngine(dataset_store=store_a)
    engine_b = PythonAnalysisEngine(dataset_store=store_b)

    d_id = store_a.save({"status": "success", "columns": ["x"], "rows": [[1]]})
    assert engine_a.analyze(d_id, "describe")["status"] == "success"
    assert engine_b.analyze(d_id, "describe")["status"] == "error"


# =============================================================================
# Category A — GroupBy Aggregation and NULL Semantics
# =============================================================================

@pytest.mark.parametrize("func_name,expected_a,expected_b", [
    ("sum", 30.0, 40.0),
    ("mean", 15.0, 40.0),
    ("count", 2, 1),
    ("min", 10.0, 40.0),
    ("max", 20.0, 40.0),
])
def test_groupby_all_aggregation_functions(store: DatasetStore, engine: PythonAnalysisEngine, func_name, expected_a, expected_b):
    d_id = store.save(
        {
            "status": "success",
            "columns": ["cat", "val"],
            "rows": [["A", 10.0], ["A", 20.0], ["B", 40.0]],
            "row_count": 3,
        }
    )
    res = engine.analyze(d_id, "groupby", {"by": "cat", "agg_column": "val", "agg_func": func_name})
    assert res["status"] == "success"
    rows = {r["cat"]: r[f"val_{func_name}"] for r in res["result"]}
    assert rows["A"] == expected_a
    assert rows["B"] == expected_b


def test_groupby_multiple_columns_and_null_keys(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save(
        {
            "status": "success",
            "columns": ["region", "channel", "revenue"],
            "rows": [
                ["North", "Online", 100],
                ["North", "Store", 50],
                [None, "Online", 200],      # NULL grouping key in region
                ["North", None, 75],         # NULL grouping key in channel
            ],
            "row_count": 4,
        }
    )
    res = engine.analyze(
        d_id,
        "groupby",
        {"by": ["region", "channel"], "agg_column": "revenue", "agg_func": "sum"},
    )
    assert res["status"] == "success"
    results = res["result"]
    assert len(results) == 4
    # Ensure None keys are preserved as None (JSON null)
    null_region_rows = [r for r in results if r["region"] is None]
    assert len(null_region_rows) == 1
    assert null_region_rows[0]["channel"] == "Online"
    assert null_region_rows[0]["revenue_sum"] == 200


def test_groupby_all_null_aggregation_group(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save(
        {
            "status": "success",
            "columns": ["grp", "val"],
            "rows": [["G1", None], ["G1", None], ["G2", 50.0]],
            "row_count": 3,
        }
    )
    res_mean = engine.analyze(d_id, "groupby", {"by": "grp", "agg_column": "val", "agg_func": "mean"})
    assert res_mean["status"] == "success"
    mean_map = {r["grp"]: r["val_mean"] for r in res_mean["result"]}
    assert mean_map["G1"] is None  # all-null group produces null
    assert mean_map["G2"] == 50.0

    res_count = engine.analyze(d_id, "groupby", {"by": "grp", "agg_column": "val", "agg_func": "count"})
    assert res_count["status"] == "success"
    count_map = {r["grp"]: r["val_count"] for r in res_count["result"]}
    assert count_map["G1"] == 0  # Pandas count counts non-nulls
    assert count_map["G2"] == 1


def test_groupby_unsupported_function_or_missing_column(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save({"status": "success", "columns": ["x", "y"], "rows": [[1, 2]]})
    res1 = engine.analyze(d_id, "groupby", {"by": "x", "agg_column": "y", "agg_func": "median"})
    assert res1["status"] == "error"
    assert res1["error_type"] == "InvalidParameterError"

    res2 = engine.analyze(d_id, "groupby", {"by": "x", "agg_column": "nonexistent", "agg_func": "sum"})
    assert res2["status"] == "error"
    assert res2["error_type"] == "InvalidColumnError"


# =============================================================================
# Category B — Growth Rate Date Granularity Validation
# =============================================================================

def test_growth_rate_mixed_granularity_rejected(store: DatasetStore, engine: PythonAnalysisEngine):
    # Mixed monthly (YYYY-MM) and daily (YYYY-MM-DD)
    d_id = store.save(
        {
            "status": "success",
            "columns": ["period", "rev"],
            "rows": [
                ["2018-01", 100],
                ["2018-01-15", 120],
                ["2018-02", 140],
            ],
            "row_count": 3,
        }
    )
    res = engine.analyze(d_id, "growth_rate", {"period_column": "period", "value_column": "rev"})
    assert res["status"] == "error"
    assert res["error_type"] == "InvalidParameterError"
    assert "Incompatible mixed temporal granularity" in res["error_message"]


def test_growth_rate_homogeneous_daily_dates(store: DatasetStore, engine: PythonAnalysisEngine):
    # Homogeneous daily dates
    d_id = store.save(
        {
            "status": "success",
            "columns": ["date", "sales"],
            "rows": [["2023-05-03", 30], ["2023-05-01", 10], ["2023-05-02", 20]],
            "row_count": 3,
        }
    )
    res = engine.analyze(d_id, "growth_rate", {"period_column": "date", "value_column": "sales"})
    assert res["status"] == "success"
    assert len(res["result"]) == 3
    assert res["result"][0]["period"] == "2023-05-01" and res["result"][0]["growth_rate"] is None
    assert res["result"][1]["period"] == "2023-05-02" and res["result"][1]["growth_rate"] == 1.0


def test_growth_rate_homogeneous_iso_timestamps(store: DatasetStore, engine: PythonAnalysisEngine):
    # Homogeneous ISO timestamps
    d_id = store.save(
        {
            "status": "success",
            "columns": ["ts", "v"],
            "rows": [["2023-01-01T12:00:00", 100], ["2023-01-01T13:00:00", 150]],
            "row_count": 2,
        }
    )
    res = engine.analyze(d_id, "growth_rate", {"period_column": "ts", "value_column": "v"})
    assert res["status"] == "success"
    assert res["result"][1]["growth_rate"] == 0.5


# =============================================================================
# Category C — NULL Group Handling in Growth Rate
# =============================================================================

def test_growth_rate_null_group_preservation(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save(
        {
            "status": "success",
            "columns": ["grp", "period", "val"],
            "rows": [
                ["A", "2023-01", 100.0],
                ["A", "2023-02", 120.0],
                [None, "2023-01", 50.0],
                [None, "2023-02", 75.0],
            ],
            "row_count": 4,
        }
    )
    res = engine.analyze(
        d_id,
        "growth_rate",
        {"period_column": "period", "value_column": "val", "group_column": "grp"},
    )
    assert res["status"] == "success"
    rows = res["result"]
    assert len(rows) == 4

    # Group A
    a_rows = [r for r in rows if r["group"] == "A"]
    assert len(a_rows) == 2
    assert a_rows[1]["growth_rate"] == 0.2

    # NULL Group (must be preserved with group: null)
    null_rows = [r for r in rows if r["group"] is None]
    assert len(null_rows) == 2
    assert null_rows[0]["period"] == "2023-01" and null_rows[0]["growth_rate"] is None
    assert null_rows[1]["period"] == "2023-02" and null_rows[1]["growth_rate"] == 0.5

    # Full JSON serialization with allow_nan=False
    dumped = json.dumps(res, allow_nan=False)
    assert '"group": null' in dumped


# =============================================================================
# Category D — Decimal and Nonstandard JSON Types
# =============================================================================

def test_decimal_support_in_analysis(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save(
        {
            "status": "success",
            "columns": ["price", "tax"],
            "rows": [[Decimal("10.50"), Decimal("1.50")], [Decimal("20.00"), Decimal("3.00")]],
            "row_count": 2,
        }
    )
    res = engine.analyze(d_id, "describe", {"columns": ["price"]})
    assert res["status"] == "success"
    assert res["result"]["price"]["mean"] == 15.25
    assert json.dumps(res, allow_nan=False)


def test_clean_json_value_exhaustive():
    data = {
        "decimal": Decimal("99.99"),
        "nan_decimal": Decimal("NaN"),
        "inf_decimal": Decimal("Infinity"),
        "f32": np.float32(1.5),
        "f64": np.float64(2.5),
        "i64": np.int64(100),
        "b_val": np.bool_(False),
        "pd_na": pd.NA,
        "pd_nat": pd.NaT,
        "np_nan": np.nan,
        "pos_inf": float("inf"),
        "neg_inf": float("-inf"),
        "ts": pd.Timestamp("2023-01-01T00:00:00"),
        "nested_dict": {"inner_dec": Decimal("5.5"), "inner_inf": float("inf")},
        "nested_list": [Decimal("1.1"), pd.NA, [float("-inf"), 42]],
    }
    cleaned = _clean_json_value(data)
    assert cleaned["decimal"] == 99.99
    assert cleaned["nan_decimal"] is None
    assert cleaned["inf_decimal"] is None
    assert cleaned["f32"] == 1.5 and isinstance(cleaned["f32"], float)
    assert cleaned["f64"] == 2.5 and isinstance(cleaned["f64"], float)
    assert cleaned["i64"] == 100 and isinstance(cleaned["i64"], int)
    assert cleaned["b_val"] is False and isinstance(cleaned["b_val"], bool)
    assert cleaned["pd_na"] is None
    assert cleaned["pd_nat"] is None
    assert cleaned["np_nan"] is None
    assert cleaned["pos_inf"] is None
    assert cleaned["neg_inf"] is None
    assert cleaned["ts"] == "2023-01-01T00:00:00"
    assert cleaned["nested_dict"]["inner_dec"] == 5.5
    assert cleaned["nested_dict"]["inner_inf"] is None
    assert cleaned["nested_list"] == [1.1, None, [None, 42]]

    # Ensure json.dumps passes cleanly without any NaN or invalid token
    dumped = json.dumps(cleaned, allow_nan=False)
    assert isinstance(dumped, str)


# =============================================================================
# Category E — Independent Output Limits Tests
# =============================================================================

def test_groupby_output_limit_enforced(store: DatasetStore):
    engine = PythonAnalysisEngine(store, max_table_rows=3)
    d_id = store.save(
        {
            "status": "success",
            "columns": ["grp", "val"],
            "rows": [[f"G_{i}", i] for i in range(10)],  # 10 groups
            "row_count": 10,
        }
    )
    res = engine.analyze(d_id, "groupby", {"by": "grp", "agg_column": "val", "agg_func": "sum"})
    assert res["status"] == "success"
    assert len(res["result"]) == 3
    assert any("bounded to first 3 groups (out of 10 total)" in w for w in res["warnings"])


def test_outlier_preview_limit_enforced(store: DatasetStore):
    engine = PythonAnalysisEngine(store, max_preview_rows=2)
    # 10 items around 10.0, 5 extreme outliers
    normal_rows = [[10.0 + (i * 0.1)] for i in range(15)]
    outliers = [[1000.0], [2000.0], [3000.0], [4000.0], [5000.0]]
    d_id = store.save(
        {
            "status": "success",
            "columns": ["val"],
            "rows": normal_rows + outliers,
            "row_count": 20,
        }
    )
    res = engine.analyze(d_id, "detect_outliers", {"column": "val"})
    assert res["status"] == "success"
    # outlier_count reflects full computation
    assert res["result"]["outlier_count"] == 5
    # preview is bounded to max_preview_rows
    assert len(res["result"]["outliers_preview"]) == 2


def test_correlation_dimension_limit(store: DatasetStore):
    engine = PythonAnalysisEngine(store, max_corr_columns=3)
    d_id = store.save(
        {
            "status": "success",
            "columns": ["a", "b", "c", "d"],
            "rows": [[1, 2, 3, 4], [2, 3, 4, 5]],
            "row_count": 2,
        }
    )
    res = engine.analyze(d_id, "correlation", {"columns": ["a", "b", "c", "d"]})
    assert res["status"] == "error"
    assert res["error_type"] == "InvalidParameterError"
    assert "exceeds maximum allowed correlation limit (3)" in res["error_message"]

    res_ok = engine.analyze(d_id, "correlation", {"columns": ["a", "b", "c"]})
    assert res_ok["status"] == "success"


@pytest.mark.parametrize("invalid_val", [0, -1, 1.5, True, False, "10", None])
def test_engine_configuration_validation_rejections(store: DatasetStore, invalid_val):
    with pytest.raises(ValueError, match="must be a strictly positive integer"):
        PythonAnalysisEngine(store, max_preview_rows=invalid_val)
    with pytest.raises(ValueError, match="must be a strictly positive integer"):
        PythonAnalysisEngine(store, max_table_rows=invalid_val)
    with pytest.raises(ValueError, match="must be a strictly positive integer"):
        PythonAnalysisEngine(store, max_corr_columns=invalid_val)


# =============================================================================
# Category F — Empty, All-NULL, and Nonfinite Numerical Data
# =============================================================================

def test_all_null_numeric_column_statistics(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save(
        {
            "status": "success",
            "columns": ["all_null"],
            "rows": [[None], [None], [None]],
            "row_count": 3,
        }
    )
    res = engine.analyze(d_id, "describe", {"columns": ["all_null"]})
    assert res["status"] == "success"
    stats = res["result"]["all_null"]
    assert stats["count"] == 0
    assert stats["mean"] is None
    assert stats["std"] is None
    assert stats["min"] is None
    assert stats["max"] is None
    assert json.dumps(res, allow_nan=False)


def test_single_observation_statistics(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save(
        {
            "status": "success",
            "columns": ["val"],
            "rows": [[42.0]],
            "row_count": 1,
        }
    )
    res = engine.analyze(d_id, "describe", {"columns": ["val"]})
    assert res["status"] == "success"
    stats = res["result"]["val"]
    assert stats["count"] == 1
    assert stats["mean"] == 42.0
    assert stats["std"] is None  # sample std undefined for N=1
    assert stats["min"] == 42.0
    assert stats["max"] == 42.0


def test_nonfinite_values_in_statistics(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save(
        {
            "status": "success",
            "columns": ["val"],
            "rows": [[10.0], [float("nan")], [20.0]],
            "row_count": 3,
        }
    )
    res = engine.analyze(d_id, "describe", {"columns": ["val"]})
    assert res["status"] == "success"
    stats = res["result"]["val"]
    assert stats["count"] == 2
    assert stats["mean"] == 15.0
    assert json.dumps(res, allow_nan=False)


def test_insufficient_data_for_iqr_outliers(store: DatasetStore, engine: PythonAnalysisEngine):
    # Less than 4 observations
    d_id = store.save(
        {
            "status": "success",
            "columns": ["val"],
            "rows": [[10.0], [20.0], [30.0]],
            "row_count": 3,
        }
    )
    res = engine.analyze(d_id, "detect_outliers", {"column": "val"})
    assert res["status"] == "success"
    outlier_info = res["result"]
    assert outlier_info["outlier_count"] == 0
    assert outlier_info["q1"] is None
    assert outlier_info["q3"] is None
    assert outlier_info["iqr"] is None
    assert outlier_info["outliers_preview"] == []


def test_growth_rate_nonfinite_values(store: DatasetStore, engine: PythonAnalysisEngine):
    d_id = store.save(
        {
            "status": "success",
            "columns": ["period", "val"],
            "rows": [["2023-01", 100.0], ["2023-02", None], ["2023-03", 150.0]],
            "row_count": 3,
        }
    )
    res = engine.analyze(d_id, "growth_rate", {"period_column": "period", "value_column": "val"})
    assert res["status"] == "success"
    results = res["result"]
    assert results[0]["growth_rate"] is None
    assert results[1]["growth_rate"] is None  # missing current value
    assert results[2]["growth_rate"] is None  # missing previous value
