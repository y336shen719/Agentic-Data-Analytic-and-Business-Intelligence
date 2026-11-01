"""Comprehensive unit and integration tests for ChartGenerationEngine (Step 6.1).

Verifies:
1. Chart Type Coverage (Bar, Line, Scatter, Histogram, Box, Heatmap).
2. Input validation & fail-closed error handling.
3. Resource limits (rows, categories, heatmap columns, figure bytes, title length).
4. Data integrity, immutability, and chronological order preservation.
5. Compatibility with PythonAnalysisEngine workflows.
6. Figure JSON reconstruction and Plotly validity.
"""

from __future__ import annotations

import base64
import copy
from decimal import Decimal
import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pytest


def _extract_trace_values(val: Any) -> list:
    """Extract coordinate values from Plotly trace, supporting both plain lists and Plotly 7 binary-encoded bdata dicts."""
    if isinstance(val, dict) and "bdata" in val:
        raw_bytes = base64.b64decode(val["bdata"])
        dtype = np.dtype(val.get("dtype", "f8"))
        return np.frombuffer(raw_bytes, dtype=dtype).tolist()
    return list(val)

from tools.chart_generation import (
    DEFAULT_MAX_CATEGORIES,
    DEFAULT_MAX_CHART_ROWS,
    DEFAULT_MAX_FIGURE_BYTES,
    DEFAULT_MAX_HEATMAP_COLUMNS,
    SUPPORTED_CHART_TYPES,
    ChartGenerationEngine,
    DataLimitExceededError,
    EmptyDatasetError,
    InvalidChartTypeError,
    InvalidColumnError,
    InvalidParameterError,
    analysis_result_to_dataset_payload,
)
from tools.dataset_store import DatasetNotFoundError, DatasetStore
from tools.python_analysis import PythonAnalysisEngine


# ==============================================================================
# Test Fixtures & Helpers
# ==============================================================================


@pytest.fixture
def empty_store() -> DatasetStore:
    return DatasetStore()


@pytest.fixture
def store_with_sales_data(empty_store: DatasetStore) -> Tuple[DatasetStore, str]:
    """Store with sample e-commerce sales dataset."""
    sql_result = {
        "status": "success",
        "columns": ["order_date", "product_category", "state", "price", "freight_value", "review_score", "is_outlier"],
        "rows": [
            ["2023-01-15", "health_beauty", "SP", 50.0, 10.0, 5, False],
            ["2023-02-15", "health_beauty", "RJ", 75.0, 15.0, 4, False],
            ["2023-03-15", "watches_gifts", "SP", 150.0, 20.0, 5, False],
            ["2023-04-15", "watches_gifts", "MG", 200.0, 25.0, 3, False],
            ["2023-05-15", "bed_bath_table", "SP", 80.0, 12.0, 4, False],
            ["2023-06-15", "bed_bath_table", "RJ", 90.0, 14.0, 5, False],
            ["2023-07-15", "electronics", "SP", 500.0, 50.0, 1, True],  # Outlier
        ],
        "row_count": 7,
    }
    dataset_id = empty_store.save(sql_result, query="SELECT * FROM orders")
    return empty_store, dataset_id


@pytest.fixture
def engine(store_with_sales_data: Tuple[DatasetStore, str]) -> ChartGenerationEngine:
    store, _ = store_with_sales_data
    return ChartGenerationEngine(dataset_store=store)


# ==============================================================================
# 1. Engine Initialization & Configuration Tests
# ==============================================================================


def test_engine_initialization_defaults(empty_store: DatasetStore):
    engine = ChartGenerationEngine(dataset_store=empty_store)
    assert engine.dataset_store is empty_store
    assert engine.max_chart_rows == DEFAULT_MAX_CHART_ROWS
    assert engine.max_categories == DEFAULT_MAX_CATEGORIES
    assert engine.max_heatmap_columns == DEFAULT_MAX_HEATMAP_COLUMNS
    assert engine.max_figure_bytes == DEFAULT_MAX_FIGURE_BYTES


def test_engine_initialization_invalid_store():
    with pytest.raises(TypeError, match="must be a DatasetStore instance"):
        ChartGenerationEngine(dataset_store=None)  # type: ignore
    with pytest.raises(TypeError, match="must be a DatasetStore instance"):
        ChartGenerationEngine(dataset_store="not_a_store")  # type: ignore


@pytest.mark.parametrize(
    "param_name,invalid_val",
    [
        ("max_chart_rows", 0),
        ("max_chart_rows", -10),
        ("max_chart_rows", True),
        ("max_chart_rows", "5000"),
        ("max_categories", 0),
        ("max_categories", -1),
        ("max_categories", False),
        ("max_heatmap_columns", -5),
        ("max_figure_bytes", 0),
        ("max_title_length", -1),
        ("max_bins", 0),
    ],
)
def test_engine_initialization_invalid_parameters(empty_store: DatasetStore, param_name: str, invalid_val: Any):
    kwargs = {param_name: invalid_val}
    with pytest.raises(ValueError, match="must be a strictly positive integer"):
        ChartGenerationEngine(dataset_store=empty_store, **kwargs)


# ==============================================================================
# 2. Chart Type Coverage Tests (Six Types)
# ==============================================================================


def test_generate_bar_chart_vertical(store_with_sales_data: Tuple[DatasetStore, str]):
    store, ds_id = store_with_sales_data
    engine = ChartGenerationEngine(dataset_store=store)

    result = engine.generate(
        dataset_id=ds_id,
        chart_type="bar",
        x_column="product_category",
        y_column="price",
        params={"title": "Category Sales", "sort": "desc"},
    )

    assert result["status"] == "success"
    assert result["chart_type"] == "bar"
    assert result["dataset_id"] == ds_id
    assert result["title"] == "Category Sales"
    assert "figure" in result
    # Check Plotly reconstruction
    fig = pio.from_json(json.dumps(result["figure"]))
    assert isinstance(fig, go.Figure)
    assert result["metadata"]["x_column"] == "product_category"
    assert result["metadata"]["y_column"] == "price"
    assert result["metadata"]["orientation"] == "v"
    assert result["metadata"]["rows_used"] == 7


def test_generate_bar_chart_horizontal(store_with_sales_data: Tuple[DatasetStore, str]):
    store, ds_id = store_with_sales_data
    engine = ChartGenerationEngine(dataset_store=store)

    result = engine.generate(
        dataset_id=ds_id,
        chart_type="bar",
        x_column="price",
        y_column="product_category",
        params={"orientation": "h", "sort": "asc"},
    )

    assert result["status"] == "success"
    assert result["metadata"]["orientation"] == "h"
    fig = pio.from_json(json.dumps(result["figure"]))
    assert isinstance(fig, go.Figure)


def test_generate_bar_chart_top_n(store_with_sales_data: Tuple[DatasetStore, str]):
    store, ds_id = store_with_sales_data
    engine = ChartGenerationEngine(dataset_store=store)

    result = engine.generate(
        dataset_id=ds_id,
        chart_type="bar",
        x_column="product_category",
        y_column="price",
        params={"top_n": 3},
    )

    assert result["status"] == "success"
    assert result["metadata"]["rows_used"] == 3
    assert result["metadata"]["selection_method"] == "top_3"


def test_generate_line_chart_chronological(store_with_sales_data: Tuple[DatasetStore, str]):
    store, ds_id = store_with_sales_data
    engine = ChartGenerationEngine(dataset_store=store)

    result = engine.generate(
        dataset_id=ds_id,
        chart_type="line",
        x_column="order_date",
        y_column="price",
        params={"sort_x": True, "markers": True},
    )

    assert result["status"] == "success"
    assert result["chart_type"] == "line"
    assert result["metadata"]["sort_x"] is True
    assert result["metadata"]["markers"] is True
    fig = pio.from_json(json.dumps(result["figure"]))
    assert isinstance(fig, go.Figure)


def test_generate_scatter_plot(store_with_sales_data: Tuple[DatasetStore, str]):
    store, ds_id = store_with_sales_data
    engine = ChartGenerationEngine(dataset_store=store)

    result = engine.generate(
        dataset_id=ds_id,
        chart_type="scatter",
        x_column="price",
        y_column="freight_value",
        params={"color_column": "state"},
    )

    assert result["status"] == "success"
    assert result["chart_type"] == "scatter"
    assert result["metadata"]["color_column"] == "state"
    fig = pio.from_json(json.dumps(result["figure"]))
    assert isinstance(fig, go.Figure)


def test_generate_scatter_plot_with_outlier_column(store_with_sales_data: Tuple[DatasetStore, str]):
    store, ds_id = store_with_sales_data
    engine = ChartGenerationEngine(dataset_store=store)

    result = engine.generate(
        dataset_id=ds_id,
        chart_type="scatter",
        x_column="price",
        y_column="freight_value",
        params={"outlier_column": "is_outlier"},
    )

    assert result["status"] == "success"
    assert result["metadata"]["outlier_column"] == "is_outlier"
    fig = pio.from_json(json.dumps(result["figure"]))
    assert isinstance(fig, go.Figure)


def test_generate_histogram(store_with_sales_data: Tuple[DatasetStore, str]):
    store, ds_id = store_with_sales_data
    engine = ChartGenerationEngine(dataset_store=store)

    result = engine.generate(
        dataset_id=ds_id,
        chart_type="histogram",
        x_column="price",
        params={"nbins": 5, "title": "Price Distribution"},
    )

    assert result["status"] == "success"
    assert result["chart_type"] == "histogram"
    assert result["metadata"]["nbins"] == 5
    fig = pio.from_json(json.dumps(result["figure"]))
    assert isinstance(fig, go.Figure)


def test_generate_box_plot(store_with_sales_data: Tuple[DatasetStore, str]):
    store, ds_id = store_with_sales_data
    engine = ChartGenerationEngine(dataset_store=store)

    # Box plot grouped by state
    result = engine.generate(
        dataset_id=ds_id,
        chart_type="box",
        x_column="state",
        y_column="price",
        params={"points": "outliers"},
    )

    assert result["status"] == "success"
    assert result["chart_type"] == "box"
    assert result["metadata"]["points"] == "outliers"
    fig = pio.from_json(json.dumps(result["figure"]))
    assert isinstance(fig, go.Figure)


def test_generate_heatmap_auto_columns(store_with_sales_data: Tuple[DatasetStore, str]):
    store, ds_id = store_with_sales_data
    engine = ChartGenerationEngine(dataset_store=store)

    result = engine.generate(
        dataset_id=ds_id,
        chart_type="heatmap",
        params={"correlation_method": "pearson", "annotated": True},
    )

    assert result["status"] == "success"
    assert result["chart_type"] == "heatmap"
    assert result["metadata"]["correlation_method"] == "pearson"
    assert set(result["metadata"]["columns"]).issubset({"price", "freight_value", "review_score", "is_outlier"})
    fig = pio.from_json(json.dumps(result["figure"]))
    assert isinstance(fig, go.Figure)


def test_generate_heatmap_explicit_columns(store_with_sales_data: Tuple[DatasetStore, str]):
    store, ds_id = store_with_sales_data
    engine = ChartGenerationEngine(dataset_store=store)

    result = engine.generate(
        dataset_id=ds_id,
        chart_type="heatmap",
        params={"columns": ["price", "freight_value"], "correlation_method": "spearman"},
    )

    assert result["status"] == "success"
    assert result["metadata"]["columns"] == ["price", "freight_value"]
    assert result["metadata"]["correlation_method"] == "spearman"


# ==============================================================================
# 3. Input Validation & Error Handling Tests
# ==============================================================================


def test_validation_missing_dataset_id(engine: ChartGenerationEngine):
    result = engine.generate(dataset_id="", chart_type="bar", x_column="cat", y_column="num")
    assert result["status"] == "error"
    assert result["error_type"] == "InvalidParameterError"


def test_validation_dataset_not_found(engine: ChartGenerationEngine):
    result = engine.generate(dataset_id="ds_nonexistent", chart_type="bar", x_column="cat", y_column="num")
    assert result["status"] == "error"
    assert result["error_type"] == "DatasetNotFoundError"


def test_validation_unsupported_chart_type(engine: ChartGenerationEngine, store_with_sales_data: Tuple[DatasetStore, str]):
    _, ds_id = store_with_sales_data
    result = engine.generate(dataset_id=ds_id, chart_type="pie", x_column="cat", y_column="num")
    assert result["status"] == "error"
    assert result["error_type"] == "InvalidChartTypeError"
    assert "pie" in result["error_message"]


def test_validation_empty_dataset(empty_store: DatasetStore):
    sql_result = {"status": "success", "columns": ["x", "y"], "rows": [], "row_count": 0}
    empty_id = empty_store.save(sql_result)
    engine = ChartGenerationEngine(dataset_store=empty_store)

    result = engine.generate(dataset_id=empty_id, chart_type="bar", x_column="x", y_column="y")
    assert result["status"] == "error"
    assert result["error_type"] == "EmptyDatasetError"


def test_validation_missing_required_columns(engine: ChartGenerationEngine, store_with_sales_data: Tuple[DatasetStore, str]):
    _, ds_id = store_with_sales_data
    # Bar chart missing x_column
    res1 = engine.generate(dataset_id=ds_id, chart_type="bar", x_column=None, y_column="price")
    assert res1["status"] == "error"
    assert res1["error_type"] == "InvalidParameterError"

    # Line chart missing y_column
    res2 = engine.generate(dataset_id=ds_id, chart_type="line", x_column="order_date", y_column=None)
    assert res2["status"] == "error"
    assert res2["error_type"] == "InvalidParameterError"


def test_validation_nonexistent_columns(engine: ChartGenerationEngine, store_with_sales_data: Tuple[DatasetStore, str]):
    _, ds_id = store_with_sales_data
    res = engine.generate(dataset_id=ds_id, chart_type="scatter", x_column="nonexistent", y_column="price")
    assert res["status"] == "error"
    assert res["error_type"] == "InvalidColumnError"
    assert "nonexistent" in res["error_message"]


def test_validation_incompatible_data_type_for_numeric_axis(engine: ChartGenerationEngine, store_with_sales_data: Tuple[DatasetStore, str]):
    _, ds_id = store_with_sales_data
    # product_category is string, cannot be y_column in vertical bar chart
    res = engine.generate(dataset_id=ds_id, chart_type="bar", x_column="state", y_column="product_category")
    assert res["status"] == "error"
    assert res["error_type"] == "InvalidColumnError"


def test_validation_invalid_parameters(engine: ChartGenerationEngine, store_with_sales_data: Tuple[DatasetStore, str]):
    _, ds_id = store_with_sales_data
    # Invalid sort option
    res1 = engine.generate(
        dataset_id=ds_id, chart_type="bar", x_column="state", y_column="price", params={"sort": "random"}
    )
    assert res1["status"] == "error"
    assert res1["error_type"] == "InvalidParameterError"

    # Invalid orientation
    res2 = engine.generate(
        dataset_id=ds_id, chart_type="bar", x_column="state", y_column="price", params={"orientation": "diagonal"}
    )
    assert res2["status"] == "error"
    assert res2["error_type"] == "InvalidParameterError"

    # Invalid box points option
    res3 = engine.generate(
        dataset_id=ds_id, chart_type="box", y_column="price", params={"points": "unsupported"}
    )
    assert res3["status"] == "error"
    assert res3["error_type"] == "InvalidParameterError"

    # Invalid nbins
    res4 = engine.generate(
        dataset_id=ds_id, chart_type="histogram", x_column="price", params={"nbins": -5}
    )
    assert res4["status"] == "error"
    assert res4["error_type"] == "InvalidParameterError"


def test_validation_invalid_outlier_column(engine: ChartGenerationEngine, store_with_sales_data: Tuple[DatasetStore, str]):
    _, ds_id = store_with_sales_data
    res = engine.generate(
        dataset_id=ds_id,
        chart_type="scatter",
        x_column="price",
        y_column="freight_value",
        params={"outlier_column": "fabricated_outliers"},
    )
    assert res["status"] == "error"
    assert res["error_type"] == "InvalidColumnError"
    assert "fabricated_outliers" in res["error_message"]


# ==============================================================================
# 4. Resource Limits & Bounded Execution Tests
# ==============================================================================


def test_limit_max_chart_rows_exceeded(empty_store: DatasetStore):
    # Create dataset with 15 rows, engine with limit 10
    rows = [[f"cat_{i}", float(i)] for i in range(15)]
    ds_id = empty_store.save({"status": "success", "columns": ["cat", "val"], "rows": rows, "row_count": 15})

    engine = ChartGenerationEngine(dataset_store=empty_store, max_chart_rows=10)
    result = engine.generate(dataset_id=ds_id, chart_type="bar", x_column="cat", y_column="val")

    assert result["status"] == "error"
    assert result["error_type"] == "DataLimitExceededError"
    assert "max_chart_rows" in result["error_message"]


def test_limit_max_categories_exceeded_without_top_n(empty_store: DatasetStore):
    # 25 unique categories, max_categories = 20
    rows = [[f"unique_cat_{i}", 10.0] for i in range(25)]
    ds_id = empty_store.save({"status": "success", "columns": ["cat", "val"], "rows": rows, "row_count": 25})

    engine = ChartGenerationEngine(dataset_store=empty_store, max_categories=20)
    result = engine.generate(dataset_id=ds_id, chart_type="bar", x_column="cat", y_column="val")

    assert result["status"] == "error"
    assert result["error_type"] == "DataLimitExceededError"
    assert "max_categories" in result["error_message"]


def test_limit_max_categories_with_top_n_succeeds(empty_store: DatasetStore):
    # 25 unique categories, max_categories = 20, but top_n = 5
    rows = [[f"unique_cat_{i}", float(i)] for i in range(25)]
    ds_id = empty_store.save({"status": "success", "columns": ["cat", "val"], "rows": rows, "row_count": 25})

    engine = ChartGenerationEngine(dataset_store=empty_store, max_categories=20)
    result = engine.generate(
        dataset_id=ds_id, chart_type="bar", x_column="cat", y_column="val", params={"top_n": 5}
    )

    assert result["status"] == "success"
    assert result["metadata"]["rows_used"] == 5


def test_limit_max_heatmap_columns_exceeded(empty_store: DatasetStore):
    # 6 numeric columns, limit is 3
    columns = [f"num_{i}" for i in range(6)]
    rows = [[float(i) for i in range(6)]]
    ds_id = empty_store.save({"status": "success", "columns": columns, "rows": rows, "row_count": 1})

    engine = ChartGenerationEngine(dataset_store=empty_store, max_heatmap_columns=3)
    result = engine.generate(dataset_id=ds_id, chart_type="heatmap")

    assert result["status"] == "error"
    assert result["error_type"] == "DataLimitExceededError"
    assert "max_heatmap_columns" in result["error_message"]


def test_limit_max_figure_bytes_enforced(store_with_sales_data: Tuple[DatasetStore, str]):
    store, ds_id = store_with_sales_data
    # Set unrealistically tiny byte limit
    engine = ChartGenerationEngine(dataset_store=store, max_figure_bytes=100)
    result = engine.generate(dataset_id=ds_id, chart_type="bar", x_column="product_category", y_column="price")

    assert result["status"] == "error"
    assert result["error_type"] == "DataLimitExceededError"
    assert "max_figure_bytes" in result["error_message"]


def test_limit_title_length_enforced(engine: ChartGenerationEngine, store_with_sales_data: Tuple[DatasetStore, str]):
    _, ds_id = store_with_sales_data
    giant_title = "A" * 300
    result = engine.generate(
        dataset_id=ds_id, chart_type="bar", x_column="state", y_column="price", params={"title": giant_title}
    )
    assert result["status"] == "error"
    assert result["error_type"] == "InvalidParameterError"
    assert "title length" in result["error_message"]


# ==============================================================================
# 5. Data Integrity & Immutability Tests
# ==============================================================================


def test_dataset_store_immutability(store_with_sales_data: Tuple[DatasetStore, str]):
    store, ds_id = store_with_sales_data
    original_record = copy.deepcopy(store.get(ds_id))

    engine = ChartGenerationEngine(dataset_store=store)
    # Perform bar chart generation with sorting and top_n filtering
    res = engine.generate(
        dataset_id=ds_id, chart_type="bar", x_column="product_category", y_column="price", params={"top_n": 3, "sort": "desc"}
    )
    assert res["status"] == "success"

    # Verify original record in store is byte-for-byte identical
    current_record = store.get(ds_id)
    assert current_record.rows == original_record.rows
    assert current_record.columns == original_record.columns
    assert current_record.row_count == original_record.row_count


def test_decimals_handled_safely(empty_store: DatasetStore):
    """Ensure DuckDB Decimal objects serialize cleanly to float without error."""
    rows = [
        ["Product A", Decimal("19.99")],
        ["Product B", Decimal("49.50")],
        ["Product C", Decimal("99.00")],
    ]
    ds_id = empty_store.save({"status": "success", "columns": ["item", "price"], "rows": rows, "row_count": 3})
    engine = ChartGenerationEngine(dataset_store=empty_store)

    result = engine.generate(dataset_id=ds_id, chart_type="bar", x_column="item", y_column="price")
    assert result["status"] == "success"
    # Ensure figure JSON was generated and is valid
    fig = pio.from_json(json.dumps(result["figure"]))
    assert isinstance(fig, go.Figure)


def test_correlation_matrix_mathematical_accuracy(empty_store: DatasetStore):
    """Verify heatmap correlation values match mathematical expectations for known vectors."""
    # x and y perfectly correlated (1.0), x and z perfectly anti-correlated (-1.0)
    rows = [
        [1.0, 2.0, 10.0],
        [2.0, 4.0, 8.0],
        [3.0, 6.0, 6.0],
        [4.0, 8.0, 4.0],
    ]
    ds_id = empty_store.save({"status": "success", "columns": ["x", "y", "z"], "rows": rows, "row_count": 4})
    engine = ChartGenerationEngine(dataset_store=empty_store)

    result = engine.generate(dataset_id=ds_id, chart_type="heatmap", params={"columns": ["x", "y", "z"]})
    assert result["status"] == "success"

    fig_dict = result["figure"]
    # The z-data in heatmap should have 1.0 on diagonal, 1.0 for (x,y), -1.0 for (x,z)
    z_matrix = fig_dict["data"][0]["z"]
    assert math.isclose(z_matrix[0][0], 1.0, rel_tol=1e-5)
    assert math.isclose(z_matrix[0][1], 1.0, rel_tol=1e-5)
    assert math.isclose(z_matrix[0][2], -1.0, rel_tol=1e-5)


# ==============================================================================
# 6. Python Analysis Engine Compatibility & Data Handoff Tests
# ==============================================================================


def test_adapter_growth_rate_to_line_chart(empty_store: DatasetStore):
    """Demonstrates handoff: Python growth_rate analytical output converted to dataset and plotted as Line chart."""
    analysis_res = {
        "status": "success",
        "operation": "growth_rate",
        "result": {
            "periods": [
                {"period": "2023-01", "value": 100.0, "previous_value": None, "growth_rate": None},
                {"period": "2023-02", "value": 120.0, "previous_value": 100.0, "growth_rate": 0.20},
                {"period": "2023-03", "value": 150.0, "previous_value": 120.0, "growth_rate": 0.25},
            ],
            "overall_growth_rate": 0.50,
        },
    }

    payload = analysis_result_to_dataset_payload(analysis_res)
    assert payload["status"] == "success"
    assert "growth_rate" in payload["columns"]

    # Store payload into dataset store
    ds_growth = empty_store.save(payload, query="growth_rate_analysis")

    # Generate Line chart of growth_rate
    engine = ChartGenerationEngine(dataset_store=empty_store)
    result = engine.generate(
        dataset_id=ds_growth,
        chart_type="line",
        x_column="period",
        y_column="growth_rate",
        params={"title": "Monthly Growth Rate (%)"},
    )

    assert result["status"] == "success"
    assert result["title"] == "Monthly Growth Rate (%)"
    fig = pio.from_json(json.dumps(result["figure"]))
    assert isinstance(fig, go.Figure)


def test_adapter_groupby_to_bar_chart(empty_store: DatasetStore):
    """Demonstrates handoff: Python groupby aggregation converted to dataset and plotted as Bar chart."""
    analysis_res = {
        "status": "success",
        "operation": "groupby",
        "result": {
            "groups": [
                {"category": "furniture", "total_revenue": 15000.0, "order_count": 120},
                {"category": "electronics", "total_revenue": 35000.0, "order_count": 250},
                {"category": "books", "total_revenue": 5000.0, "order_count": 80},
            ],
            "row_count": 3,
        },
    }

    payload = analysis_result_to_dataset_payload(analysis_res)
    ds_groupby = empty_store.save(payload, query="groupby_analysis")

    engine = ChartGenerationEngine(dataset_store=empty_store)
    result = engine.generate(
        dataset_id=ds_groupby,
        chart_type="bar",
        x_column="category",
        y_column="total_revenue",
        params={"sort": "desc"},
    )

    assert result["status"] == "success"
    assert result["metadata"]["x_column"] == "category"
    assert result["metadata"]["y_column"] == "total_revenue"


def test_adapter_describe_to_box_plot_compatibility(empty_store: DatasetStore):
    """Tests plotting distribution directly from stored dataset corresponding to describe() statistics."""
    # When describe() summarizes a numeric column, the underlying dataset is plotted via box/histogram
    rows = [[float(v)] for v in [10, 12, 14, 15, 18, 20, 22, 25, 29, 30, 85]]
    ds_id = empty_store.save({"status": "success", "columns": ["delivery_days"], "rows": rows, "row_count": len(rows)})

    engine = ChartGenerationEngine(dataset_store=empty_store)
    # Box plot displays visual distribution corresponding to describe quantiles
    result = engine.generate(dataset_id=ds_id, chart_type="box", y_column="delivery_days")
    assert result["status"] == "success"
    assert "Tukey" in result["metadata"]["note"]


# ==============================================================================
# 7. Determinism & Output Schema Integrity Tests
# ==============================================================================


def test_determinism_identical_runs(store_with_sales_data: Tuple[DatasetStore, str]):
    store, ds_id = store_with_sales_data
    engine = ChartGenerationEngine(dataset_store=store)

    run1 = engine.generate(
        dataset_id=ds_id,
        chart_type="bar",
        x_column="product_category",
        y_column="price",
        params={"sort": "desc"},
    )
    run2 = engine.generate(
        dataset_id=ds_id,
        chart_type="bar",
        x_column="product_category",
        y_column="price",
        params={"sort": "desc"},
    )

    assert run1["status"] == run2["status"] == "success"
    assert run1["metadata"] == run2["metadata"]
    # Figures serialized to JSON must be identical
    assert json.dumps(run1["figure"], sort_keys=True) == json.dumps(run2["figure"], sort_keys=True)


def test_error_schema_structure(engine: ChartGenerationEngine):
    result = engine.generate(dataset_id="ds_missing", chart_type="line", x_column="a", y_column="b")
    assert result["status"] == "error"
    assert "dataset_id" in result
    assert "chart_type" in result
    assert "error_type" in result
    assert "error_message" in result
    # JSON-serializable
    assert json.dumps(result)


# ==============================================================================
# 8. Step 6.1 Targeted Verification Tests (Fixes A, B, C, D, E)
# ==============================================================================


# --- Fix A: Undefined Correlations in Heatmap ---


def test_fix_a_undefined_correlation_preserved_as_null(empty_store: DatasetStore):
    """Undefined correlation due to zero variance (constant column) must be null in JSON, not 0.0."""
    rows = [
        [1.0, 2.0, 5.0],
        [2.0, 4.0, 5.0],
        [3.0, 6.0, 5.0],
        [4.0, 8.0, 5.0],
    ]
    ds_id = empty_store.save({"status": "success", "columns": ["x", "y", "constant"], "rows": rows, "row_count": 4})
    engine = ChartGenerationEngine(dataset_store=empty_store)

    result = engine.generate(dataset_id=ds_id, chart_type="heatmap", params={"columns": ["x", "y", "constant"]})
    assert result["status"] == "success"

    fig_dict = result["figure"]
    z_matrix = fig_dict["data"][0]["z"]
    text_matrix = fig_dict["data"][0]["text"]

    # (x, y) correlation must be 1.0
    assert math.isclose(z_matrix[0][1], 1.0, rel_tol=1e-5)
    # (x, constant) and (y, constant) must be None (JSON null), NOT 0.0
    assert z_matrix[0][2] is None
    assert z_matrix[1][2] is None
    assert z_matrix[2][0] is None
    assert z_matrix[2][1] is None
    assert z_matrix[2][2] is None

    # Text annotations for undefined must be 'N/A'
    assert text_matrix[0][2] == "N/A"
    assert text_matrix[2][2] == "N/A"

    # Warning must be present in metadata
    assert "warning" in result["metadata"]
    assert "undefined" in result["metadata"]["warning"].lower()
    assert result["metadata"]["undefined_correlations_count"] > 0


# --- Fix B: Strict Numeric Conversion & Missing Values ---


def test_fix_b_invalid_non_null_string_rejected(empty_store: DatasetStore):
    """Columns containing invalid non-null strings like 'unknown' must be rejected with InvalidColumnError."""
    rows = [
        ["cat_a", "100"],
        ["cat_b", "200"],
        ["cat_c", "unknown"],
        ["cat_d", "400"],
    ]
    ds_id = empty_store.save({"status": "success", "columns": ["cat", "val"], "rows": rows, "row_count": 4})
    engine = ChartGenerationEngine(dataset_store=empty_store)

    result = engine.generate(dataset_id=ds_id, chart_type="bar", x_column="cat", y_column="val")
    assert result["status"] == "error"
    assert result["error_type"] == "InvalidColumnError"
    assert "unknown" in result["error_message"]


def test_fix_b_infinities_strictly_rejected(empty_store: DatasetStore):
    """Columns with positive/negative infinity must be rejected with InvalidColumnError."""
    rows = [
        ["a", 10.0],
        ["b", float("inf")],
        ["c", 30.0],
    ]
    ds_id = empty_store.save({"status": "success", "columns": ["cat", "val"], "rows": rows, "row_count": 3})
    engine = ChartGenerationEngine(dataset_store=empty_store)

    res_inf = engine.generate(dataset_id=ds_id, chart_type="bar", x_column="cat", y_column="val")
    assert res_inf["status"] == "error"
    assert res_inf["error_type"] == "InvalidColumnError"
    assert "non-finite" in res_inf["error_message"].lower() or "infinity" in res_inf["error_message"].lower()


def test_fix_b_decimal_infinity_rejected(empty_store: DatasetStore):
    """Decimal Infinity must be rejected with InvalidColumnError."""
    rows = [
        ["a", Decimal("10.0")],
        ["b", Decimal("Infinity")],
    ]
    ds_id = empty_store.save({"status": "success", "columns": ["cat", "val"], "rows": rows, "row_count": 2})
    engine = ChartGenerationEngine(dataset_store=empty_store)

    res = engine.generate(dataset_id=ds_id, chart_type="bar", x_column="cat", y_column="val")
    assert res["status"] == "error"
    assert res["error_type"] == "InvalidColumnError"
    assert "non-finite" in res["error_message"].lower() or "infinite" in res["error_message"].lower()


def test_fix_b_all_nulls_rejected(empty_store: DatasetStore):
    """A column containing only null values must be rejected with InvalidColumnError."""
    rows = [
        ["a", None],
        ["b", None],
    ]
    ds_id = empty_store.save({"status": "success", "columns": ["cat", "val"], "rows": rows, "row_count": 2})
    engine = ChartGenerationEngine(dataset_store=empty_store)

    res = engine.generate(dataset_id=ds_id, chart_type="bar", x_column="cat", y_column="val")
    assert res["status"] == "error"
    assert res["error_type"] == "InvalidColumnError"
    assert "null" in res["error_message"].lower() or "missing" in res["error_message"].lower()


def test_fix_b_mixed_nulls_reported_in_metadata(empty_store: DatasetStore):
    """Valid numbers mixed with nulls must be preserved; excluded rows must be accurately reported."""
    rows = [
        ["2023-01", 100.0],
        ["2023-02", None],
        ["2023-03", 150.0],
        ["2023-04", 200.0],
    ]
    ds_id = empty_store.save({"status": "success", "columns": ["month", "revenue"], "rows": rows, "row_count": 4})
    engine = ChartGenerationEngine(dataset_store=empty_store)

    res = engine.generate(dataset_id=ds_id, chart_type="line", x_column="month", y_column="revenue")
    assert res["status"] == "success"
    assert res["metadata"]["total_rows"] == 4
    assert res["metadata"]["rows_used"] == 3
    assert res["metadata"]["excluded_rows"] == 1


# --- Fix C: PythonAnalysisEngine Compatibility with Real Outputs ---


def test_fix_c_growth_rate_real_engine_output(empty_store: DatasetStore):
    """Test adapter with genuine output directly from PythonAnalysisEngine.analyze('growth_rate').

    Verifies:
    1. First undefined growth rate is excluded from the plot.
    2. Plotted y-values are approx [0.20, 0.25, 0.20].
    3. Plotted x-values preserve correct chronological periods.
    4. total_rows == 4, rows_used == 3, excluded_rows == 1.
    5. No percentage rescaling (0.20 not 20.0).
    """
    from tools.python_analysis import PythonAnalysisEngine

    sales_rows = [
        ["2023-01", 100.0],
        ["2023-02", 120.0],
        ["2023-03", 150.0],
        ["2023-04", 180.0],
    ]
    ds_id = empty_store.save({"status": "success", "columns": ["month", "revenue"], "rows": sales_rows, "row_count": 4})

    analysis_engine = PythonAnalysisEngine(dataset_store=empty_store)
    analysis_res = analysis_engine.analyze(
        dataset_id=ds_id,
        operation="growth_rate",
        params={"period_column": "month", "value_column": "revenue"},
    )
    assert analysis_res["status"] == "success"

    # Adapter converts real analysis result to dataset payload
    payload = analysis_result_to_dataset_payload(analysis_res)
    assert payload["status"] == "success"
    assert "growth_rate" in payload["columns"]
    assert len(payload["rows"]) == 4

    # Save to DatasetStore
    ds_growth = empty_store.save(payload)

    # Plot as Line chart
    chart_engine = ChartGenerationEngine(dataset_store=empty_store)
    chart_res = chart_engine.generate(
        dataset_id=ds_growth,
        chart_type="line",
        x_column="period",
        y_column="growth_rate",
    )
    assert chart_res["status"] == "success"
    # The first period has growth_rate: None, so total_rows=4, rows_used=3, excluded_rows=1
    assert chart_res["metadata"]["total_rows"] == 4
    assert chart_res["metadata"]["rows_used"] == 3
    assert chart_res["metadata"]["excluded_rows"] == 1

    fig = pio.from_json(json.dumps(chart_res["figure"]))
    assert isinstance(fig, go.Figure)
    trace = fig.data[0]
    plotted_x = _extract_trace_values(trace.x)
    plotted_y = _extract_trace_values(trace.y)

    # Verified coordinates:
    assert plotted_x == ["2023-02", "2023-03", "2023-04"]
    assert plotted_y == pytest.approx([0.20, 0.25, 0.20])
    # No silent percentage rescaling
    assert all(y < 1.0 for y in plotted_y)


def test_fix_c_groupby_real_engine_output(empty_store: DatasetStore):
    """Test adapter with genuine output directly from PythonAnalysisEngine.analyze('groupby')."""
    from tools.python_analysis import PythonAnalysisEngine

    rows = [
        ["electronics", 100.0],
        ["electronics", 200.0],
        ["furniture", 300.0],
    ]
    ds_id = empty_store.save({"status": "success", "columns": ["category", "price"], "rows": rows, "row_count": 3})

    analysis_engine = PythonAnalysisEngine(dataset_store=empty_store)
    analysis_res = analysis_engine.analyze(
        dataset_id=ds_id,
        operation="groupby",
        params={"by": "category", "agg_column": "price", "agg_func": "sum"},
    )
    assert analysis_res["status"] == "success"

    payload = analysis_result_to_dataset_payload(analysis_res)
    assert payload["status"] == "success"
    assert "price_sum" in payload["columns"]

    ds_grouped = empty_store.save(payload)
    chart_engine = ChartGenerationEngine(dataset_store=empty_store)
    chart_res = chart_engine.generate(
        dataset_id=ds_grouped,
        chart_type="bar",
        x_column="category",
        y_column="price_sum",
    )
    assert chart_res["status"] == "success"
    assert chart_res["metadata"]["rows_used"] == 2


def test_fix_c_correlation_real_engine_output_and_precomputed_heatmap(empty_store: DatasetStore):
    """Test correlation output adapter preserves row labels, numerical matrix values, and works with is_precomputed: True."""
    from tools.python_analysis import PythonAnalysisEngine

    # Deterministic dataset with nontrivial positive and negative correlations
    rows = [
        [1.0, 10.0, 100.0],
        [2.0, 20.0, 90.0],
        [3.0, 15.0, 80.0],
        [4.0, 30.0, 70.0],
    ]
    ds_id = empty_store.save({"status": "success", "columns": ["w", "x", "y"], "rows": rows, "row_count": 4})

    analysis_engine = PythonAnalysisEngine(dataset_store=empty_store)
    analysis_res = analysis_engine.analyze(
        dataset_id=ds_id,
        operation="correlation",
        params={"columns": ["w", "x", "y"]},
    )
    assert analysis_res["status"] == "success"
    raw_corr_matrix = analysis_res["result"]

    payload = analysis_result_to_dataset_payload(analysis_res)
    assert payload["status"] == "success"
    # Row label must be preserved
    assert payload["columns"][0] == "column_name"
    assert payload["columns"][1:] == ["w", "x", "y"]

    ds_corr = empty_store.save(payload)
    chart_engine = ChartGenerationEngine(dataset_store=empty_store)
    chart_res = chart_engine.generate(
        dataset_id=ds_corr,
        chart_type="heatmap",
        params={"is_precomputed": True, "columns": ["w", "x", "y"]},
    )
    assert chart_res["status"] == "success"
    assert chart_res["metadata"]["is_precomputed"] is True
    assert chart_res["metadata"]["correlation_method"] == "precomputed"

    fig = pio.from_json(json.dumps(chart_res["figure"]))
    assert isinstance(fig, go.Figure)
    trace = fig.data[0]

    # Required assertions:
    # 1. x-axis labels match correlation matrix column names
    assert list(trace.x) == ["w", "x", "y"]
    # 2. y-axis labels preserve original matrix row labels in correct order
    assert list(trace.y) == ["w", "x", "y"]
    # 3. plotted z-values match correlations returned by PythonAnalysisEngine
    for i, row_name in enumerate(["w", "x", "y"]):
        for j, col_name in enumerate(["w", "x", "y"]):
            expected = raw_corr_matrix[row_name][col_name]
            actual = trace.z[i][j]
            assert actual == pytest.approx(expected, abs=0.01)
    # 4. All plotted values remain bounded in [-1.0, 1.0]
    assert all(-1.0 <= v <= 1.0 for row in trace.z for v in row if v is not None)


def test_precomputed_correlation_heatmap_with_undefined_values(empty_store: DatasetStore):
    """Verify precomputed correlation heatmap preserves undefined correlations as null and annotations as 'N/A'."""
    columns = ["column_name", "A", "B", "C"]
    rows = [
        ["A", 1.0, 0.75, None],
        ["B", 0.75, 1.0, -0.45],
        ["C", None, -0.45, 1.0],
    ]
    ds_id = empty_store.save({"status": "success", "columns": columns, "rows": rows, "row_count": len(rows)})

    chart_engine = ChartGenerationEngine(dataset_store=empty_store)
    chart_res = chart_engine.generate(
        dataset_id=ds_id, chart_type="heatmap", params={"is_precomputed": True, "columns": ["A", "B", "C"]}
    )

    assert chart_res["status"] == "success"
    assert chart_res["metadata"]["is_precomputed"] is True
    assert chart_res["metadata"]["undefined_correlations_count"] == 2

    fig = pio.from_json(json.dumps(chart_res["figure"]))
    assert isinstance(fig, go.Figure)
    trace = fig.data[0]
    assert list(trace.x) == ["A", "B", "C"]
    assert list(trace.y) == ["A", "B", "C"]

    # (A, C) and (C, A) must be None (JSON null), NOT 0.0
    assert trace.z[0][2] is None
    assert trace.z[2][0] is None
    # Text annotations must be 'N/A'
    assert trace.text[0][2] == "N/A"
    assert trace.text[2][0] == "N/A"

    # Valid entries remain accurate
    assert trace.z[0][1] == pytest.approx(0.75)
    assert trace.z[1][2] == pytest.approx(-0.45)
    assert trace.text[0][1] == "0.75"
    assert all(-1.0 <= v <= 1.0 for row in trace.z for v in row if v is not None)


def test_bar_chart_top_n_with_duplicate_categories(empty_store: DatasetStore):
    """Verify Top-N selects Top-N input rows without silent category aggregation."""
    columns = ["category", "revenue"]
    rows = [
        ["Electronics", 100.0],
        ["Electronics", 200.0],
        ["Furniture", 250.0],
        ["Books", 180.0],
    ]
    ds_id = empty_store.save({"status": "success", "columns": columns, "rows": rows, "row_count": len(rows)})
    engine = ChartGenerationEngine(dataset_store=empty_store)

    result = engine.generate(
        dataset_id=ds_id,
        chart_type="bar",
        x_column="category",
        y_column="revenue",
        params={"top_n": 2, "sort": "desc"},
    )

    assert result["status"] == "success"
    assert result["metadata"]["selection_method"] == "top_2"
    assert result["metadata"]["rows_used"] == 2
    assert result["metadata"]["total_rows"] == 4

    fig = pio.from_json(json.dumps(result["figure"]))
    trace = fig.data[0]
    plotted_categories = list(trace.x)
    plotted_revenues = _extract_trace_values(trace.y)

    # Exactly two input observations selected: Furniture (250.0) and Electronics (200.0)
    assert plotted_revenues == [250.0, 200.0]
    assert plotted_categories == ["Furniture", "Electronics"]
    # Verify no silent aggregation: Electronics is 200.0, not 300.0
    assert 300.0 not in plotted_revenues


def test_fix_c_describe_real_engine_output(empty_store: DatasetStore):
    """Test describe output adapter converts summary statistics to table without mistaking them for raw observations."""
    from tools.python_analysis import PythonAnalysisEngine

    rows = [[10.0], [20.0], [30.0], [40.0]]
    ds_id = empty_store.save({"status": "success", "columns": ["price"], "rows": rows, "row_count": 4})

    analysis_engine = PythonAnalysisEngine(dataset_store=empty_store)
    analysis_res = analysis_engine.analyze(
        dataset_id=ds_id,
        operation="describe",
        params={"columns": ["price"]},
    )
    assert analysis_res["status"] == "success"

    payload = analysis_result_to_dataset_payload(analysis_res)
    assert payload["status"] == "success"
    assert payload["columns"] == ["statistic", "price"]
    stat_names = [r[0] for r in payload["rows"]]
    assert "mean" in stat_names
    assert "median" in stat_names


def test_fix_c_detect_outliers_adapter_rejection(empty_store: DatasetStore):
    """Test that detect_outliers cannot be adapted into a raw dataset without indicator column."""
    analysis_res = {
        "status": "success",
        "operation": "detect_outliers",
        "result": {
            "column": "price",
            "outlier_count": 2,
            "outliers_preview": [{"row_index": 1, "value": 500.0}],
        },
    }
    with pytest.raises(ValueError, match="detect_outliers"):
        analysis_result_to_dataset_payload(analysis_res)


# --- Fix D: Strict Boolean Parameter Validation ---


@pytest.mark.parametrize(
    "param_name,invalid_val",
    [
        ("sort_x", "false"),
        ("sort_x", 0),
        ("sort_x", None),
        ("markers", "true"),
        ("markers", 1),
    ],
)
def test_fix_d_line_boolean_coercion_rejected(empty_store: DatasetStore, param_name: str, invalid_val: Any):
    """Strings like 'false' or numbers like 0 must be strictly rejected as invalid boolean values."""
    rows = [["2023-01", 10.0], ["2023-02", 20.0]]
    ds_id = empty_store.save({"status": "success", "columns": ["d", "v"], "rows": rows, "row_count": 2})
    engine = ChartGenerationEngine(dataset_store=empty_store)

    res = engine.generate(
        dataset_id=ds_id,
        chart_type="line",
        x_column="d",
        y_column="v",
        params={param_name: invalid_val},
    )
    assert res["status"] == "error"
    assert res["error_type"] == "InvalidParameterError"
    assert "boolean" in res["error_message"].lower()


def test_fix_d_box_points_strict_type(empty_store: DatasetStore):
    """Box plot points parameter must accept 'outliers', 'all', or False, rejecting True or strings."""
    rows = [["a", 10.0], ["b", 20.0]]
    ds_id = empty_store.save({"status": "success", "columns": ["c", "v"], "rows": rows, "row_count": 2})
    engine = ChartGenerationEngine(dataset_store=empty_store)

    res_true = engine.generate(
        dataset_id=ds_id, chart_type="box", y_column="v", params={"points": True}
    )
    assert res_true["status"] == "error"
    assert res_true["error_type"] == "InvalidParameterError"

    res_valid_false = engine.generate(
        dataset_id=ds_id, chart_type="box", y_column="v", params={"points": False}
    )
    assert res_valid_false["status"] == "success"
    assert res_valid_false["metadata"]["points"] is False


# --- Fix E: Category Cardinality & Trace Limits Across All Charts ---


def test_fix_e_line_color_cardinality_limit(empty_store: DatasetStore):
    """High cardinality color_column in line chart must raise DataLimitExceededError."""
    # 25 categories when max_categories is 10
    rows = [[f"2023-01", float(i), f"cat_{i}"] for i in range(25)]
    ds_id = empty_store.save({"status": "success", "columns": ["d", "v", "grp"], "rows": rows, "row_count": 25})
    engine = ChartGenerationEngine(dataset_store=empty_store, max_categories=10)

    res = engine.generate(
        dataset_id=ds_id, chart_type="line", x_column="d", y_column="v", params={"color_column": "grp"}
    )
    assert res["status"] == "error"
    assert res["error_type"] == "DataLimitExceededError"
    assert "max_categories" in res["error_message"]


def test_fix_e_scatter_color_and_outlier_cardinality_limit(empty_store: DatasetStore):
    """High cardinality in scatter color_column or outlier_column must raise DataLimitExceededError."""
    rows = [[float(i), float(i * 2), f"cat_{i}"] for i in range(30)]
    ds_id = empty_store.save({"status": "success", "columns": ["x", "y", "grp"], "rows": rows, "row_count": 30})
    engine = ChartGenerationEngine(dataset_store=empty_store, max_categories=10)

    # Color column test
    res_color = engine.generate(
        dataset_id=ds_id, chart_type="scatter", x_column="x", y_column="y", params={"color_column": "grp"}
    )
    assert res_color["status"] == "error"
    assert res_color["error_type"] == "DataLimitExceededError"

    # Outlier column test
    res_outlier = engine.generate(
        dataset_id=ds_id, chart_type="scatter", x_column="x", y_column="y", params={"outlier_column": "grp"}
    )
    assert res_outlier["status"] == "error"
    assert res_outlier["error_type"] == "DataLimitExceededError"


def test_fix_e_histogram_color_cardinality_limit(empty_store: DatasetStore):
    """High cardinality color_column in histogram must raise DataLimitExceededError."""
    rows = [[float(i), f"cat_{i}"] for i in range(20)]
    ds_id = empty_store.save({"status": "success", "columns": ["x", "grp"], "rows": rows, "row_count": 20})
    engine = ChartGenerationEngine(dataset_store=empty_store, max_categories=10)

    res = engine.generate(
        dataset_id=ds_id, chart_type="histogram", x_column="x", params={"color_column": "grp"}
    )
    assert res["status"] == "error"
    assert res["error_type"] == "DataLimitExceededError"


def test_fix_e_box_grouping_cardinality_limit(empty_store: DatasetStore):
    """High cardinality grouping in box plot x_column must raise DataLimitExceededError."""
    rows = [[f"cat_{i}", float(i)] for i in range(25)]
    ds_id = empty_store.save({"status": "success", "columns": ["grp", "y"], "rows": rows, "row_count": 25})
    engine = ChartGenerationEngine(dataset_store=empty_store, max_categories=10)

    res = engine.generate(
        dataset_id=ds_id, chart_type="box", x_column="grp", y_column="y"
    )
    assert res["status"] == "error"
    assert res["error_type"] == "DataLimitExceededError"

