"""Unit and isolation tests for ChartRegistry (Step 6.2 Hardening).

Verifies:
1. Save-time and Get-time Figure immutability & isolation.
2. Save-time and List-time Metadata immutability & isolation.
3. Memory capacity limits and accounting on successes and failures.
4. Clear and lifecycle behavior.
5. Cross-session instance isolation.
"""

from __future__ import annotations

import copy
import pytest

from tools.chart_registry import (
    DEFAULT_MAX_CHARTS,
    DEFAULT_MAX_REGISTRY_BYTES,
    ChartNotFoundError,
    ChartRecord,
    ChartRegistry,
    ChartRegistryCapacityError,
    InvalidChartRecordError,
    _estimate_figure_bytes,
)


# ==============================================================================
# C1 & C4: Deep-Copy Isolation & Immutability Tests
# ==============================================================================


def test_save_time_figure_isolation():
    """Verify modifying the original figure dict after save does not mutate the registry."""
    registry = ChartRegistry()
    original_fig = {
        "data": [{"type": "scatter", "x": [1, 2], "y": [3, 4], "nested": {"key": "val1"}}],
        "layout": {"title": "Original Title"},
    }

    cid = registry.save(original_fig, "ds_1", "scatter", "Scatter Title")

    # Mutate original figure deeply
    original_fig["data"][0]["x"].append(999)
    original_fig["data"][0]["nested"]["key"] = "mutated"
    original_fig["layout"]["title"] = "Mutated Title"

    # Stored record must remain pristine
    rec = registry.get(cid)
    assert rec.figure["data"][0]["x"] == [1, 2]
    assert rec.figure["data"][0]["nested"]["key"] == "val1"
    assert rec.figure["layout"]["title"] == "Original Title"


def test_get_time_figure_isolation():
    """Verify modifying figure dict retrieved via get() does not mutate registry internal state."""
    registry = ChartRegistry()
    fig = {
        "data": [{"type": "bar", "x": ["a", "b"], "y": [10, 20], "config": {"color": "blue"}}],
        "layout": {"barmode": "group"},
    }

    cid = registry.save(fig, "ds_1", "bar", "Bar Title")

    # Retrieve and mutate deeply
    rec1 = registry.get(cid)
    rec1.figure["data"][0]["x"].append("c")
    rec1.figure["data"][0]["config"]["color"] = "red"
    rec1.figure["layout"]["barmode"] = "stack"

    # Retrieve again: internal state must be unaffected
    rec2 = registry.get(cid)
    assert rec2.figure["data"][0]["x"] == ["a", "b"]
    assert rec2.figure["data"][0]["config"]["color"] == "blue"
    assert rec2.figure["layout"]["barmode"] == "group"


def test_save_time_metadata_isolation():
    """Verify modifying original metadata dict after save does not mutate the registry."""
    registry = ChartRegistry()
    fig = {"data": [], "layout": {}}
    orig_meta = {"tags": ["finance", "monthly"], "nested": {"level": 1}}

    cid = registry.save(fig, "ds_1", "line", "Line Title", metadata=orig_meta)

    # Mutate original metadata
    orig_meta["tags"].append("mutated")
    orig_meta["nested"]["level"] = 999

    rec = registry.get(cid)
    assert rec.metadata["tags"] == ["finance", "monthly"]
    assert rec.metadata["nested"]["level"] == 1


def test_list_time_metadata_isolation():
    """Verify modifying metadata returned by list_charts() does not mutate the registry."""
    registry = ChartRegistry()
    fig = {"data": [], "layout": {}}
    orig_meta = {"columns": ["revenue", "units"], "options": {"log_scale": False}}

    cid = registry.save(fig, "ds_1", "scatter", "Scatter Title", metadata=orig_meta)

    # Call list_charts() and mutate returned metadata
    chart_list_1 = registry.list_charts()
    assert len(chart_list_1) == 1
    chart_list_1[0]["metadata"]["columns"].append("mutated_column")
    chart_list_1[0]["metadata"]["options"]["log_scale"] = True

    # Call list_charts() again and verify registry metadata is intact
    chart_list_2 = registry.list_charts()
    assert chart_list_2[0]["metadata"]["columns"] == ["revenue", "units"]
    assert chart_list_2[0]["metadata"]["options"]["log_scale"] is False

    # Also verify get() is intact
    rec = registry.get(cid)
    assert rec.metadata["columns"] == ["revenue", "units"]
    assert rec.metadata["options"]["log_scale"] is False


# ==============================================================================
# C2: Memory Capacity Tests & Accounting
# ==============================================================================


def test_memory_limit_enforcement_and_capacity_error():
    """Verify max_memory_bytes raises ChartRegistryCapacityError when exceeded."""
    small_limit = 2000
    registry = ChartRegistry(max_charts=10, max_memory_bytes=small_limit)

    small_fig = {"data": [{"x": [1]}], "layout": {}}
    est_small = _estimate_figure_bytes(small_fig)
    assert est_small < small_limit

    # First save succeeds
    cid1 = registry.save(small_fig, "ds_1", "scatter", "Small 1")
    assert cid1 in registry
    assert len(registry) == 1
    assert registry.total_memory_bytes == est_small

    # Bulky figure that exceeds remaining memory
    bulky_fig = {"data": [{"x": list(range(10000))}], "layout": {}}
    est_bulky = _estimate_figure_bytes(bulky_fig)
    assert est_small + est_bulky > small_limit

    # Attempting to save bulky figure raises ChartRegistryCapacityError
    with pytest.raises(ChartRegistryCapacityError) as exc_info:
        registry.save(bulky_fig, "ds_2", "scatter", "Bulky")

    assert "exceed memory limit" in str(exc_info.value)

    # Failed save must not increment chart count or alter total_memory_bytes
    assert len(registry) == 1
    assert registry.total_memory_bytes == est_small

    # Registry remains fully usable after failed save if capacity permits
    another_small = {"data": [{"x": [2]}], "layout": {}}
    est_another = _estimate_figure_bytes(another_small)
    if est_small + est_another <= small_limit:
        cid2 = registry.save(another_small, "ds_3", "scatter", "Small 2")
        assert cid2 in registry
        assert len(registry) == 2
        assert registry.total_memory_bytes == est_small + est_another


def test_chart_count_limit_enforcement():
    """Verify max_charts limit raises ChartRegistryCapacityError without modifying memory counter."""
    registry = ChartRegistry(max_charts=2, max_memory_bytes=1000000)

    fig = {"data": [], "layout": {}}
    cid1 = registry.save(fig, "ds_1", "bar", "Bar 1")
    cid2 = registry.save(fig, "ds_2", "bar", "Bar 2")
    assert len(registry) == 2
    mem_before = registry.total_memory_bytes

    with pytest.raises(ChartRegistryCapacityError, match="maximum capacity of 2 charts"):
        registry.save(fig, "ds_3", "bar", "Bar 3")

    assert len(registry) == 2
    assert registry.total_memory_bytes == mem_before


# ==============================================================================
# C3: Clear & Lifecycle Tests
# ==============================================================================


def test_clear_and_lifecycle():
    """Verify clear() resets chart count, memory counter, and invalidates previous IDs."""
    registry = ChartRegistry(max_charts=5, max_memory_bytes=50000)

    fig = {"data": [{"x": [1, 2]}], "layout": {}}
    cid1 = registry.save(fig, "ds_1", "line", "Line 1")
    cid2 = registry.save(fig, "ds_2", "bar", "Bar 2")

    assert len(registry) == 2
    assert registry.total_memory_bytes > 0

    # Evict all
    registry.clear()

    assert len(registry) == 0
    assert registry.total_memory_bytes == 0
    assert registry.list_charts() == []

    # Previously stored IDs are gone
    assert cid1 not in registry
    assert cid2 not in registry
    with pytest.raises(ChartNotFoundError):
        registry.get(cid1)
    with pytest.raises(ChartNotFoundError):
        registry.get(cid2)

    # New charts can be saved after clearing
    cid3 = registry.save(fig, "ds_3", "histogram", "Histogram 3")
    assert cid3 in registry
    assert len(registry) == 1
    assert registry.total_memory_bytes > 0
    # Opaque non-deterministic ID
    assert cid3 != cid1
    assert cid3 != cid2
    assert cid3.startswith("chart_")


# ==============================================================================
# C5: Session Isolation & Validation
# ==============================================================================


def test_session_isolation_independent_registries():
    """Verify independent ChartRegistry instances never share state or memory accounting."""
    reg_a = ChartRegistry()
    reg_b = ChartRegistry()

    fig = {"data": [], "layout": {}}
    cid_a = reg_a.save(fig, "ds_a", "box", "Box A")

    assert cid_a in reg_a
    assert cid_a not in reg_b
    assert len(reg_a) == 1
    assert len(reg_b) == 0
    assert reg_a.total_memory_bytes > 0
    assert reg_b.total_memory_bytes == 0

    with pytest.raises(ChartNotFoundError):
        reg_b.get(cid_a)


def test_invalid_chart_registry_constructor_arguments():
    """Verify constructor rejects non-positive or non-integer arguments."""
    with pytest.raises(ValueError):
        ChartRegistry(max_charts=0)
    with pytest.raises(ValueError):
        ChartRegistry(max_charts=-1)
    with pytest.raises(ValueError):
        ChartRegistry(max_charts=True)
    with pytest.raises(ValueError):
        ChartRegistry(max_memory_bytes=0)
    with pytest.raises(ValueError):
        ChartRegistry(max_memory_bytes=-100)
    with pytest.raises(ValueError):
        ChartRegistry(max_memory_bytes=False)


def test_save_invalid_record_arguments():
    """Verify save() rejects invalid figures, dataset IDs, and unsupported chart types."""
    registry = ChartRegistry()
    valid_fig = {"data": [], "layout": {}}

    # Empty figure
    with pytest.raises(InvalidChartRecordError, match="non-empty dictionary"):
        registry.save({}, "ds_1", "line", "Title")

    # Non-dict figure
    with pytest.raises(InvalidChartRecordError, match="non-empty dictionary"):
        registry.save("not a dict", "ds_1", "line", "Title")  # type: ignore

    # Invalid dataset_id
    with pytest.raises(InvalidChartRecordError, match="non-empty string"):
        registry.save(valid_fig, "", "line", "Title")
    with pytest.raises(InvalidChartRecordError, match="non-empty string"):
        registry.save(valid_fig, "   ", "line", "Title")

    # Unsupported chart type
    with pytest.raises(InvalidChartRecordError, match="chart_type must be one of"):
        registry.save(valid_fig, "ds_1", "pie", "Title")

    # Non-string title
    with pytest.raises(InvalidChartRecordError, match="title must be a string"):
        registry.save(valid_fig, "ds_1", "line", 12345)  # type: ignore
