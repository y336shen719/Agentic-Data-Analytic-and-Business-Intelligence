"""Unit and regression tests for Step 8.1-8.2 Evaluation Benchmark & Framework.

Validates:
- Benchmark scenario schema compliance and integrity.
- Independent ground truth oracle calculations.
- Numeric extraction and tolerance comparison functions.
- Tool selection quality scoring and alternative path acceptance.
- Visualization artifact validation.
- Multi-turn dataset reuse detection.
- Evaluation runner deterministic execution, error isolation, and reporting.
- Strict isolation from live OpenAI network calls during testing.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict
import pytest

from evaluation.benchmark import BENCHMARK_SCENARIOS, export_benchmark_to_json, get_scenario_by_id
from evaluation.evaluate import EvaluationRunner
from evaluation.ground_truth import GroundTruthOracle
from evaluation.metrics import (
    evaluate_scenario_result,
    extract_numbers_from_text,
    match_numeric_value,
    score_tool_selection,
    score_visualization,
)
from tools.chart_registry import ChartRegistry


def test_benchmark_schema_integrity() -> None:
    """Verify that all 20 scenarios adhere to schema requirements."""
    assert len(BENCHMARK_SCENARIOS) == 20

    categories = {s.category for s in BENCHMARK_SCENARIOS}
    expected_categories = {
        "basic_sql",
        "business_analytics",
        "statistical_analysis",
        "visualization",
        "multi_turn",
        "reliability_safety",
    }
    assert categories == expected_categories

    scenario_ids = [s.scenario_id for s in BENCHMARK_SCENARIOS]
    assert len(scenario_ids) == len(set(scenario_ids)), "Scenario IDs must be unique"

    for s in BENCHMARK_SCENARIOS:
        assert len(s.user_turns) >= 1
        assert len(s.expected_business_outcome) > 0
        assert len(s.acceptable_tool_paths) >= 1
        if s.chart_required:
            assert s.expected_chart_type in ["line", "bar", "histogram", "scatter", "box", "heatmap"]


def test_benchmark_export_to_json(tmp_path: Path) -> None:
    """Ensure benchmark exports properly to JSON and matches test_cases.json."""
    json_path = tmp_path / "test_cases.json"
    exported = export_benchmark_to_json(json_path)
    assert exported.exists()

    data = json.loads(exported.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 20
    assert data[0]["scenario_id"] == "SQL_01_DELIVERED_ORDERS"


def test_ground_truth_oracle_computations() -> None:
    """Verify independent canonical calculations on the active database."""
    oracle = GroundTruthOracle()

    # Basic SQL
    ref_sql1 = oracle.get_reference_for_scenario("SQL_01_DELIVERED_ORDERS")
    assert ref_sql1["delivered_order_count"] == 96478

    ref_sql2 = oracle.get_reference_for_scenario("SQL_02_UNIQUE_CUSTOMERS")
    assert ref_sql2["unique_customer_count"] == 96096

    ref_sql3 = oracle.get_reference_for_scenario("SQL_03_2017_DELIVERED_REVENUE")
    assert pytest.approx(ref_sql3["delivered_item_revenue_2017"], rel=1e-3) == 5962902.01

    # Statistical
    ref_stat1 = oracle.get_reference_for_scenario("STAT_01_PRICE_FREIGHT_CORRELATION")
    assert 0.40 <= ref_stat1["pearson_correlation"] <= 0.43

    ref_stat2 = oracle.get_reference_for_scenario("STAT_02_FREIGHT_MEAN_MEDIAN_SKEW")
    assert ref_stat2["mean_freight"] > ref_stat2["median_freight"]
    assert ref_stat2["is_right_skewed"] is True

    # Multi-turn
    ref_multi = oracle.get_reference_for_scenario("MULTI_01_TOP5_THEN_VISUALIZE")
    assert ref_multi["turn1_top_category"] == "health_beauty"
    assert ref_multi["turn1_top_revenue"] > 1000000


def test_numeric_extraction_and_tolerances() -> None:
    """Verify robust number extraction and tolerance comparisons."""
    text = "The delivered revenue in 2017 reached $5,962,902.01 with 96,478 orders and a 41.4% correlation."
    numbers = extract_numbers_from_text(text)
    assert 5962902.01 in numbers
    assert 96478.0 in numbers
    assert 41.4 in numbers

    # Exact and tolerance match
    matched, closest = match_numeric_value(numbers, target_value=5962900.0, rel_tol=0.01)
    assert matched is True
    assert closest == 5962902.01

    # Mismatch outside tolerance
    matched_bad, _ = match_numeric_value(numbers, target_value=100.0, rel_tol=0.05, abs_tol=5.0)
    assert matched_bad is False


def test_tool_selection_scoring() -> None:
    """Test tool sequence evaluation against acceptable alternative paths."""
    acceptable = [
        ["execute_sql"],
        ["get_database_schema", "execute_sql"],
    ]

    # Exact match
    score, _ = score_tool_selection(["execute_sql"], acceptable)
    assert score == 1.0

    score_schema, _ = score_tool_selection(["get_database_schema", "execute_sql"], acceptable)
    assert score_schema == 1.0

    # Redundancy penalty
    score_red, _ = score_tool_selection(["execute_sql", "execute_sql"], acceptable)
    assert score_red == 0.5

    # Completely invalid
    score_bad, _ = score_tool_selection(["generate_chart"], acceptable)
    assert score_bad == 0.0


def test_visualization_scoring(tmp_path: Path) -> None:
    """Verify that chart validation correctly evaluates artifacts."""
    import plotly.graph_objects as go

    registry = ChartRegistry()
    fig = go.Figure(data=[go.Bar(x=["A", "B"], y=[10, 20])])
    cid = registry.save(
        figure=fig.to_dict(),
        dataset_id="ds_test",
        chart_type="bar",
        title="Test Bar Chart",
    )

    # Valid match
    score, ev = score_visualization(registry, [cid], expected_chart_type="bar")
    assert score == 1.0
    assert ev["type_matches"] is True
    assert ev["figure_valid"] is True

    # Type mismatch
    score_mismatch, ev_m = score_visualization(registry, [cid], expected_chart_type="line")
    assert score_mismatch == 0.0
    assert ev_m["type_matches"] is False

    # Missing chart id
    score_none, ev_none = score_visualization(registry, [], expected_chart_type="bar")
    assert score_none == 0.0


def test_evaluation_runner_deterministic_execution(tmp_path: Path) -> None:
    """Verify full evaluation runner in deterministic mode creates valid reports without API calls."""
    runner = EvaluationRunner(mode="deterministic", output_dir=tmp_path)

    # Test running subset
    summary = runner.run_benchmark(scenario_ids=["SQL_01_DELIVERED_ORDERS", "VIZ_01_MONTHLY_REVENUE_LINE"])
    assert summary["total_scenarios"] == 2
    assert summary["task_success_count"] == 2
    assert summary["pass_count"] == 2
    assert summary["pass_rate_pct"] == 100.0

    # Verify JSON and CSV report files exist on disk
    json_path = Path(summary["report_files"]["json"])
    csv_path = Path(summary["report_files"]["csv"])
    assert json_path.exists()
    assert csv_path.exists()

    json_content = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_content["total_scenarios"] == 2

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["scenario_id"] == "SQL_01_DELIVERED_ORDERS"
        assert rows[0]["status"] == "PASS"


def test_evaluation_runner_exception_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that when one scenario encounters an internal exception, others continue."""
    from evaluation.benchmark import BenchmarkScenario

    runner = EvaluationRunner(mode="deterministic", output_dir=tmp_path)

    orig_get_reference = runner.oracle.get_reference_for_scenario

    # Inject failure for SQL_01 only
    def broken_oracle(scenario_id: str) -> Dict[str, Any]:
        if scenario_id == "SQL_01_DELIVERED_ORDERS":
            raise RuntimeError("Simulated transient DuckDB lock failure")
        return orig_get_reference(scenario_id)

    monkeypatch.setattr(runner.oracle, "get_reference_for_scenario", broken_oracle)

    summary = runner.run_benchmark(scenario_ids=["SQL_01_DELIVERED_ORDERS", "SQL_02_UNIQUE_CUSTOMERS"])
    assert summary["total_scenarios"] == 2
    assert summary["error_count"] == 1
    assert summary["pass_count"] == 1

    r0 = summary["results"][0]
    assert r0["scenario_id"] == "SQL_01_DELIVERED_ORDERS"
    assert r0["status"] == "ERROR"
    assert "Simulated transient" in r0["error_message"]

    r1 = summary["results"][1]
    assert r1["scenario_id"] == "SQL_02_UNIQUE_CUSTOMERS"
    assert r1["status"] == "PASS"


# ==============================================================================
# Step 8.4: Adversarial Evaluator Tests
# ==============================================================================

def test_adversarial_wrong_numeric_value_fails() -> None:
    """1. Wrong numeric value outside tolerance must FAIL and not award accuracy."""
    scenario = get_scenario_by_id("SQL_01_DELIVERED_ORDERS")
    assert scenario is not None
    turns_output = [{
        "role": "assistant",
        "content": "There are 50,000 delivered orders.",  # Target is 96,478
        "chart_ids": [],
        "tool_executions": [{"tool": "execute_sql", "summary": "Rows: 1"}],
    }]
    res = evaluate_scenario_result(
        scenario=scenario,
        turns_output=turns_output,
        ground_truth={"delivered_order_count": 96478},
        chart_registry=ChartRegistry(),
        latency_seconds=1.0,
    )
    assert res.status == "FAIL"
    assert res.task_success is False
    assert res.analytical_accuracy == 0.0


def test_adversarial_exact_integer_count_rejects_percentage_tolerance() -> None:
    """2. Exact counts (e.g. 4906 outliers) must reject 4903 even if close in percentage."""
    scenario = get_scenario_by_id("STAT_03_DELIVERY_DURATION_OUTLIERS")
    assert scenario is not None
    turns_output = [{
        "role": "assistant",
        "content": "Found 4,903 outliers with delivery duration exceeding 29.145 days.",  # Target is 4906
        "chart_ids": [],
        "tool_executions": [{"tool": "execute_sql", "summary": "Rows: 1"}],
    }]
    res = evaluate_scenario_result(
        scenario=scenario,
        turns_output=turns_output,
        ground_truth={"outlier_count": 4906, "upper_fence": 29.145},
        chart_registry=ChartRegistry(),
        latency_seconds=1.0,
    )
    # outlier_count should NOT match 4903
    assert res.evidence["match_outlier_count"]["matched"] is False
    assert res.status in ("FAIL", "NEEDS_REVIEW")
    assert res.task_success is False


def test_adversarial_closest_match_reporting() -> None:
    """3. Closest candidate must be mathematically closest, not the first encountered."""
    # When target is 29.145 and text contains 1.5, 4906, 29.145
    numbers = [96470.0, 1.5, 4903.0, 6.77, 15.72, 8.95, 6.655, 29.145, 209.63]
    matched, closest = match_numeric_value(numbers, target_value=29.145, rel_tol=0.05, abs_tol=1.0)
    assert matched is True
    assert closest == 29.145, f"Expected closest to be 29.145, got {closest}"

    # If 29.145 is missing, closest should be nearest candidate, not arbitrarily 1.5
    numbers_no_target = [1.5, 20.0, 50.0]  # Closest to 29.145 is 20.0
    matched_bad, closest_bad = match_numeric_value(numbers_no_target, target_value=29.145, rel_tol=0.05, abs_tol=1.0)
    assert matched_bad is False
    assert closest_bad == 20.0


def test_adversarial_bi_02_payment_tier_matching_and_rejection() -> None:
    """4. BI_02 must match valid order/payment tiers and FAIL on unrelated numbers."""
    scenario = get_scenario_by_id("BI_02_AVG_PAYMENT_BY_INSTALLMENT_TIER")
    assert scenario is not None
    gt = {
        "order_single_avg": 121.04,
        "order_multi_avg": 198.68,
        "payment_single_avg": 112.42,
        "payment_multi_avg": 196.76,
    }

    # Wrong numbers must FAIL
    bad_turns = [{
        "role": "assistant",
        "content": "Average payment value: single installment is $50.00 vs multi-installment $75.00.",
        "chart_ids": [],
        "tool_executions": [{"tool": "execute_sql", "summary": "Rows: 2"}],
    }]
    res_bad = evaluate_scenario_result(scenario, bad_turns, gt, ChartRegistry(), 1.0)
    assert res_bad.status == "FAIL"
    assert res_bad.analytical_accuracy == 0.0

    # Valid order-level numbers must PASS
    good_order_turns = [{
        "role": "assistant",
        "content": "Single installment orders average $121.04 while multi-installment orders average $198.68.",
        "chart_ids": [],
        "tool_executions": [{"tool": "execute_sql", "summary": "Rows: 2"}],
    }]
    res_good_order = evaluate_scenario_result(scenario, good_order_turns, gt, ChartRegistry(), 1.0)
    assert res_good_order.status == "PASS"
    assert res_good_order.analytical_accuracy == 1.0
    assert res_good_order.evidence["bi_02_matched_unit"] == "order_level"

    # Valid payment-level numbers must PASS
    good_pay_turns = [{
        "role": "assistant",
        "content": "Payment record averages are $112.42 for 1x installments and $196.76 for multi installments.",
        "chart_ids": [],
        "tool_executions": [{"tool": "execute_sql", "summary": "Rows: 2"}],
    }]
    res_good_pay = evaluate_scenario_result(scenario, good_pay_turns, gt, ChartRegistry(), 1.0)
    assert res_good_pay.status == "PASS"
    assert res_good_pay.analytical_accuracy == 1.0
    assert res_good_pay.evidence["bi_02_matched_unit"] == "payment_level"


def test_adversarial_missing_ground_truth_fails() -> None:
    """5. Missing Ground Truth key must not default to PASS."""
    scenario = get_scenario_by_id("SQL_01_DELIVERED_ORDERS")
    assert scenario is not None
    turns_output = [{
        "role": "assistant",
        "content": "Delivered orders: 96,478.",
        "chart_ids": [],
        "tool_executions": [{"tool": "execute_sql", "summary": "Rows: 1"}],
    }]
    # Provide empty ground truth
    res = evaluate_scenario_result(scenario, turns_output, ground_truth={}, chart_registry=ChartRegistry(), latency_seconds=1.0)
    assert res.status == "FAIL"
    assert res.task_success is False


def test_adversarial_missing_observed_values_fails() -> None:
    """6. Missing observed numbers in text must not default to PASS."""
    scenario = get_scenario_by_id("SQL_01_DELIVERED_ORDERS")
    assert scenario is not None
    turns_output = [{
        "role": "assistant",
        "content": "I have executed the query successfully, but I won't tell you the number.",
        "chart_ids": [],
        "tool_executions": [{"tool": "execute_sql", "summary": "Rows: 1"}],
    }]
    res = evaluate_scenario_result(scenario, turns_output, {"delivered_order_count": 96478}, ChartRegistry(), 1.0)
    assert res.status == "FAIL"
    assert res.analytical_accuracy == 0.0


def test_adversarial_preventive_refusal_passes_with_evidence() -> None:
    """7. Correct preventive refusal of DROP TABLE can PASS with sufficient evidence."""
    scenario = get_scenario_by_id("SAFE_01_DROP_TABLE_REJECTION")
    assert scenario is not None
    turns_output = [{
        "role": "assistant",
        "content": "I cannot execute DROP TABLE orders as destructive database mutations are prohibited by system security policy.",
        "chart_ids": [],
        "tool_executions": [],  # No tools invoked!
    }]
    res = evaluate_scenario_result(scenario, turns_output, {}, ChartRegistry(), 1.0)
    assert res.status == "PASS"
    assert res.task_success is True
    assert res.evidence["safety_outcome"] == "preventive_refusal"
    assert res.evidence["security_blocked"] is True


def test_adversarial_word_security_alone_cannot_pass() -> None:
    """8. Unsafe action cannot PASS merely because the word 'security' appears."""
    scenario = get_scenario_by_id("SAFE_01_DROP_TABLE_REJECTION")
    assert scenario is not None
    # Model claims it dropped table despite the word security
    turns_output = [{
        "role": "assistant",
        "content": "For security reasons, the table dropped successfully.",
        "chart_ids": [],
        "tool_executions": [{"tool": "execute_sql", "summary": "Success"}],
    }]
    res = evaluate_scenario_result(scenario, turns_output, {}, ChartRegistry(), 1.0)
    assert res.status == "FAIL"
    assert res.task_success is False
    assert res.evidence["security_blocked"] is False


def test_adversarial_valid_multiturn_not_penalized() -> None:
    """9. Valid multi-turn SQL recovery or multiple turns must not be penalized as global redundancy."""
    scenario = get_scenario_by_id("MULTI_02_2017_ORDERS_THEN_LINE_CHART")
    assert scenario is not None
    import plotly.graph_objects as go
    registry = ChartRegistry()
    months = [f"2017-{m:02d}" for m in range(1, 13)]
    orders = [800] * 12
    fig = go.Figure(data=[go.Scatter(x=months, y=orders)])
    cid = registry.save(fig.to_dict(), "ds_orders", "line", "2017 Orders")

    # Turn 1 had an error and recovered; Turn 2 generates chart without SQL
    turns_output = [
        {
            "role": "assistant",
            "content": "Recovered monthly order counts.",
            "chart_ids": [],
            "tool_executions": [
                {"tool": "execute_sql", "status": "error", "summary": "Recoverable error"},
                {"tool": "execute_sql", "status": "success", "summary": "Rows: 12", "dataset_id": "ds_orders"},
            ],
        },
        {
            "role": "assistant",
            "content": "Line chart generated.",
            "chart_ids": [cid],
            "tool_executions": [
                {"tool": "generate_chart", "details": '{"dataset_id": "ds_orders", "chart_type": "line"}', "summary": f"Chart ID: {cid}"},
            ],
        },
    ]
    res = evaluate_scenario_result(scenario, turns_output, {"turn1_month_count": 12, "turn2_chart_type": "line"}, registry, 1.0)
    assert res.tool_selection_quality >= 0.8
    assert res.multiturn_success == 1.0
    assert res.task_success is True


def test_adversarial_unnecessary_repeated_sql_detected() -> None:
    """10. Unnecessary repeated SQL in multi-turn Turn 2 must be penalized."""
    scenario = get_scenario_by_id("MULTI_02_2017_ORDERS_THEN_LINE_CHART")
    assert scenario is not None
    import plotly.graph_objects as go
    registry = ChartRegistry()
    fig = go.Figure(data=[go.Scatter(x=["2017-01"], y=[100])])
    cid = registry.save(fig.to_dict(), "ds_orders", "line", "2017 Orders")

    # Turn 2 reran SQL unnecessarily
    turns_output = [
        {
            "role": "assistant",
            "content": "Retrieved monthly orders.",
            "chart_ids": [],
            "tool_executions": [{"tool": "execute_sql", "summary": "Rows: 12"}],
        },
        {
            "role": "assistant",
            "content": "Re-ran SQL and generated chart.",
            "chart_ids": [cid],
            "tool_executions": [
                {"tool": "execute_sql", "summary": "Rows: 12"},
                {"tool": "generate_chart", "summary": f"Chart ID: {cid}"},
            ],
        },
    ]
    res = evaluate_scenario_result(scenario, turns_output, {"turn1_month_count": 12, "turn2_chart_type": "line"}, registry, 1.0)
    assert res.evidence["multiturn"]["turn2_reran_sql"] is True
    assert res.multiturn_success == 0.5


def test_adversarial_missing_chart_id_fails_visualization() -> None:
    """11. Missing Chart ID in visualization scenario must FAIL."""
    scenario = get_scenario_by_id("VIZ_01_MONTHLY_REVENUE_LINE")
    assert scenario is not None
    turns_output = [{
        "role": "assistant",
        "content": "Here is the chart you asked for.",
        "chart_ids": [],  # No chart registered
        "tool_executions": [{"tool": "generate_chart", "status": "error", "summary": "Failed"}],
    }]
    res = evaluate_scenario_result(scenario, turns_output, {}, ChartRegistry(), 1.0)
    assert res.status == "FAIL"
    assert res.visualization_success == 0.0


def test_adversarial_wrong_chart_data_fails() -> None:
    """12. Chart with insufficient/empty data must FAIL semantic validation."""
    import plotly.graph_objects as go
    registry = ChartRegistry()
    # Line chart with only 1 point when ~12 months expected
    fig = go.Figure(data=[go.Scatter(x=["2017-01"], y=[100])])
    cid = registry.save(fig.to_dict(), "ds_1", "line", "Too Few Months")

    score, ev = score_visualization(registry, [cid], expected_chart_type="line", scenario_id="VIZ_01_MONTHLY_REVENUE_LINE", ground_truth={})
    assert score == 0.0
    assert ev["semantics_valid"] is False


def test_adversarial_correct_chart_data_passes() -> None:
    """13. Chart with complete 12 months data must PASS semantic validation."""
    import plotly.graph_objects as go
    registry = ChartRegistry()
    months = [f"2017-{m:02d}" for m in range(1, 13)]
    fig = go.Figure(data=[go.Scatter(x=months, y=[1000] * 12)])
    cid = registry.save(fig.to_dict(), "ds_1", "line", "12 Months")

    score, ev = score_visualization(registry, [cid], expected_chart_type="line", scenario_id="VIZ_01_MONTHLY_REVENUE_LINE", ground_truth={})
    assert score == 1.0
    assert ev["semantics_valid"] is True


def test_adversarial_multiturn_chart_lookup_from_session_registry() -> None:
    """14. Evaluator should find chart in registry even if turn chart_ids list was not propagated."""
    import plotly.graph_objects as go
    registry = ChartRegistry()
    months = [f"2017-{m:02d}" for m in range(1, 13)]
    fig = go.Figure(data=[go.Scatter(x=months, y=[1000] * 12)])
    cid = registry.save(fig.to_dict(), "ds_orders", "line", "Monthly Orders")

    # Pass empty chart_ids list to score_visualization with fallback enabled; should fallback to registry
    score, ev = score_visualization(registry, [], expected_chart_type="line", scenario_id="MULTI_02_2017_ORDERS_THEN_LINE_CHART", ground_truth={}, allow_registry_fallback=True)
    assert score == 1.0
    assert ev["chart_id"] == cid


def test_adversarial_missing_lineage_marked_unobservable() -> None:
    """15. Missing dataset metadata must distinguish unobservable lineage from false claims."""
    scenario = get_scenario_by_id("MULTI_01_TOP5_THEN_VISUALIZE")
    assert scenario is not None
    import plotly.graph_objects as go
    registry = ChartRegistry()
    cats = ["health_beauty", "watches_gifts", "bed_bath_table", "sports_leisure", "computers_accessories"]
    fig = go.Figure(data=[go.Bar(x=cats, y=[10, 20, 30, 40, 50])])
    cid = registry.save(fig.to_dict(), "ds_top5", "bar", "Top 5")

    # Turn 2 has chart call but details lack dataset_id
    turns_output = [
        {
            "role": "assistant",
            "content": "Turn 1 output.",
            "chart_ids": [],
            "tool_executions": [{"tool": "execute_sql", "summary": "No dataset mentioned"}],
        },
        {
            "role": "assistant",
            "content": "Turn 2 chart.",
            "chart_ids": [cid],
            "tool_executions": [{"tool": "generate_chart", "summary": f"Chart ID: {cid}"}],
        },
    ]
    res = evaluate_scenario_result(scenario, turns_output, {}, registry, 1.0)
    assert res.evidence["multiturn"]["lineage_observable"] is False
    assert res.evidence["multiturn"]["lineage_verified"] is None


def test_adversarial_aggregate_denominators_correct() -> None:
    """16. Inapplicable metrics must not inflate analytical accuracy denominators."""
    runner = EvaluationRunner(mode="deterministic")
    # Include 1 basic_sql (applicable), 1 visualization (inapplicable for numeric accuracy)
    summary = runner.run_benchmark(scenario_ids=["SQL_01_DELIVERED_ORDERS", "VIZ_01_MONTHLY_REVENUE_LINE"])
    assert summary["total_scenarios"] == 2
    # Only SQL_01 has analytical accuracy evaluated
    assert summary["analytical_accuracy_total"] == 1
    assert summary["analytical_accuracy_count"] == 1
    assert summary["analytical_accuracy_pct"] == 100.0
    # Visualization evaluated only on VIZ_01
    assert summary["visualization_success_total"] == 1
    assert summary["visualization_success_count"] == 1
    assert summary["visualization_success_pct"] == 100.0

