"""Automated Evaluation Runner for the Agentic Data Analyst benchmark.

Supports two distinct execution modes:
1. Deterministic Mode (Default): Runs with simulated/scripted responses without OpenAI API calls,
   validating tool orchestration, extraction, metrics calculation, and reporting in CI/pytest.
2. Live API Mode (--live): Runs against the live OpenAI model and Olist DuckDB database,
   measuring real-world analytical accuracy, tool path selection, and latency.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agent.ui_session import AgentSession, create_agent_session, execute_session_turn
from evaluation.benchmark import BENCHMARK_SCENARIOS, BenchmarkScenario, get_scenario_by_id
from evaluation.ground_truth import GroundTruthOracle
from evaluation.metrics import ScenarioEvaluationResult, evaluate_scenario_result
from tools.chart_registry import ChartRegistry
from tools.dataset_store import DatasetStore

logger = logging.getLogger("eval_runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def run_deterministic_scenario_mock(
    scenario: BenchmarkScenario,
    ground_truth: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], ChartRegistry]:
    """Simulate realistic agent execution turns deterministically for CI and tests."""
    registry = ChartRegistry()
    store = DatasetStore()

    turns_output: List[Dict[str, Any]] = []

    if scenario.scenario_id == "SQL_01_DELIVERED_ORDERS":
        turns_output.append({
            "role": "assistant",
            "content": f"There are {ground_truth.get('delivered_order_count', 96478):,} delivered orders.",
            "chart_ids": [],
            "tool_executions": [{"tool": "execute_sql", "summary": "Rows: 1"}],
        })

    elif scenario.scenario_id == "SQL_02_UNIQUE_CUSTOMERS":
        turns_output.append({
            "role": "assistant",
            "content": f"The total number of unique customers is {ground_truth.get('unique_customer_count', 96096):,}.",
            "chart_ids": [],
            "tool_executions": [{"tool": "execute_sql", "summary": "Rows: 1"}],
        })

    elif scenario.scenario_id == "SQL_03_2017_DELIVERED_REVENUE":
        rev = ground_truth.get("delivered_item_revenue_2017", 5962902.01)
        turns_output.append({
            "role": "assistant",
            "content": f"Total delivered item revenue in 2017 was ${rev:,.2f}.",
            "chart_ids": [],
            "tool_executions": [{"tool": "execute_sql", "summary": "Rows: 1"}],
        })

    elif scenario.scenario_id == "SQL_04_TOP_STATE_BY_CUSTOMERS":
        st = ground_truth.get("top_state", "SP")
        cnt = ground_truth.get("customer_count", 41746)
        turns_output.append({
            "role": "assistant",
            "content": f"State {st} has the highest customer count with {cnt:,} customers.",
            "chart_ids": [],
            "tool_executions": [{"tool": "execute_sql", "summary": "Rows: 1"}],
        })

    elif scenario.scenario_id == "BI_01_TOP_5_CATEGORIES_REVENUE":
        top_name = ground_truth.get("top_category_name", "health_beauty")
        top_rev = ground_truth.get("top_category_revenue", 1233131.72)
        turns_output.append({
            "role": "assistant",
            "content": f"Top category is {top_name} with delivered revenue of ${top_rev:,.2f}.",
            "chart_ids": [],
            "tool_executions": [{"tool": "execute_sql", "summary": "Rows: 5"}],
        })

    elif scenario.scenario_id == "BI_02_AVG_PAYMENT_BY_INSTALLMENT_TIER":
        single_val = ground_truth.get("order_single_avg", 121.04)
        multi_val = ground_truth.get("order_multi_avg", 198.68)
        turns_output.append({
            "role": "assistant",
            "content": f"Average payment value for single installment orders is ${single_val:.2f}, compared to ${multi_val:.2f} for multi-installment orders.",
            "chart_ids": [],
            "tool_executions": [{"tool": "execute_sql", "summary": "Rows: 2"}],
        })

    elif scenario.scenario_id == "BI_03_REPEAT_PURCHASE_RATE":
        pct = ground_truth.get("repeat_rate_pct", 3.12)
        turns_output.append({
            "role": "assistant",
            "content": f"The repeat purchase rate among unique customers is {pct:.2f}%.",
            "chart_ids": [],
            "tool_executions": [{"tool": "execute_sql", "summary": "Rows: 1"}],
        })

    elif scenario.scenario_id == "BI_04_CANCELLATION_RATE":
        pct = ground_truth.get("canceled_rate_pct", 0.63)
        turns_output.append({
            "role": "assistant",
            "content": f"The overall order cancellation rate is {pct:.2f}%.",
            "chart_ids": [],
            "tool_executions": [{"tool": "execute_sql", "summary": "Rows: 1"}],
        })

    elif scenario.scenario_id == "STAT_01_PRICE_FREIGHT_CORRELATION":
        corr = ground_truth.get("pearson_correlation", 0.4142)
        turns_output.append({
            "role": "assistant",
            "content": f"The Pearson correlation between product price and freight value is {corr:.4f}.",
            "chart_ids": [],
            "tool_executions": [{"tool": "execute_sql", "summary": "Rows: 1"}],
        })

    elif scenario.scenario_id == "STAT_02_FREIGHT_MEAN_MEDIAN_SKEW":
        mean_v = ground_truth.get("mean_freight", 19.99)
        med_v = ground_truth.get("median_freight", 16.26)
        turns_output.append({
            "role": "assistant",
            "content": f"Mean freight is {mean_v:.2f} and median is {med_v:.2f}. The distribution is right-skewed.",
            "chart_ids": [],
            "tool_executions": [{"tool": "execute_sql", "summary": "Rows: 1"}],
        })

    elif scenario.scenario_id == "STAT_03_DELIVERY_DURATION_OUTLIERS":
        outliers = ground_truth.get("outlier_count", 4906)
        fence = ground_truth.get("upper_fence", 29.145)
        turns_output.append({
            "role": "assistant",
            "content": f"Found {outliers:,} outliers with delivery duration exceeding {fence:.3f} days.",
            "chart_ids": [],
            "tool_executions": [
                {"tool": "execute_sql", "summary": "Rows: 1000"},
                {"tool": "execute_python_analysis", "summary": "detect_outliers"},
            ],
        })

    elif scenario.scenario_id == "STAT_04_REVIEW_SCORE_STATS":
        mean_s = ground_truth.get("mean_score", 4.08)
        std_s = ground_truth.get("std_score", 1.35)
        turns_output.append({
            "role": "assistant",
            "content": f"Review scores have a mean of {mean_s:.2f} and standard deviation of {std_s:.2f}.",
            "chart_ids": [],
            "tool_executions": [{"tool": "execute_sql", "summary": "Rows: 1"}],
        })

    elif scenario.chart_required and scenario.category == "visualization":
        import plotly.graph_objects as go
        if scenario.scenario_id == "VIZ_01_MONTHLY_REVENUE_LINE":
            months = [f"2017-{m:02d}" for m in range(1, 13)]
            revs = [500000.0] * 12
            fig = go.Figure(data=[go.Scatter(x=months, y=revs, mode="lines+markers")])
        elif scenario.scenario_id == "VIZ_02_TOP_5_CATEGORIES_BAR":
            cats = ["health_beauty", "watches_gifts", "bed_bath_table", "sports_leisure", "computers_accessories"]
            revs = [1233131, 1166176, 1023434, 954852, 915234]
            fig = go.Figure(data=[go.Bar(x=cats, y=revs)])
        elif scenario.scenario_id == "VIZ_03_FREIGHT_HISTOGRAM":
            fig = go.Figure(data=[go.Histogram(x=[10.0, 15.0, 20.0, 25.0, 30.0, 50.0, 100.0] * 10)])
        elif scenario.scenario_id == "VIZ_04_REVIEW_SCORE_DISTRIBUTION_BAR":
            fig = go.Figure(data=[go.Bar(x=[1, 2, 3, 4, 5], y=[11424, 3151, 8179, 19142, 57328])])
        else:
            fig = go.Figure(data=[go.Bar(x=["A", "B", "C", "D", "E"], y=[10, 20, 30, 40, 50])])

        cid = registry.save(
            figure=fig.to_dict(),
            dataset_id="ds_mock_1",
            chart_type=scenario.expected_chart_type or "bar",
            title=f"Mock {scenario.expected_chart_type} chart",
        )
        turns_output.append({
            "role": "assistant",
            "content": f"Here is the requested {scenario.expected_chart_type} visualization.",
            "chart_ids": [cid],
            "tool_executions": [
                {"tool": "execute_sql", "summary": "Rows: 10"},
                {"tool": "generate_chart", "summary": f"Status: success | Chart ID: {cid}"},
            ],
        })

    elif scenario.scenario_id == "MULTI_01_TOP5_THEN_VISUALIZE":
        import plotly.graph_objects as go
        # Turn 1: SQL
        turns_output.append({
            "role": "assistant",
            "content": "Top 5 categories: health_beauty ($1.23M), watches_gifts, bed_bath_table, sports_leisure, computers_accessories.",
            "chart_ids": [],
            "tool_executions": [{"tool": "execute_sql", "summary": "Status: completed | Rows: 5 | Dataset ID: ds_top5"}],
        })
        # Turn 2: Chart (reusing dataset, no SQL)
        cats = ["health_beauty", "watches_gifts", "bed_bath_table", "sports_leisure", "computers_accessories"]
        revs = [1233131, 1166176, 1023434, 954852, 915234]
        fig = go.Figure(data=[go.Bar(x=cats, y=revs)])
        cid = registry.save(figure=fig.to_dict(), dataset_id="ds_top5", chart_type="bar", title="Top 5 Categories")
        turns_output.append({
            "role": "assistant",
            "content": "Here is the bar chart visualization.",
            "chart_ids": [cid],
            "tool_executions": [{"tool": "generate_chart", "details": '{"dataset_id": "ds_top5", "chart_type": "bar"}', "summary": f"Status: success | Chart ID: {cid}"}],
        })
        # Turn 3: Explanation without tools
        turns_output.append({
            "role": "assistant",
            "content": "Health & Beauty leads Watches & Gifts by $66,954.74 (+5.7%).",
            "chart_ids": [],
            "tool_executions": [],
        })

    elif scenario.scenario_id == "MULTI_02_2017_ORDERS_THEN_LINE_CHART":
        import plotly.graph_objects as go
        # Turn 1: SQL
        turns_output.append({
            "role": "assistant",
            "content": "Monthly order counts for 2017 retrieved for 12 months.",
            "chart_ids": [],
            "tool_executions": [{"tool": "execute_sql", "summary": "Status: completed | Rows: 12 | Dataset ID: ds_orders"}],
        })
        # Turn 2: Chart
        months = [f"2017-{m:02d}" for m in range(1, 13)]
        orders = [800, 1780, 2682, 2404, 3700, 3245, 4026, 4331, 4285, 4631, 7544, 5673]
        fig = go.Figure(data=[go.Scatter(x=months, y=orders, mode="lines+markers")])
        cid = registry.save(figure=fig.to_dict(), dataset_id="ds_orders", chart_type="line", title="2017 Monthly Orders")
        turns_output.append({
            "role": "assistant",
            "content": "Here is the monthly order trend line chart.",
            "chart_ids": [cid],
            "tool_executions": [{"tool": "generate_chart", "details": '{"dataset_id": "ds_orders", "chart_type": "line"}', "summary": f"Status: success | Chart ID: {cid}"}],
        })

    elif scenario.scenario_id == "SAFE_01_DROP_TABLE_REJECTION":
        turns_output.append({
            "role": "assistant",
            "content": "Execution stopped: the SQL query was rejected by security policy (SQLValidationError). Drop table is not permitted.",
            "chart_ids": [],
            "tool_executions": [{"tool": "execute_sql", "status": "error", "summary": "SQLValidationError: DROP TABLE is not permitted."}],
        })

    elif scenario.scenario_id == "SAFE_02_SYNTAX_ERROR_RECOVERY":
        cnt = ground_truth.get("customer_count", 41746)
        turns_output.append({
            "role": "assistant",
            "content": f"Customer count in SP is {cnt:,}.",
            "chart_ids": [],
            "tool_executions": [{"tool": "execute_sql", "summary": "Rows: 1"}],
        })

    else:
        turns_output.append({
            "role": "assistant",
            "content": "Default response.",
            "chart_ids": [],
            "tool_executions": [{"tool": "execute_sql", "summary": "Rows: 1"}],
        })

    return turns_output, registry


class EvaluationRunner:
    """Manages end-to-end execution and scoring of benchmark scenarios."""

    def __init__(
        self,
        mode: str = "deterministic",
        db_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.mode = mode.lower()
        self.db_path = db_path
        self.oracle = GroundTruthOracle(db_path=db_path)
        self.output_dir = Path(output_dir or (Path(__file__).parent / "results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_scenario(self, scenario: BenchmarkScenario) -> ScenarioEvaluationResult:
        """Execute a single scenario across its turn sequence and return evaluated result."""
        t0 = time.time()

        try:
            ground_truth = self.oracle.get_reference_for_scenario(scenario.scenario_id)

            if self.mode == "live":
                # Real execution in isolated fresh session
                session = create_agent_session(db_path=self.db_path)
                turns_output: List[Dict[str, Any]] = []

                for turn_idx, turn_prompt in enumerate(scenario.user_turns):
                    turn_res = execute_session_turn(
                        session,
                        turn_prompt,
                        source="evaluation",
                        scenario_id=scenario.scenario_id,
                        turn_index=turn_idx,
                    )
                    turns_output.append(turn_res)

                latency = time.time() - t0
                return evaluate_scenario_result(
                    scenario=scenario,
                    turns_output=turns_output,
                    ground_truth=ground_truth,
                    chart_registry=session.chart_registry,
                    latency_seconds=latency,
                )

            else:
                # Deterministic execution
                turns_output, mock_registry = run_deterministic_scenario_mock(scenario, ground_truth)
                latency = time.time() - t0
                return evaluate_scenario_result(
                    scenario=scenario,
                    turns_output=turns_output,
                    ground_truth=ground_truth,
                    chart_registry=mock_registry,
                    latency_seconds=latency,
                )

        except Exception as exc:
            logger.exception("Failed to execute scenario %s: %s", scenario.scenario_id, exc)
            return ScenarioEvaluationResult(
                scenario_id=scenario.scenario_id,
                status="ERROR",
                task_success=False,
                analytical_accuracy=0.0,
                tool_selection_quality=0.0,
                visualization_success=0.0 if scenario.chart_required else None,
                multiturn_success=0.0 if scenario.category == "multi_turn" else None,
                latency_seconds=time.time() - t0,
                actual_tool_sequence=[],
                error_message=str(exc),
            )

    def run_benchmark(
        self,
        scenario_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Execute all or selected benchmark scenarios and produce summary statistics."""
        targets = [
            s for s in BENCHMARK_SCENARIOS
            if (scenario_ids is None or s.scenario_id in scenario_ids)
        ]

        logger.info("Starting benchmark run in mode=%s for %d scenarios", self.mode, len(targets))
        results: List[ScenarioEvaluationResult] = []

        for idx, scenario in enumerate(targets, 1):
            logger.info("[%d/%d] Running scenario: %s", idx, len(targets), scenario.scenario_id)
            res = self.run_scenario(scenario)
            results.append(res)
            logger.info("  -> Status: %s | Success: %s | Latency: %.2fs", res.status, res.task_success, res.latency_seconds)

        summary = self._compute_aggregate_metrics(results)
        report_files = self._save_reports(results, summary)
        summary["report_files"] = report_files
        return summary

    def _compute_aggregate_metrics(self, results: List[ScenarioEvaluationResult]) -> Dict[str, Any]:
        """Aggregate performance and reliability statistics across scenarios."""
        total = len(results)
        if total == 0:
            return {"total_scenarios": 0}

        passed_count = sum(1 for r in results if r.status == "PASS")
        task_success_count = sum(1 for r in results if r.task_success)
        needs_review_count = sum(1 for r in results if r.status == "NEEDS_REVIEW")
        fail_count = sum(1 for r in results if r.status == "FAIL")
        error_count = sum(1 for r in results if r.status == "ERROR")

        latencies = [r.latency_seconds for r in results]
        p50_latency = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

        accuracies = [r.analytical_accuracy for r in results if r.analytical_accuracy is not None]
        avg_accuracy = (sum(accuracies) / len(accuracies)) if accuracies else 1.0

        tool_qualities = [r.tool_selection_quality for r in results]
        avg_tool_quality = (sum(tool_qualities) / len(tool_qualities)) if tool_qualities else 1.0

        viz_scores = [r.visualization_success for r in results if r.visualization_success is not None]
        avg_viz = (sum(viz_scores) / len(viz_scores)) if viz_scores else 1.0

        multiturn_scores = [r.multiturn_success for r in results if r.multiturn_success is not None]
        avg_multiturn = (sum(multiturn_scores) / len(multiturn_scores)) if multiturn_scores else 1.0

        # Breakdown by category
        categories: Dict[str, Dict[str, Any]] = {}
        for r in results:
            sc = get_scenario_by_id(r.scenario_id)
            cat = sc.category if sc else "unknown"
            if cat not in categories:
                categories[cat] = {"total": 0, "pass": 0, "success": 0}
            categories[cat]["total"] += 1
            if r.status == "PASS":
                categories[cat]["pass"] += 1
            if r.task_success:
                categories[cat]["success"] += 1

        for cat, data in categories.items():
            data["pass_rate"] = round(data["pass"] / data["total"], 3) if data["total"] else 0.0

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "mode": self.mode,
            "total_scenarios": total,
            "pass_count": passed_count,
            "task_success_count": task_success_count,
            "needs_review_count": needs_review_count,
            "fail_count": fail_count,
            "error_count": error_count,
            "pass_rate_pct": round((passed_count / total) * 100.0, 1),
            "task_success_rate_pct": round((task_success_count / total) * 100.0, 1),
            "analytical_accuracy_pct": round(avg_accuracy * 100.0, 1),
            "analytical_accuracy_count": sum(1 for a in accuracies if a >= 0.99),
            "analytical_accuracy_total": len(accuracies),
            "tool_selection_quality_pct": round(avg_tool_quality * 100.0, 1),
            "visualization_success_pct": round(avg_viz * 100.0, 1),
            "visualization_success_count": sum(1 for v in viz_scores if v >= 0.99),
            "visualization_success_total": len(viz_scores),
            "multiturn_success_pct": round(avg_multiturn * 100.0, 1),
            "multiturn_success_count": sum(1 for m in multiturn_scores if m >= 0.99),
            "multiturn_success_total": len(multiturn_scores),
            "latency_p50_seconds": round(p50_latency, 3),
            "latency_p95_seconds": round(p95_latency, 3),
            "category_breakdown": categories,
            "results": [r.to_dict() for r in results],
        }

    def _save_reports(
        self,
        results: List[ScenarioEvaluationResult],
        summary: Dict[str, Any],
    ) -> Dict[str, str]:
        """Save structured JSON and CSV reports into the evaluation results directory."""
        ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"eval_report_{self.mode}_{ts_str}.json"
        csv_path = self.output_dir / f"eval_report_{self.mode}_{ts_str}.csv"

        # 1. Save JSON
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # 2. Save CSV
        fieldnames = [
            "scenario_id",
            "status",
            "task_success",
            "analytical_accuracy",
            "tool_selection_quality",
            "visualization_success",
            "multiturn_success",
            "latency_seconds",
            "tools_called",
            "error_message",
        ]
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "scenario_id": r.scenario_id,
                    "status": r.status,
                    "task_success": r.task_success,
                    "analytical_accuracy": r.analytical_accuracy,
                    "tool_selection_quality": r.tool_selection_quality,
                    "visualization_success": r.visualization_success,
                    "multiturn_success": r.multiturn_success,
                    "latency_seconds": round(r.latency_seconds, 3),
                    "tools_called": " -> ".join(r.actual_tool_sequence),
                    "error_message": r.error_message or "",
                })

        logger.info("Evaluation reports successfully generated:\n  - JSON: %s\n  - CSV: %s", json_path, csv_path)
        return {"json": str(json_path), "csv": str(csv_path)}


def print_summary_table(summary: Dict[str, Any]) -> None:
    """Print an aesthetic tabular console summary of evaluation outcomes."""
    print("\n" + "=" * 80)
    print(f" AGENTIC DATA ANALYST — BENCHMARK EVALUATION SUMMARY ({summary.get('mode', '').upper()})")
    print("=" * 80)
    print(f"Total Scenarios Evaluated : {summary.get('total_scenarios')}")
    print(f"Task Success Rate        : {summary.get('task_success_rate_pct')}% ({summary.get('task_success_count')}/{summary.get('total_scenarios')})")
    print(f"Strict Pass Rate          : {summary.get('pass_rate_pct')}% ({summary.get('pass_count')}/{summary.get('total_scenarios')})")
    print(f"Needs Review Rate         : {summary.get('needs_review_count')}")
    acc_count = summary.get('analytical_accuracy_count', 0)
    acc_total = summary.get('analytical_accuracy_total', 0)
    print(f"Analytical Accuracy       : {summary.get('analytical_accuracy_pct')}% ({acc_count}/{acc_total})")
    print(f"Tool Selection Quality    : {summary.get('tool_selection_quality_pct')}%")
    viz_count = summary.get('visualization_success_count', 0)
    viz_total = summary.get('visualization_success_total', 0)
    print(f"Visualization Success     : {summary.get('visualization_success_pct')}% ({viz_count}/{viz_total})")
    mt_count = summary.get('multiturn_success_count', 0)
    mt_total = summary.get('multiturn_success_total', 0)
    print(f"Multi-turn Success        : {summary.get('multiturn_success_pct')}% ({mt_count}/{mt_total})")
    print(f"Latency P50 / P95         : {summary.get('latency_p50_seconds')}s / {summary.get('latency_p95_seconds')}s")
    print("-" * 80)
    print(f"{'Category':<24} | {'Total':<6} | {'Pass':<6} | {'Success':<8} | {'Pass Rate':<10}")
    print("-" * 80)
    for cat, d in summary.get("category_breakdown", {}).items():
        print(f"{cat:<24} | {d['total']:<6} | {d['pass']:<6} | {d['success']:<8} | {d['pass_rate']*100:.1f}%")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic Data Analyst Evaluation Benchmark Runner")
    parser.add_argument("--live", action="store_true", help="Run in Live API Mode using OpenAI")
    parser.add_argument("--scenario", type=str, default=None, help="Filter by specific scenario ID")
    parser.add_argument("--out-dir", type=str, default=None, help="Custom output directory for reports")
    parser.add_argument("--langsmith", action="store_true", help="Publish evaluation results to LangSmith experiment")
    args = parser.parse_args()

    mode = "live" if args.live else "deterministic"
    scenario_ids = [args.scenario] if args.scenario else None

    if args.langsmith:
        from evaluation.langsmith_integration import run_langsmith_evaluation
        summary = run_langsmith_evaluation(
            scenario_ids=scenario_ids,
            live=args.live,
            output_dir=args.out_dir,
        )
    else:
        runner = EvaluationRunner(mode=mode, output_dir=args.out_dir)
        summary = runner.run_benchmark(scenario_ids=scenario_ids)

    print_summary_table(summary)


if __name__ == "__main__":
    main()
