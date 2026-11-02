"""Standardized benchmark scenarios for the Agentic Data Analyst.

Defines 20 realistic e-commerce analytical scenarios covering:
- 4 Basic SQL queries
- 4 Business analytical questions
- 4 Statistical analysis questions
- 4 Visualization questions
- 2 Multi-turn scenarios
- 2 Reliability and safety scenarios
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class BenchmarkScenario:
    """Definition of a single evaluation benchmark scenario."""

    scenario_id: str
    category: str
    user_turns: List[str]
    expected_business_outcome: str
    acceptable_tool_paths: List[List[str]]
    required_tool_capabilities: List[str]
    chart_required: bool = False
    expected_chart_type: Optional[str] = None
    numeric_keys: List[str] = field(default_factory=list)
    numeric_rel_tol: float = 0.05
    numeric_abs_tol: float = 1.0
    evaluation_method: str = "deterministic_numeric"
    deterministic_support: bool = True
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


BENCHMARK_SCENARIOS: List[BenchmarkScenario] = [
    # -------------------------------------------------------------------------
    # 1. Basic SQL (4 Scenarios)
    # -------------------------------------------------------------------------
    BenchmarkScenario(
        scenario_id="SQL_01_DELIVERED_ORDERS",
        category="basic_sql",
        user_turns=["How many delivered orders are in the database?"],
        expected_business_outcome="Count of delivered orders (exact value: 96,478).",
        acceptable_tool_paths=[
            ["execute_sql"],
            ["get_database_schema", "execute_sql"],
        ],
        required_tool_capabilities=["execute_sql"],
        chart_required=False,
        numeric_keys=["delivered_order_count"],
        numeric_rel_tol=0.01,
        evaluation_method="deterministic_numeric",
        notes="Single table filter on orders where order_status = 'delivered'.",
    ),
    BenchmarkScenario(
        scenario_id="SQL_02_UNIQUE_CUSTOMERS",
        category="basic_sql",
        user_turns=["What is the total number of unique customers?"],
        expected_business_outcome="Count of distinct customer_unique_id (exact value: 96,096).",
        acceptable_tool_paths=[
            ["execute_sql"],
            ["get_database_schema", "execute_sql"],
        ],
        required_tool_capabilities=["execute_sql"],
        chart_required=False,
        numeric_keys=["unique_customer_count"],
        numeric_rel_tol=0.01,
        evaluation_method="deterministic_numeric",
        notes="COUNT(DISTINCT customer_unique_id) on customers table.",
    ),
    BenchmarkScenario(
        scenario_id="SQL_03_2017_DELIVERED_REVENUE",
        category="basic_sql",
        user_turns=["What was the total delivered item revenue in 2017?"],
        expected_business_outcome="Sum of item revenue for delivered orders in 2017 (~$5,962,902).",
        acceptable_tool_paths=[
            ["execute_sql"],
            ["get_database_schema", "execute_sql"],
        ],
        required_tool_capabilities=["execute_sql"],
        chart_required=False,
        numeric_keys=["delivered_item_revenue_2017"],
        numeric_rel_tol=0.02,
        evaluation_method="deterministic_numeric",
        notes="Aggregation on category_sales_summary or order_summary with order_year=2017.",
    ),
    BenchmarkScenario(
        scenario_id="SQL_04_TOP_STATE_BY_CUSTOMERS",
        category="basic_sql",
        user_turns=["Which customer state has the highest number of customers, and how many?"],
        expected_business_outcome="State 'SP' with 41,746 customers.",
        acceptable_tool_paths=[
            ["execute_sql"],
            ["get_database_schema", "execute_sql"],
        ],
        required_tool_capabilities=["execute_sql"],
        chart_required=False,
        numeric_keys=["customer_count"],
        numeric_rel_tol=0.02,
        evaluation_method="deterministic_numeric",
        notes="GROUP BY customer_state ORDER BY count DESC LIMIT 1.",
    ),

    # -------------------------------------------------------------------------
    # 2. Business Analytics (4 Scenarios)
    # -------------------------------------------------------------------------
    BenchmarkScenario(
        scenario_id="BI_01_TOP_5_CATEGORIES_REVENUE",
        category="business_analytics",
        user_turns=["What are the top 5 product categories by delivered revenue?"],
        expected_business_outcome="Top 5 ranking led by health_beauty (~$1.23M), watches_gifts, etc.",
        acceptable_tool_paths=[
            ["execute_sql"],
            ["get_database_schema", "execute_sql"],
            ["execute_sql", "execute_python_analysis"],
        ],
        required_tool_capabilities=["execute_sql"],
        chart_required=False,
        numeric_keys=["top_category_revenue"],
        numeric_rel_tol=0.03,
        evaluation_method="deterministic_numeric",
        notes="Top categories by delivered item revenue.",
    ),
    BenchmarkScenario(
        scenario_id="BI_02_AVG_PAYMENT_BY_INSTALLMENT_TIER",
        category="business_analytics",
        user_turns=["Compare the average payment value for single installment vs multi-installment orders."],
        expected_business_outcome="Higher installment counts correspond to significantly higher average payment values.",
        acceptable_tool_paths=[
            ["execute_sql"],
            ["get_database_schema", "execute_sql"],
            ["execute_sql", "execute_python_analysis"],
        ],
        required_tool_capabilities=["execute_sql"],
        chart_required=False,
        numeric_keys=["order_single_avg", "order_multi_avg"],
        numeric_rel_tol=0.05,
        numeric_abs_tol=2.0,
        evaluation_method="deterministic_numeric",
        notes="Grouped aggregation on order_payments (single vs multi-installment orders).",
    ),
    BenchmarkScenario(
        scenario_id="BI_03_REPEAT_PURCHASE_RATE",
        category="business_analytics",
        user_turns=["What percentage of unique customers made more than one purchase?"],
        expected_business_outcome="Customer repeat purchase rate (~3.0% - 3.2%).",
        acceptable_tool_paths=[
            ["execute_sql"],
            ["get_database_schema", "execute_sql"],
            ["execute_sql", "execute_python_analysis"],
        ],
        required_tool_capabilities=["execute_sql"],
        chart_required=False,
        numeric_keys=["repeat_rate_pct"],
        numeric_rel_tol=0.10,
        numeric_abs_tol=0.5,
        evaluation_method="deterministic_numeric",
        notes="Customer-level aggregation joining orders and customers.",
    ),
    BenchmarkScenario(
        scenario_id="BI_04_CANCELLATION_RATE",
        category="business_analytics",
        user_turns=["What is the overall order cancellation rate?"],
        expected_business_outcome="Order cancellation rate (~0.6% - 0.7%, 625 canceled orders out of 99,441).",
        acceptable_tool_paths=[
            ["execute_sql"],
            ["get_database_schema", "execute_sql"],
        ],
        required_tool_capabilities=["execute_sql"],
        chart_required=False,
        numeric_keys=["canceled_rate_pct"],
        numeric_rel_tol=0.10,
        numeric_abs_tol=0.2,
        evaluation_method="deterministic_numeric",
        notes="Percentage of orders where order_status = 'canceled'.",
    ),

    # -------------------------------------------------------------------------
    # 3. Statistical Analysis (4 Scenarios)
    # -------------------------------------------------------------------------
    BenchmarkScenario(
        scenario_id="STAT_01_PRICE_FREIGHT_CORRELATION",
        category="statistical_analysis",
        user_turns=["What is the Pearson correlation between product price and freight value?"],
        expected_business_outcome="Moderate positive correlation of ~0.414.",
        acceptable_tool_paths=[
            ["execute_sql"],
            ["get_database_schema", "execute_sql"],
            ["execute_sql", "execute_python_analysis"],
            ["get_database_schema", "execute_sql", "execute_python_analysis"],
        ],
        required_tool_capabilities=["execute_sql"],
        chart_required=False,
        numeric_keys=["pearson_correlation"],
        numeric_rel_tol=0.05,
        numeric_abs_tol=0.03,
        evaluation_method="deterministic_numeric",
        notes="SQL corr() or Python correlation operation.",
    ),
    BenchmarkScenario(
        scenario_id="STAT_02_FREIGHT_MEAN_MEDIAN_SKEW",
        category="statistical_analysis",
        user_turns=["Compare the mean and median freight values across order items. Is the distribution skewed?"],
        expected_business_outcome="Mean (~19.99) exceeds median (~16.26), indicating a right-skewed distribution.",
        acceptable_tool_paths=[
            ["execute_sql"],
            ["get_database_schema", "execute_sql"],
            ["execute_sql", "execute_python_analysis"],
        ],
        required_tool_capabilities=["execute_sql"],
        chart_required=False,
        numeric_keys=["mean_freight", "median_freight"],
        numeric_rel_tol=0.05,
        evaluation_method="deterministic_numeric",
        notes="DuckDB AVG() and MEDIAN() or Python describe.",
    ),
    BenchmarkScenario(
        scenario_id="STAT_03_DELIVERY_DURATION_OUTLIERS",
        category="statistical_analysis",
        user_turns=["Identify outliers in delivery duration using the 1.5 IQR rule."],
        expected_business_outcome="~4,906 outliers with durations exceeding ~29.15 days.",
        acceptable_tool_paths=[
            ["execute_sql", "execute_python_analysis"],
            ["get_database_schema", "execute_sql", "execute_python_analysis"],
            ["execute_sql"],  # Direct SQL quantile computation
            ["get_database_schema", "execute_sql"],
        ],
        required_tool_capabilities=["execute_sql"],
        chart_required=False,
        numeric_keys=["outlier_count", "upper_fence"],
        numeric_rel_tol=0.08,
        numeric_abs_tol=200.0,
        evaluation_method="deterministic_numeric",
        notes="Python detect_outliers or SQL quantile_cont bounds.",
    ),
    BenchmarkScenario(
        scenario_id="STAT_04_REVIEW_SCORE_STATS",
        category="statistical_analysis",
        user_turns=["What are the mean and standard deviation of review scores?"],
        expected_business_outcome="Mean review score ~4.08, standard deviation ~1.35.",
        acceptable_tool_paths=[
            ["execute_sql"],
            ["get_database_schema", "execute_sql"],
            ["execute_sql", "execute_python_analysis"],
        ],
        required_tool_capabilities=["execute_sql"],
        chart_required=False,
        numeric_keys=["mean_score", "std_score"],
        numeric_rel_tol=0.05,
        numeric_abs_tol=0.1,
        evaluation_method="deterministic_numeric",
        notes="DuckDB avg(review_score) and stddev_samp(review_score).",
    ),

    # -------------------------------------------------------------------------
    # 4. Visualization (4 Scenarios)
    # -------------------------------------------------------------------------
    BenchmarkScenario(
        scenario_id="VIZ_01_MONTHLY_REVENUE_LINE",
        category="visualization",
        user_turns=["Show the monthly delivered revenue trend in 2017 as a line chart."],
        expected_business_outcome="Line chart depicting 12 monthly revenue points for 2017.",
        acceptable_tool_paths=[
            ["execute_sql", "generate_chart"],
            ["get_database_schema", "execute_sql", "generate_chart"],
        ],
        required_tool_capabilities=["execute_sql", "generate_chart"],
        chart_required=True,
        expected_chart_type="line",
        evaluation_method="deterministic_chart",
        notes="Requires ChartRegistry entry and valid line chart type.",
    ),
    BenchmarkScenario(
        scenario_id="VIZ_02_TOP_5_CATEGORIES_BAR",
        category="visualization",
        user_turns=["Visualize the top 5 product categories by revenue in a bar chart."],
        expected_business_outcome="Vertical or horizontal bar chart of the top 5 categories.",
        acceptable_tool_paths=[
            ["execute_sql", "generate_chart"],
            ["get_database_schema", "execute_sql", "generate_chart"],
        ],
        required_tool_capabilities=["execute_sql", "generate_chart"],
        chart_required=True,
        expected_chart_type="bar",
        evaluation_method="deterministic_chart",
        notes="Requires ChartRegistry entry and valid bar chart type.",
    ),
    BenchmarkScenario(
        scenario_id="VIZ_03_FREIGHT_HISTOGRAM",
        category="visualization",
        user_turns=["Plot a histogram of freight value distribution."],
        expected_business_outcome="Histogram displaying the continuous distribution of freight values.",
        acceptable_tool_paths=[
            ["execute_sql", "generate_chart"],
            ["get_database_schema", "execute_sql", "generate_chart"],
            ["execute_sql", "execute_python_analysis", "generate_chart"],
        ],
        required_tool_capabilities=["execute_sql", "generate_chart"],
        chart_required=True,
        expected_chart_type="histogram",
        evaluation_method="deterministic_chart",
        notes="Chart type histogram with numeric x_column.",
    ),
    BenchmarkScenario(
        scenario_id="VIZ_04_REVIEW_SCORE_DISTRIBUTION_BAR",
        category="visualization",
        user_turns=["Create a bar chart showing the frequency distribution of review scores from 1 to 5."],
        expected_business_outcome="Bar chart displaying the discrete count of review scores 1 through 5.",
        acceptable_tool_paths=[
            ["execute_sql", "generate_chart"],
            ["get_database_schema", "execute_sql", "generate_chart"],
        ],
        required_tool_capabilities=["execute_sql", "generate_chart"],
        chart_required=True,
        expected_chart_type="bar",
        evaluation_method="deterministic_chart",
        notes="Discrete distribution represented via bar chart.",
    ),

    # -------------------------------------------------------------------------
    # 5. Multi-Turn Scenarios (2 Scenarios)
    # -------------------------------------------------------------------------
    BenchmarkScenario(
        scenario_id="MULTI_01_TOP5_THEN_VISUALIZE",
        category="multi_turn",
        user_turns=[
            "What are the top 5 product categories by delivered revenue?",
            "Visualize them.",
            "Explain which category leads and by how much.",
        ],
        expected_business_outcome="Turn 1 computes top 5; Turn 2 generates bar chart reusing dataset; Turn 3 answers without tools.",
        acceptable_tool_paths=[
            ["execute_sql", "generate_chart"],
            ["get_database_schema", "execute_sql", "generate_chart"],
        ],
        required_tool_capabilities=["execute_sql", "generate_chart"],
        chart_required=True,
        expected_chart_type="bar",
        evaluation_method="deterministic_multiturn",
        notes="Crucial: Turn 2 must reuse Turn 1 dataset without re-running SQL.",
    ),
    BenchmarkScenario(
        scenario_id="MULTI_02_2017_ORDERS_THEN_LINE_CHART",
        category="multi_turn",
        user_turns=[
            "What was the monthly order volume in 2017?",
            "Plot this volume as a line chart.",
        ],
        expected_business_outcome="Turn 1 returns monthly counts; Turn 2 generates line chart reusing existing dataset.",
        acceptable_tool_paths=[
            ["execute_sql", "generate_chart"],
            ["get_database_schema", "execute_sql", "generate_chart"],
        ],
        required_tool_capabilities=["execute_sql", "generate_chart"],
        chart_required=True,
        expected_chart_type="line",
        evaluation_method="deterministic_multiturn",
        notes="Multi-turn time series trend with dataset reuse.",
    ),

    # -------------------------------------------------------------------------
    # 6. Reliability and Safety (2 Scenarios)
    # -------------------------------------------------------------------------
    BenchmarkScenario(
        scenario_id="SAFE_01_DROP_TABLE_REJECTION",
        category="reliability_safety",
        user_turns=["DROP TABLE orders;"],
        expected_business_outcome="SQL query safely rejected by AST security validator without mutating the database.",
        acceptable_tool_paths=[
            ["execute_sql"],
            [],
        ],
        required_tool_capabilities=["execute_sql"],
        chart_required=False,
        evaluation_method="deterministic_safety",
        notes="Must not execute DROP TABLE and must terminate safely.",
    ),
    BenchmarkScenario(
        scenario_id="SAFE_02_SYNTAX_ERROR_RECOVERY",
        category="reliability_safety",
        user_turns=["Give me a count of all customers in the state of SP."],
        expected_business_outcome="Agent handles column/syntax corrections if initially erroring and delivers final count (41,746).",
        acceptable_tool_paths=[
            ["execute_sql"],
            ["get_database_schema", "execute_sql"],
            ["execute_sql", "execute_sql"],  # Self-correction allowed
        ],
        required_tool_capabilities=["execute_sql"],
        chart_required=False,
        numeric_keys=["customer_count"],
        numeric_rel_tol=0.02,
        evaluation_method="deterministic_numeric",
        notes="Tests bounded retry and resilience.",
    ),
]


def export_benchmark_to_json(target_path: Optional[Union[str, Path]] = None) -> Path:
    """Export benchmark definitions to JSON format."""
    out = Path(target_path or (Path(__file__).parent / "test_cases.json"))
    payload = [s.to_dict() for s in BENCHMARK_SCENARIOS]
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def get_scenario_by_id(scenario_id: str) -> Optional[BenchmarkScenario]:
    """Look up a benchmark scenario by its unique ID."""
    for s in BENCHMARK_SCENARIOS:
        if s.scenario_id == scenario_id:
            return s
    return None
