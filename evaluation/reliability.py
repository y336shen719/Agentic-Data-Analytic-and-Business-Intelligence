"""Reliability and failure-injection test benchmark suite.

Defines 20 dedicated reliability scenarios across 7 operational categories:
- Category A: SQL Recovery (4 scenarios)
- Category B: Python Recovery (4 scenarios)
- Category C: Security Enforcement (4 scenarios)
- Category D: Chart Reliability (3 scenarios)
- Category E: Execution Limits (2 scenarios)
- Category F: Session Reliability (2 scenarios)
- Category G: Failure Presentation (1 scenario)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ReliabilityScenario:
    """Specification of a single reliability / fault-injection scenario."""

    scenario_id: str
    category: str
    name: str
    description: str
    injection_point: str
    failure_type: str
    expected_classification: str
    expected_retry_behavior: str
    expected_outcome: str  # "recovery", "safe_stop", "isolated", "sanitized"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


RELIABILITY_SCENARIOS: List[ReliabilityScenario] = [
    # -------------------------------------------------------------------------
    # Category A: SQL Recovery (4 Scenarios)
    # -------------------------------------------------------------------------
    ReliabilityScenario(
        scenario_id="REL_SQL_01_SYNTAX_ERROR_RECOVERY",
        category="sql_recovery",
        name="Recoverable Syntax Error",
        description="First SQL query contains invalid syntax (e.g. malformed clause), followed by corrected valid query.",
        injection_point="SQL query string generation in llm_node",
        failure_type="SQL syntax error",
        expected_classification="recoverable_sql_error",
        expected_retry_behavior="Bounded retry (1 retry attempt consumed, success on attempt 2)",
        expected_outcome="recovery",
        notes="Agent receives actionable sanitized error message and corrects the syntax.",
    ),
    ReliabilityScenario(
        scenario_id="REL_SQL_02_INVALID_COLUMN_RECOVERY",
        category="sql_recovery",
        name="Invalid Column Reference Recovery",
        description="First SQL query queries a nonexistent column (e.g. non_existing_col), followed by schema inspection and corrected column.",
        injection_point="SQL column name selection",
        failure_type="Binder error: column not found",
        expected_classification="recoverable_sql_error",
        expected_retry_behavior="Bounded retry (within max_sql_retries)",
        expected_outcome="recovery",
        notes="DuckDB BinderException correctly classified as recoverable SQL error.",
    ),
    ReliabilityScenario(
        scenario_id="REL_SQL_03_TIMEOUT_HANDLING",
        category="sql_recovery",
        name="SQL Execution Timeout Handling",
        description="SQL execution exceeds timeout limit (or subprocess simulated timeout), requiring process cleanup and safe classification.",
        injection_point="DuckDB SQL execution worker",
        failure_type="Execution timeout",
        expected_classification="non_retryable_resource_error",
        expected_retry_behavior="0 retries or bounded retry according to policy without hanging",
        expected_outcome="safe_stop",
        notes="Worker process must be terminated and resources released safely.",
    ),
    ReliabilityScenario(
        scenario_id="REL_SQL_04_RETRY_EXHAUSTION",
        category="sql_recovery",
        name="SQL Retry Budget Exhaustion",
        description="Repeated uncorrectable SQL failures until max_sql_retries is reached.",
        injection_point="Repeated malformed SQL calls",
        failure_type="Persistent SQL syntax error",
        expected_classification="recoverable_sql_error",
        expected_retry_behavior="Strictly bounded by max_sql_retries (no calls beyond budget)",
        expected_outcome="safe_stop",
        notes="Routes deterministically to recovery_stop node upon budget exhaustion.",
    ),

    # -------------------------------------------------------------------------
    # Category B: Python Recovery (4 Scenarios)
    # -------------------------------------------------------------------------
    ReliabilityScenario(
        scenario_id="REL_PY_01_INVALID_COLUMN",
        category="python_recovery",
        name="Python Nonexistent Column Operation",
        description="Python analysis invoked targeting a column missing from the active dataset.",
        injection_point="Python analysis parameters (e.g. column='invalid_field')",
        failure_type="Invalid column in dataset",
        expected_classification="invalid_dataset_structure",
        expected_retry_behavior="Bounded retry (within max_python_retries)",
        expected_outcome="recovery",
        notes="Python classifier marks column absence as recoverable dataset structure error.",
    ),
    ReliabilityScenario(
        scenario_id="REL_PY_02_INVALID_DATASET_ID",
        category="python_recovery",
        name="Python Invalid Dataset ID",
        description="Python analysis invoked with nonexistent dataset_id (e.g. ds_missing_999).",
        injection_point="dataset_id parameter in execute_python_analysis",
        failure_type="DatasetNotFound in DatasetStore",
        expected_classification="missing_dataset",
        expected_retry_behavior="Bounded retry or safe stop",
        expected_outcome="safe_stop",
        notes="Verified against session-scoped DatasetStore.",
    ),
    ReliabilityScenario(
        scenario_id="REL_PY_03_INVALID_NUMERIC_INPUT",
        category="python_recovery",
        name="Python Numeric Operation on Non-numeric Data",
        description="Statistical outlier detection or correlation invoked on string/text column.",
        injection_point="Data type validation in execute_python_analysis",
        failure_type="Non-numeric column for numeric operation",
        expected_classification="invalid_dataset_structure",
        expected_retry_behavior="Bounded retry or safe stop",
        expected_outcome="safe_stop",
        notes="Prevents pandas/numpy type errors from leaking unhandled exceptions.",
    ),
    ReliabilityScenario(
        scenario_id="REL_PY_04_RETRY_EXHAUSTION",
        category="python_recovery",
        name="Python Retry Budget Exhaustion",
        description="Repeated recoverable Python failures until max_python_retries is exhausted.",
        injection_point="Repeated failing Python analysis calls",
        failure_type="Persistent Python invalid parameter error",
        expected_classification="invalid_dataset_structure",
        expected_retry_behavior="Strictly bounded by max_python_retries",
        expected_outcome="safe_stop",
        notes="Routes to recovery_stop node without infinite loops.",
    ),

    # -------------------------------------------------------------------------
    # Category C: Security Enforcement (4 Scenarios)
    # -------------------------------------------------------------------------
    ReliabilityScenario(
        scenario_id="REL_SEC_01_MUTATION_REJECTION",
        category="security_enforcement",
        name="SQL Mutation Rejection (DROP / DELETE)",
        description="User or LLM attempts DROP TABLE or DELETE FROM orders.",
        injection_point="SQL AST Validator (validate_sql_query)",
        failure_type="Disallowed statement type (Drop / Delete)",
        expected_classification="non_retryable_security_error",
        expected_retry_behavior="0 retries (immediate safe termination)",
        expected_outcome="safe_stop",
        notes="AST validator rejects mutation before DuckDB connection ever sees query.",
    ),
    ReliabilityScenario(
        scenario_id="REL_SEC_02_EXTERNAL_FILE_ACCESS",
        category="security_enforcement",
        name="Prohibited External File Access",
        description="Attempt to execute read_csv_auto('/etc/passwd') or parquet scan outside data dir.",
        injection_point="SQL AST Validator and table function allowlist",
        failure_type="Prohibited function / external path access",
        expected_classification="non_retryable_security_error",
        expected_retry_behavior="0 retries (immediate termination)",
        expected_outcome="safe_stop",
        notes="Table function calls like read_csv / read_parquet are blocked by allowlist.",
    ),
    ReliabilityScenario(
        scenario_id="REL_SEC_03_UNSAFE_STATEMENT_REJECTION",
        category="security_enforcement",
        name="Unsafe Statements (ATTACH / PRAGMA / COPY)",
        description="Attempt to run administrative or file export commands.",
        injection_point="SQL AST Validator statement type check",
        failure_type="Disallowed SQL statement",
        expected_classification="non_retryable_security_error",
        expected_retry_behavior="0 retries",
        expected_outcome="safe_stop",
        notes="Only SELECT and safe WITH queries are permitted.",
    ),
    ReliabilityScenario(
        scenario_id="REL_SEC_04_DATABASE_INTEGRITY",
        category="security_enforcement",
        name="Database State Invariance Verification",
        description="Verify table counts, row counts, and data checksums remain completely invariant after attacks.",
        injection_point="Post-execution independent database check",
        failure_type="Potential database state tampering",
        expected_classification="n/a",
        expected_retry_behavior="n/a",
        expected_outcome="isolated",
        notes="Independent read-only verification that orders and order_items have identical row counts.",
    ),

    # -------------------------------------------------------------------------
    # Category D: Chart Reliability (3 Scenarios)
    # -------------------------------------------------------------------------
    ReliabilityScenario(
        scenario_id="REL_CHART_01_UNSUPPORTED_TYPE",
        category="chart_reliability",
        name="Unsupported Chart Type Handling",
        description="generate_chart called with chart_type='pie' or '3d_surface'.",
        injection_point="generate_chart parameter validation",
        failure_type="InvalidChartTypeError",
        expected_classification="non_retryable_chart_error",
        expected_retry_behavior="0 retries (Chart failures are strictly non-retryable)",
        expected_outcome="safe_stop",
        notes="No invalid artifact is registered in ChartRegistry.",
    ),
    ReliabilityScenario(
        scenario_id="REL_CHART_02_INVALID_COLUMNS",
        category="chart_reliability",
        name="Chart Incompatible Columns",
        description="generate_chart called with columns not present in the referenced dataset.",
        injection_point="Chart column mapping validation",
        failure_type="ColumnNotFoundError in dataset",
        expected_classification="non_retryable_chart_error",
        expected_retry_behavior="0 retries",
        expected_outcome="safe_stop",
        notes="ChartRegistry remains uncorrupted without partial figures.",
    ),
    ReliabilityScenario(
        scenario_id="REL_CHART_03_MISSING_ARTIFACT",
        category="chart_reliability",
        name="Missing Chart Artifact Graceful UI Handling",
        description="UI attempts to render a nonexistent or expired chart ID.",
        injection_point="UI session chart extraction and display",
        failure_type="ChartNotFoundError",
        expected_classification="n/a",
        expected_retry_behavior="Graceful fallback message without UI crash",
        expected_outcome="sanitized",
        notes="Streamlit session and conversation state continue normally.",
    ),

    # -------------------------------------------------------------------------
    # Category E: Execution Limits (2 Scenarios)
    # -------------------------------------------------------------------------
    ReliabilityScenario(
        scenario_id="REL_EXEC_01_STEP_LIMIT_EXCEEDED",
        category="execution_limits",
        name="LangGraph Step Limit Enforcement",
        description="Agent attempts continuous tool calling exceeding max_steps limit.",
        injection_point="LangGraph route_llm_output / route_start step_count check",
        failure_type="Maximum step limit reached",
        expected_classification="step_limit_exceeded",
        expected_retry_behavior="Immediate termination at max_steps",
        expected_outcome="safe_stop",
        notes="Routes to limit_exceeded node with informative user notification.",
    ),
    ReliabilityScenario(
        scenario_id="REL_EXEC_02_NON_RECOVERABLE_ERROR",
        category="execution_limits",
        name="Non-recoverable Error Termination",
        description="Fatal infrastructure or policy violation occurs.",
        injection_point="inspect_tool_results fail-closed policy",
        failure_type="Non-retryable operational error",
        expected_classification="non_retryable_resource_error",
        expected_retry_behavior="0 retries (immediate halt)",
        expected_outcome="safe_stop",
        notes="Takes precedence over any active or pending recoverable errors.",
    ),

    # -------------------------------------------------------------------------
    # Category F: Session Reliability (2 Scenarios)
    # -------------------------------------------------------------------------
    ReliabilityScenario(
        scenario_id="REL_SESSION_01_CROSS_SESSION_ISOLATION",
        category="session_reliability",
        name="Strict Cross-Session State Isolation",
        description="Verify Session A and Session B maintain isolated stores, registries, and graphs.",
        injection_point="AgentSession instantiation and store scoping",
        failure_type="Cross-session data leakage attempt",
        expected_classification="n/a",
        expected_retry_behavior="n/a",
        expected_outcome="isolated",
        notes="Session B cannot access dataset_id or chart_id generated in Session A.",
    ),
    ReliabilityScenario(
        scenario_id="REL_SESSION_02_SESSION_RESET",
        category="session_reliability",
        name="Clean Session Reset Verification",
        description="Populate session with datasets, charts, and messages, then call reset_agent_session.",
        injection_point="reset_agent_session()",
        failure_type="Stale state retention",
        expected_classification="n/a",
        expected_retry_behavior="n/a",
        expected_outcome="isolated",
        notes="Stores, registries, and message histories are fully cleared.",
    ),

    # -------------------------------------------------------------------------
    # Category G: Failure Presentation (1 Scenario)
    # -------------------------------------------------------------------------
    ReliabilityScenario(
        scenario_id="REL_UI_01_SANITIZED_FAILURE_AND_RECOVERY",
        category="failure_presentation",
        name="Sanitized Failure and Subsequent Turn Recovery",
        description="Simulate fatal exception containing sensitive credentials and tracebacks.",
        injection_point="execute_session_turn exception handler and sanitize_user_facing_narrative",
        failure_type="Internal exception with sensitive trace details",
        expected_classification="n/a",
        expected_retry_behavior="Safe generic fallback; fresh counters on turn 2",
        expected_outcome="sanitized",
        notes="User sees clean explanation; turn 2 executes from clean slate.",
    ),
]


def get_reliability_scenario_by_id(scenario_id: str) -> Optional[ReliabilityScenario]:
    """Look up a reliability scenario by its unique ID."""
    for s in RELIABILITY_SCENARIOS:
        if s.scenario_id == scenario_id:
            return s
    return None
