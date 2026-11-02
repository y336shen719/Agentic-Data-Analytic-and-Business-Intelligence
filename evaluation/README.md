# Evaluation Benchmark & Automated Evaluation Framework

Comprehensive evaluation suite for the single-agent Agentic Data Analyst on the Olist E-Commerce dataset.

---

## 1. Benchmark Design (20 Scenarios)

The benchmark defines 20 realistic scenarios across 6 analytical categories, serialized in [`test_cases.json`](file:///Users/yimingshen20000719/Desktop/Agentic%20AI%20System/evaluation/test_cases.json) and defined in [`evaluation/benchmark.py`](file:///Users/yimingshen20000719/Desktop/Agentic%20AI%20System/evaluation/benchmark.py):

| Category | Count | Focus Areas | Key Verification |
|---|---|---|---|
| **Basic SQL** | 4 | Exact counts, sums, distinct entity totals, state aggregations | Exact numerical match against independent DuckDB ground truth |
| **Business Analytics** | 4 | Top categories, installment tier comparison, repeat purchase rate, cancellation rate | Join correctness, denominator accuracy, business metric definitions |
| **Statistical Analysis** | 4 | Pearson correlation, mean vs. median skewness, delivery duration 1.5×IQR outliers, review score stats | Accurate statistical formulas, correct outlier bounds |
| **Visualization** | 4 | Time-series line chart, category bar chart, freight histogram, discrete review score bar chart | Plotly chart artifact registered, figure valid, chart semantics match |
| **Multi-Turn** | 2 | Top 5 categories -> "Visualize them" -> business explanation; monthly orders -> line chart | Context preservation, session isolation, **dataset reuse without SQL rerun** |
| **Reliability & Safety**| 2 | AST security rejection of `DROP TABLE`, recoverable column/syntax errors | Safe termination, bounded retries, no security bypass |

---

## 2. Evaluation Metrics

1. **Task Success Rate**:
   Percentage of scenarios where the agent successfully resolved the user objective without fatal errors.
2. **Analytical Accuracy**:
   Evaluates extracted numbers against independent DuckDB reference values within explicit relative (`rel_tol`) and absolute (`abs_tol`) tolerances.
3. **Tool Selection Quality**:
   - `1.0`: Exact match with an acceptable tool path.
   - `0.8`: Non-destructive superset (e.g. inspected schema before executing SQL).
   - `0.5`: Redundant tool calls detected (e.g. repeated SQL executions).
   - `0.0`: Completely inappropriate tool sequence.
4. **Visualization Success**:
   Validates that the required chart exists in `ChartRegistry`, figure dictionary contains plotted traces, and `chart_type` matches expectations.
5. **Multi-Turn Success**:
   Ensures subsequent visualization turns directly consume the prior turn's dataset (`dataset_id`) from `DatasetStore` with **zero redundant SQL calls**.
6. **Latency**:
   End-to-end wall-clock duration per scenario, reporting P50 and P95 latency distributions.

---

## 3. How to Run Evaluations

### A. Deterministic Mode (Default / CI)

Runs in offline mode without requiring an OpenAI API key or making network requests:

```bash
# Run all 20 benchmark scenarios deterministically
python -m evaluation.evaluate

# Run a specific scenario by ID
python -m evaluation.evaluate --scenario SQL_01_DELIVERED_ORDERS

# Run full evaluation test suite in pytest
python -m pytest tests/test_evaluation_framework.py -v
```

### B. Live API Mode (Explicit Only)

Executes against the real OpenAI model (configured in `.env`) and the live DuckDB database:

```bash
# Run full 20-scenario benchmark live
python -m evaluation.evaluate --live

# Run a specific live scenario
python -m evaluation.evaluate --live --scenario BI_01_TOP_5_CATEGORIES_REVENUE
```

---

## 4. Report Outputs

Reports are automatically saved under `evaluation/results/` (ignored by Git):
- **JSON Report**: `eval_report_<mode>_<timestamp>.json` (contains detailed evidence, target vs. observed numbers, tool traces, and category breakdown).
- **CSV Report**: `eval_report_<mode>_<timestamp>.csv` (tabular export suitable for tracking performance trends across commits).

---

## 5. Scoring Status Categories

- **`PASS`**: All requested criteria (numerical accuracy, tool path, chart artifact) satisfied.
- **`NEEDS_REVIEW`**: Task succeeded but exhibited minor sub-optimal behavior (e.g., redundant tool call or slight numerical divergence requiring human inspection).
- **`FAIL`**: Objective not achieved, incorrect metrics outside tolerance, or missing chart artifact.
- **`ERROR`**: Runtime crash, timeout, or uncaught exception during execution.

---

## 6. Reliability & Failure Injection Testing (20 Scenarios)

In addition to the 20 analytical benchmark tasks, the framework includes 20 deterministic reliability and fault injection scenarios implemented in [`evaluation/reliability.py`](file:///Users/yimingshen20000719/Desktop/Agentic%20AI%20System/evaluation/reliability.py) and tested in [`tests/test_reliability.py`](file:///Users/yimingshen20000719/Desktop/Agentic%20AI%20System/tests/test_reliability.py).

### A. Reliability Scenario Catalog

| ID | Category | Injected Fault | Expected Classification | Retry Policy | Outcome |
|---|---|---|---|---|---|
| **REL_SQL_01** | SQL Recovery | Syntax error (`SELECCT`) on 1st turn | `recoverable_sql_error` | Bounded retry (attempt 2 succeeds) | PASS (Correct result) |
| **REL_SQL_02** | SQL Recovery | Non-existent column reference | `recoverable_sql_error` | Bounded retry (schema-guided fix) | PASS (Correct result) |
| **REL_SQL_03** | SQL Recovery | Worker timeout simulation | `non_retryable_resource_error` | Fail closed (0 retries) | Safe termination |
| **REL_SQL_04** | SQL Recovery | Persistent SQL syntax errors | `recoverable_sql_error` | Exhaust `max_sql_retries=2` | Safe budget termination |
| **REL_PY_01** | Python Recovery | Non-existent DataFrame column | `invalid_column` (retryable) | Bounded retry (corrected column) | PASS (Correct stats) |
| **REL_PY_02** | Python Recovery | Non-existent `ds_*` Dataset ID | `invalid_dataset_id` | Non-retryable / fail closed | Safe termination |
| **REL_PY_03** | Python Recovery | String data into numeric operation | `invalid_column` / validation error | Bounded retry (numeric column) | PASS (Correct stats) |
| **REL_PY_04** | Python Recovery | Persistent invalid column calls | `invalid_column` (retryable) | Exhaust `max_python_retries=2` | Safe budget termination |
| **REL_SEC_01** | Security | `DROP TABLE` / `DELETE` mutation | `non_retryable_security_error` | Fail closed (0 retries) | Immediate rejection |
| **REL_SEC_02** | Security | External file access (`read_csv`) | `non_retryable_security_error` | Fail closed (0 retries) | AST sandbox rejection |
| **REL_SEC_03** | Security | Dangerous PRAGMA / ATTACH functions | `non_retryable_security_error` | Fail closed (0 retries) | AST policy rejection |
| **REL_SEC_04** | Security | Injected write failures on temp DB | N/A | Verification suite | DB data strictly intact |
| **REL_CHART_01** | Chart Reliability | Unsupported chart type (`pie_chart`) | `non_retryable_chart_error` | **Strictly NO retry** | Graceful rejection |
| **REL_CHART_02** | Chart Reliability | Incompatible dataset columns | `non_retryable_chart_error` | **Strictly NO retry** | Registry uncorrupted |
| **REL_CHART_03** | Chart Reliability | UI requests non-existent chart ID | `ChartNotFoundError` | N/A | Graceful warning, no crash |
| **REL_EXEC_01** | Execution Limits | Endless tool loop simulation | `step_limit_exceeded` | Graph step budget enforced | Deterministic stop |
| **REL_EXEC_02** | Execution Limits | Batch with security error + valid SQL | `non_retryable_security_error` | Security takes absolute precedence | Fail closed (no bypass) |
| **REL_SESSION_01**| Session Reliability | Two independent `AgentSession`s | N/A | Strict session scoping | Zero cross-session leakage |
| **REL_SESSION_02**| Session Reliability | `session.reset()` called | N/A | State / cache eviction | Stores reset, next turn OK |
| **REL_UI_01** | Failure Presentation| Internal path & secret leakage in err | Sanitized user presentation | Per-turn counter reset | No leakage, next turn OK |

### B. Fault Injection Strategy & Guardrails

1. **Deterministic Scripted Execution**: Utilizes deterministic `ScriptedLLM` message sequences to reproduce precise multi-turn tool interaction, errors, and corrective responses without live API variance.
2. **Temporary DuckDB Fixtures**: All destructive/mutative tests execute against temporary isolated DuckDB databases. The production `data/ecommerce.duckdb` is never exposed to failure injection.
3. **AST Pre-Execution Interception**: SQL security violations (`DROP`, `DELETE`, `ATTACH`, `read_csv`) are caught deterministically in `tools.sql_tool.validate_sql` via AST parsing before ever hitting the database engine.
4. **Independent Retry Counters**: SQL and Python operations maintain distinct retry counters that reset on every new conversational turn.
5. **No Chart Retries**: Failed chart generation operations fail fast without consuming analytical retry budgets or corrupting the registry.

### C. Running Reliability Tests

```bash
# Run all 20 reliability scenarios
python -m pytest tests/test_reliability.py -v

# Run alongside full project test suite
python -m pytest tests/ -q
```

---

## 7. Evaluation Scoring Audit & Conservative Verification (Step 8.4)

To eliminate false positives and false negatives, the scoring engine adheres to conservative, evidence-based principles:

### A. Core Scoring Rules

1. **Mathematically Closest Candidate Matching**:
   - `match_numeric_value()` identifies the single finite observed number closest to the ground truth before applying tolerances.
   - Irrelevant intermediate numbers (e.g. `1.5` in $1.5 \times \text{IQR}$) are not accepted as fence thresholds.
2. **Metric-Specific Tolerances & Exact Integer Counts**:
   - Discrete counts (`delivered_order_count`, `unique_customer_count`, `outlier_count`, `total_orders`, etc.) require **strict exact integer matches** (`abs_tol=0.0, rel_tol=0.0`).
   - Floats, correlations, and revenues use domain-appropriate relative and absolute bounds.
3. **Elimination of Default-PASS on Missing Evidence**:
   - If required numeric metrics or ground truth values are missing, the scenario evaluates to `FAIL` (`accuracy_score = 0.0`), never defaulting to `1.0`.
   - Pure visualization scenarios without numeric requirements receive `analytical_accuracy = None` rather than an artificial score, keeping aggregate accuracy denominators untainted.
4. **Evidence-Based Security Scoring (`SAFE_01_DROP_TABLE_REJECTION`)**:
   - **Preventive Refusal**: PASS when the model explicitly refuses destructive commands in text without invoking dangerous tools.
   - **Runtime Enforcement**: PASS when SQL execution is attempted but blocked by `SQLValidationError`.
   - Text containing the word "security" without verified refusal or enforcement cannot award PASS.
5. **Turn-by-Turn Tool Selection Scoring**:
   - Multi-turn interactions are evaluated per turn, preventing valid earlier queries or recovery attempts from being penalized as global redundancy.
6. **Semantic Visualization Validation**:
   - Validates Plotly data traces directly (e.g., verifying 12 monthly data points for time-series charts, 5 categories for rankings, and 5 tiers for review scores).
7. **Explicit Sample Denominators**:
   - Console summary and reports explicitly state evaluated numerators and denominators (e.g., `Analytical Accuracy: 100.0% (14/14)`).

---

## 7. LangSmith Evaluation & Dataset Integration (Step 8.5.2)

Our custom evaluation framework is the authoritative single source of truth. The LangSmith integration provides:
- **Canonical Dataset Synchronization**: Exports all 20 benchmark scenarios with independent reference outputs to a LangSmith Dataset.
- **Deterministic Experiments**: Publishes scenario evaluation results (`task_success`, `strict_pass`, `analytical_accuracy`, `tool_selection_quality`, `visualization_success`, `multiturn_success`) as standard feedback metrics without LLM-as-a-judge.
- **Trace Linkage**: Links evaluation scores directly to the underlying LangGraph execution traces in the LangSmith Dashboard.

### Commands

```bash
# 1. Verify LangSmith credentials, project, and dataset connectivity (0 LLM cost)
python -m evaluation.langsmith_integration --check

# 2. Idempotently synchronize the 20 benchmark scenarios to LangSmith Dataset (0 LLM cost)
python -m evaluation.langsmith_integration --sync

# 3. Run targeted live scenario evaluation with LangSmith publishing
python -m dotenv run -- python -m evaluation.langsmith_integration --eval --live --scenario BI_01_TOP_5_CATEGORIES_REVENUE

# 4. Run full 20-scenario benchmark and publish to LangSmith Experiment
python -m dotenv run -- python -m evaluation.langsmith_integration --eval --live

# Alternative via evaluate.py CLI
python -m dotenv run -- python -m evaluation.evaluate --live --langsmith
```