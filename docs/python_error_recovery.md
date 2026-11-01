# Python Analysis Error Classification & Bounded Automatic Retry (Step 5.4)

## 1. Overview

In Step 5.4, the stateful single-agent architecture was extended with a safe, deterministic, bounded automatic recovery mechanism for failed `execute_python_analysis` tool calls.

The mechanism allows the LLM to inspect structured analytical errors (e.g., misspelled column names, invalid analytical parameters, or missing dataset references) and perform self-correction while strictly bounding retry attempts and preventing infinite loops.

```text
User Question
    |
    v
LangGraph LLM Node
    |
    v
execute_python_analysis
    |
    v
PythonAnalysisEngine.analyze()
    |
    v
Structured Tool Result
    |
    v
inspect_tool_results
    |
    +-- Success --------------------------> Continue Agent (python_retry_count reset to 0)
    |
    +-- Recoverable Error
    |       |
    |       v
    |   Classify Error (agent/python_error_classifier.py)
    |       |
    |       v
    |   Check Python Retry Budget (max_python_retries, default 2)
    |       |
    |       +-- Available --> LLM Self-Correction --> Retry
    |       |
    |       +-- Exhausted --> Safe Recovery Stop (recovery_stop_node)
    |
    +-- Non-Recoverable Error ------------> Safe Recovery Stop
```

---

## 2. Python Error Classification

Pure, deterministic classification is implemented in `agent/python_error_classifier.py`:

```python
def classify_python_result(result: dict) -> dict:
    ...
```

### Classification Matrix

| Exception / Error Type | Category | Retryable | Recovery Action | Recovery Strategy |
|---|---|---|---|---|
| `InvalidColumnError` | `invalid_column` | `True` | `correct_python_arguments` | LLM corrects column names based on SQL metadata |
| `InvalidParameterError` | `invalid_parameter` | `True` | `correct_python_arguments` | LLM corrects parameter values or structure |
| `InvalidOperationError` | `invalid_operation` | `True` | `correct_python_arguments` | LLM selects an allowlisted operation (bounded by retries) |
| `DuplicatePeriodError` | `invalid_dataset_structure` | `True` | `regenerate_dataset` | LLM constructs aggregating SQL query |
| `EmptyDatasetError` | `invalid_dataset_structure` | `True` | `regenerate_dataset` | LLM adjusts SQL filter / query |
| `DatasetNotFoundError` | `missing_dataset` | `True` | `recover_dataset_reference` | LLM reuses existing valid dataset ID or regenerates via SQL |
| `DataFrameConstructionError` | `internal_failure` | `False` | `none` | Non-retryable internal failure |
| `InternalExecutionError` | `internal_failure` | `False` | `none` | Non-retryable calculation crash |
| Explicit Policy Violation (`SecurityPolicyError`, etc.) | `policy_violation` | `False` | `none` | Non-retryable execution policy violation |
| Unknown error | `unknown_error` | `False` | `none` | Non-retryable |
| Missing / invalid result | `malformed_result` | `False` | `none` | Non-retryable |

### Policy Violations vs. Operation Name Corrections

- **Accidental operation-name errors** (e.g. `operation="summary_statistics"` instead of `"describe"`): Handled as `InvalidOperationError`, allowing bounded correction up to `max_python_retries`.
- **Explicit policy violations**: When an explicit, trusted policy signal is present (such as `SecurityPolicyError`, `ExecutionPolicyError`, `PolicyViolationError`, or a `policy_violation=True` flag), the error is classified as `policy_violation` (`retryable=False`).
- **Engine contracts preserved**: The classifier does not invent heuristic regex/keyword rules over user intent. The engine continues strictly enforcing the five allowlisted operations (`describe`, `correlation`, `growth_rate`, `detect_outliers`, `groupby`).

### Dataset Recovery Strategy

When `DatasetNotFoundError` occurs (`action: recover_dataset_reference`):
1. The LLM first checks previous trusted SQL `ToolMessage`s in conversation history for an existing valid `dataset_id`. If one exists that matches the desired data, the LLM reuses that ID without re-executing SQL.
2. If no valid dataset exists in history, or the data structure requires regeneration (e.g. `DuplicatePeriodError`), the LLM executes an authorized SQL query to materialize a new dataset.
3. Dataset IDs must never be fabricated.

---

## 3. AgentState Extensions

In `agent/state.py`:

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int
    last_sql: Optional[str]
    last_error: Optional[str]
    sql_retry_count: int
    last_error_category: Optional[str]
    python_retry_count: int       # Added in Step 5.4
    pending_python_recovery: bool # Tracks active Python recovery across intermediate SQL calls
```

### Retry Counter Semantics

The `python_retry_count` tracks **corrective retries** after an initial failure for a specific analytical task:

```text
Initial Python call fails:      python_retry_count = 0, pending_python_recovery = True
First corrective retry fails:   python_retry_count = 1, pending_python_recovery = True
Second corrective retry fails:  python_retry_count = 2, pending_python_recovery = True
Further retries blocked:        Routed to recovery_stop_node
```

1. **Initial Python failure** sets `pending_python_recovery = True` while `python_retry_count` remains 0.
2. **Corrective retries** increment `python_retry_count` by 1 upon execution while `pending_python_recovery` is True.
3. **Successful recovery** resets `python_retry_count = 0`, sets `pending_python_recovery = False`, and clears `last_error` and `last_error_category`.
4. **Subsequent independent operations** receive their own full retry budget.
5. **Pre-execution guard** prevents parallel tool calls from exceeding remaining budget.

---

## 4. SQL and Python Retry Independence & Cross-Tool State

SQL and Python retry budgets and counters are completely independent:

1. **No cross-contamination**:
   - SQL errors only increment `sql_retry_count`.
   - Python errors only increment `python_retry_count`.
2. **Cross-Tool State Bridge (`pending_python_recovery`)**:
   - When Python fails with `missing_dataset` or `invalid_dataset_structure`, the LLM executes SQL to regenerate data.
   - Successful intermediate SQL executions do not reset or bypass the pending Python retry budget because `pending_python_recovery` remains True.
   - Resuming the Python call correctly consumes from the remaining Python retry budget.
   - Alternating between tools cannot exceed `max_python_retries`.
3. **Cumulative Limits**:
   - Both tools remain strictly bounded by global `max_steps` (default 12).
   - `max_sql_retries` (default 2) and `max_python_retries` (default 2) enforce tool-specific bounds.

---

## 5. Multi-Tool Batch Error Precedence Policy

When the LLM outputs multiple tool calls in a single response, `inspect_tool_results` resolves batch outcomes deterministically:

1. **Fail Closed (Non-Retryable Errors)**:
   - Any non-retryable error (e.g. `SecurityPolicyError`, internal failure) takes absolute priority and immediately routes to `recovery_stop`.
   - A non-retryable error cannot be masked by any other tool result in the batch.
2. **Preservation of Recoverable Errors**:
   - Recoverable failures cannot be masked by unrelated successes in the same batch.
   - If a Python call fails in a batch alongside a successful Python or SQL call, the failure is preserved as `last_error`, `pending_python_recovery = True`, and the agent is routed to self-correction.
3. **Pure SQL Intra-Batch Retries**:
   - In a pure SQL batch, an earlier SQL failure followed by a later successful query is treated as an intra-batch correction, incrementing `sql_retry_count` and clearing `last_error` on final success.
4. **ToolMessage Integrity**:
   - All tool results are preserved as individual `ToolMessage`s in conversation history so the LLM has complete visibility into execution results.

