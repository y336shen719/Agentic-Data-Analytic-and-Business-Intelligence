# Planned Architecture

The system is planned as a stateful single-agent workflow:

```text
User
  → Streamlit
  → LangGraph Agent
  → Tool Selection
  → SQL / Python / Chart Tools
  → Tool Results
  → Agent Loop
  → Final Report
```

This describes intended future behavior. The current repository does not yet
implement the agent, tools, data loading, or analysis workflow.

## Component responsibilities

- **Streamlit UI:** Accept a natural-language business question and display
  progress, errors, and the final report. The current app is a landing page
  only.
- **LangGraph orchestration:** Manage the stateful, bounded single-agent loop,
  route tool calls, and determine whether to continue or finish.
- **LLM:** Interpret the user request, select from the available tools, and
  synthesize a report grounded in returned tool results. The model must not be
  treated as an authority for data it has not received.
- **SQL Tool:** Execute validated analytical queries through a read-only
  DuckDB connection, with appropriate execution and result limits.
- **Python Analysis Tool:** Perform approved statistical operations on
  returned data using constrained, validated inputs; it must not execute
  arbitrary generated code.
- **Chart Tool:** Generate charts from validated analysis outputs and return
  references or displayable results to the application.
- **Agent State:** Carry the user request, relevant intermediate messages,
  tool calls/results, and error information across graph steps.
- **Error Handling:** Surface tool and model errors explicitly, retry only
  within configured bounds, and stop safely when limits are reached.
- **Evaluation:** Run defined test cases against implemented behavior and
  report measured outcomes. Evaluation data and metrics must never be
  fabricated.

---

## Implemented Architecture: SQL-to-Python Data Handoff (Step 5.3)

In Step 5.3, `tools/sql_tool.py`, `tools/dataset_store.py`, `tools/python_analysis.py`, and `agent/tool_wrappers.py` are integrated into the stateful single-agent LangGraph workflow:

```text
User Question
    |
    v
LangGraph Agent (llm_node)
    |
    v
get_database_schema / execute_sql
    |
    v
DuckDB Query Execution (Read-Only)
    |
    v
DatasetStore.save(sql_result) -> dataset_id
    |
    v
execute_sql ToolMessage (dataset_id, columns, row_count, preview)
    |
    v
LLM (selects operation and provides params)
    |
    v
execute_python_analysis(dataset_id, operation, params)
    |
    v
PythonAnalysisEngine.analyze() -> DatasetStore.get(dataset_id)
    |
    v
Controlled Pandas/NumPy Operations -> Clean JSON Result
    |
    v
LLM Final Answer (Grounded in Tool Outputs)
```

### 1. Shared DatasetStore Injection

SQL and Python tools share an identical, injected `DatasetStore` instance to eliminate redundant SQL executions and keep large raw query outputs out of LLM token context:

```python
from agent.graph import create_agent_graph
from agent.tool_wrappers import create_python_analysis_tools, create_sql_tools
from tools.dataset_store import DatasetStore

# 1. Instantiate shared DatasetStore
dataset_store = DatasetStore()

# 2. Inject into SQL and Python tool factories
sql_tools = create_sql_tools(dataset_store=dataset_store)
python_tools = create_python_analysis_tools(dataset_store=dataset_store)

# 3. Concatenate and register with LangGraph agent
all_tools = sql_tools + python_tools
graph = create_agent_graph(tools=all_tools)
```

### 2. Supported Python Operations

The engine and tool strictly forbid arbitrary Python code execution. Five allowlisted operations are supported:

1. `describe`: Numerical summary statistics (count, mean, std, min, 25%, 50%, 75%, max).
2. `correlation`: Pearson correlation matrix across numerical columns.
3. `growth_rate`: Chronological period-over-period percentage growth (supports grouped growth with NULL-group preservation).
4. `detect_outliers`: IQR-based outlier detection with configurable multiplier (default 1.5).
5. `groupby`: Secondary grouping and aggregations (`sum`, `mean`, `count`, `min`, `max`) on materialized datasets.

### 3. Python Error Recovery and Bounded Automatic Retry (Step 5.4)

When `execute_python_analysis` encounters an invalid parameter, missing dataset ID, or invalid column, it returns a structured JSON error payload:

```json
{
  "status": "error",
  "dataset_id": "ds_example",
  "operation": "correlation",
  "error_type": "InvalidColumnError",
  "error_message": "Column 'foo' does not exist in dataset."
}
```

In Step 5.4:
- The error is deterministically classified by `agent.python_error_classifier.classify_python_result`.
- `inspect_tool_results` in LangGraph updates `AgentState.python_retry_count` and routes recoverable errors back to the LLM for self-correction.
- Retries are strictly bounded by `max_python_retries` (default 2), preventing infinite correction loops.
- Python and SQL retry counters operate completely independently.
- See detailed documentation in [docs/python_error_recovery.md](file:///Users/yimingshen20000719/Desktop/Agentic%20AI%20System/docs/python_error_recovery.md).

---

## Implemented Architecture: Four-Tool Agent & ChartRegistry (Step 6.2)

Step 6.2 integrates the `ChartGenerationEngine` into the single-agent LangGraph workflow, establishing a four-tool system with shared dataset storage and lightweight chart references:

```text
User Question
    |
    v
LangGraph Agent (llm_node)
    |
    v
[Tool Selection]
 ├── get_database_schema
 ├── execute_sql ─────────> saves sql_result ───┐
 ├── execute_python_analysis ───────────────────┤
 │    (auto-registers derived dataset)          v
 │                                       Shared DatasetStore
 └── generate_chart <───────────────────────────┘
      │
      v
  Plotly Figure generated
      │
      v
  ChartRegistry.save(figure) ──> stores full Figure JSON in memory
      │
      v
  Lightweight ToolMessage (chart_id, metadata, title)
      │
      v
LLM (synthesizes answer grounded in chart_id without JSON bloat)
```

### 1. Four-Tool Agent Registration & Dependency Injection

The agent supports four distinct tools sharing the same `DatasetStore` and session-scoped `ChartRegistry`:

```python
from agent.graph import create_agent_graph
from agent.tool_wrappers import (
    create_chart_generation_tools,
    create_python_analysis_tools,
    create_sql_tools,
)
from tools.chart_registry import ChartRegistry
from tools.dataset_store import DatasetStore

# Shared session storage
dataset_store = DatasetStore()
chart_registry = ChartRegistry()

# Inject into tool factories
sql_tools = create_sql_tools(dataset_store=dataset_store)
py_tools = create_python_analysis_tools(dataset_store=dataset_store)
chart_tools = create_chart_generation_tools(
    dataset_store=dataset_store,
    chart_registry=chart_registry,
)

# Combine tools and compile graph
all_tools = sql_tools + py_tools + chart_tools
graph = create_agent_graph(tools=all_tools)
```

### 2. Automatic Derived Analysis Dataset Registration

When `execute_python_analysis` succeeds, supported operations are automatically adapted and saved back to the shared `DatasetStore`:
- Supported operations: `growth_rate`, `groupby`, `correlation`, `describe`.
- Return schema augmentation: includes `derived_dataset_id` (e.g. `ds_...`) and `derived_dataset_status` (`"registered"`).
- Original source dataset immutability: the source dataset and original analytical calculations remain unchanged.
- Unsupported handoff (`detect_outliers`): outlier detection outputs bounds and previews rather than full row-level indicator flags. Derived dataset creation is omitted (`derived_dataset_id: None`, `derived_dataset_status: "unsupported"`, `derived_dataset_error: "This operation does not produce a complete chart-ready dataset."`), while preserving the valid analytical output.
- Registration failure reporting: if dataset storage fails (e.g. `DatasetCapacityError`), the tool safely reports `derived_dataset_status: "failed"` and an actionable `derived_dataset_error` without converting the successful analysis into a calculation failure or leaking exception traces.

### 3. Lightweight Chart References vs. UI Retrieval & Lifecycle Ownership

To prevent context-window saturation and high token costs, the full Plotly Figure JSON is **never** emitted into the LLM conversation:
- **LLM-facing ToolMessage:** Compact JSON containing `chart_id`, `dataset_id`, `chart_type`, `title`, and execution `metadata` (e.g. `rows_used`, `total_rows`).
- **Application Registry:** `ChartRegistry` retains the full Plotly figure dictionary in memory with bounded retention (FIFO / memory limits) and cross-session isolation.
- **Engine Consistency:** When preconfigured engines are supplied alongside a `DatasetStore`, factories strictly validate object identity (`engine.dataset_store is dataset_store`), preventing silent dependency diversion.
- **Session Lifecycle & Ownership:** The graph invoker (e.g. Streamlit session state) owns and maintains references to the shared `DatasetStore` and `ChartRegistry` instances across user turns. Global singletons are strictly avoided, ensuring multi-session isolation.
- **Future UI (Step 6.3):** The frontend queries `chart_registry.get(chart_id)` to render native Plotly figures interactively.

### 4. Chart Tool Result Inspection & Non-Retryable Error Policy (Step 6.2 Fix)

The LangGraph node `inspect_tool_results` explicitly inspects `generate_chart` invocations alongside SQL and Python:
- **Success Handling:** Valid chart results (`status == "success"`) allow the LLM to proceed without modifying `sql_retry_count` or `python_retry_count`. Chart success does not clear pending SQL or Python recovery state unless the respective tool actually succeeded.
- **Non-Retryable Chart Error Policy:** Chart generation failures (including `InvalidChartTypeError`, `InvalidColumnError`, `DatasetNotFoundError`, `ChartRegistryCapacityError`, or malformed/ToolNode failures) are deterministically classified under `category = NON_RETRYABLE_CHART_ERROR`.
- **Termination via `recovery_stop`:** Chart errors immediately route to `recovery_stop` with an actionable, safe message (`Execution stopped: chart generation failed and will not be automatically retried. Error: <error>`), preventing unbounded tool loops or misreporting as SQL/Python retry exhaustion.
- **Mixed-Batch Precedence:** In mixed tool batches, non-retryable chart errors take absolute precedence over recoverable errors or unrelated tool successes (Fail Closed).
- **Independent Budgets & Isolation:** Chart tool invocations never increment SQL or Python retry counters.
- **Future Scope:** Automatic, bounded chart self-correction policies (e.g. argument retries) are deferred to future revisions.



