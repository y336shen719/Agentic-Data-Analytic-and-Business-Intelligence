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


