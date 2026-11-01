# SQL-to-Python Data Handoff: Dataset Store & References

## 1. Purpose of the Dataset Store

The **Dataset Store** (`tools/dataset_store.py`) solves the LLM context-window and token-cost problem in data analytics agent workflows.

In typical SQL analytical workloads, queries often return hundreds or thousands of rows (e.g. 1,000 rows across 10 columns). Storing the full JSON payload directly in `AgentState.messages` and `ToolMessage.content`:
- Bloats the conversation context window rapidly.
- Exceeds model token limits and causes extreme latency.
- Increases API inference costs unnecessarily.
- Risks model hallucinations on large numerical arrays.

The Dataset Store decouples **full structured analytical data storage** from **agent conversation history**. Full query results are saved in an application-controlled in-memory store, while the LLM receives a compact metadata reference (`dataset_id`) and a small, bounded preview (e.g., 10 rows).

---

## 2. Architecture & Data Handoff Workflow

```text
User / LLM
    │
    ▼ (1) execute_sql(query="SELECT ...")
[execute_sql Tool Wrapper] (agent/tool_wrappers.py)
    │
    ├─► (2) raw_execute_sql() -> [DuckDB Read-Only Worker]
    │        │
    │        └─► Full Result Dict: {columns, rows: 1000 items, row_count, truncated}
    │
    ├─► (3) DatasetStore.save(result, query)
    │        │
    │        └─► Stores full records, calculates bytes, checks limits
    │        └─► Returns opaque reference: "ds_a1b2c3d4..."
    │
    └─► (4) Returns compact tool result:
             {
               "status": "success",
               "dataset_id": "ds_a1b2c3d4...",
               "columns": [...],
               "row_count": 1000,
               "preview_row_count": 10,
               "rows": [... 10 preview rows ...],
               "truncated": false
             }
    │
    ▼ (5) ToolMessage -> LangGraph -> inspect_tool_results -> LLM
[LLM Conversation Context]
    │
    ▼ (6) Future Step 5.2: LLM invokes python_analysis(dataset_id="ds_a1b2c3d4...", code=...)
[Python Analysis Tool]
    │
    └─► (7) DatasetStore.get("ds_a1b2c3d4...") -> loads full DataFrame for computation
```

---

## 3. Dataset Representation & Lifecycle

### Internal Record (`DatasetRecord`)
Stored as an immutable dataclass in memory:
- `dataset_id`: Unique opaque identifier (e.g., `ds_<uuid4_hex>`). Never leaks file paths, database filenames, or system internals.
- `columns`: List of column name strings (`List[str]`).
- `rows`: Full query rows (`List[List[Any]]`).
- `row_count`: Total integer row count.
- `created_at`: POSIX timestamp float.
- `source_sql`: The originating analytical query string for auditability and lineage tracking.
- `truncated`: Boolean indicating whether SQL row limit was reached.
- `size_bytes`: Estimated memory payload footprint.

### Data Mutation Protection
`DatasetStore.save()` and `DatasetStore.get()` perform deep copies of column names and row lists. Modifying objects returned from the store or inputs prior to storage cannot corrupt internal stored datasets.

---

## 4. Tool-Facing Response Format & Compatibility

When `dataset_store` is configured in `create_sql_tools(..., dataset_store=store)`:
- `status`: `"success"` (or `"error"` if execution failed).
- `dataset_id`: Unique reference string.
- `columns`: Full column headers.
- `row_count`: Full number of rows available in the dataset store.
- `preview_row_count`: Number of preview rows included in this message.
- `rows`: Bounded preview slice (`rows[:preview_rows_limit]`).
- `truncated`: Preserves underlying SQL execution truncation flag.
- `execution_time_ms`: Execution duration reported by the SQL engine.

### Error Recovery Compatibility
`agent/sql_error_classifier.py` and `inspect_tool_results` in `agent/graph.py` expect `columns` and `rows` lists for success detection. Because `rows` contains the bounded preview list, compatibility is preserved:
- Error classification continues to detect `SUCCESS` deterministically.
- `sql_retry_count` and `last_error` clearing function identically.
- If a SQL error occurs, no dataset is created and the standard error payload is returned unchanged.

---

## 5. Storage Limits, Isolation, and Session Ownership

### Resource Bounds
- `max_datasets` (default 50): Restricts the total number of datasets concurrent in memory. Reaching the limit raises `DatasetCapacityError`.
- `max_memory_bytes` (default 50 MB): Approximate serialized memory limit. Storing a result that would exceed available capacity raises `DatasetCapacityError`.

### Session Isolation
- `DatasetStore` is an application-level service instantiated per graph or user session.
- Datasets are never shared across independent `DatasetStore` instances.
- Random opaque UUIDs prevent enumeration but are scoped to explicit store instances owned by that session.

---

## 6. Interaction with Existing SQL Row Limits

The Dataset Store strictly respects the security parameters established in Step 3:
- If a query hits DuckDB execution row limits (`max_rows=1000`), `truncated=True` is preserved in both the stored `DatasetRecord` and the tool response.
- The Dataset Store never executes additional SQL queries or bypasses `max_rows` constraints.

---

## 7. Known Limitations of In-Memory Storage

1. **Process Volatility**: Datasets reside in process RAM. If the application restarts or crashes, all datasets are cleared.
2. **Single-Instance Prototype**: Because storage is in-memory, requests distributed across multiple server workers or microservices cannot access datasets created on another worker unless a shared state or sticky session is used.
3. **Capacity Boundaries**: Analytical queries generating gigantic results (> 50MB) must be filtered or aggregated in SQL first before landing in the Dataset Store.

---

## 8. Future Integration Points

- **Step 5.2–5.3 (Python Analysis Tool)**: The Python execution tool will accept `dataset_id`, retrieve the dataset via `dataset_store.get(dataset_id)`, convert it into a Pandas DataFrame, and execute sandboxed analytical Python code.
- **Chart Generation Tool**: Visualizations will consume datasets via `dataset_id` to generate charts without reloading data from SQL.
