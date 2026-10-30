# SQL Error Classification (Step 4.6.1)

`agent/sql_error_classifier.py::classify_sql_result(result)` turns an `execute_sql`
result into a category plus a `retryable` flag. It is pure and deterministic: no I/O,
no LLM, no input mutation. It is **not wired into the graph yet**.

## Categories

| category | retryable | source `error_type` |
|---|---|---|
| `success` (rows or zero rows) | no | `status == "success"` with `columns`/`rows` lists |
| `recoverable_sql_error` | **yes** | `ParserException`, `SyntaxException`, `BinderException`, `CatalogException`, `ConversionException`, `TypeMismatchException`, `InvalidTypeException`, and `SQLValidationError` whose message starts with `Malformed SQL statement:` |
| `non_retryable_security_error` | no | every other `SQLValidationError` (disallowed table/schema/function, DDL/DML, file access, multi-statement), `PermissionException` |
| `non_retryable_resource_error` | no | `TimeoutError`, `OutOfMemoryException`, `InterruptException`, `ProcessError`, `PayloadSizeError`, `IPCError`, `OSError`, `IOException`, `FileNotFoundError`, `ValueError` |
| `non_retryable_unknown_error` | no | any other `error_type` |
| `malformed_result` | no | non-dict, bad/missing `status`, missing `error_type`/`error_message`, success without `columns`/`rows` |

Error output: `{status, category, retryable, error_type, error_message}`.
Success output: the original result plus `category="success"`, `retryable=False`.

## Why unknown and malformed are non-retryable

A retry means another LLM call plus another SQL execution. Retrying something we do not
understand risks looping on a systemic failure or probing the safety layer. Only
well-understood, fixable SQL mistakes are retried.

## Why classification keys on `error_type`

The tool already returns structured types (DuckDB exception class names from the worker,
plus `SQLValidationError`, `TimeoutError`, `ProcessError`, ...). Message text is never
searched for keywords. The single exception is the anchored prefix
`Malformed SQL statement:`, because `SQLValidationError` is used for both sqlglot parse
failures (fixable) and security rejections (not).

## Graph integration (Step 4.6.2–4.6.3)

`tools -> inspect_tool_results -> (llm | recovery_stop)`; see `agent/graph.py`.

- `inspect_tool_results` walks the `execute_sql` tool calls of the latest AI message in order,
  matches each to its `ToolMessage` by `tool_call_id`, parses the JSON content (or text blocks),
  and calls `classify_sql_result`. `get_database_schema` and other tools are ignored.
  Missing, mismatched, `status="error"` (ToolNode-level) or non-JSON messages are *malformed* → stop.
- State: `last_error` (`"<type>: <message>"`, `None` after success), `last_error_category`
  (the classifier category, needed for routing), `sql_retry_count` = SQL executions that ran while
  `last_error` was set (initial attempt excluded; cumulative for the request, not reset on success).
- Routing: success / non-SQL → `llm`; recoverable and `sql_retry_count < max_sql_retries` → `llm`
  (the error is already in the `ToolMessage` history); anything else → `recovery_stop`
  (explanatory `AIMessage`, then `END`).
- `max_sql_retries` (default 2) bounds retries. A pre-execution guard in the LLM router sends a
  response to `recovery_stop` (all pending calls get "not executed" `ToolMessage`s) if its SQL calls
  exceed the remaining budget, so parallel calls cannot bypass it.
- `max_steps` (LLM invocations) still applies independently.
