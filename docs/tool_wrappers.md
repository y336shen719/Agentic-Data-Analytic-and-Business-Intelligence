# SQL Tool Wrappers (Step 4.3)

This document explains the architecture of the LangChain tool wrappers in `agent/tool_wrappers.py` and how they interface between the OpenAI model and the DuckDB analytical engine.

---

## 1. What are Tool Wrappers?

Tool Wrappers are adapter functions that bridge lower-level application capabilities (like database discovery and SQL execution) to the framework's tool interface (`langchain_core.tools.BaseTool`).

They serve as an **enforcement boundary** between untrusted model outputs and trusted system calls:
- They export Pydantic-compatible JSON schemas so the LLM knows what functions exist and how to call them.
- They intercept LLM tool invocations, validate incoming parameters, and inject fixed application configuration (such as database file paths, timeouts, and table allowlists).

---

## 2. Why are Wrappers Separate from `tools/sql_tool.py`?

The separation of concerns between `tools/sql_tool.py` and `agent/tool_wrappers.py` is intentional:

1. **Framework Independence**: `tools/sql_tool.py` is pure Python and DuckDB without dependencies on LangChain, LangGraph, or OpenAI. It can be tested, benchmarked, or integrated into CLI utilities, REST APIs, or alternative LLM frameworks.
2. **Security Decoupling**: Security logic (AST analysis, read-only sandboxing, process-level timeouts) belongs to the database tool layer. Wrappers simply adapt calling conventions and must never be the only defense line.
3. **Configuration Injection**: Application policies (e.g. `max_rows=1000`, `timeout_seconds=10.0`, production database path) are injected at the wrapper layer so the LLM cannot tamper with them.

---

## 3. How Tool Schemas are Exposed to the LLM

When tools are bound to the model (`llm.bind_tools([schema_tool, sql_tool])`), LangChain extracts:
- **Function Name**: e.g., `get_database_schema`, `execute_sql`.
- **Description**: Extracted from the docstring to guide when and how the model should pick the tool.
- **Parameters Schema**: Generated from function type annotations as a JSON Schema.

For example, `execute_sql` generates:
```json
{
  "name": "execute_sql",
  "description": "Safely execute a read-only analytical SQL query against the database...",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "title": "Query",
        "type": "string"
      }
    },
    "required": ["query"]
  }
}
```

And `get_database_schema` exposes:
```json
{
  "name": "get_database_schema",
  "description": "Discover database schema including all tables, views, columns...",
  "parameters": {
    "type": "object",
    "properties": {}
  }
}
```

---

## 4. Which Arguments Can the LLM Control?

The LLM is granted minimal privileges:

| Tool | LLM-Controlled Arguments | Application-Controlled Arguments (Hidden from LLM) |
| :--- | :--- | :--- |
| `get_database_schema` | **None** (`{}`) | `db_path` |
| `execute_sql` | `query: str` (The SQL SELECT string) | `db_path`, `max_rows`, `timeout_seconds`, `allowed_tables` |

The LLM **cannot** specify:
- Database connection strings or file paths.
- Memory limits or thread counts.
- Row limit overrides beyond `MAX_PERMITTED_ROWS`.
- Timeout overrides beyond `MAX_PERMITTED_TIMEOUT_SECONDS`.
- Table authorization allowlists.

---

## 5. Why SQL Security Must Remain Inside the Underlying SQL Tool

Security cannot rely on prompt instructions or wrapper-level string checks alone. The core database tool (`tools/sql_tool.py`) enforces:
1. **SQLGlot AST Parsing & Validation**: Guarantees only read-only `SELECT` queries (with CTEs, JOINs, and aggregates) are parsed. Disallows modifying DDL/DML, file readers (`read_csv`, etc.), and administrative commands.
2. **Scope-Aware Table Validation**: Ensures queries only access allowed tables and views in the `main` schema, preventing unauthorized schema traversals or fake CTE aliasing.
3. **DuckDB Sandboxing**: Read-only connection mode, `enable_external_access=false`, and `lock_configuration=true`.
4. **Isolated Process Execution & Framing**: Bounded subprocess execution with strict Pipe framing so runaway queries or worker crashes never hang the host application.

Keeping security inside `sql_tool.py` ensures that even if an LLM is jailbroken or generates adversarial queries, the execution engine rejects the query safely.
