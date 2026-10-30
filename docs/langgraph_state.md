# LangGraph State Management & Minimal Graph (Step 4.4)

This document explains the foundational principles of LangGraph state management, graph topologies, and deterministic node execution as implemented in `agent/state.py`.

---

## 1. What is State in LangGraph?

In LangGraph, **State** is a centralized, typed data structure (typically a Python `TypedDict` or Pydantic model) that serves as the single source of truth across a workflow.

Every Node in the graph reads from this central state and outputs state updates. Rather than passing ad-hoc parameters between isolated functions, the State coordinates all conversational history, analytical metadata, intermediate steps, and error diagnostics.

In our system, `AgentState` is defined in `agent/state.py`:

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int
    last_sql: Optional[str]
    last_error: Optional[str]
    sql_retry_count: int
```

- Secrets, database credentials, and raw connections are strictly kept outside the state schema.

---

## 2. What are Nodes?

**Nodes** are individual Python functions representing units of computation in the graph:

```python
def my_node(state: AgentState) -> Dict[str, Any]:
    ...
    return {"step_count": state["step_count"] + 1}
```

- **Input**: A snapshot of the current `AgentState`.
- **Output**: A dictionary containing **partial state updates** (only the fields that changed).
- In our minimal demonstration:
  - `inspect_state` (Node A): Reads incoming state, inspects the latest `HumanMessage`, and increments `step_count`.
  - `update_state` (Node B): Returns a deterministic confirmation `AIMessage`, updates `last_sql`, and increments `step_count`.

---

## 3. What are Edges?

**Edges** define the execution flow and routing rules connecting Nodes:

- **Standard Edges**: Deterministic transitions from one node to the next.
- **Entry and Exit Points**: Explicit transitions from the pseudo-node `START` to the entry node, and from the terminal node to `END`.
- Minimal graph topology:
  ```text
  START -> inspect_state -> update_state -> END
  ```
- Conditional edges (to be implemented in later stages) will route execution dynamically based on state values (e.g., deciding whether to call a tool or reply to the user).

---

## 4. What is a State Reducer?

By default, when a node returns a dictionary key (such as `step_count: 2`), LangGraph applies **overwrite semantics**: the old value is replaced with the new value.

A **Reducer** is a custom function that dictates how updates to a specific state key should be merged with its existing value. Reducers are declared using `typing.Annotated`:

```python
messages: Annotated[list[BaseMessage], add_messages]
```

When a node emits `{"messages": [new_msg]}`, LangGraph passes `(existing_messages, [new_msg])` to the reducer function rather than overwriting the entire history.

---

## 5. How `add_messages` Differs from Normal List Replacement

If `messages` used default overwrite semantics, a node returning `{"messages": [AIMessage(...)]}` would completely wipe out previous `HumanMessage` and system messages.

The `add_messages` reducer provides intelligent append and upsert semantics:

1. **Append**: New messages without matching IDs are appended to the end of the message history.
2. **ID-Based Upsert**: If a newly returned message has the same `id` as an existing message in state, `add_messages` replaces that specific message in-place. This is vital for streaming tokens, updating tool calls, or editing prior messages.
3. **Format Normalization**: It automatically normalizes message structures (e.g. dict representations to `BaseMessage` objects).

---

## 6. Why Nodes Return Partial State Updates

Nodes in LangGraph do **not** need to construct or return the entire `AgentState`. They only return the specific fields they modified.

Benefits:
- **Encapsulation**: Node A (`inspect_state`) only cares about `step_count`; it does not need to know or touch `last_sql` or `sql_retry_count`.
- **Safety**: Returning partial dictionaries prevents accidental overwrites or state corruption of unmanaged fields.
- **Maintainability**: New state fields can be added to `AgentState` without updating every existing node.

---

## 7. Why `.compile()` is Required Before `.invoke()`

A `StateGraph` object is a mutable builder where nodes and edges are registered.

Calling `.compile()`:
1. Validates graph integrity (ensuring all node references exist, edges connect to valid nodes, and no orphan components violate topology).
2. Initializes internal state management channels and reducer dispatch pipelines.
3. Returns a `CompiledStateGraph` instance exposing the execution interface (`.invoke()`, `.stream()`, `.astream_events()`).

---

## 8. Educational Scope & Boundary Notice

This minimal demonstration is intentionally:
- **100% Offline**: Zero external network or OpenAI API calls are made.
- **No SQL Execution**: No DuckDB queries are run.
- **No Agent Loop**: Does not implement dynamic tool execution loops, tool calling nodes, or error-recovery cycles yet.

Those capabilities will be introduced progressively in subsequent steps (Step 4.5+).
