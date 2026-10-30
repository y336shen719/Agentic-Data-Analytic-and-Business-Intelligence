# Tool Calling Fundamentals (Step 4.2)

Demo: `python -m scripts.demo_tool_calling` (live, billable). Offline tests: `tests/test_tool_calling.py`.

## What is tool / function calling?

The model can't run code. You describe functions (name, description, JSON-schema parameters) and the model may answer with a *structured request* to call one: "call `multiply` with `{a: 123, b: 456}`". Your application decides whether and how to run it, then reports the result back.

## Why the LLM doesn't execute Python itself

The model only emits text/JSON. Execution stays in our process, so we can enforce an allowlist of tools, validate arguments, cap the number of calls, and apply timeouts. Never `eval`/`exec` model output.

## What `bind_tools()` does

`llm.bind_tools([multiply])` returns a new runnable that sends each tool's schema (derived from the `@tool` function's name, docstring and type hints) with every request. It doesn't run anything; it only tells the model what it may ask for. The original `llm` is unchanged.

## What `AIMessage.tool_calls` contains

A list of dicts, one per requested call:

```python
{"name": "multiply", "args": {"a": 123, "b": 456}, "id": "call_abc123", "type": "tool_call"}
```

`args` is model-generated and untrusted: validate it before use. A response may hold several calls, or none.

## Why `ToolMessage.tool_call_id` must match

OpenAI pairs each result with the request that produced it via the id. Every `tool_calls[i].id` must be answered by a `ToolMessage` with the same `tool_call_id`, placed right after the `AIMessage`; otherwise the API rejects the history. Errors are therefore returned as error `ToolMessage`s rather than skipped.

## Message order in one cycle

```
HumanMessage  -> AIMessage(tool_calls=[...]) -> ToolMessage(tool_call_id=...) -> AIMessage(final answer)
```

The demo runs exactly one such cycle (no loop). Looping until the model stops asking for tools comes later with LangGraph.
