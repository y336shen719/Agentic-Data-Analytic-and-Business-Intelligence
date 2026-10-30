"""End-to-End Live Demonstration of the LangGraph Agentic Data Analyst.

Usage:
    python -m scripts.demo_agent_graph

Requirements:
    Requires a valid OPENAI_API_KEY set in .env or the system environment.
    This script makes live LLM calls and is intended for manual smoke testing,
    not automated unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.config import ConfigError, load_llm_config
from agent.graph import create_agent_graph
from agent.state import create_initial_state


def _format_content(content: Any) -> str:
    """Format message content whether it is a string or structured content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
            else:
                text_parts.append(str(block))
        return "\n".join(text_parts)
    return str(content)


def _summarize_tool_result(name: str, content: str) -> str:
    """Provide a concise summary of tool results without dumping huge schema strings."""
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            if name == "get_database_schema":
                table_names = list(data.keys())
                return f"[Schema Discovery] Successfully found {len(table_names)} tables: {', '.join(table_names[:5])}..."
            if name == "execute_sql":
                status = data.get("status")
                row_count = data.get("row_count", 0)
                time_ms = data.get("execution_time_ms", 0)
                rows = data.get("rows", [])
                sample = rows[0] if rows else None
                return f"[SQL Execution] Status: {status}, Rows: {row_count}, Time: {time_ms:.1f}ms, Sample: {sample}"
    except Exception:
        pass

    # Fallback to truncated preview
    clean = content.strip().replace("\n", " ")
    if len(clean) > 120:
        return clean[:120] + "..."
    return clean


def main() -> None:
    print("=" * 75)
    print("Step 4.5 — Agentic Data Analyst: LangGraph Workflow Live Demo")
    print("=" * 75)

    # 1. Verify Configuration
    try:
        config = load_llm_config()
        if not config.api_key:
            raise ConfigError("OPENAI_API_KEY is not configured in .env or environment.")
        print(f"Configuration: model={config.model}, timeout={config.timeout_seconds}s")
    except ConfigError as err:
        print(f"Configuration Error: {err}")
        print("Please copy .env.example to .env and configure your OPENAI_API_KEY.")
        sys.exit(1)

    # 2. Build the Agent Graph
    print("\nBuilding LangGraph agent workflow...")
    agent_graph = create_agent_graph(max_steps=10)
    print("Graph compiled successfully (Topology: START -> llm -> tools -> llm -> END).")

    # 3. Formulate Question
    question = "How many orders are in the database? First check the schema, then execute a SQL query."
    print(f"\nUser Question:\n  \"{question}\"")

    initial_state = create_initial_state(
        messages=[HumanMessage(content=question)],
    )

    # 4. Invoke the Workflow
    print("\nInvoking Agent Workflow...\n" + "-" * 75)
    final_state = agent_graph.invoke(initial_state)
    print("-" * 75)

    # 5. Print Execution Event Log
    print("\nExecution History:")
    for idx, msg in enumerate(final_state["messages"], start=1):
        if isinstance(msg, HumanMessage):
            print(f"\n[{idx}] User: {msg.content}")
        elif isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                print(f"\n[{idx}] AI Decision (Tool Calling Requested):")
                for tc in tool_calls:
                    tc_name = tc.get("name")
                    tc_args = tc.get("args")
                    print(f"    - Requested Tool: `{tc_name}` with args: {tc_args}")
            if msg.content:
                formatted_text = _format_content(msg.content)
                if not tool_calls:
                    print(f"\n[{idx}] AI Final Answer:")
                    print(f"    {formatted_text}")
                else:
                    print(f"    Thought / Message: {formatted_text}")
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            summary = _summarize_tool_result(tool_name, str(msg.content))
            print(f"\n[{idx}] Tool Output (`{tool_name}`):")
            print(f"    {summary}")

    # 6. Print Summary
    print("\n" + "=" * 75)
    print("Workflow Execution Summary:")
    print(f"  Total Steps Taken : {final_state.get('step_count')}")
    print(f"  Last SQL Query    : {final_state.get('last_sql')}")
    print(f"  Total Messages    : {len(final_state.get('messages', []))}")
    print("=" * 75)


if __name__ == "__main__":
    main()
