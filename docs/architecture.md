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
