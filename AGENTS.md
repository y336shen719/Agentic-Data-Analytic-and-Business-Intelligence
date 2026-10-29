# Engineering Instructions

- Follow the stateful single-agent architecture. Do not introduce
  multi-agent orchestration without explicit approval.
- Use LangGraph for agent orchestration when that functionality is implemented.
- Keep each tool modular, narrowly scoped, and independently testable.
- Use DuckDB read-only connections for analytical queries.
- Never execute unrestricted LLM-generated Python code.
- Validate all tool inputs before performing work.
- Use bounded retries, timeouts, and execution/result limits.
- Do not fabricate data, tool results, or evaluation metrics. Report missing
  data and failures explicitly.
- Keep credentials and secrets outside source control. Never hardcode API keys.
- Write simple, readable, maintainable Python code and avoid unnecessary
  dependencies.
- Add unit tests for implemented functionality.
- Do not alter Git configuration, switch branches, or commit/push changes
  without approval.
