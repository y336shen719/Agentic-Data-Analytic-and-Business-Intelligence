# Agentic Data Analyst

**Autonomous E-Commerce Business Intelligence Agent** is a portfolio project
to explore how a stateful AI agent could answer natural-language questions
about e-commerce business data.

> **Status: Functional.** This repository contains an autonomous Business Intelligence
> agent built on LangGraph, DuckDB, and Plotly, with a full Streamlit chat interface.
> The agent executes validated, read-only SQL, bounded Python statistical analysis,
> and interactive Plotly chart generation.

## Business problem and objective

E-commerce teams often need to investigate business performance across
multiple datasets and turn questions into useful analysis. This project implements
a stateful, single-agent system that breaks down questions, selects appropriate
tools, inspects results, and provides a grounded report.
The system supports multi-step, multi-turn analysis while keeping data access and
execution strictly bounded.

## Dataset

The data source is the [Olist Brazilian E-Commerce Public
Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). The
ingested DuckDB schema is documented in [data/schema.md](data/schema.md).

## Architecture

The architecture is a **stateful single agent with multiple tools**:
Streamlit accepts user requests, LangGraph orchestrates an agent loop, and
the LLM selects from constrained SQL, Python analysis, and chart tools.
Tool results and errors are recorded in agent state so the agent can recover
from errors or return a grounded final report with interactive Plotly figures.
The design is documented in [docs/architecture.md](docs/architecture.md).

## Technology stack

- Python 3.11
- LangGraph
- LangChain OpenAI
- DuckDB
- pandas and NumPy
- Matplotlib
- Streamlit
- Pytest
- python-dotenv
- LangSmith tracing (planned/optional)

## Repository structure

```text
.
├── app.py
├── agent/                 # Planned state, prompts, and graph orchestration
├── data/                  # Planned data preparation and schema documentation
├── docs/                  # Architecture documentation
├── evaluation/            # Planned evaluation cases and runner
├── tests/                 # Planned unit and agent tests
└── tools/                  # Planned SQL, analysis, and chart tools
```

The Python modules outside the landing page currently contain only brief
module descriptions. They are intentionally not placeholder implementations.

## Local development setup

### Requirements

- Python 3.11
- Conda (for the existing `agentic-ai` environment) or another Python 3.11
  environment

Activate the existing Conda environment:

```bash
conda activate agentic-ai
```

Install the project's declared dependencies:

```bash
python -m pip install -r requirements.txt
```

Dependencies are declared without pinned versions. They are not installed
automatically by this project setup.

## Run the Streamlit application

After installing dependencies and configuring `.env` (or having an OpenAI API key ready), run:

```bash
streamlit run app.py
```

The application provides:
- Natural-language business intelligence queries against the Olist e-commerce database.
- Interactive Plotly charts rendered from session-scoped chart specifications.
- Expandable tool execution audit steps (SQL query, rows returned, Python analysis).
- Multi-turn conversation persistence and a sidebar button to clear the conversation.

## Environment variables

Copy the example file and configure your credentials:

```bash
cp .env.example .env
```

Configure `OPENAI_API_KEY` in `.env` or input it directly in the Streamlit sidebar. Never commit credentials to source control.

## LangSmith Observability & Tracing (Step 8.5.1)

LangSmith tracing is fully supported and **strictly opt-in**. Tracing provides complete visibility into:
- LangGraph agent graph executions and node transitions (`llm_node`, `tools`).
- Tool calls (`execute_sql`, `execute_python_analysis`, `generate_chart`, `get_database_schema`).
- Execution duration, token usage, tool input parameters, and error recovery events.
- Multi-turn conversation association via safe session metadata.

### Configuration

Add the following to your `.env` file:

```dotenv
# Opt-in flag: set to true to enable LangSmith tracing (default: false)
LANGSMITH_TRACING=true

# Your LangSmith API key (from https://smith.langchain.com)
LANGSMITH_API_KEY=lsv2_pt_...

# Target project name (default: agentic-data-analyst)
LANGSMITH_PROJECT=agentic-data-analyst
```

### Privacy & Data Protection Safeguards
- **Sensitive Key Redaction**: API keys, credentials, tokens, and password fields in metadata are automatically redacted (`[REDACTED]`) before trace transmission.
- **Compact Tool Payloads**: The agent only passes lightweight previews (up to 5 rows) and chart IDs in tool messages. Full DuckDB tables and large Plotly figure JSON structures are preserved in session memory and are **never** uploaded to LangSmith traces.
- **Graceful Deactivation**: If `LANGSMITH_TRACING=true` but no API key is configured, tracing gracefully disables with a warning without breaking agent execution.

### Verification Smoke Test

To verify tracing against a live scenario without full benchmark cost:
```bash
python -m dotenv run -- python -m evaluation.evaluate --live --scenario VIZ_01_MONTHLY_REVENUE_LINE
```
View the resulting runs in the [LangSmith Dashboard](https://smith.langchain.com/) under the configured project.

## Security considerations

- Keep API keys, `.env`, and Streamlit secrets out of source control.
- Treat user questions and model-produced tool arguments as untrusted input.
- Restrict analytical database access to read-only connections and validate
  query inputs before execution.
- Do not execute unrestricted LLM-generated Python code.
- Apply bounded retries, timeouts, row/result limits, and other execution
  limits when tools are implemented.
- Never fabricate dataset contents, tool results, or evaluation metrics.

See [AGENTS.md](AGENTS.md) for engineering constraints for future work.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
