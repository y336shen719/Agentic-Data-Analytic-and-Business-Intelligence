# Agentic Data Analyst

**Autonomous E-Commerce Business Intelligence Agent** is a portfolio project
to explore how a stateful AI agent could answer natural-language questions
about e-commerce business data.

> **Status: Under development.** This repository currently contains project
> scaffolding and a minimal Streamlit landing page. The agent, SQL tools,
> analytics, charts, data pipeline, and evaluation workflow are planned and
> are not yet implemented.

## Business problem and objective

E-commerce teams often need to investigate business performance across
multiple datasets and turn questions into useful analysis. This project aims
to build a stateful, single-agent system that can break down those questions,
select appropriate tools, inspect results, and provide a grounded report.
The eventual system is intended to support multi-step analysis while keeping
data access and execution constrained.

## Dataset

The planned data source is the [Olist Brazilian E-Commerce Public
Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). The
dataset has not been downloaded or loaded into this repository. Its future
DuckDB schema will be documented in [data/schema.md](data/schema.md) after
the data preparation workflow is implemented and verified.

## Planned architecture

The planned architecture is a **stateful single agent with multiple tools**:
Streamlit accepts a user request, LangGraph orchestrates an agent loop, and
the LLM may select from constrained SQL, Python analysis, and chart tools.
Tool results and errors are recorded in agent state so the agent can continue
or return a grounded final report. The design is documented in
[docs/architecture.md](docs/architecture.md).

This architecture is a plan, not a description of currently working agent
behavior. No LLM is called and no LangGraph graph is initialized by the
current application.

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

After installing dependencies, run:

```bash
streamlit run app.py
```

The current landing page does not require credentials, connect to an LLM, or
provide chat functionality.

## Environment variables

Copy the example file if you want a local template for future integrations:

```bash
cp .env.example .env
```

Set credentials only in the local `.env` file or another secure secrets
manager. Never commit real credentials. The current landing page does not load
or use these variables; API keys are not required to run it.

## Planned development milestones

1. Define and validate a reproducible Olist data preparation workflow.
2. Load prepared data into DuckDB and document its verified schema.
3. Implement and unit-test constrained, read-only SQL access.
4. Add statistical analysis and chart tools with validated inputs.
5. Define agent state, prompts, and bounded LangGraph orchestration.
6. Connect the Streamlit interface to the agent and add robust error handling.
7. Add evaluation cases and measured evaluation reporting.
8. Add optional LangSmith tracing and document operational safeguards.

These are planned milestones; they are not claims of implemented functionality.

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
