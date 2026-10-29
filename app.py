"""Streamlit landing page for the Agentic Data Analyst project."""

import streamlit as st


st.set_page_config(page_title="Agentic Data Analyst", page_icon="📊")

st.title("Agentic Data Analyst")
st.subheader("Autonomous E-Commerce Business Intelligence Agent")
st.write(
    "A portfolio project exploring a stateful AI agent for analyzing "
    "e-commerce business data through natural language."
)

st.info("This project is currently under development.")

st.header("Planned technology stack")
st.markdown(
    """
- Python 3.11
- LangGraph and LangChain OpenAI
- DuckDB, pandas, and NumPy
- Matplotlib and Streamlit
- Pytest and python-dotenv
"""
)

st.header("Planned core features")
st.markdown(
    """
- Stateful single-agent orchestration with LangGraph
- Natural-language business questions with LLM tool calling
- Validated, read-only analytical SQL against DuckDB
- Python statistical analysis and automated chart generation
- Bounded error handling, retries, and agent evaluation
- Optional LangSmith tracing
"""
)
