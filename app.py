"""Streamlit web application for the Agentic Data Analyst.

Provides a natural-language business intelligence chat interface powered by a stateful
LangGraph agent querying the Olist Brazilian E-Commerce DuckDB database, executing
Python statistical analyses, and rendering interactive Plotly charts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import plotly.graph_objects as go
import streamlit as st

from agent.config import DEFAULT_OPENAI_MODEL
from agent.tracing import (
    has_diagnostic_password_configured,
    run_synthetic_trace_diagnostic,
    setup_langsmith_environment,
    sync_secrets_to_environment,
    verify_diagnostic_access,
)
from agent.ui_session import (
    execute_session_turn,
    get_or_create_session,
    reset_agent_session,
    resolve_api_key,
)
from data.prepare_data import DEFAULT_DB_PATH
from tools.chart_registry import ChartNotFoundError, ChartRegistry

# 1. Page Configuration
st.set_page_config(
    page_title="Agentic Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Synchronize Streamlit secrets to process environment in main execution thread
sync_secrets_to_environment()
setup_langsmith_environment()

# 2. Sidebar Navigation and Configuration
st.sidebar.title("📊 Data Analyst Agent")
st.sidebar.markdown(
    "Autonomous E-Commerce Business Intelligence assistant powered by a "
    "stateful LangGraph agent, DuckDB, and Plotly."
)

st.sidebar.header("📁 Dataset Information")
st.sidebar.markdown(
    """
    **Olist Brazilian E-Commerce**
    - **Engine**: DuckDB (Read-Only)
    - **Orders**: ~99k orders (2016–2018)
    - **Core Tables**:
      - `orders` (lifecycle & status)
      - `order_items` (items & prices)
      - `customers` (locations & users)
      - `products` (categories & weights)
      - `order_payments` (payment methods)
      - `order_reviews` (ratings & comments)
      - `sellers` (merchant locations)
    """
)

st.sidebar.header("⚙️ Configuration")
api_key_input = st.sidebar.text_input(
    "OpenAI API Key (optional)",
    type="password",
    placeholder="Using configured credentials if available",
    help="Set in environment, Streamlit secrets, or enter your key here.",
)
actual_user_key = api_key_input.strip() if api_key_input and api_key_input.strip() else None
resolved_key = resolve_api_key(actual_user_key)

model_name = st.sidebar.selectbox(
    "Model",
    options=[DEFAULT_OPENAI_MODEL, "gpt-4o", "gpt-4o-mini", "o3-mini"],
    index=0,
    help="Select the OpenAI model for reasoning and tool calling.",
)

# Example questions
EXAMPLE_QUESTIONS = [
    "What are the top 5 product categories by total delivered revenue?",
    "Show the monthly revenue trend for delivered orders in 2017.",
    "What is the correlation between product price and freight value?",
    "Analyze the distribution of order review scores with a histogram.",
    "Which Brazilian states have the highest average freight value?",
]

st.sidebar.header("💡 Example Questions")
for idx, q_text in enumerate(EXAMPLE_QUESTIONS):
    if st.sidebar.button(q_text, key=f"example_q_{idx}", use_container_width=True):
        st.session_state["pending_prompt"] = q_text

st.sidebar.header("Session Management")
if st.sidebar.button("🗑️ Clear Conversation", use_container_width=True, type="secondary"):
    if "agent_session" in st.session_state and st.session_state["agent_session"]:
        reset_agent_session(st.session_state["agent_session"])
    st.session_state["pending_prompt"] = None
    st.rerun()

st.sidebar.header("🛠️ Developer Settings")
show_debug = st.sidebar.checkbox(
    "Show Technical Tool Executions",
    value=False,
    help="Display SQL queries, Python arguments, and tool execution metadata for debugging.",
)

# Admin Diagnostics (Disabled by default, requires DIAGNOSTIC_PASSWORD in secrets/env)
if has_diagnostic_password_configured():
    with st.sidebar.expander("🔒 Admin Diagnostics", expanded=False):
        st.caption("Administrator-only synthetic probe to verify cloud LangSmith tracing.")
        diag_password = st.text_input(
            "Diagnostic Password",
            type="password",
            key="diag_pwd_input",
            help="Requires DIAGNOSTIC_PASSWORD in Streamlit Secrets.",
        )
        if st.button("Run Diagnostic Probe", key="btn_run_diagnostic", use_container_width=True):
            if not verify_diagnostic_access(diag_password):
                st.error("Invalid password or diagnostics disabled.")
            else:
                with st.spinner("Executing direct synthetic trace probe..."):
                    diag_result = run_synthetic_trace_diagnostic()
                if diag_result.get("status") == "success" and diag_result.get("read_back_verified"):
                    st.success("✅ LangSmith trace successfully created and verified on server!")
                    st.json(diag_result)
                elif diag_result.get("status") == "error":
                    st.error(f"❌ Diagnostic failed: {diag_result.get('error')}")
                    st.json(diag_result)
                else:
                    st.warning("⚠️ Trace submitted, but read-back could not be verified.")
                    st.json(diag_result)

# 3. Main Header
st.title("📊 Agentic Data Analyst")
st.caption(
    "Autonomous Business Intelligence: natural-language SQL, Python statistical analysis, "
    "and interactive Plotly visualization."
)

# 4. Check Database Existence
db_file = Path(DEFAULT_DB_PATH)
if not db_file.exists():
    st.error(
        f"Database file not found at `{DEFAULT_DB_PATH}`. "
        "Please run `python -m data.prepare_data` to initialize the DuckDB database."
    )
    st.stop()

# 5. Check API Key
if not resolved_key:
    st.warning(
        "⚠️ **OpenAI API Key Required**: Please provide your OpenAI API key in the sidebar "
        "or define `OPENAI_API_KEY` in your `.env` file to interact with the agent."
    )

# 6. Retrieve or Reconfigure Session
session = None
if resolved_key:
    session = get_or_create_session(
        st.session_state,
        db_path=DEFAULT_DB_PATH,
        api_key=resolved_key,
        model=model_name,
    )


def render_assistant_message(
    msg: Dict[str, Any],
    chart_registry: ChartRegistry,
    show_debug: bool = False,
) -> None:
    """Render a single assistant message with its narrative, charts, and optional debug steps."""
    with st.chat_message("assistant", avatar="🤖"):
        # 1. Render assistant narrative first (result-first UI)
        st.markdown(msg.get("content", ""))

        # 2. Render associated interactive Plotly figures
        chart_ids = msg.get("chart_ids", [])
        for cid in chart_ids:
            try:
                record = chart_registry.get(cid)
                fig = go.Figure(record.figure)
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{cid}")
            except ChartNotFoundError:
                st.info("The requested visualization is no longer available in this session.")
            except Exception:
                # Sanitized user-friendly warning without leaking internal exceptions or paths
                st.warning("Unable to display the interactive chart due to a rendering issue.")

        # 3. Render tool execution diagnostics only if developer toggle is enabled
        tool_steps = msg.get("tool_executions", [])
        if show_debug and tool_steps:
            with st.expander(f"🛠️ Tool Executions ({len(tool_steps)})", expanded=False):
                for step in tool_steps:
                    st.markdown(f"**{step.get('title', 'Tool Execution')}**")
                    details = step.get("details")
                    if details:
                        lang = step.get("language", "text")
                        st.code(details, language=lang)
                    summary = step.get("summary")
                    if summary:
                        st.caption(summary)
                    st.divider()


# 7. Consolidated Chat History Rendering
if session is not None and session.ui_messages:
    for msg in session.ui_messages:
        role = msg.get("role", "assistant")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(msg.get("content", ""))
        else:
            render_assistant_message(msg, session.chart_registry, show_debug=show_debug)
else:
    # Welcome banner when no messages exist
    st.info(
        "👋 Welcome! Ask any business question about the Olist e-commerce database, "
        "or select one of the example questions in the sidebar to get started."
    )

# 8. Handle User Input
prompt: Optional[str] = None
if "pending_prompt" in st.session_state and st.session_state["pending_prompt"]:
    prompt = st.session_state.pop("pending_prompt")
else:
    chat_disabled = not bool(resolved_key)
    chat_placeholder = (
        "Ask a business analytics question..."
        if resolved_key
        else "Please configure your OpenAI API Key in the sidebar..."
    )
    prompt = st.chat_input(chat_placeholder, disabled=chat_disabled)

if prompt and session is not None:
    # Display the user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process via LangGraph with spinner
    with st.spinner("Analyzing data and generating insights..."):
        execute_session_turn(session, prompt)

    # Rerun to cleanly render through the consolidated history path
    st.rerun()
