"""Streamlit web application for Data Analytics & Business Intelligence Agent.

Provides a natural-language business intelligence interface powered by a stateful
LangGraph agent querying the Olist Brazilian E-Commerce DuckDB database, executing
Python statistical analyses, and rendering interactive Plotly charts within an
enterprise SaaS layout with multi-page navigation and dedicated Database Explorer.
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from agent.config import DEFAULT_OPENAI_MODEL
from agent.tracing import (
    _has_streamlit_context,
    has_diagnostic_password_configured,
    run_synthetic_trace_diagnostic,
    setup_langsmith_environment,
    sync_secrets_to_environment,
    verify_diagnostic_access,
)
from agent.ui_session import (
    reset_agent_session,
    resolve_api_key,
)
from data.prepare_data import DEFAULT_DB_PATH
from views.analytics import (
    BASE_TABLE_DESCRIPTIONS,
    VIEW_DESCRIPTIONS,
    load_database_schema,
    render_analytics_page,
    render_assistant_message,
)
from views.database_explorer import render_database_explorer_page

# Re-exports for backward compatibility
__all__ = [
    "BASE_TABLE_DESCRIPTIONS",
    "VIEW_DESCRIPTIONS",
    "load_database_schema",
    "render_assistant_message",
    "render_analytics_page",
    "render_database_explorer_page",
]

# Assets & Path Resolution
ASSETS_DIR = Path(__file__).parent / "assets"
FAVICON_PATH = ASSETS_DIR / "favicon.svg"
STYLES_PATH = ASSETS_DIR / "styles.css"

# ==============================================================================
# 1. Page Configuration & Custom CSS Injection
# ==============================================================================

page_icon_val = str(FAVICON_PATH) if FAVICON_PATH.exists() else None

st.set_page_config(
    page_title="Data Analytics | BI Agent",
    page_icon=page_icon_val,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Synchronize secrets and initialize LangSmith tracing before any agent work
sync_secrets_to_environment()
setup_langsmith_environment()

# Inject external enterprise stylesheet
if STYLES_PATH.exists():
    css_content = STYLES_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


# ==============================================================================
# 2. Application Access Gate (Password Protection)
# ==============================================================================

def get_app_access_password() -> Optional[str]:
    """Retrieve APP_ACCESS_PASSWORD from environment or Streamlit Secrets."""
    val: Optional[str] = os.environ.get("APP_ACCESS_PASSWORD")
    if not val and _has_streamlit_context():
        try:
            val = st.secrets.get("APP_ACCESS_PASSWORD")
        except Exception:
            val = None
    if val:
        val = str(val).strip()
    return val or None


def check_app_access() -> bool:
    """Validate application access gate.

    Fails closed if APP_ACCESS_PASSWORD is not configured.
    Renders login form if unauthenticated.
    Returns True only when the session is authenticated.
    """
    configured_password = get_app_access_password()
    if not configured_password:
        st.error(
            "Application configuration error: Access password is not configured. "
            "Please configure APP_ACCESS_PASSWORD in Streamlit Secrets or environment variables."
        )
        return False

    if st.session_state.get("authenticated", False):
        return True

    col_l, col_center, col_r = st.columns([1, 1.8, 1])
    with col_center:
        st.subheader("Data Analytics & Business Intelligence Agent")
        st.caption("Enterprise Analytics Assistant. Please enter the access password to continue.")

        with st.form("access_gate_form", clear_on_submit=False):
            entered_password = st.text_input(
                "Access Password",
                type="password",
                placeholder="Enter password",
                key="app_access_password_input",
                help="Application access password configured in Streamlit Secrets or environment.",
            )
            submitted = st.form_submit_button("Enter", type="primary", width="stretch")
            if submitted:
                if entered_password and hmac.compare_digest(
                    entered_password.strip().encode("utf-8"),
                    configured_password.encode("utf-8"),
                ):
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Incorrect password. Please try again.")

    return False


if not check_app_access():
    st.stop()


# ==============================================================================
# 3. Pre-flight Checks
# ==============================================================================

# Database existence check
db_file = Path(DEFAULT_DB_PATH)
if not db_file.exists():
    st.error(
        f"Database file not found at `{DEFAULT_DB_PATH}`. "
        "Please run `python -m data.prepare_data` to initialize the DuckDB database."
    )
    st.stop()


# ==============================================================================
# 4. Multi-Page Navigation Setup
# ==============================================================================

page_analytics = st.Page(
    "views/analytics.py",
    title="Analytics Workspace",
    icon=":material/home:",
    default=True,
)
page_db = st.Page(
    "views/database_explorer.py",
    title="Database Explorer",
    icon=":material/database:",
)

pg = st.navigation([page_analytics, page_db], position="hidden")


# ==============================================================================
# 5. Sidebar Navigation and Configuration
# ==============================================================================

st.sidebar.title("Data Analytics & BI Agent")
st.sidebar.caption("Business Intelligence Copilot")

st.sidebar.page_link(page_analytics, label="Analytics Workspace", icon=":material/home:")
st.sidebar.page_link(page_db, label="Database Explorer", icon=":material/database:")
st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

# Example questions matching modern copilot cards
EXAMPLE_QUESTIONS = [
    "What are the top 5 product categories by total revenue?",
    "Show the monthly revenue trend for delivered orders in 2017.",
    "What is the correlation between product price and freight value?",
    "Analyze the distribution of order review scores with a histogram.",
    "Which Brazilian states have the highest average freight value?",
]

with st.sidebar.container(key="example_questions_container"):
    st.header("Example Questions")
    for idx, q_text in enumerate(EXAMPLE_QUESTIONS):
        if st.button(q_text, key=f"example_q_{idx}", width="stretch"):
            st.session_state["pending_prompt"] = q_text
            if pg != page_analytics:
                st.switch_page(page_analytics)
            else:
                st.rerun()

with st.sidebar.container(key="session_management_container"):
    st.header("Session Management")
    if st.button("Clear Conversation", icon=":material/delete:", width="stretch", type="secondary"):
        if "agent_session" in st.session_state and st.session_state["agent_session"]:
            reset_agent_session(st.session_state["agent_session"])
        st.session_state["pending_prompt"] = None
        st.rerun()

    if st.button("Log Out", icon=":material/logout:", width="stretch"):
        st.session_state["authenticated"] = False
        st.session_state.pop("pending_prompt", None)
        st.rerun()

with st.sidebar.container(key="developer_settings_container"):
    st.header("Developer Settings")
    st.checkbox(
        ":material/settings: Show Technical Tool Executions",
        value=False,
        key="show_debug",
        help="Display SQL queries, Python arguments, and tool execution metadata for debugging.",
    )

# Admin Diagnostics (Disabled by default, requires DIAGNOSTIC_PASSWORD in secrets/env)
if has_diagnostic_password_configured():
    with st.sidebar.expander("Admin Diagnostics", expanded=False):
        st.caption("Administrator-only synthetic probe to verify cloud LangSmith tracing.")
        diag_password = st.text_input(
            "Diagnostic Password",
            type="password",
            key="diag_pwd_input",
            help="Requires DIAGNOSTIC_PASSWORD in Streamlit Secrets.",
        )
        if st.button("Run Diagnostic Probe", key="btn_run_diagnostic", width="stretch"):
            if not verify_diagnostic_access(diag_password):
                st.error("Invalid password or diagnostics disabled.")
            else:
                with st.spinner("Executing direct synthetic trace probe..."):
                    diag_result = run_synthetic_trace_diagnostic()
                if diag_result.get("status") == "success" and diag_result.get("read_back_verified"):
                    st.success("LangSmith trace successfully created and verified on server.")
                    st.json(diag_result)
                elif diag_result.get("status") == "error":
                    st.error(f"Diagnostic failed: {diag_result.get('error')}")
                    st.json(diag_result)
                else:
                    st.warning("Trace submitted, but read-back could not be verified.")
                    st.json(diag_result)


# ==============================================================================
# 6. Execute Selected Page
# ==============================================================================

pg.run()
