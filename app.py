"""Streamlit web application for Data Analytics & Business Intelligence Agent.

Provides a natural-language business intelligence interface powered by a stateful
LangGraph agent querying the Olist Brazilian E-Commerce DuckDB database, executing
Python statistical analyses, and rendering interactive Plotly charts within an
independently scrollable SaaS dashboard layout.
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
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
    execute_session_turn,
    get_or_create_session,
    reset_agent_session,
    resolve_api_key,
)
from data.prepare_data import DEFAULT_DB_PATH
from tools.chart_registry import ChartNotFoundError, ChartRegistry
from tools.sql_tool import get_database_schema

# ==============================================================================
# Descriptive Metadata for Base Tables and Views
# ==============================================================================

BASE_TABLE_DESCRIPTIONS: Dict[str, str] = {
    "customers": "Customer account records and shipping location details.",
    "orders": "Core order transaction records, fulfillment timestamps, and delivery statuses.",
    "order_items": "Line items per order, specifying product, seller, price, and freight value.",
    "order_payments": "Payment transactions, payment methods, installments, and payment values.",
    "order_reviews": "Customer post-delivery review ratings (1-5) and survey text comments.",
    "products": "Product catalog dimensions, weights, and category classifications.",
    "sellers": "Seller account records and merchant geographic locations.",
    "geolocation": "Brazilian postal zip code coordinates (latitude, longitude) and city/state mappings.",
    "category_translation": "Portuguese to English translations for product category names.",
}

VIEW_DESCRIPTIONS: Dict[str, str] = {
    "order_summary": "Pre-aggregated order-level analytical view with revenue, item counts, payment totals, and review scores.",
    "category_sales_summary": "Monthly category-level sales, item volumes, gross revenue, and active seller metrics.",
}

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

# Synchronize Streamlit secrets to process environment in main execution thread
sync_secrets_to_environment()
setup_langsmith_environment()

# Inject external enterprise SaaS stylesheet
if STYLES_PATH.exists():
    try:
        st.markdown(f"<style>{STYLES_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    except Exception:
        pass


# ==============================================================================
# 2. Application Access Gate (Password Protection)
# ==============================================================================

def get_app_access_password() -> Optional[str]:
    """Retrieve APP_ACCESS_PASSWORD from Streamlit Secrets or environment."""
    val: Optional[str] = None
    if _has_streamlit_context():
        try:
            val = st.secrets.get("APP_ACCESS_PASSWORD")
        except Exception:
            val = None
    if not val:
        val = os.environ.get("APP_ACCESS_PASSWORD")
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
            submitted = st.form_submit_button("Enter", type="primary", use_container_width=True)
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
# 3. Sidebar Navigation and Configuration
# ==============================================================================

st.sidebar.title("Data Analytics & BI Agent")
st.sidebar.markdown(
    "Autonomous E-Commerce Business Intelligence assistant powered by a "
    "stateful LangGraph agent, DuckDB, and Plotly."
)

st.sidebar.header("Model Selection")
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

st.sidebar.header("Example Questions")
for idx, q_text in enumerate(EXAMPLE_QUESTIONS):
    if st.sidebar.button(q_text, key=f"example_q_{idx}", use_container_width=True):
        st.session_state["pending_prompt"] = q_text
        st.rerun()

st.sidebar.header("Session Management")
if st.sidebar.button("Clear Conversation", use_container_width=True, type="secondary"):
    if "agent_session" in st.session_state and st.session_state["agent_session"]:
        reset_agent_session(st.session_state["agent_session"])
    st.session_state["pending_prompt"] = None
    st.rerun()

if st.sidebar.button("Log Out", use_container_width=True):
    st.session_state["authenticated"] = False
    st.session_state.pop("pending_prompt", None)
    st.rerun()

st.sidebar.header("Developer Settings")
show_debug = st.sidebar.checkbox(
    "Show Technical Tool Executions",
    value=False,
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
        if st.button("Run Diagnostic Probe", key="btn_run_diagnostic", use_container_width=True):
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
# 4. Main Area Header and Pre-flight Checks
# ==============================================================================

st.title("Data Analytics & Business Intelligence Agent")
st.caption(
    "Autonomous Business Intelligence: natural-language SQL, Python statistical analysis, "
    "and interactive Plotly visualization."
)

# Database existence check
db_file = Path(DEFAULT_DB_PATH)
if not db_file.exists():
    st.error(
        f"Database file not found at `{DEFAULT_DB_PATH}`. "
        "Please run `python -m data.prepare_data` to initialize the DuckDB database."
    )
    st.stop()

# Server-side OpenAI API Key resolution
resolved_key = resolve_api_key(None)
if not resolved_key:
    st.warning(
        "OpenAI API Key is not configured on the server. "
        "Please configure OPENAI_API_KEY in Streamlit Secrets or the local environment."
    )

# Retrieve or initialize stateful agent session
session = None
if resolved_key:
    session = get_or_create_session(
        st.session_state,
        db_path=DEFAULT_DB_PATH,
        api_key=resolved_key,
        model=model_name,
    )


# ==============================================================================
# 5. Section B (Upper): Dataset Explorer
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_database_schema(db_path: str) -> Dict[str, Any]:
    """Inspect and return the database schema using DuckDB metadata queries."""
    return get_database_schema(db_path)


st.header("Dataset Explorer")
st.markdown(
    """
    <div style="margin-bottom: 0.65rem;">
        <span class="metadata-pill"><strong>Dataset</strong>: Olist Brazilian E-Commerce</span>
        <span class="metadata-pill"><strong>Engine</strong>: DuckDB (Read-Only)</span>
        <span class="metadata-pill"><strong>Volume</strong>: ~99,000 orders</span>
        <span class="metadata-pill"><strong>Period</strong>: 2016–2018</span>
    </div>
    """,
    unsafe_allow_html=True,
)

schema_error: Optional[str] = None
base_tables: List[Dict[str, Any]] = []
analytical_views: List[Dict[str, Any]] = []

try:
    schema_data = load_database_schema(DEFAULT_DB_PATH)
    base_tables = schema_data.get("tables", [])
    analytical_views = schema_data.get("views", [])
except Exception as exc:
    schema_error = f"Unable to introspect database schema: {exc}"

if schema_error:
    st.warning(schema_error)
else:
    # Height-constrained container prevents schema expansion from displacing the chat panel
    with st.container(height=260, border=False):
        tab_tables, tab_views = st.tabs(
            [f"Base Tables ({len(base_tables)})", f"Analytical Views ({len(analytical_views)})"]
        )
        with tab_tables:
            if not base_tables:
                st.info("No base tables found in database.")
            for tbl in base_tables:
                tbl_name = tbl["name"]
                with st.expander(tbl_name, expanded=False):
                    desc = BASE_TABLE_DESCRIPTIONS.get(tbl_name)
                    if desc:
                        st.caption(desc)
                    cols = tbl.get("columns", [])
                    if cols:
                        col_records = []
                        for c in cols:
                            nullable_val = c.get("nullable")
                            nullable_str = "Yes" if nullable_val is True else ("No" if nullable_val is False else "—")
                            rec = {
                                "Column": c["name"],
                                "Data Type": c["data_type"],
                                "Nullable": nullable_str,
                            }
                            if "description" in c:
                                rec["Description"] = c["description"]
                            col_records.append(rec)
                        st.dataframe(col_records, width="stretch", hide_index=True)
                    else:
                        st.caption("No column metadata available.")

        with tab_views:
            if not analytical_views:
                st.info("No analytical views found in database.")
            for vw in analytical_views:
                vw_name = vw["name"]
                with st.expander(f"{vw_name} (Analytical View)", expanded=False):
                    purpose = VIEW_DESCRIPTIONS.get(vw_name) or vw.get("purpose")
                    if purpose:
                        st.caption(f"**Purpose**: {purpose}")
                    if vw.get("grain"):
                        st.caption(f"**Grain**: {vw['grain']}")
                    if vw.get("warning"):
                        st.caption(f"**Warning**: {vw['warning']}")
                    cols = vw.get("columns", [])
                    if cols:
                        col_records = []
                        for c in cols:
                            nullable_val = c.get("nullable")
                            nullable_str = "Yes" if nullable_val is True else ("No" if nullable_val is False else "—")
                            rec = {
                                "Column": c["name"],
                                "Data Type": c["data_type"],
                                "Nullable": nullable_str,
                            }
                            if "description" in c:
                                rec["Description"] = c["description"]
                            col_records.append(rec)
                        st.dataframe(col_records, width="stretch", hide_index=True)
                    else:
                        st.caption("No column metadata available.")


# ==============================================================================
# 6. Section B (Middle): Question Entry
# ==============================================================================

with st.form("query_form", clear_on_submit=True):
    col_input, col_submit = st.columns([5, 1])
    with col_input:
        query_input = st.text_input(
            "Question",
            placeholder="Ask a business analytics question...",
            label_visibility="collapsed",
            disabled=not bool(resolved_key),
        )
    with col_submit:
        form_submitted = st.form_submit_button(
            "Submit",
            type="primary",
            use_container_width=True,
            disabled=not bool(resolved_key),
        )

# Identify question from form submission or sidebar example selection
active_prompt: Optional[str] = None
if st.session_state.get("pending_prompt"):
    active_prompt = st.session_state.pop("pending_prompt")
elif form_submitted and query_input and query_input.strip():
    active_prompt = query_input.strip()

if active_prompt and session is not None:
    with st.spinner("Analyzing data and generating insights..."):
        execute_session_turn(session, active_prompt)
    st.rerun()


# ==============================================================================
# 7. Section C (Lower): Dedicated Scrollable Conversation History Panel
# ==============================================================================

def render_assistant_message(
    msg: Dict[str, Any],
    chart_registry: ChartRegistry,
    show_debug: bool = False,
) -> None:
    """Render a single assistant message with its narrative, charts, and optional debug steps."""
    with st.chat_message("assistant"):
        # 1. Render assistant narrative first (result-first UI)
        st.markdown(msg.get("content", ""))

        # 2. Render associated interactive Plotly figures
        chart_ids = msg.get("chart_ids", [])
        for cid in chart_ids:
            try:
                record = chart_registry.get(cid)
                fig = go.Figure(record.figure)
                st.plotly_chart(fig, width="stretch", key=f"chart_{cid}")
            except ChartNotFoundError:
                st.info("The requested visualization is no longer available in this session.")
            except Exception:
                st.warning("Unable to display the interactive chart due to a rendering issue.")

        # 3. Render tool execution diagnostics only if developer toggle is enabled
        tool_steps = msg.get("tool_executions", [])
        if show_debug and tool_steps:
            with st.expander(f"Tool Executions ({len(tool_steps)})", expanded=False):
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


st.header("Conversation History")

# Native scrollable chat container with automatic scroll-to-bottom on new messages
with st.container(height=520, border=True, autoscroll=True):
    if session is not None and session.ui_messages:
        for msg in session.ui_messages:
            role = msg.get("role", "assistant")
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(msg.get("content", ""))
            else:
                render_assistant_message(msg, session.chart_registry, show_debug=show_debug)
    else:
        st.info(
            "Welcome to the Data Analytics & Business Intelligence Agent. "
            "Ask a business analytics question above, or select an example question from the sidebar to begin."
        )
