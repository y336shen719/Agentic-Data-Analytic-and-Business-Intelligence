"""Analytics Workspace page module.

Preserves the existing stateful LangGraph analytical workspace, compact dataset
explorer, conversational history, and Plotly chart visualization layout while
integrating the model selector directly into the question input bar.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set

import plotly.graph_objects as go
import streamlit as st

from agent.config import DEFAULT_OPENAI_MODEL
from agent import ui_session
from data.prepare_data import DEFAULT_DB_PATH
from tools.chart_registry import ChartNotFoundError, ChartRegistry
from tools.sql_tool import db_connection, get_database_schema

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


@st.cache_data(show_spinner=False)
def load_database_schema(db_path: str) -> Dict[str, Any]:
    """Inspect and return the database schema using DuckDB metadata queries."""
    return get_database_schema(db_path)


@st.cache_data(show_spinner=False)
def get_table_row_counts(db_path: str, table_names: tuple[str, ...]) -> Dict[str, int]:
    """Retrieve row counts for all given tables in a single cached pass."""
    counts: Dict[str, int] = {}
    try:
        with db_connection(db_path) as con:
            for tbl in table_names:
                if re.fullmatch(r"^[a-zA-Z0-9_]+$", tbl):
                    try:
                        res = con.execute(f'SELECT count(*) FROM "{tbl}"').fetchone()
                        if res and res[0] is not None:
                            counts[tbl] = int(res[0])
                    except Exception:
                        pass
    except Exception:
        pass
    return counts


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


def render_analytics_page() -> None:
    """Render the primary Analytics Workspace page."""
    # Server-side OpenAI API Key resolution
    resolved_key = ui_session.resolve_api_key(None)
    if not resolved_key:
        st.warning(
            "OpenAI API Key is not configured on the server. "
            "Please configure OPENAI_API_KEY in Streamlit Secrets or the local environment."
        )

    # Determine effective model from session state or default
    current_model = st.session_state.get("selected_model", DEFAULT_OPENAI_MODEL)

    # Retrieve or initialize stateful agent session
    session = None
    if resolved_key:
        session = ui_session.get_or_create_session(
            st.session_state,
            db_path=DEFAULT_DB_PATH,
            api_key=resolved_key,
            model=current_model,
        )

    # ==========================================================================
    # Section B (Upper): Compact Database Explorer
    # ==========================================================================
    st.header("Analytics Workspace")
    st.markdown(
        """
        <div style="margin-bottom: 0.65rem;">
            <span class="metadata-pill"><strong>Dataset</strong>: Olist Brazilian E-Commerce</span>
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
        # Whitelist of authorized table and view identifiers
        all_table_names: Set[str] = {t["name"] for t in base_tables} | {v["name"] for v in analytical_views}
        row_counts = get_table_row_counts(str(DEFAULT_DB_PATH), tuple(sorted(all_table_names)))

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
                    total_rows = row_counts.get(tbl_name)
                    rows_str = f"{total_rows:,} rows" if total_rows is not None else "DuckDB table"
                    with st.expander(f"{tbl_name} • {rows_str}", expanded=False):
                        desc = BASE_TABLE_DESCRIPTIONS.get(tbl_name)
                        if desc:
                            st.caption(desc)
                        cols = tbl.get("columns", [])
                        if cols:
                            col_records = []
                            for c in cols:
                                c_name = c["name"]
                                nullable_val = c.get("nullable")
                                nullable_str = "Yes" if nullable_val is True else ("No" if nullable_val is False else "Unknown")
                                rec = {
                                    "Column": c_name,
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
                    total_rows = row_counts.get(vw_name)
                    rows_str = f"{total_rows:,} rows" if total_rows is not None else "DuckDB view"
                    with st.expander(f"{vw_name} • {rows_str}", expanded=False):
                        if vw.get("notes"):
                            st.caption(f"**Notes**: {vw['notes']}")
                        cols = vw.get("columns", [])
                        if cols:
                            col_records = []
                            for c in cols:
                                c_name = c["name"]
                                nullable_val = c.get("nullable")
                                nullable_str = "Yes" if nullable_val is True else ("No" if nullable_val is False else "Unknown")
                                rec = {
                                    "Column": c_name,
                                    "Data Type": c["data_type"],
                                    "Nullable": nullable_str,
                                }
                                if "description" in c:
                                    rec["Description"] = c["description"]
                                col_records.append(rec)
                            st.dataframe(col_records, width="stretch", hide_index=True)
                        else:
                            st.caption("No column metadata available.")

    # ==========================================================================
    # Section B (Middle): Question Entry (Unified Modern LLM Input Bar)
    # ==========================================================================
    with st.form("query_form", clear_on_submit=True):
        query_input = st.text_input(
            "Question",
            placeholder="Ask a business analytics question about this database...",
            label_visibility="collapsed",
            disabled=not bool(resolved_key),
            key="query_input",
        )
        col_model, col_spacer, col_submit = st.columns([2.4, 5.8, 1.8], vertical_alignment="center")
        with col_model:
            model_options = [DEFAULT_OPENAI_MODEL, "gpt-4o", "gpt-4o-mini", "o3-mini"]
            current_idx = model_options.index(current_model) if current_model in model_options else 0
            selected_model = st.selectbox(
                "Model",
                options=model_options,
                index=current_idx,
                format_func=lambda m: "GPT-4o" if m == "gpt-4o" else ("gpt-4o-mini" if m == "gpt-4o-mini" else ("o3-mini" if m == "o3-mini" else m)),
                label_visibility="collapsed",
                key="selected_model_dropdown",
                help="Select the OpenAI model for reasoning and tool calling.",
                disabled=not bool(resolved_key),
            )
            st.session_state["selected_model"] = selected_model
        with col_submit:
            form_submitted = st.form_submit_button(
                "Submit",
                type="primary",
                icon=":material/send:",
                width="stretch",
                disabled=not bool(resolved_key),
            )

    # Identify question from form submission or sidebar example selection
    active_prompt: Optional[str] = None
    if st.session_state.get("pending_prompt"):
        active_prompt = st.session_state.pop("pending_prompt")
    elif form_submitted and query_input and query_input.strip():
        active_prompt = query_input.strip()

    if active_prompt and session is not None:
        # Reconfigure session model on-the-fly if user altered dropdown
        if selected_model and session.configured_model != selected_model:
            ui_session.reconfigure_session(session, model=selected_model, api_key=resolved_key)

    # ==========================================================================
    # Section C (Lower): Dedicated Scrollable Conversation History Panel
    # ==========================================================================
    st.header("Conversation History")
    show_debug = bool(st.session_state.get("show_debug", False))

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
        elif not active_prompt:
            st.info(
                "Welcome to the Data Analytics & Business Intelligence Agent. "
                "Ask a business analytics question above, or select an example question from the sidebar to begin."
            )

        # Show user question immediately in chat and display assistant loading indicator while generating insights
        if active_prompt and session is not None:
            with st.chat_message("user"):
                st.markdown(active_prompt)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing data and generating insights..."):
                    ui_session.execute_session_turn(session, active_prompt)
            st.rerun()


if __name__ == "__main__":
    render_analytics_page()
