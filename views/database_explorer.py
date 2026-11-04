"""Database Explorer page module.

Provides a dedicated, professional database inspection interface for the
Olist Brazilian E-Commerce DuckDB database. Features schema inspection,
read-only table previews (first 5 rows), and an interactive Entity-Relationship
Diagram (ERD) based on verified relational integrity and logical lookups.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Collection, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import streamlit as st

from data.prepare_data import DEFAULT_DB_PATH
from views.analytics import BASE_TABLE_DESCRIPTIONS, get_table_row_counts, load_database_schema
from tools.sql_tool import db_connection

# ==============================================================================
# Verified Key Constraints & Relational Metadata
# ==============================================================================

TABLE_PRIMARY_KEYS: Dict[str, List[str]] = {
    "customers": ["customer_id"],
    "orders": ["order_id"],
    "order_items": ["order_id", "order_item_id"],
    "order_payments": ["order_id", "payment_sequential"],
    "order_reviews": ["review_id", "order_id"],
    "products": ["product_id"],
    "sellers": ["seller_id"],
    "geolocation": [],
    "category_translation": ["product_category_name"],
}

TABLE_FOREIGN_KEYS: Dict[str, Dict[str, str]] = {
    "orders": {
        "customer_id": "customers.customer_id",
    },
    "order_items": {
        "order_id": "orders.order_id",
        "product_id": "products.product_id",
        "seller_id": "sellers.seller_id",
    },
    "order_payments": {
        "order_id": "orders.order_id",
    },
    "order_reviews": {
        "order_id": "orders.order_id",
    },
    "products": {
        "product_category_name": "category_translation.product_category_name (Lookup)",
    },
    "customers": {
        "customer_zip_code_prefix": "geolocation.geolocation_zip_code_prefix (Spatial)",
    },
    "sellers": {
        "seller_zip_code_prefix": "geolocation.geolocation_zip_code_prefix (Spatial)",
    },
}

VERIFIED_RELATIONSHIPS: List[Dict[str, str]] = [
    {
        "source_table": "orders",
        "source_column": "customer_id",
        "target_table": "customers",
        "target_column": "customer_id",
        "cardinality": "N:1",
        "type": "Referential Integrity (Verified)",
        "description": "Each commercial order is placed by exactly one registered customer.",
    },
    {
        "source_table": "order_items",
        "source_column": "order_id",
        "target_table": "orders",
        "target_column": "order_id",
        "cardinality": "N:1",
        "type": "Referential Integrity (Verified)",
        "description": "Line items fulfilled within a parent order.",
    },
    {
        "source_table": "order_items",
        "source_column": "product_id",
        "target_table": "products",
        "target_column": "product_id",
        "cardinality": "N:1",
        "type": "Referential Integrity (Verified)",
        "description": "Catalog product identified in an order item line.",
    },
    {
        "source_table": "order_items",
        "source_column": "seller_id",
        "target_table": "sellers",
        "target_column": "seller_id",
        "cardinality": "N:1",
        "type": "Referential Integrity (Verified)",
        "description": "Merchant seller fulfilling the specific line item.",
    },
    {
        "source_table": "order_payments",
        "source_column": "order_id",
        "target_table": "orders",
        "target_column": "order_id",
        "cardinality": "N:1",
        "type": "Referential Integrity (Verified)",
        "description": "Payment installment or split method records for an order.",
    },
    {
        "source_table": "order_reviews",
        "source_column": "order_id",
        "target_table": "orders",
        "target_column": "order_id",
        "cardinality": "N:1",
        "type": "Referential Integrity (Verified)",
        "description": "Customer post-fulfillment satisfaction review ratings.",
    },
    {
        "source_table": "products",
        "source_column": "product_category_name",
        "target_table": "category_translation",
        "target_column": "product_category_name",
        "cardinality": "N:1",
        "type": "Logical Lookup (Verified in Views)",
        "description": "English translation dictionary for Portuguese category names.",
    },
    {
        "source_table": "customers",
        "source_column": "customer_zip_code_prefix",
        "target_table": "geolocation",
        "target_column": "geolocation_zip_code_prefix",
        "cardinality": "N:N",
        "type": "Logical Spatial Reference",
        "description": "Geographic coordinates mapped to customer ZIP code prefix.",
    },
    {
        "source_table": "sellers",
        "source_column": "seller_zip_code_prefix",
        "target_table": "geolocation",
        "target_column": "geolocation_zip_code_prefix",
        "cardinality": "N:N",
        "type": "Logical Spatial Reference",
        "description": "Geographic coordinates mapped to merchant ZIP code prefix.",
    },
]


# ==============================================================================
# Deterministic, Safe Data Access Helpers
# ==============================================================================


def fetch_table_preview_rows(
    db_path: Union[str, Path],
    table_name: str,
    allowed_tables: Collection[str],
) -> pd.DataFrame:
    """Fetch first 5 rows for a validated table in read-only DuckDB mode.

    Guarantees:
    - Table identifier is strictly whitelisted against schema metadata.
    - SQL identifier uses safe double-quoting.
    - Connection is enforced read-only.
    - LIMIT 5 bounded execution.
    - Completely deterministic without LLM or Agent SQL Tool involvement.
    """
    if table_name not in allowed_tables:
        raise ValueError(f"Table identifier '{table_name}' is not in the authorized schema whitelist.")

    if not re.fullmatch(r"^[a-zA-Z0-9_]+$", table_name):
        raise ValueError(f"Invalid table identifier format: '{table_name}'.")

    with db_connection(db_path) as con:
        sql = f'SELECT * FROM "{table_name}" LIMIT 5;'
        return con.execute(sql).df()


def generate_erd_dot() -> str:
    """Construct Graphviz DOT representation for all 9 Olist base tables."""
    return """digraph Olist_ERD {
    graph [
        rankdir=LR,
        bgcolor="transparent",
        size="11.8,5.2!",
        ratio="compress",
        nodesep=0.28,
        ranksep=0.46,
        concentrate=false,
        overlap=false,
        splines=spline,
        fontname="Arial, Helvetica, sans-serif",
        fontsize=10.5
    ];
    node [
        shape=none,
        fontname="Arial, Helvetica, sans-serif",
        fontsize=10.5,
        fontcolor="#F3F4F6",
        margin=0
    ];
    edge [
        fontname="Arial, Helvetica, sans-serif",
        fontsize=9.5,
        color="#6366F1",
        arrowhead="vee",
        arrowsize=0.8,
        penwidth=1.2
    ];

    customers [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="4" bgcolor="#111827" color="#374151">
        <tr><td colspan="2" bgcolor="#1D4ED8" align="center"><font color="#EFF6FF"><b> customers </b></font></td></tr>
        <tr><td align="center" width="26"><font color="#818CF8"><b>PK</b></font></td><td align="left"><font color="#E2E8F0">customer_id&#160;&#160;</font></td></tr>
        <tr><td align="center" width="26"><font color="#38BDF8"><b>FK</b></font></td><td align="left"><font color="#CBD5E1">zip_code_prefix&#160;&#160;</font></td></tr>
    </table>>];

    orders [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="4" bgcolor="#111827" color="#374151">
        <tr><td colspan="2" bgcolor="#4338CA" align="center"><font color="#EEF2FF"><b> orders </b></font></td></tr>
        <tr><td align="center" width="26"><font color="#818CF8"><b>PK</b></font></td><td align="left"><font color="#E2E8F0">order_id&#160;&#160;</font></td></tr>
        <tr><td align="center" width="26"><font color="#38BDF8"><b>FK</b></font></td><td align="left"><font color="#CBD5E1">customer_id&#160;&#160;</font></td></tr>
        <tr><td align="center" width="26"></td><td align="left"><font color="#CBD5E1">order_status&#160;&#160;</font></td></tr>
    </table>>];

    order_items [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="4" bgcolor="#111827" color="#374151">
        <tr><td colspan="2" bgcolor="#5B21B6" align="center"><font color="#F5F3FF"><b> order_items </b></font></td></tr>
        <tr><td align="center" width="26"><font color="#818CF8"><b>PK</b></font></td><td align="left"><font color="#E2E8F0">order_id, item_id&#160;&#160;</font></td></tr>
        <tr><td align="center" width="26"><font color="#38BDF8"><b>FK</b></font></td><td align="left"><font color="#CBD5E1">product_id&#160;&#160;</font></td></tr>
        <tr><td align="center" width="26"><font color="#38BDF8"><b>FK</b></font></td><td align="left"><font color="#CBD5E1">seller_id&#160;&#160;</font></td></tr>
    </table>>];

    order_payments [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="4" bgcolor="#111827" color="#374151">
        <tr><td colspan="2" bgcolor="#0369A1" align="center"><font color="#F0F9FF"><b> order_payments </b></font></td></tr>
        <tr><td align="center" width="26"><font color="#818CF8"><b>PK</b></font></td><td align="left"><font color="#E2E8F0">order_id, seq&#160;&#160;</font></td></tr>
        <tr><td align="center" width="26"></td><td align="left"><font color="#CBD5E1">payment_type&#160;&#160;</font></td></tr>
    </table>>];

    order_reviews [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="4" bgcolor="#111827" color="#374151">
        <tr><td colspan="2" bgcolor="#0E7490" align="center"><font color="#ECFEFF"><b> order_reviews </b></font></td></tr>
        <tr><td align="center" width="26"><font color="#818CF8"><b>PK</b></font></td><td align="left"><font color="#E2E8F0">review_id&#160;&#160;</font></td></tr>
        <tr><td align="center" width="26"><font color="#38BDF8"><b>FK</b></font></td><td align="left"><font color="#CBD5E1">order_id&#160;&#160;</font></td></tr>
    </table>>];

    products [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="4" bgcolor="#111827" color="#374151">
        <tr><td colspan="2" bgcolor="#047857" align="center"><font color="#ECFDF5"><b> products </b></font></td></tr>
        <tr><td align="center" width="26"><font color="#818CF8"><b>PK</b></font></td><td align="left"><font color="#E2E8F0">product_id&#160;&#160;</font></td></tr>
        <tr><td align="center" width="26"><font color="#38BDF8"><b>FK</b></font></td><td align="left"><font color="#CBD5E1">category_name&#160;&#160;</font></td></tr>
    </table>>];

    category_translation [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="4" bgcolor="#111827" color="#374151">
        <tr><td colspan="2" bgcolor="#0F766E" align="center"><font color="#F0FDFA"><b> category_translation </b></font></td></tr>
        <tr><td align="center" width="26"><font color="#818CF8"><b>PK</b></font></td><td align="left"><font color="#E2E8F0">category_name&#160;&#160;</font></td></tr>
        <tr><td align="center" width="26"></td><td align="left"><font color="#CBD5E1">name_english&#160;&#160;</font></td></tr>
    </table>>];

    sellers [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="4" bgcolor="#111827" color="#374151">
        <tr><td colspan="2" bgcolor="#B45309" align="center"><font color="#FFFBEB"><b> sellers </b></font></td></tr>
        <tr><td align="center" width="26"><font color="#818CF8"><b>PK</b></font></td><td align="left"><font color="#E2E8F0">seller_id&#160;&#160;</font></td></tr>
        <tr><td align="center" width="26"><font color="#38BDF8"><b>FK</b></font></td><td align="left"><font color="#CBD5E1">zip_code_prefix&#160;&#160;</font></td></tr>
    </table>>];

    geolocation [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="4" bgcolor="#111827" color="#374151">
        <tr><td colspan="2" bgcolor="#334155" align="center"><font color="#F8FAFC"><b> geolocation </b></font></td></tr>
        <tr><td align="center" width="26"><font color="#818CF8"><b>PK</b></font></td><td align="left"><font color="#E2E8F0">zip_code_prefix&#160;&#160;</font></td></tr>
        <tr><td align="center" width="26"></td><td align="left"><font color="#CBD5E1">city, state&#160;&#160;</font></td></tr>
    </table>>];

    /* Relationships with verified cardinalities */
    orders -> customers [dir=back];
    order_items -> orders [dir=back];
    order_payments -> orders [dir=back];
    order_reviews -> orders [dir=back];
    order_items -> products [dir=back];
    order_items -> sellers [dir=back];
    products -> category_translation [dir=back];
    customers -> geolocation [style="dashed", color="#475569", penwidth=1.0, dir=none, arrowhead=none];
    sellers -> geolocation [style="dashed", color="#475569", penwidth=1.0, dir=none, arrowhead=none];
}"""


# ==============================================================================
# Modal Dialogs (st.dialog)
# ==============================================================================

@st.dialog("Table Preview", width="large")
def preview_table_dialog(
    table_name: str,
    allowed_tables: Collection[str],
    total_rows: Optional[int] = None,
) -> None:
    """Display interactive modal preview containing the first 5 rows of a table."""
    st.subheader(f"Table Preview: {table_name}")
    if total_rows is not None:
        st.caption(f"Showing first 5 rows of {total_rows:,} total rows")
    else:
        st.caption("Showing first 5 rows")

    try:
        df = fetch_table_preview_rows(DEFAULT_DB_PATH, table_name, allowed_tables=allowed_tables)
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception as exc:
        st.error(f"Failed to fetch table preview: {exc}")


@st.dialog("Entity-Relationship Diagram (ERD)", width="large")
def show_erd_dialog() -> None:
    """Display Entity-Relationship Diagram in an interactive modal dialog."""
    st.subheader("Entity-Relationship Diagram (ERD)")
    st.caption(
        "Interactive relational topology of the 9 Olist base tables in DuckDB. "
        "Solid arrows denote referential integrity (N:1); dashed lines denote spatial mappings."
    )
    dot_str = generate_erd_dot()
    st.graphviz_chart(dot_str, width="stretch")


def _handle_database_refresh() -> None:
    """Invalidate cached schema and row counts to force fresh database inspection."""
    load_database_schema.clear()
    get_table_row_counts.clear()
    try:
        import sys
        vde = sys.modules.get("views.database_explorer")
        if vde and hasattr(vde, "get_table_row_counts") and vde.get_table_row_counts is not get_table_row_counts:
            vde.get_table_row_counts.clear()
    except Exception:
        pass
    st.session_state["_db_schema_refreshed"] = True


# ==============================================================================
# Main Database Explorer Page View
# ==============================================================================

def render_database_explorer_page() -> None:
    """Render the dedicated Database Explorer interface."""
    # Header layout with metadata pills matching Analytics Workspace (Dataset, Volume, Period)
    st.header("Database Explorer")

    # Load dynamic schema introspection
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
        st.error(schema_error)
        return

    # Whitelist of authorized table and view identifiers
    all_table_names: Set[str] = {t["name"] for t in base_tables} | {v["name"] for v in analytical_views}

    # Fetch cached row counts for all tables and views
    row_counts = get_table_row_counts(str(DEFAULT_DB_PATH), tuple(sorted(all_table_names)))

    # Action toolbar: Metadata pills on the left, Refresh button positioned on the far right
    col_pills, col_refresh = st.columns([0.84, 0.16], vertical_alignment="center")
    with col_pills:
        st.markdown(
            """
            <div style="margin-bottom: 0.35rem;">
                <span class="metadata-pill"><strong>Dataset</strong>: Olist Brazilian E-Commerce</span>
                <span class="metadata-pill"><strong>Volume</strong>: ~99,000 orders</span>
                <span class="metadata-pill"><strong>Period</strong>: 2016–2018</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_refresh:
        if st.button(
            "Refresh",
            key="btn_refresh_schema",
            icon=":material/refresh:",
            width="stretch",
            on_click=_handle_database_refresh,
        ):
            if not st.session_state.get("_db_schema_refreshed"):
                _handle_database_refresh()
                st.rerun()

    if st.session_state.pop("_db_schema_refreshed", False):
        st.toast("Database schema and row counts refreshed.")

    tab_base, tab_views, tab_erd = st.tabs([
        f"Base Tables ({len(base_tables)})",
        f"Analytical Views ({len(analytical_views)})",
        "ER Diagram",
    ])

    # ==========================================================================
    # Tab 1: Base Tables
    # ==========================================================================
    with tab_base:
        if not base_tables:
            st.info("No base tables found in database.")
        for tbl in base_tables:
            tbl_name = tbl["name"]
            total_rows = row_counts.get(tbl_name)
            rows_str = f"{total_rows:,} rows" if total_rows is not None else "DuckDB table"

            col_exp, col_prev = st.columns([5.3, 1.1])
            with col_exp:
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

            with col_prev:
                if st.button("Preview", key=f"btn_prev_base_{tbl_name}", width="stretch"):
                    preview_table_dialog(tbl_name, allowed_tables=all_table_names, total_rows=total_rows)

    # ==========================================================================
    # Tab 2: Analytical Views
    # ==========================================================================
    with tab_views:
        if not analytical_views:
            st.info("No analytical views found in database.")
        for vw in analytical_views:
            vw_name = vw["name"]
            total_rows = row_counts.get(vw_name)
            rows_str = f"{total_rows:,} rows" if total_rows is not None else "DuckDB view"

            col_exp, col_prev = st.columns([5.3, 1.1])
            with col_exp:
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

            with col_prev:
                if st.button("Preview", key=f"btn_prev_view_{vw_name}", width="stretch"):
                    preview_table_dialog(vw_name, allowed_tables=all_table_names, total_rows=total_rows)

    # ==========================================================================
    # Tab 3: ER Diagram
    # ==========================================================================
    with tab_erd:
        st.subheader("Entity-Relationship Diagram (ERD)")
        st.caption(
            '<span class="erd-tab-caption">Interactive relational topology of the 9 Olist base tables in DuckDB. '
            'Solid lines denote verified referential integrity; dashed lines denote spatial mappings.</span>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <style>
            /* Tab 3 ERD subheader & caption font reduction */
            div[data-testid="stTabPanel"][id="2"] h3,
            div[data-testid="stTabPanel"]:nth-of-type(3) h3,
            div[data-testid="stTabPanel"][id="2"] [data-testid="stHeadingWithActionElements"] h3,
            div[data-testid="stTabPanel"]:nth-of-type(3) [data-testid="stHeadingWithActionElements"] h3 {
                font-size: 1.05rem !important;
                font-weight: 600 !important;
                margin-top: 0.15rem !important;
                margin-bottom: 0.1rem !important;
                color: var(--text-primary, #F9FAFB) !important;
            }

            div[data-testid="stTabPanel"][id="2"] [data-testid="stCaptionContainer"] p,
            div[data-testid="stTabPanel"]:nth-of-type(3) [data-testid="stCaptionContainer"] p,
            .erd-tab-caption {
                font-size: 0.85rem !important;
                color: #94A3B8 !important;
                line-height: 1.45 !important;
                margin-bottom: 0.4rem !important;
            }

            /* Graphviz chart container: professional blueprint canvas */
            [data-testid="stGraphVizChart"], [data-testid="stGraphvizChart"], .stGraphVizChart {
                background: radial-gradient(circle at 50% 35%, rgba(30, 41, 59, 0.45) 0%, rgba(15, 23, 42, 0.8) 100%), #0D111A !important;
                border: 1px solid rgba(55, 65, 81, 0.85) !important;
                border-radius: 10px !important;
                box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.4), inset 0 1px 0 0 rgba(255, 255, 255, 0.05) !important;
                max-width: 1080px !important;
                max-height: 540px !important;
                margin: 0.35rem auto 0.85rem auto !important;
            }
            [data-testid="stGraphVizChart"] svg, [data-testid="stGraphvizChart"] svg, .stGraphVizChart svg {
                max-height: 500px !important;
                max-width: 100% !important;
                width: auto !important;
                height: auto !important;
                object-fit: contain !important;
                display: block !important;
                margin: 0 auto !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        dot_str = generate_erd_dot()
        st.graphviz_chart(dot_str, width="stretch")

        with st.expander("Verified Relationships Reference Table", expanded=True):
            rel_records = []
            for r in VERIFIED_RELATIONSHIPS:
                rel_records.append({
                    "Relationship": f"{r['source_table']} → {r['target_table']}",
                    "Key Columns": f"{r['source_column']} = {r['target_column']}",
                    "Cardinality": r["cardinality"],
                    "Constraint Type": r["type"],
                    "Description": r["description"],
                })
            st.dataframe(rel_records, width="stretch", hide_index=True)


if __name__ == "__main__":
    render_database_explorer_page()
