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
from views.analytics import BASE_TABLE_DESCRIPTIONS, VIEW_DESCRIPTIONS, load_database_schema
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
        nodesep=0.7,
        ranksep=1.1,
        fontname="Helvetica, Arial, sans-serif",
        fontsize=10
    ];
    node [
        shape=none,
        fontname="Helvetica, Arial, sans-serif",
        fontsize=9
    ];
    edge [
        fontname="Helvetica, Arial, sans-serif",
        fontsize=8,
        color="#6366f1",
        fontcolor="#4338ca",
        penwidth=1.5
    ];

    customers [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="3" bgcolor="#ffffff" style="rounded">
        <tr><td bgcolor="#3b82f6" align="center"><font color="#ffffff"><b>customers</b></font></td></tr>
        <tr><td align="left">🔑 <b>customer_id</b> : VARCHAR (PK)</td></tr>
        <tr><td align="left">customer_unique_id : VARCHAR</td></tr>
        <tr><td align="left">customer_zip_code_prefix : VARCHAR</td></tr>
        <tr><td align="left">customer_city : VARCHAR</td></tr>
        <tr><td align="left">customer_state : VARCHAR</td></tr>
    </table>>];

    orders [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="3" bgcolor="#ffffff" style="rounded">
        <tr><td bgcolor="#4f46e5" align="center"><font color="#ffffff"><b>orders</b></font></td></tr>
        <tr><td align="left">🔑 <b>order_id</b> : VARCHAR (PK)</td></tr>
        <tr><td align="left">🔗 customer_id : VARCHAR (FK)</td></tr>
        <tr><td align="left">order_status : VARCHAR</td></tr>
        <tr><td align="left">order_purchase_timestamp : TIMESTAMP</td></tr>
        <tr><td align="left">order_delivered_customer_date : TIMESTAMP</td></tr>
    </table>>];

    order_items [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="3" bgcolor="#ffffff" style="rounded">
        <tr><td bgcolor="#6366f1" align="center"><font color="#ffffff"><b>order_items</b></font></td></tr>
        <tr><td align="left">🔑 <b>order_id</b> : VARCHAR (PK, FK)</td></tr>
        <tr><td align="left">🔑 <b>order_item_id</b> : INTEGER (PK)</td></tr>
        <tr><td align="left">🔗 product_id : VARCHAR (FK)</td></tr>
        <tr><td align="left">🔗 seller_id : VARCHAR (FK)</td></tr>
        <tr><td align="left">price : DECIMAL(18,2)</td></tr>
        <tr><td align="left">freight_value : DECIMAL(18,2)</td></tr>
    </table>>];

    order_payments [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="3" bgcolor="#ffffff" style="rounded">
        <tr><td bgcolor="#0284c7" align="center"><font color="#ffffff"><b>order_payments</b></font></td></tr>
        <tr><td align="left">🔑 <b>order_id</b> : VARCHAR (PK, FK)</td></tr>
        <tr><td align="left">🔑 <b>payment_sequential</b> : INTEGER (PK)</td></tr>
        <tr><td align="left">payment_type : VARCHAR</td></tr>
        <tr><td align="left">payment_installments : INTEGER</td></tr>
        <tr><td align="left">payment_value : DECIMAL(18,2)</td></tr>
    </table>>];

    order_reviews [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="3" bgcolor="#ffffff" style="rounded">
        <tr><td bgcolor="#0891b2" align="center"><font color="#ffffff"><b>order_reviews</b></font></td></tr>
        <tr><td align="left">🔑 <b>review_id</b> : VARCHAR (PK)</td></tr>
        <tr><td align="left">🔑 <b>order_id</b> : VARCHAR (PK, FK)</td></tr>
        <tr><td align="left">review_score : INTEGER</td></tr>
        <tr><td align="left">review_creation_date : TIMESTAMP</td></tr>
    </table>>];

    products [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="3" bgcolor="#ffffff" style="rounded">
        <tr><td bgcolor="#059669" align="center"><font color="#ffffff"><b>products</b></font></td></tr>
        <tr><td align="left">🔑 <b>product_id</b> : VARCHAR (PK)</td></tr>
        <tr><td align="left">🔗 product_category_name : VARCHAR (FK)</td></tr>
        <tr><td align="left">product_weight_g : INTEGER</td></tr>
        <tr><td align="left">product_length_cm : INTEGER</td></tr>
    </table>>];

    category_translation [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="3" bgcolor="#ffffff" style="rounded">
        <tr><td bgcolor="#10b981" align="center"><font color="#ffffff"><b>category_translation</b></font></td></tr>
        <tr><td align="left">🔑 <b>product_category_name</b> : VARCHAR (PK)</td></tr>
        <tr><td align="left">product_category_name_english : VARCHAR</td></tr>
    </table>>];

    sellers [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="3" bgcolor="#ffffff" style="rounded">
        <tr><td bgcolor="#d97706" align="center"><font color="#ffffff"><b>sellers</b></font></td></tr>
        <tr><td align="left">🔑 <b>seller_id</b> : VARCHAR (PK)</td></tr>
        <tr><td align="left">seller_zip_code_prefix : VARCHAR</td></tr>
        <tr><td align="left">seller_city : VARCHAR</td></tr>
        <tr><td align="left">seller_state : VARCHAR</td></tr>
    </table>>];

    geolocation [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="3" bgcolor="#ffffff" style="rounded">
        <tr><td bgcolor="#64748b" align="center"><font color="#ffffff"><b>geolocation</b></font></td></tr>
        <tr><td align="left"><b>geolocation_zip_code_prefix</b> : VARCHAR</td></tr>
        <tr><td align="left">geolocation_lat : DOUBLE</td></tr>
        <tr><td align="left">geolocation_lng : DOUBLE</td></tr>
        <tr><td align="left">geolocation_city : VARCHAR</td></tr>
        <tr><td align="left">geolocation_state : VARCHAR</td></tr>
    </table>>];

    /* Relationships with verified cardinalities */
    orders -> customers [label="N:1\\ncustomer_id", dir=back];
    order_items -> orders [label="N:1\\norder_id", dir=back];
    order_payments -> orders [label="N:1\\norder_id", dir=back];
    order_reviews -> orders [label="N:1\\norder_id", dir=back];
    order_items -> products [label="N:1\\nproduct_id", dir=back];
    order_items -> sellers [label="N:1\\nseller_id", dir=back];
    products -> category_translation [label="N:1\\ncategory_name", dir=back];
    customers -> geolocation [label="N:N\\nzip_prefix", style="dashed", color="#94a3b8", fontcolor="#64748b", dir=none];
    sellers -> geolocation [label="N:N\\nzip_prefix", style="dashed", color="#94a3b8", fontcolor="#64748b", dir=none];
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
        st.caption(f"Showing first 5 rows of **{total_rows:,}** total rows • DuckDB Read-Only")
    else:
        st.caption("Showing first 5 rows • DuckDB Read-Only")

    try:
        df = fetch_table_preview_rows(DEFAULT_DB_PATH, table_name, allowed_tables=allowed_tables)
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception as exc:
        st.error(f"Failed to fetch table preview: {exc}")


@st.dialog("Entity-Relationship Diagram", width="large")
def show_erd_dialog() -> None:
    """Display Entity-Relationship Diagram and verified relationships table."""
    st.subheader("Entity-Relationship Diagram (ERD)")
    st.caption(
        "Interactive relational topology of the 9 Olist base tables in DuckDB. "
        "Solid lines denote verified referential integrity; dashed lines denote spatial mappings."
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


# ==============================================================================
# Main Database Explorer Page View
# ==============================================================================

def render_database_explorer_page() -> None:
    """Render the dedicated Database Explorer interface."""
    # Header layout with prominent but compact ER Diagram button
    col_header, col_erd = st.columns([5, 1.4])
    with col_header:
        st.header("Database Explorer")
        st.caption("Olist Brazilian E-Commerce | DuckDB")
    with col_erd:
        st.write("")  # Visual alignment spacer
        if st.button("ER Diagram", key="btn_open_erd_modal", type="primary", width="stretch"):
            show_erd_dialog()

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

    # Database overview metadata pills
    st.markdown(
        f"""
        <div style="margin-bottom: 0.85rem;">
            <span class="metadata-pill"><strong>Database Engine</strong>: DuckDB (Embedded)</span>
            <span class="metadata-pill"><strong>Base Tables</strong>: {len(base_tables)}</span>
            <span class="metadata-pill"><strong>Analytical Views</strong>: {len(analytical_views)}</span>
            <span class="metadata-pill"><strong>Access Mode</strong>: Read-Only (Enforced)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_base, tab_views = st.tabs(
        [f"Base Tables ({len(base_tables)})", f"Analytical Views ({len(analytical_views)})"]
    )

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
                        pks = TABLE_PRIMARY_KEYS.get(tbl_name, [])
                        fks = TABLE_FOREIGN_KEYS.get(tbl_name, {})

                        for c in cols:
                            c_name = c["name"]
                            nullable_val = c.get("nullable")
                            nullable_str = "Yes" if nullable_val is True else ("No" if nullable_val is False else "Unknown")

                            is_pk = c_name in pks
                            fk_target = fks.get(c_name)

                            if is_pk and fk_target:
                                key_str = f"PK, FK → {fk_target}"
                            elif is_pk:
                                key_str = "PK"
                            elif fk_target:
                                key_str = f"FK → {fk_target}"
                            else:
                                key_str = "—"

                            rec = {
                                "Column": c_name,
                                "Data Type": c["data_type"],
                                "Nullable": nullable_str,
                                "Key Constraint": key_str,
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
                with st.expander(f"{vw_name} (Analytical View) • {rows_str}", expanded=False):
                    purpose = VIEW_DESCRIPTIONS.get(vw_name) or vw.get("purpose")
                    if purpose:
                        st.caption(f"**Purpose**: {purpose}")
                    if vw.get("grain"):
                        st.caption(f"**Grain**: {vw['grain']}")
                    if vw.get("warning"):
                        st.warning(f"**Warning**: {vw['warning']}")
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


if __name__ == "__main__":
    render_database_explorer_page()
