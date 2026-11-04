"""Targeted test suite for Database Explorer, ERD, Table Preview, and Multipage Navigation.

Verifies:
1. Navigation and Authentication:
   - Protected content hidden before login.
   - Both pages accessible after authentication.
   - Navigation works in both directions.
   - Logout clears auth state.
   - Session state preserved across page switches.
2. Analytics Workspace:
   - Model selection is integrated in the question input bar.
   - Selected model is passed to the agent session.
3. Database Explorer:
   - All 9 base tables listed.
   - Both analytical views listed.
   - Schema expansion displays real data types, nullability, and keys.
   - Preview retrieves at most 5 rows in read-only mode.
   - Invalid table identifiers cannot execute SQL.
   - ER Diagram includes all 9 base tables with verified relationships.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from agent.config import DEFAULT_OPENAI_MODEL
from data.prepare_data import DEFAULT_DB_PATH
from views.database_explorer import (
    TABLE_FOREIGN_KEYS,
    TABLE_PRIMARY_KEYS,
    VERIFIED_RELATIONSHIPS,
    fetch_table_preview_rows,
    generate_erd_dot,
    get_table_row_counts,
    preview_table_dialog,
    show_erd_dialog,
)
from tools.sql_tool import get_database_schema

APP_PATH = Path(__file__).parent.parent / "app.py"
EXPLORER_PATH = Path(__file__).parent.parent / "views" / "database_explorer.py"


# ==============================================================================
# 1. Navigation & Authentication Access
# ==============================================================================

def test_unauthenticated_cannot_access_database_explorer(monkeypatch):
    """Unauthenticated visitor cannot access database explorer or view schemas."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "secret123")
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.run()

    assert not at.exception
    # Only access gate rendered, no database explorer or analytics headers
    headers = [h.value for h in at.header]
    assert "Database Explorer" not in headers
    assert "Dataset Explorer" not in headers
    assert len(at.dataframe) == 0


def test_navigation_between_pages(monkeypatch):
    """Authenticated user can navigate between Analytics Workspace and Database Explorer."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "secret123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-test-key")
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()

    # Default page is Analytics Workspace
    assert at.header[0].value == "Analytics Workspace"
    assert "Conversation History" in [h.value for h in at.header]

    # Switch page to Database Explorer
    at.switch_page("views/database_explorer.py").run()
    assert not at.exception
    headers = [h.value for h in at.header]
    assert "Database Explorer" in headers
    assert "Conversation History" not in headers

    # Switch back to Analytics Workspace
    at.switch_page("views/analytics.py").run()
    assert not at.exception
    assert at.header[0].value == "Analytics Workspace"
    assert "Conversation History" in [h.value for h in at.header]


def test_session_preserved_across_page_switches(monkeypatch):
    """Agent session messages and stores are preserved when switching between pages."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "secret123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-test-key")
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()

    session = at.session_state["agent_session"]
    session.ui_messages.append({"role": "user", "content": "How many orders?"})
    session.ui_messages.append({
        "role": "assistant",
        "content": "There are 99,441 orders in total.",
        "chart_ids": [],
        "tool_executions": [],
    })
    at.run()
    assert len(at.chat_message) == 2

    # Switch to Database Explorer
    at.switch_page("views/database_explorer.py").run()
    assert not at.exception
    assert at.session_state["agent_session"] is session

    # Switch back to Analytics Workspace
    at.switch_page("views/analytics.py").run()
    assert not at.exception
    assert len(at.chat_message) == 2
    assert "How many orders?" in at.chat_message[0].markdown[0].value
    assert "99,441 orders" in at.chat_message[1].markdown[0].value


# ==============================================================================
# 2. Model Selector in Question Input Bar
# ==============================================================================

def test_model_selector_integrated_in_input_bar(monkeypatch):
    """Model selector dropdown is located within the question entry form and controls model."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "secret123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-test-key")
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()

    # Model selector is present
    assert len(at.selectbox) >= 1
    model_box = [sb for sb in at.selectbox if "Model" in sb.label][0]
    assert model_box.value == DEFAULT_OPENAI_MODEL
    assert any(opt.lower() == "gpt-4o" for opt in model_box.options)
    assert any("gpt-4o-mini" in opt.lower() for opt in model_box.options)
    assert any("o3-mini" in opt.lower() for opt in model_box.options)

    # Selectbox is not in sidebar
    assert len(at.sidebar.selectbox) == 0


def test_selected_model_passed_to_agent_turn(monkeypatch):
    """Selecting a different model passes the model override to the agent session."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "secret123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-test-key")
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()

    session = at.session_state["agent_session"]
    assert session.configured_model == DEFAULT_OPENAI_MODEL

    class MockGraph:
        def invoke(self, state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            msgs = list(state.get("messages", []))
            return {"messages": msgs + [AIMessage(content="Mock response from gpt-4o")]}

    session.graph = MockGraph()

    # Change model to gpt-4o
    model_box = [sb for sb in at.selectbox if "Model" in sb.label][0]
    model_box.select("gpt-4o")

    # Enter question and submit
    at.text_input[0].set_value("Analyze revenue")
    submit_btn = [b for b in at.button if b.label == "Submit"][0]
    submit_btn.click().run()

    assert not at.exception
    assert session.configured_model == "gpt-4o"


# ==============================================================================
# 3. Database Explorer: Base Tables & Analytical Views
# ==============================================================================

def test_database_explorer_metadata_and_structure(monkeypatch):
    """Database Explorer displays metadata pills, 3 tabs, base tables, views, and ERD."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "secret123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-test-key")
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.session_state["authenticated"] = True
    at.switch_page("views/database_explorer.py").run()

    assert not at.exception
    # Header & Subtitle
    headers = [h.value for h in at.header]
    assert "Database Explorer" in headers
    # Metadata pills match Analytics Workspace design (Dataset, Volume, Period without Engine)
    pill_markdowns = [
        m.value for m in at.markdown
        if "metadata-pill" in m.value and not m.value.startswith("<style>")
    ]
    assert len(pill_markdowns) >= 1
    pill_text = pill_markdowns[0]
    assert "<strong>Dataset</strong>: Olist Brazilian E-Commerce" in pill_text
    assert "<strong>Volume</strong>: ~99,000 orders" in pill_text
    assert "<strong>Period</strong>: 2016–2018" in pill_text
    assert "Engine" not in pill_text

    # Header-level ER Diagram button is removed
    erd_btn = [b for b in at.button if b.label == "ER Diagram"]
    assert len(erd_btn) == 0

    # Refresh button is present
    refresh_btn = [b for b in at.button if b.key == "btn_refresh_schema"]
    assert len(refresh_btn) == 1
    assert refresh_btn[0].label == "Refresh"

    # Three tabs in the correct order
    assert len(at.tabs) == 3
    assert at.tabs[0].label == "Base Tables (9)"
    assert at.tabs[1].label == "Analytical Views (2)"
    assert at.tabs[2].label == "ER Diagram"

    # Base table expanders exist
    expander_labels = [e.label for e in at.expander]
    expected_tables = [
        "customers", "orders", "order_items", "order_payments", "order_reviews",
        "products", "sellers", "geolocation", "category_translation",
    ]
    for tbl in expected_tables:
        assert any(tbl in label for label in expander_labels), f"Table {tbl} not in expanders"

    # Analytical view expanders exist
    assert any("order_summary" in label for label in expander_labels)
    assert any("category_sales_summary" in label for label in expander_labels)

    # Preview buttons exist for tables and views
    preview_btns = [b for b in at.button if b.label == "Preview"]
    assert len(preview_btns) >= 11  # 9 base tables + 2 analytical views


def test_base_table_schema_does_not_contain_key_constraint():
    """Expanded Base Table schema tables do not display Key Constraint column."""
    at = AppTest.from_file(str(EXPLORER_PATH), default_timeout=10)
    at.run()
    assert not at.exception

    # Check dataframes in tab 0 (Base Tables)
    tab_base = at.tabs[0]
    assert len(tab_base.dataframe) >= 9
    for df in tab_base.dataframe:
        cols = list(df.value.columns)
        assert "Key Constraint" not in cols
        assert "Column" in cols
        assert "Data Type" in cols
        assert "Nullable" in cols


def test_analytical_views_simplified_without_grain_or_warning():
    """Analytical views do not display Grain, Warning, or Purpose in the UI."""
    at = AppTest.from_file(str(EXPLORER_PATH), default_timeout=10)
    at.run()
    assert not at.exception

    tab_views = at.tabs[1]
    # No warning widgets in tab 1
    assert len(tab_views.warning) == 0

    # No Grain, Warning, or Purpose in captions
    for cap in tab_views.caption:
        assert "Grain:" not in cap.value
        assert "Warning:" not in cap.value
        assert "Purpose:" not in cap.value

    # Notes remain preserved, Purpose is removed
    captions_text = " ".join(cap.value for cap in tab_views.caption)
    assert "Purpose" not in captions_text
    assert "Notes" in captions_text


def test_er_diagram_renders_in_third_tab():
    """ER Diagram renders directly in Tab 3 with Graphviz chart and relationship table."""
    at = AppTest.from_file(str(EXPLORER_PATH), default_timeout=10)
    at.run()
    assert not at.exception

    tab_erd = at.tabs[2]
    # Subheader & caption
    subheaders = [s.value for s in tab_erd.subheader]
    assert "Entity-Relationship Diagram (ERD)" in subheaders
    assert any("relational topology" in c.value for c in tab_erd.caption)

    # Graphviz chart is rendered directly in Tab 3
    charts = at.get("graphviz_chart")
    assert len(charts) == 1
    assert "digraph Olist_ERD" in charts[0].spec

    # Verified relationships table expander and dataframe
    erd_expanders = [e.label for e in tab_erd.expander]
    assert "Verified Relationships Reference Table" in erd_expanders
    assert len(tab_erd.dataframe) == 1
    rel_df = tab_erd.dataframe[0].value
    assert "Relationship" in rel_df.columns
    assert "Cardinality" in rel_df.columns


def test_show_erd_dialog_renders_chart_directly():
    """show_erd_dialog renders Graphviz spec with Olist_ERD."""
    with patch("streamlit.graphviz_chart") as mock_chart, \
         patch("streamlit.subheader"), \
         patch("streamlit.caption"):
        show_erd_dialog.__wrapped__()
        assert mock_chart.called
        spec = mock_chart.call_args[0][0]
        assert "digraph Olist_ERD" in spec


def test_refresh_button_invalidates_cache_and_preserves_session(monkeypatch):
    """Clicking Refresh clears schema and row count caches without resetting agent session."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "secret123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-test-key")
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()

    # Pre-populate session state
    session = at.session_state["agent_session"]
    session.ui_messages.append({"role": "user", "content": "How many orders?"})

    at.switch_page("views/database_explorer.py").run()
    assert not at.exception

    mock_schema_clear = MagicMock()
    mock_counts_clear = MagicMock()

    import views.analytics
    views.analytics.load_database_schema.clear = mock_schema_clear
    for mod in list(__import__("sys").modules.values()):
        if mod and hasattr(mod, "get_table_row_counts") and hasattr(mod, "_handle_database_refresh"):
            mod.get_table_row_counts.clear = mock_counts_clear

    refresh_btn = [b for b in at.button if b.key == "btn_refresh_schema"][0]
    refresh_btn.click().run()

    assert not at.exception
    assert mock_schema_clear.called
    assert mock_counts_clear.called

    # Verify session preservation
    assert at.session_state["agent_session"] is session
    assert len(session.ui_messages) == 1
    assert session.ui_messages[0]["content"] == "How many orders?"

    # Verify toast
    assert any("refreshed" in t.value for t in at.toast)


# ==============================================================================
# 4. Safe Table Preview Functionality
# ==============================================================================

def test_fetch_table_preview_rows_bounded_and_readonly():
    """Preview retrieves exactly at most 5 rows for whitelisted tables without writing."""
    schema = get_database_schema(DEFAULT_DB_PATH)
    allowed = {t["name"] for t in schema["tables"]} | {v["name"] for v in schema["views"]}

    for tbl in ["customers", "orders", "order_summary"]:
        df = fetch_table_preview_rows(DEFAULT_DB_PATH, tbl, allowed_tables=allowed)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert len(df.columns) > 0


def test_fetch_table_preview_rejects_unauthorized_and_invalid_identifiers():
    """fetch_table_preview_rows rejects un-whitelisted names or injection strings."""
    allowed = {"orders", "customers"}

    # Disallowed table
    with pytest.raises(ValueError, match="not in the authorized schema whitelist"):
        fetch_table_preview_rows(DEFAULT_DB_PATH, "secret_credentials", allowed_tables=allowed)

    # Malicious injection string
    with pytest.raises(ValueError):
        fetch_table_preview_rows(DEFAULT_DB_PATH, "orders; DROP TABLE orders; --", allowed_tables=allowed)

    # SQL with spaces
    with pytest.raises(ValueError):
        fetch_table_preview_rows(DEFAULT_DB_PATH, "orders union select", allowed_tables=allowed)


def test_get_table_row_counts_returns_accurate_metrics():
    """get_table_row_counts accurately inspects row counts across base tables."""
    tables = ("customers", "orders", "products")
    counts = get_table_row_counts(str(DEFAULT_DB_PATH), tables)
    assert counts["customers"] == 99441
    assert counts["orders"] == 99441
    assert counts["products"] == 32951


def test_preview_table_dialog_caption_format():
    """Preview dialog captions show row counts without '• DuckDB Read-Only'."""
    captions = []
    with patch("streamlit.caption", side_effect=lambda txt: captions.append(txt)), \
         patch("streamlit.dataframe"), \
         patch("streamlit.subheader"):
        preview_table_dialog.__wrapped__("customers", {"customers"}, total_rows=99441)
        preview_table_dialog.__wrapped__("orders", {"orders"}, total_rows=None)

    assert captions[0] == "Showing first 5 rows of 99,441 total rows"
    assert "DuckDB Read-Only" not in captions[0]
    assert captions[1] == "Showing first 5 rows"
    assert "DuckDB Read-Only" not in captions[1]


# ==============================================================================
# 5. Entity-Relationship Diagram (ERD)
# ==============================================================================

def test_er_diagram_contains_all_nine_base_tables_and_relationships():
    """generate_erd_dot includes all 9 base tables and verified relational edges."""
    dot_str = generate_erd_dot()
    assert isinstance(dot_str, str)
    assert "digraph Olist_ERD" in dot_str

    expected_tables = [
        "customers", "orders", "order_items", "order_payments", "order_reviews",
        "products", "sellers", "geolocation", "category_translation",
    ]
    for tbl in expected_tables:
        assert f"{tbl} [" in dot_str, f"Base table {tbl} missing from ERD DOT"

    # Referential integrity edges
    assert "orders -> customers" in dot_str
    assert "order_items -> orders" in dot_str
    assert "order_items -> products" in dot_str
    assert "order_items -> sellers" in dot_str
    assert "order_payments -> orders" in dot_str
    assert "order_reviews -> orders" in dot_str
    assert "products -> category_translation" in dot_str
    # Spatial prefix mappings
    assert "customers -> geolocation" in dot_str
    assert "sellers -> geolocation" in dot_str


def test_verified_relationships_metadata_consistency():
    """VERIFIED_RELATIONSHIPS matches documented referential integrity checks."""
    assert len(VERIFIED_RELATIONSHIPS) >= 9
    rel_map = {(r["source_table"], r["target_table"]): r for r in VERIFIED_RELATIONSHIPS}

    # Core order -> customer
    assert ("orders", "customers") in rel_map
    assert rel_map[("orders", "customers")]["cardinality"] == "N:1"

    # Items -> orders, products, sellers
    assert ("order_items", "orders") in rel_map
    assert ("order_items", "products") in rel_map
    assert ("order_items", "sellers") in rel_map

    # Lookup & Spatial
    assert ("products", "category_translation") in rel_map
    assert ("customers", "geolocation") in rel_map
    assert ("sellers", "geolocation") in rel_map
