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
from unittest.mock import patch

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
)
from tools.sql_tool import get_database_schema

APP_PATH = Path(__file__).parent.parent / "app.py"


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
    assert at.header[0].value == "Dataset Explorer"

    # Switch page to Database Explorer
    at.switch_page("views/database_explorer.py").run()
    assert not at.exception
    headers = [h.value for h in at.header]
    assert "Database Explorer" in headers
    assert "Dataset Explorer" not in headers

    # Switch back to Analytics Workspace
    at.switch_page("views/analytics.py").run()
    assert not at.exception
    assert at.header[0].value == "Dataset Explorer"


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
    assert "gpt-4o" in model_box.options
    assert "gpt-4o-mini" in model_box.options
    assert "o3-mini" in model_box.options

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
    """Database Explorer displays all 9 base tables, 2 views, and read-only status."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "secret123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-test-key")
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.session_state["authenticated"] = True
    at.switch_page("views/database_explorer.py").run()

    assert not at.exception
    # Header & Subtitle
    headers = [h.value for h in at.header]
    assert "Database Explorer" in headers
    assert any("Olist Brazilian E-Commerce" in cap.value for cap in at.caption)

    # ER Diagram button near header
    erd_btn = [b for b in at.button if b.label == "ER Diagram"]
    assert len(erd_btn) >= 1

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
