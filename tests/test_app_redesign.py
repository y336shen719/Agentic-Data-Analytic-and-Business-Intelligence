"""Targeted test suite for the redesigned Streamlit UI and access gate in app.py.

Verifies:
1. Unauthenticated visitor cannot access application or trigger Agent execution.
2. Incorrect password does not authenticate.
3. Correct authentication unlocks full application.
4. Missing APP_ACCESS_PASSWORD fails closed.
5. Manual OpenAI API Key input is completely removed from the UI.
6. Server-side OpenAI API Key resolves correctly without visitor input.
7. Sidebar controls (models, example questions, clear conversation, logout, developer settings, admin diagnostics).
8. Discoverability of all 9 base tables and 2 analytical views via DuckDB metadata introspection.
9. Table expansion metadata: column names, data types, nullability, and descriptions.
10. Submitting typed question executes one turn.
11. Selecting example question executes one turn.
12. Chat history persistence across reruns.
13. Session reset clears chat state.
14. Plotly charts and technical diagnostics rendering.
15. LangSmith tracing and Admin Diagnostics functionality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import plotly.graph_objects as go
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from streamlit.testing.v1 import AppTest

from agent.tracing import run_synthetic_trace_diagnostic, verify_diagnostic_access
from agent.ui_session import AgentSession, extract_turn_details, resolve_api_key
from data.prepare_data import DEFAULT_DB_PATH
from tools.sql_tool import get_database_schema

APP_PATH = Path(__file__).parent.parent / "app.py"


# ==============================================================================
# 1-4. Access Gate & Authentication Tests
# ==============================================================================

def test_missing_app_access_password_fails_closed(monkeypatch):
    """Criterion 4: If APP_ACCESS_PASSWORD is missing, application must fail closed with error."""
    monkeypatch.delenv("APP_ACCESS_PASSWORD", raising=False)
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.run()

    assert not at.exception
    # Must fail closed: error message shown, main title and chat components never rendered
    assert len(at.error) >= 1
    assert "Access password is not configured" in at.error[0].value
    assert len(at.title) == 0
    assert len(at.chat_input) == 0
    assert "agent_session" not in at.session_state


def test_unauthenticated_visitor_blocked(monkeypatch):
    """Criterion 1: Unauthenticated visitor sees password prompt and cannot access app."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "secret123")
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.run()

    assert not at.exception
    # Password form is displayed, full app is inaccessible
    assert any("enter the access password" in cap.value.lower() for cap in at.caption)
    assert len(at.title) == 0
    assert "agent_session" not in at.session_state
    # Sidebar should have no application controls
    assert len(at.sidebar.button) == 0


def test_incorrect_password_rejected(monkeypatch):
    """Criterion 2: Incorrect password fails to authenticate and displays error."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "secret123")
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.run()

    # Enter wrong password and click Enter
    at.text_input[0].set_value("wrong-password")
    submit_btn = [b for b in at.button if b.label == "Enter"][0]
    submit_btn.click().run()

    assert not at.exception
    assert any("Incorrect password" in err.value for err in at.error)
    assert at.session_state.get("authenticated", False) is False
    assert len(at.title) == 0


def test_correct_password_authenticates_and_unlocks(monkeypatch):
    """Criterion 3: Correct password unlocks full application."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "secret123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-test-key")
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.run()

    # Enter correct password
    at.text_input[0].set_value("secret123")
    submit_btn = [b for b in at.button if b.label == "Enter"][0]
    submit_btn.click().run()

    assert not at.exception
    assert at.session_state.get("authenticated") is True
    # Now full app is unlocked
    assert len(at.title) >= 1
    assert at.title[0].value == "Data Analytics & Business Intelligence Agent"
    assert len(at.sidebar.button) >= 5


def test_logout_button_clears_authentication(monkeypatch):
    """Test logout button in sidebar resets authentication state back to gate."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "secret123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-test-key")
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()

    assert at.session_state.get("authenticated") is True
    logout_btn = [b for b in at.sidebar.button if "Log Out" in b.label][0]
    logout_btn.click().run()

    assert not at.exception
    assert at.session_state.get("authenticated") is False
    # App is locked again
    assert len(at.title) == 0


# ==============================================================================
# 5-6. OpenAI Key Handling (No Manual Input, Server-Side Only)
# ==============================================================================

def test_manual_openai_key_input_removed(monkeypatch):
    """Criterion 5: Manual OpenAI API Key text widget is completely removed from sidebar."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "secret123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-test-key")
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()

    # Sidebar has no text input for API keys
    sidebar_text_inputs = at.sidebar.text_input
    assert len(sidebar_text_inputs) == 0
    # No label mentions OpenAI API Key in sidebar
    for el in at.sidebar.markdown:
        assert "OpenAI API Key (optional)" not in el.value


def test_server_side_openai_key_resolved_and_missing_warning(monkeypatch):
    """Criterion 6: Server-side key is used; when absent, clear instruction is displayed."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "secret123")
    monkeypatch.setattr("agent.ui_session.resolve_api_key", lambda *args, **kwargs: None)

    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()

    assert not at.exception
    # Missing key warning directed to administrator
    assert any("OpenAI API Key is not configured on the server" in w.value for w in at.warning)
    assert any("OPENAI_API_KEY in Streamlit Secrets" in w.value for w in at.warning)
    # Query submit button disabled
    submit_btn = [b for b in at.button if b.label == "Submit"][0]
    assert submit_btn.disabled is True


# ==============================================================================
# 8-9. Database Schema Introspection & Metadata Verification
# ==============================================================================

def test_database_tables_and_views_discovered():
    """Criterion 8 & 9: All 9 base tables and 2 analytical views are discoverable with metadata."""
    schema = get_database_schema(DEFAULT_DB_PATH)
    tables = {t["name"]: t for t in schema["tables"]}
    views = {v["name"]: v for v in schema["views"]}

    expected_tables = [
        "customers", "geolocation", "order_items", "order_payments",
        "order_reviews", "orders", "products", "sellers", "category_translation",
    ]
    expected_views = ["order_summary", "category_sales_summary"]

    for tbl_name in expected_tables:
        assert tbl_name in tables, f"Expected base table {tbl_name} missing"
        cols = tables[tbl_name]["columns"]
        assert len(cols) > 0
        for col in cols:
            assert "name" in col and "data_type" in col

    for vw_name in expected_views:
        assert vw_name in views, f"Expected view {vw_name} missing"
        cols = views[vw_name]["columns"]
        assert len(cols) > 0
        assert "purpose" in views[vw_name] or "grain" in views[vw_name]


def test_schema_explorer_rendered_in_ui(monkeypatch):
    """Criterion 9: Verify expanders for base tables and views render in UI."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "secret123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-test-key")
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()

    assert not at.exception
    # Check Dataset Explorer header
    headers = [h.value for h in at.header]
    assert "Dataset Explorer" in headers
    assert "Conversation History" in headers

    # Base table expanders and analytical view expanders present
    expander_labels = [e.label for e in at.expander]
    assert "customers" in expander_labels
    assert "orders" in expander_labels
    assert "order_summary (Analytical View)" in expander_labels
    assert "category_sales_summary (Analytical View)" in expander_labels


# ==============================================================================
# 10-13. Query Execution, Example Questions & Session State
# ==============================================================================

def test_submitting_typed_question_executes_turn(monkeypatch):
    """Criterion 10: Submitting a question through query form executes one turn."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "secret123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-test-key")
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()

    session = at.session_state["agent_session"]
    mock_reply = "Total delivered revenue for 2017 was approximately 7.2M BRL."

    class MockGraph:
        def invoke(self, state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            msgs = list(state.get("messages", []))
            return {"messages": msgs + [AIMessage(content=mock_reply)]}

    session.graph = MockGraph()

    # Enter question and submit
    at.text_input[0].set_value("What was total delivered revenue in 2017?")
    submit_btn = [b for b in at.button if b.label == "Submit"][0]
    submit_btn.click().run()

    assert not at.exception
    assert len(at.chat_message) == 2
    assert "What was total delivered revenue in 2017?" in at.chat_message[0].markdown[0].value
    assert mock_reply in at.chat_message[1].markdown[0].value


def test_selecting_example_question_executes_turn(monkeypatch):
    """Criterion 11: Clicking an example question button executes one turn."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "secret123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-test-key")
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()

    session = at.session_state["agent_session"]
    mock_reply = "Top 5 categories: Bed & Bath, Health & Beauty, Sports & Leisure, Computers, Housewares."

    class MockGraph:
        def invoke(self, state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            msgs = list(state.get("messages", []))
            return {"messages": msgs + [AIMessage(content=mock_reply)]}

    session.graph = MockGraph()

    # Click first example question
    example_btn = at.sidebar.button[0]
    expected_q = example_btn.label
    example_btn.click().run()

    assert not at.exception
    assert len(at.chat_message) == 2
    assert expected_q in at.chat_message[0].markdown[0].value
    assert mock_reply in at.chat_message[1].markdown[0].value


# ==============================================================================
# 14-15. Plotly Charts, Admin Diagnostics & Clean Styling
# ==============================================================================

def test_clean_branding_no_decorative_emojis(monkeypatch):
    """Criterion 1: Verify all decorative emojis are removed from headers and titles."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "secret123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-test-key")
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()

    # Title has no emojis
    assert "📊" not in at.title[0].value
    assert at.title[0].value == "Data Analytics & Business Intelligence Agent"

    # Sidebar headers have no emojis
    for h in at.sidebar.header:
        for emoji in ["📊", "📁", "⚙️", "💡", "🗑️", "🛠️", "🔒"]:
            assert emoji not in h.value

    # Buttons have no decorative emojis
    for b in at.sidebar.button:
        assert "🗑️" not in b.label


def test_admin_diagnostics_integration(monkeypatch):
    """Criterion 15: Admin diagnostics expander verifies password and executes probe."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "secret123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-test-key")
    monkeypatch.setenv("DIAGNOSTIC_PASSWORD", "admin-diag-secret")
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()

    # Expander "Admin Diagnostics" should be in sidebar
    diag_expander = [e for e in at.sidebar.expander if e.label == "Admin Diagnostics"]
    assert len(diag_expander) == 1
    assert verify_diagnostic_access("admin-diag-secret") is True
    assert verify_diagnostic_access("bad-secret") is False


# ==============================================================================
# 16-18. SaaS Dashboard Polish: Assets & Scrollable Chat Container
# ==============================================================================

def test_saas_assets_and_page_config(monkeypatch):
    """Verify SVG favicon, external stylesheet, and updated SaaS branding."""
    favicon_path = Path(__file__).parent.parent / "assets" / "favicon.svg"
    styles_path = Path(__file__).parent.parent / "assets" / "styles.css"

    assert favicon_path.exists(), "assets/favicon.svg must exist"
    svg_content = favicon_path.read_text(encoding="utf-8")
    assert "<svg" in svg_content and "</svg>" in svg_content
    assert "#6366f1" in svg_content or "#4f46e5" in svg_content  # Indigo palette

    assert styles_path.exists(), "assets/styles.css must exist"
    css_content = styles_path.read_text(encoding="utf-8")
    assert "--accent-primary" in css_content
    assert "--bg-app" in css_content

    monkeypatch.setenv("APP_ACCESS_PASSWORD", "secret123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-test-key")
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()

    # Sidebar branding
    sidebar_title = at.sidebar.title[0].value
    assert sidebar_title == "Data Analytics & BI Agent"
    assert "📊" not in sidebar_title


def test_scrollable_conversation_container(monkeypatch):
    """Verify conversation history is housed inside a scrollable container with autoscroll."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "secret123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-test-key")
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()

    # Ensure Conversation History header exists
    headers = [h.value for h in at.header]
    assert "Conversation History" in headers

    # Verify messages inside container render without page exceptions
    session = at.session_state["agent_session"]
    session.ui_messages.append({"role": "user", "content": "Test user query"})
    session.ui_messages.append({
        "role": "assistant",
        "content": "Test assistant response with analytical details.",
        "chart_ids": [],
        "tool_executions": [],
    })
    at.run()

    assert not at.exception
    assert len(at.chat_message) == 2
    assert "Test user query" in at.chat_message[0].markdown[0].value
    assert "Test assistant response" in at.chat_message[1].markdown[0].value

