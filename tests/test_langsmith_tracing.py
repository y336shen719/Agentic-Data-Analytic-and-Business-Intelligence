"""Tests for Step 8.5.1 LangSmith Tracing Integration.

Verifies:
1. Tracing disabled by default.
2. Tracing enabled with valid configuration.
3. Missing LangSmith API key handling (graceful deactivation, no crash).
4. Streamlit entry-point instrumentation and metadata.
5. Evaluation CLI entry-point instrumentation and metadata.
6. Custom tool trace metadata cleanliness (no full tables or raw figures).
7. Correct chart metadata propagation.
8. Safe multi-turn session association across turns.
9. Redaction of sensitive information (API keys, secrets, tokens).
10. No behavioral change or overhead when tracing is disabled.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from agent.tracing import (
    LangSmithConfig,
    create_runnable_tracing_config,
    load_langsmith_config,
    redact_sensitive_metadata,
    setup_langsmith_environment,
)
from agent.ui_session import create_agent_session, execute_session_turn


def test_tracing_disabled_by_default():
    """Verify that tracing is disabled by default when no environment variables are set."""
    cfg = load_langsmith_config(env={})
    assert cfg.enabled is False
    assert cfg.api_key is None
    assert cfg.project == "agentic-data-analyst"


def test_tracing_enabled_with_valid_config():
    """Verify that tracing parses correctly when valid environment variables are supplied."""
    env = {
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": "lsv2_pt_testkey123456",
        "LANGSMITH_PROJECT": "ecommerce-bi-analytics",
    }
    cfg = load_langsmith_config(env=env)
    assert cfg.enabled is True
    assert cfg.api_key == "lsv2_pt_testkey123456"
    assert cfg.project == "ecommerce-bi-analytics"


def test_missing_api_key_gracefully_disables_tracing():
    """Verify that missing API key when LANGSMITH_TRACING=true does not crash and deactivates tracing."""
    env = {
        "LANGSMITH_TRACING": "true",
        # No API KEY provided
    }
    cfg = load_langsmith_config(env=env)
    assert cfg.enabled is False
    assert cfg.api_key is None


def test_setup_langsmith_environment_sync():
    """Verify that setup_langsmith_environment sets matching LANGCHAIN_* and LANGSMITH_* variables."""
    # When enabled
    cfg_enabled = LangSmithConfig(
        enabled=True,
        api_key="lsv2_pt_dummykey",
        project="test-proj",
        endpoint="https://api.smith.langchain.com",
    )
    is_active = setup_langsmith_environment(cfg_enabled)
    assert is_active is True
    assert os.environ.get("LANGSMITH_TRACING") == "true"
    assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    assert os.environ.get("LANGSMITH_API_KEY") == "lsv2_pt_dummykey"
    assert os.environ.get("LANGCHAIN_PROJECT") == "test-proj"

    # When disabled
    cfg_disabled = LangSmithConfig(enabled=False, api_key=None)
    is_active_dis = setup_langsmith_environment(cfg_disabled)
    assert is_active_dis is False
    assert os.environ.get("LANGSMITH_TRACING") == "false"
    assert os.environ.get("LANGCHAIN_TRACING_V2") == "false"


def test_streamlit_entry_point_config():
    """Verify RunnableConfig generated for Streamlit entry point."""
    config = create_runnable_tracing_config(
        source="streamlit",
        session_id="session_streamlit_12345678",
        model_name="gpt-6-luna",
        turn_index=0,
    )
    assert "Agent Turn (streamlit - session_" in config["run_name"]
    assert "agentic-data-analyst" in config["tags"]
    assert "streamlit" in config["tags"]
    assert config["metadata"]["source"] == "streamlit"
    assert config["metadata"]["session_id"] == "session_"
    assert config["metadata"]["model"] == "gpt-6-luna"
    assert config["metadata"]["turn_index"] == 0


def test_evaluation_cli_entry_point_config():
    """Verify RunnableConfig generated for Evaluation benchmark scenarios."""
    config = create_runnable_tracing_config(
        source="evaluation",
        session_id="session_eval_abcdefgh",
        scenario_id="VIZ_01_MONTHLY_REVENUE_LINE",
        model_name="gpt-6-luna",
        turn_index=1,
    )
    assert config["run_name"] == "Agent Turn (VIZ_01_MONTHLY_REVENUE_LINE)"
    assert "evaluation" in config["tags"]
    assert "VIZ_01_MONTHLY_REVENUE_LINE" in config["tags"]
    assert config["metadata"]["scenario_id"] == "VIZ_01_MONTHLY_REVENUE_LINE"
    assert config["metadata"]["turn_index"] == 1


def test_sensitive_metadata_redaction():
    """Verify that credentials and API keys are redacted before metadata transmission."""
    raw = {
        "normal_key": "normal_value",
        "api_key": "sk-proj-secret12345",
        "user_token": "bearer xyz",
        "db_password": "supersecretpassword",
        "nested": {
            "openai_api_key": "sk-1234567890",
            "safe_counter": 42,
        },
        "langsmith_direct": "lsv2_pt_somekey",
    }
    redacted = redact_sensitive_metadata(raw)
    assert redacted["normal_key"] == "normal_value"
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["user_token"] == "[REDACTED]"
    assert redacted["db_password"] == "[REDACTED]"
    assert redacted["nested"]["openai_api_key"] == "[REDACTED]"
    assert redacted["nested"]["safe_counter"] == 42
    assert redacted["langsmith_direct"] == "[REDACTED_API_KEY]"


def test_multi_turn_session_association():
    """Verify that session_id is preserved and turn_index increments across multiple turns."""
    mock_llm = MagicMock()
    session = create_agent_session(llm=mock_llm)

    # Mock graph.invoke to capture the config passed
    captured_configs = []

    def fake_invoke(state, config=None):
        captured_configs.append(config)
        return {"messages": state["messages"]}

    session.graph.invoke = fake_invoke

    # Turn 0
    execute_session_turn(session, "Turn 1 query", source="streamlit", turn_index=0)
    # Turn 1
    execute_session_turn(session, "Turn 2 query", source="streamlit", turn_index=1)

    assert len(captured_configs) == 2
    assert captured_configs[0]["metadata"]["session_id"] == session.session_id[:8]
    assert captured_configs[1]["metadata"]["session_id"] == session.session_id[:8]
    assert captured_configs[0]["metadata"]["turn_index"] == 0
    assert captured_configs[1]["metadata"]["turn_index"] == 1


def test_no_behavioral_change_when_tracing_disabled():
    """Verify that execute_session_turn works cleanly without tracing without crashing."""
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    mock_llm = MagicMock()
    session = create_agent_session(llm=mock_llm)

    def fake_invoke(state, config=None):
        return {"messages": state["messages"]}

    session.graph.invoke = fake_invoke

    turn_res = execute_session_turn(session, "Test query without tracing")
    assert turn_res["role"] == "assistant"
    assert "content" in turn_res
    assert turn_res["chart_ids"] == []


def test_tool_trace_metadata_cleanliness_and_chart_propagation():
    """Verify tool wrappers return compact metadata suitable for tracing without uploading large figures/tables."""
    from agent.tool_wrappers import create_chart_generation_tool
    from tools.dataset_store import DatasetStore
    from tools.chart_registry import ChartRegistry

    store = DatasetStore()
    registry = ChartRegistry()
    chart_tool = create_chart_generation_tool(dataset_store=store, chart_registry=registry)

    # Save a small dataset
    ds_id = store.save({
        "status": "success",
        "columns": ["category", "revenue"],
        "rows": [["electronics", 5000.0], ["apparel", 3000.0]],
        "row_count": 2,
    })

    # Invoke chart tool directly
    res = chart_tool.invoke({
        "dataset_id": ds_id,
        "chart_type": "bar",
        "x_column": "category",
        "y_column": "revenue",
    })

    assert res["status"] == "success"
    assert "chart_id" in res
    assert "figure" not in res  # Full figure JSON must NOT be returned in tool result (kept in registry)
    assert res["metadata"]["rows_used"] == 2
    assert res["metadata"]["x_column"] == "category"
    assert res["metadata"]["y_column"] == "revenue"


def test_workspace_id_resolution_and_environment_sync():
    """Verify that workspace_id is parsed and propagated to environment variables."""
    env = {
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": "lsv2_pt_orgkey_123456",
        "LANGSMITH_PROJECT": "org-project",
        "LANGSMITH_WORKSPACE_ID": "ws-uuid-998877",
    }
    cfg = load_langsmith_config(env=env)
    assert cfg.workspace_id == "ws-uuid-998877"

    is_active = setup_langsmith_environment(cfg)
    assert is_active is True
    assert os.environ.get("LANGSMITH_WORKSPACE_ID") == "ws-uuid-998877"
    assert os.environ.get("LANGCHAIN_WORKSPACE_ID") == "ws-uuid-998877"


def test_streamlit_secrets_fallback_for_tracing_config():
    """Verify that tracing config safely falls back to Streamlit secrets when env vars are missing."""
    import sys
    from types import ModuleType

    mock_st = ModuleType("streamlit")
    mock_st.secrets = {
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": "lsv2_pt_secret_token",
        "LANGSMITH_PROJECT": "secrets-project",
    }

    with patch.dict(sys.modules, {"streamlit": mock_st}):
        # Provide empty env mapping so os.environ is bypassed
        cfg = load_langsmith_config(env={})
        assert cfg.enabled is True
        assert cfg.api_key == "lsv2_pt_secret_token"
        assert cfg.project == "secrets-project"


def test_runnable_tracing_config_binds_tracer_when_active():
    """Verify that create_runnable_tracing_config injects LangChainTracer into callbacks when active."""
    env = {
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": "lsv2_pt_validkey",
        "LANGSMITH_PROJECT": "test-callbacks-project",
    }
    cfg = load_langsmith_config(env=env)
    setup_langsmith_environment(cfg)

    config = create_runnable_tracing_config(source="streamlit", session_id="test_sess_123")
    assert "callbacks" in config
    assert len(config["callbacks"]) >= 1
    tracer = config["callbacks"][0]
    assert type(tracer).__name__ == "LangChainTracer"
    assert tracer.project_name == "test-callbacks-project"

    # Reset
    setup_langsmith_environment(LangSmithConfig(enabled=False, api_key=None))
    disabled_config = create_runnable_tracing_config(source="streamlit", session_id="test_sess_123")
    assert "callbacks" not in disabled_config


def test_flush_tracing_execution_and_resilience():
    """Verify that flush_tracing calls wait_for_futures/flush and survives exceptions gracefully."""
    from agent.tracing import flush_tracing

    mock_tracer = MagicMock()
    mock_client = MagicMock()
    mock_tracer.client = mock_client

    # 1. Successful flush
    ok = flush_tracing(tracer=mock_tracer, timeout=1.0)
    assert ok is True
    mock_tracer.wait_for_futures.assert_called_once()
    mock_client.flush.assert_called_once_with(timeout=1.0)

    # 2. Resilient on exception (should return False and never raise)
    failing_tracer = MagicMock()
    failing_tracer.wait_for_futures.side_effect = RuntimeError("Network timeout to LangSmith")
    ok_fail = flush_tracing(tracer=failing_tracer, timeout=1.0)
    assert ok_fail is False


def test_execute_session_turn_triggers_flush_tracing():
    """Verify that execute_session_turn triggers synchronous flush_tracing on completion."""
    mock_llm = MagicMock()
    session = create_agent_session(llm=mock_llm)

    def fake_invoke(state, config=None):
        return {"messages": state["messages"]}

    session.graph.invoke = fake_invoke

    with patch("agent.ui_session.flush_tracing") as mock_flush:
        execute_session_turn(session, "Hello data analyst")
        mock_flush.assert_called_once_with(timeout=3.0)

