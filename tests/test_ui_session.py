"""Tests for Streamlit UI session management, multi-turn state persistence, and artifact handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import uuid
from unittest.mock import patch

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.language_models.chat_models import BaseChatModel
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from agent.state import create_initial_state
from agent.secrets import SecretReadError
from agent.ui_session import (
    DEFAULT_INCOMPLETE_MESSAGE,
    AgentSession,
    create_agent_session,
    execute_session_turn,
    extract_text_from_message_content,
    extract_turn_details,
    get_or_create_session,
    reconfigure_session,
    reset_agent_session,
    resolve_api_key,
    trim_messages_for_context,
)
from tools.chart_registry import ChartNotFoundError


class MockScriptedLLM(BaseChatModel):
    """Deterministic scripted mock LLM for testing session turns."""

    model_name: str = "mock-model"
    responses: List[BaseMessage] = []
    call_index: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.call_index < len(self.responses):
            resp = self.responses[self.call_index]
            self.call_index += 1
            return ChatResult(generations=[ChatGeneration(message=resp)])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Default mock response."))])

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self):
        return "mock-scripted-llm"


def _make_test_db(tmp_path: Path) -> str:
    import duckdb

    db_path = str(tmp_path / "test_session.duckdb")
    con = duckdb.connect(db_path)
    con.execute("CREATE TABLE orders (order_id VARCHAR, customer_id VARCHAR, status VARCHAR, revenue DOUBLE);")
    con.execute("INSERT INTO orders VALUES ('o1', 'c1', 'delivered', 150.0), ('o2', 'c2', 'delivered', 250.0);")
    con.close()
    return db_path


def test_session_initialization_defaults(tmp_path):
    """Verify create_agent_session creates isolated stores, tools, and compiled graph."""
    db_path = _make_test_db(tmp_path)
    mock_llm = MockScriptedLLM()

    session = create_agent_session(db_path=db_path, llm=mock_llm)

    assert isinstance(session, AgentSession)
    assert session.session_id.startswith("sess_")
    assert len(session.dataset_store) == 0
    assert len(session.chart_registry) == 0
    assert len(session.agent_messages) == 0
    assert len(session.ui_messages) == 0
    assert len(session.tools) >= 3  # SQL, Python, Chart tools


def test_session_resource_reuse_across_reruns(tmp_path):
    """Verify get_or_create_session reuses existing session without reinitializing."""
    db_path = _make_test_db(tmp_path)
    mock_llm = MockScriptedLLM()
    state_dict: Dict[str, Any] = {}

    s1 = get_or_create_session(state_dict, db_path=db_path, llm=mock_llm)
    s1.ui_messages.append({"role": "user", "content": "hello"})

    s2 = get_or_create_session(state_dict, db_path=db_path, llm=mock_llm)
    assert s1 is s2
    assert s2.session_id == s1.session_id
    assert len(s2.ui_messages) == 1


def test_independent_session_isolation(tmp_path):
    """Verify different sessions have distinct stores, registries, and histories."""
    db_path = _make_test_db(tmp_path)
    mock_llm = MockScriptedLLM()

    s1 = create_agent_session(db_path=db_path, llm=mock_llm)
    s2 = create_agent_session(db_path=db_path, llm=mock_llm)

    assert s1.session_id != s2.session_id
    assert s1.dataset_store is not s2.dataset_store
    assert s1.chart_registry is not s2.chart_registry

    s1.dataset_store.save({"status": "success", "columns": ["a"], "rows": [[1]], "row_count": 1})
    assert len(s1.dataset_store) == 1
    assert len(s2.dataset_store) == 0


def test_multi_turn_state_persistence_and_followup(tmp_path):
    """Verify multi-turn conversations persist context across user turns."""
    db_path = _make_test_db(tmp_path)

    # Turn 1: SQL call then answer
    # Turn 2: Follow-up direct answer referencing prior context
    llm = MockScriptedLLM()
    llm.responses = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "execute_sql",
                "args": {"query": "SELECT status, SUM(revenue) as rev FROM orders GROUP BY status"},
                "id": "c1",
            }],
        ),
        AIMessage(content="Delivered orders generated 400.0 total revenue."),
        AIMessage(content="Based on the delivered orders, average revenue was 200.0 per order."),
    ]

    session = create_agent_session(db_path=db_path, llm=llm)

    # Turn 1
    r1 = execute_session_turn(session, "What is total revenue by status?")
    assert r1["role"] == "assistant"
    assert "400.0" in r1["content"]
    assert len(r1["tool_executions"]) == 1
    assert r1["tool_executions"][0]["tool"] == "execute_sql"

    # Turn 2
    r2 = execute_session_turn(session, "What was the average revenue?")
    assert r2["role"] == "assistant"
    assert "200.0" in r2["content"]

    # Verify conversation history has both turns without duplication
    assert len(session.ui_messages) == 4  # User1, Assistant1, User2, Assistant2
    assert session.ui_messages[0]["content"] == "What is total revenue by status?"
    assert session.ui_messages[2]["content"] == "What was the average revenue?"

    # Check agent_messages holds full message sequence
    assert len(session.agent_messages) >= 4


def test_per_turn_retry_and_step_counter_reset(tmp_path):
    """Verify each turn resets per-turn counters to 0 while preserving conversation context."""
    db_path = _make_test_db(tmp_path)
    session = create_agent_session(db_path=db_path, llm=MockScriptedLLM())

    captured_inputs: List[Dict[str, Any]] = []

    class SpyGraph:
        call_count: int = 0

        def invoke(self, state: Dict[str, Any], config=None, **kwargs) -> Dict[str, Any]:
            captured_inputs.append(dict(state))
            self.call_count += 1
            current_messages = list(state.get("messages", []))

            # Simulate first turn returning nonzero execution/retry counters and error states
            if self.call_count == 1:
                return {
                    "messages": current_messages + [AIMessage(content="Turn 1 answer.")],
                    "step_count": 8,
                    "sql_retry_count": 2,
                    "python_retry_count": 2,
                    "pending_python_recovery": True,
                    "last_error": "SyntaxError: simulated error",
                    "last_error_category": "syntax_error",
                }
            else:
                return {
                    "messages": current_messages + [AIMessage(content="Turn 2 answer.")],
                    "step_count": 2,
                    "sql_retry_count": 0,
                    "python_retry_count": 0,
                    "pending_python_recovery": False,
                    "last_error": None,
                    "last_error_category": None,
                }

    session.graph = SpyGraph()

    # Turn 1
    r1 = execute_session_turn(session, "First question")
    assert r1["content"] == "Turn 1 answer."

    # Turn 2
    r2 = execute_session_turn(session, "Second question")
    assert r2["content"] == "Turn 2 answer."

    assert len(captured_inputs) == 2

    # Verify Turn 1 received zeroed counters
    turn1_input = captured_inputs[0]
    assert turn1_input["step_count"] == 0
    assert turn1_input["sql_retry_count"] == 0
    assert turn1_input["python_retry_count"] == 0
    assert turn1_input["pending_python_recovery"] is False
    assert turn1_input["last_error"] is None
    assert turn1_input["last_error_category"] is None
    assert len(turn1_input["messages"]) == 1
    assert isinstance(turn1_input["messages"][0], HumanMessage)
    assert turn1_input["messages"][0].content == "First question"

    # Verify Turn 2 received zeroed counters regardless of Turn 1 final counter values
    turn2_input = captured_inputs[1]
    assert turn2_input["step_count"] == 0
    assert turn2_input["sql_retry_count"] == 0
    assert turn2_input["python_retry_count"] == 0
    assert turn2_input["pending_python_recovery"] is False
    assert turn2_input["last_error"] is None
    assert turn2_input["last_error_category"] is None

    # Verify Turn 2 input preserves complete previous context [Human1, AI1, Human2]
    turn2_messages = turn2_input["messages"]
    assert len(turn2_messages) == 3
    assert isinstance(turn2_messages[0], HumanMessage)
    assert turn2_messages[0].content == "First question"
    assert isinstance(turn2_messages[1], AIMessage)
    assert turn2_messages[1].content == "Turn 1 answer."
    assert isinstance(turn2_messages[2], HumanMessage)
    assert turn2_messages[2].content == "Second question"

    # Verify UI history integrity
    assert len(session.ui_messages) == 4
    assert session.ui_messages[0]["content"] == "First question"
    assert session.ui_messages[1]["content"] == "Turn 1 answer."
    assert session.ui_messages[2]["content"] == "Second question"
    assert session.ui_messages[3]["content"] == "Turn 2 answer."


def test_chart_reference_extraction_and_figure_retrieval(tmp_path):
    """Verify chart tool execution stores figure in registry and extracts chart_id to UI."""
    db_path = _make_test_db(tmp_path)
    llm = MockScriptedLLM()

    session = create_agent_session(db_path=db_path, llm=llm)

    cid = session.chart_registry.save(
        figure={"data": [{"type": "bar", "x": ["delivered"], "y": [400]}], "layout": {"title": "Sales"}},
        dataset_id="ds_test",
        chart_type="bar",
        title="Sales",
    )

    sample_turn_messages = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "generate_chart",
                "args": {"chart_type": "bar", "dataset_id": "ds_test", "x_column": "status", "y_column": "rev"},
                "id": "c2",
            }],
        ),
        ToolMessage(
            content=json.dumps({"status": "success", "chart_id": cid, "chart_type": "bar"}),
            tool_call_id="c2",
            name="generate_chart",
        ),
        AIMessage(content="Here is your bar chart."),
    ]

    assistant_text, chart_ids, tool_steps = extract_turn_details(sample_turn_messages)

    assert assistant_text == "Here is your bar chart."
    assert chart_ids == [cid]
    assert len(tool_steps) == 1
    assert tool_steps[0]["tool"] == "generate_chart"

    # Verify retrieval from registry and Plotly Figure reconstruction
    record = session.chart_registry.get(cid)
    fig = go.Figure(record.figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"


def test_missing_chart_reference_handling(tmp_path):
    """Verify non-existent chart_id raises ChartNotFoundError safely."""
    db_path = _make_test_db(tmp_path)
    session = create_agent_session(db_path=db_path, llm=MockScriptedLLM())

    with pytest.raises(ChartNotFoundError):
        session.chart_registry.get("chart_nonexistent")


def test_reset_agent_session(tmp_path):
    """Verify reset_agent_session completely empties all stores and histories."""
    db_path = _make_test_db(tmp_path)
    session = create_agent_session(db_path=db_path, llm=MockScriptedLLM())

    # Populate state
    session.dataset_store.save({"status": "success", "columns": ["val"], "rows": [[10]], "row_count": 1})
    session.chart_registry.save(
        figure={"data": []},
        dataset_id="ds_1",
        chart_type="line",
        title="Title",
    )
    session.agent_messages.append(HumanMessage(content="test"))
    session.ui_messages.append({"role": "user", "content": "test"})

    assert len(session.dataset_store) == 1
    assert len(session.chart_registry) == 1
    assert len(session.agent_messages) == 1
    assert len(session.ui_messages) == 1

    # Reset
    reset_agent_session(session)

    assert len(session.dataset_store) == 0
    assert len(session.chart_registry) == 0
    assert len(session.agent_messages) == 0
    assert len(session.ui_messages) == 0


def test_safe_handling_of_graph_execution_errors_and_history_consistency(tmp_path):
    """Verify graph execution failures are sanitized and agent history remains consistent."""
    db_path = _make_test_db(tmp_path)

    class FailingLLM(BaseChatModel):
        fail_first: bool = True

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if self.fail_first:
                self.fail_first = False
                raise RuntimeError("Sensitive internal database connection failure at 192.168.1.50:/var/data/secret.db")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Recovery response."))])

        def bind_tools(self, tools, **kwargs):
            return self

        @property
        def _llm_type(self):
            return "failing-llm"

    failing_llm = FailingLLM()
    session = create_agent_session(db_path=db_path, llm=failing_llm)

    # Turn 1: Fails
    res = execute_session_turn(session, "Calculate something sensitive")

    assert res["role"] == "assistant"
    assert res.get("error") is True
    # P0: Error message must be generic and never leak internal IPs, paths, or exception text
    assert res["content"] == "I encountered an unexpected error while processing your request. Please try again."
    assert "192.168.1.50" not in res["content"]
    assert "secret.db" not in res["content"]
    assert "RuntimeError" not in res["content"]

    # P1: Check history consistency - agent_messages must have HumanMessage AND AIMessage
    assert len(session.ui_messages) == 2
    assert len(session.agent_messages) == 2
    assert isinstance(session.agent_messages[0], HumanMessage)
    assert isinstance(session.agent_messages[1], AIMessage)
    assert session.agent_messages[1].content == res["content"]

    # Turn 2: Subsequent interaction must succeed and not duplicate previous user message
    res2 = execute_session_turn(session, "Try a new question")
    assert res2["role"] == "assistant"
    assert res2["content"] == "Recovery response."
    assert len(session.ui_messages) == 4
    assert len(session.agent_messages) == 4
    assert session.ui_messages[2]["content"] == "Try a new question"
    assert session.agent_messages[2].content == "Try a new question"


def test_resolve_api_key_precedence(monkeypatch):
    """Verify precedence: User input > Streamlit secrets > Environment variable > None."""
    monkeypatch.setenv("OPENAI_API_KEY", "env_secret_key")
    monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)

    # 1. User provided input wins over all
    assert resolve_api_key("user_provided_key") == "user_provided_key"

    # 2. Mock streamlit secrets
    import streamlit as st

    class MockSecrets(dict):
        pass

    mock_sec = MockSecrets({"OPENAI_API_KEY": "secrets_key"})
    monkeypatch.setattr(st, "secrets", mock_sec)

    # User input wins over secrets
    assert resolve_api_key("user_key") == "user_key"
    # Secrets win over env
    assert resolve_api_key("") == "secrets_key"
    assert resolve_api_key(None) == "secrets_key"

    # 3. When secrets is empty or lacks key, env wins
    monkeypatch.setattr(st, "secrets", MockSecrets({}))
    assert resolve_api_key(None) == "env_secret_key"

    # 4. When neither exists, returns None
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert resolve_api_key(None) is None


@pytest.mark.parametrize("with_env_key", [False, True])
def test_resolve_api_key_file(tmp_path, monkeypatch, with_env_key):
    import streamlit as st

    monkeypatch.setattr("agent.tracing._has_streamlit_context", lambda: False)
    monkeypatch.setattr(st, "secrets", {})
    secret_file = tmp_path / "openai-key"
    secret_file.write_text(" file-test-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(secret_file))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    if with_env_key:
        monkeypatch.setenv("OPENAI_API_KEY", "environment-test-key")
    assert resolve_api_key() == "file-test-key"


def test_resolve_api_key_preserves_explicit_and_streamlit_priority(tmp_path, monkeypatch):
    import streamlit as st

    monkeypatch.setattr("agent.tracing._has_streamlit_context", lambda: True)
    monkeypatch.setattr(st, "secrets", {"OPENAI_API_KEY": "streamlit-test-key"})
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(tmp_path / "missing"))
    assert resolve_api_key("explicit-test-key") == "explicit-test-key"
    assert resolve_api_key() == "streamlit-test-key"
    monkeypatch.setattr(st, "secrets", {})
    monkeypatch.setenv("OPENAI_API_KEY", "environment-test-key")
    with pytest.raises(SecretReadError):
        resolve_api_key()


def test_trim_messages_for_context():
    """Verify trim_messages_for_context keeps last N turns at HumanMessage boundaries."""
    messages = [
        HumanMessage(content="turn 1"),
        AIMessage(content="", tool_calls=[{"name": "t1", "args": {}, "id": "1"}]),
        ToolMessage(content="ok", tool_call_id="1", name="t1"),
        AIMessage(content="ans 1"),
        HumanMessage(content="turn 2"),
        AIMessage(content="ans 2"),
        HumanMessage(content="turn 3"),
        AIMessage(content="ans 3"),
    ]

    trimmed = trim_messages_for_context(messages, max_turns=2)
    assert len(trimmed) == 4
    assert isinstance(trimmed[0], HumanMessage)
    assert trimmed[0].content == "turn 2"

    assert len(trim_messages_for_context(messages, max_turns=5)) == len(messages)


def test_bounded_multi_turn_history_and_tool_pairing():
    """Verify turn-aware history trimming respects max_messages without breaking tool pairs."""
    # Build 15 turns where each turn has Human, AI(tool_call), ToolMessage, AI(answer)
    messages: List[BaseMessage] = []
    for i in range(15):
        tid = f"call_{i}"
        messages.extend([
            HumanMessage(content=f"user question {i}"),
            AIMessage(content="", tool_calls=[{"name": "execute_sql", "args": {"query": f"Q{i}"}, "id": tid}]),
            ToolMessage(content=json.dumps({"status": "success", "rows": [i]}), tool_call_id=tid, name="execute_sql"),
            AIMessage(content=f"assistant answer {i}"),
        ])

    assert len(messages) == 60  # 15 turns * 4 messages

    # Max turns = 5 -> should keep exactly 5 * 4 = 20 messages
    trimmed_turns = trim_messages_for_context(messages, max_turns=5, max_messages=100)
    assert len(trimmed_turns) == 20
    assert isinstance(trimmed_turns[0], HumanMessage)
    assert trimmed_turns[0].content == "user question 10"

    # Message limit tighter than max_turns: max_messages = 10 -> fits 2 full turns (8 messages)
    trimmed_msgs = trim_messages_for_context(messages, max_turns=5, max_messages=10)
    assert len(trimmed_msgs) == 8  # 2 turns * 4 messages = 8 <= 10
    assert isinstance(trimmed_msgs[0], HumanMessage)
    assert trimmed_msgs[0].content == "user question 13"

    # Oversized single turn exception: newest turn has 12 messages, max_messages = 5
    oversized_turn = [
        HumanMessage(content="huge turn"),
        AIMessage(content="", tool_calls=[{"name": f"t_{j}", "args": {}, "id": f"c_{j}"} for j in range(5)]),
    ]
    for j in range(5):
        oversized_turn.append(ToolMessage(content="ok", tool_call_id=f"c_{j}", name=f"t_{j}"))
    oversized_turn.append(AIMessage(content="huge answer"))

    all_with_oversized = messages + oversized_turn
    trimmed_oversized = trim_messages_for_context(all_with_oversized, max_turns=5, max_messages=5)
    # The newest oversized turn must be retained intact (12 messages) rather than sliced
    assert len(trimmed_oversized) == len(oversized_turn)
    assert isinstance(trimmed_oversized[0], HumanMessage)
    assert trimmed_oversized[0].content == "huge turn"


def test_robust_tool_message_parsing_edge_cases():
    """Verify extract_turn_details safely handles non-dict JSON, scalars, nulls, and tool errors."""
    tcs = [
        {"name": "execute_sql", "args": {"query": "SELECT 1"}, "id": "tc1"},
        {"name": "generate_chart", "args": {"chart_type": "bar"}, "id": "tc2"},
        {"name": "execute_python_analysis", "args": {"operation": "describe"}, "id": "tc3"},
        {"name": "get_database_schema", "args": {}, "id": "tc4"},
        {"name": "generate_chart", "args": {"chart_type": "line"}, "id": "tc5"},
    ]
    aimsg = AIMessage(content="Here are the results", tool_calls=tcs)

    tool_msgs = [
        # 1. Invalid JSON string
        ToolMessage(content="{not json", tool_call_id="tc1", name="execute_sql"),
        # 2. JSON list
        ToolMessage(content="[1, 2, 3]", tool_call_id="tc2", name="generate_chart"),
        # 3. JSON null
        ToolMessage(content="null", tool_call_id="tc3", name="execute_python_analysis"),
        # 4. ToolMessage level error with valid JSON dict
        ToolMessage(content='{"status": "success"}', status="error", tool_call_id="tc4", name="get_database_schema"),
        # 5. Malformed chart: success status but empty/missing chart_id
        ToolMessage(content='{"status": "success", "chart_id": "   "}', tool_call_id="tc5", name="generate_chart"),
    ]

    narrative, chart_ids, tool_steps = extract_turn_details([aimsg, *tool_msgs])

    assert narrative == "Here are the results"
    # No invalid or empty chart IDs collected
    assert chart_ids == []
    # All 5 tool executions extracted without raising exceptions
    assert len(tool_steps) == 5
    # Status error on tool 4 was respected
    assert tool_steps[3]["status"] == "error"
    # Chart generation failure summary
    assert "Chart generation failed" in tool_steps[1]["summary"]
    assert "Chart generation failed" in tool_steps[4]["summary"]


def test_model_switching_and_reconfiguration(tmp_path):
    """Verify selecting a new model reconfigures LLM/graph while preserving stores and history."""
    db_path = _make_test_db(tmp_path)
    state_dict: Dict[str, Any] = {}

    llm1 = MockScriptedLLM(model_name="model-v1")
    llm2 = MockScriptedLLM(model_name="model-v2")

    session = get_or_create_session(state_dict, db_path=db_path, llm=llm1, model="model-v1")
    assert session.configured_model == "model-v1"
    initial_graph = session.graph

    # Store some data in session
    session.dataset_store.save({"status": "success", "columns": ["a"], "rows": [[1]], "row_count": 1})
    session.ui_messages.append({"role": "user", "content": "hello"})

    # Switch model to model-v2
    updated_session = get_or_create_session(state_dict, db_path=db_path, llm=llm2, model="model-v2")

    assert updated_session is session
    assert session.configured_model == "model-v2"
    assert session.llm is llm2
    assert session.graph is not initial_graph  # Rebuilt graph
    # History and stores are safely preserved
    assert len(session.dataset_store) == 1
    assert len(session.ui_messages) == 1


def test_credential_change_reconfigures_session(tmp_path):
    """Verify changing API key triggers reconfiguration without losing stores or history."""
    db_path = _make_test_db(tmp_path)
    state_dict: Dict[str, Any] = {}

    session = get_or_create_session(
        state_dict, db_path=db_path, api_key="sk-test-key-1", model="gpt-4o-mini"
    )
    assert session.configured_api_key == "sk-test-key-1"
    initial_graph = session.graph

    # Re-call with unchanged key & model: graph must NOT change
    s_same = get_or_create_session(
        state_dict, db_path=db_path, api_key="sk-test-key-1", model="gpt-4o-mini"
    )
    assert s_same.graph is initial_graph

    # Re-call with changed key: graph must rebuild and update configured_api_key
    s_new = get_or_create_session(
        state_dict, db_path=db_path, api_key="sk-test-key-2", model="gpt-4o-mini"
    )
    assert s_new.configured_api_key == "sk-test-key-2"
    assert s_new.graph is not initial_graph


def test_streamlit_app_startup_with_apptest(monkeypatch):
    """Verify app.py initializes cleanly with AppTest without unhandled exceptions."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "test-pass")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-apptest-key")
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()

    assert not at.exception
    # Check header and sidebar branding (redundant main title removed)
    assert at.header[0].value == "Analytics Workspace"
    assert at.sidebar.title[0].value == "Data Analytics & BI Agent"
    # Check sidebar
    assert len(at.sidebar.button) >= 5  # Example questions + clear button + logout


def test_streamlit_apptest_chat_submission_scenario_b1(monkeypatch):
    """Scenario B1: User submits prompt; verify execution, UI message display, and rerun stability."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "test-pass")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-apptest-key")
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()

    # Attach deterministic MockGraph to the session
    session = at.session_state["agent_session"]
    mock_answer = "Top categories by delivered revenue: 1. Bed & Bath, 2. Health & Beauty."

    class MockAppGraph:
        def invoke(self, state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            msgs = list(state.get("messages", []))
            return {"messages": msgs + [AIMessage(content=mock_answer)]}

    session.graph = MockAppGraph()

    # User submits prompt via query form
    user_prompt = "What are the top product categories by revenue?"
    at.text_input[0].set_value(user_prompt)
    submit_btn = [b for b in at.button if b.label == "Submit"][0]
    submit_btn.click().run()

    assert not at.exception
    # Verify messages in UI
    assert len(at.chat_message) == 2
    user_chat = at.chat_message[0]
    assistant_chat = at.chat_message[1]

    assert user_chat.name == "user"
    assert user_prompt in [m.value for m in user_chat.markdown]

    assert assistant_chat.name == "assistant"
    assert mock_answer in [m.value for m in assistant_chat.markdown]

    # Verify rerun does not duplicate messages
    at.run()
    assert len(at.chat_message) == 2
    assert not at.exception


def test_streamlit_apptest_plotly_chart_association_scenario_b2(monkeypatch):
    """Scenario B2: Verify Plotly chart retrieval, data matching, and association with assistant message."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "test-pass")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-apptest-key")
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()

    session = at.session_state["agent_session"]
    fig_data = {
        "data": [{"type": "bar", "x": ["bed_bath", "health_beauty"], "y": [1000, 800]}],
        "layout": {"title": "Top Categories"},
    }
    cid = session.chart_registry.save(
        figure=fig_data,
        dataset_id="ds_cats",
        chart_type="bar",
        title="Top Categories",
    )

    # Populate session history with chart reference
    session.ui_messages.append({"role": "user", "content": "Show revenue chart"})
    session.ui_messages.append({
        "role": "assistant",
        "content": "Here is your category revenue chart.",
        "chart_ids": [cid],
        "tool_executions": [],
    })

    with patch("streamlit.plotly_chart") as mock_plotly:
        at.run()
        assert not at.exception
        # Verify chart rendering was invoked
        assert mock_plotly.call_count == 1
        rendered_fig = mock_plotly.call_args[0][0]
        assert isinstance(rendered_fig, go.Figure)
        assert tuple(rendered_fig.data[0].x) == ("bed_bath", "health_beauty")
        assert tuple(rendered_fig.data[0].y) == (1000, 800)

        # Re-run: ensure chart is rendered from registry, not regenerated or duplicated
        at.run()
        assert not at.exception
        assert mock_plotly.call_count == 2
        # Chat messages remain exactly 2
        assert len(at.chat_message) == 2


def test_streamlit_apptest_clear_conversation_scenario_b3(monkeypatch):
    """Scenario B3: Verify clicking 'Clear Conversation' resets all session stores, histories, and UI."""
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "test-pass")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-apptest-key")
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()

    session = at.session_state["agent_session"]
    session.dataset_store.save({"status": "success", "columns": ["id"], "rows": [[1]], "row_count": 1})
    cid = session.chart_registry.save(figure={"data": []}, dataset_id="ds_1", chart_type="bar", title="t")
    session.agent_messages.append(HumanMessage(content="Hello"))
    session.ui_messages.append({"role": "user", "content": "Hello"})
    session.ui_messages.append({"role": "assistant", "content": "Hi", "chart_ids": [cid]})

    # Re-render to show populated state
    at.run()
    assert len(at.chat_message) == 2
    assert len(session.dataset_store) == 1
    assert len(session.chart_registry) == 1

    # Click Clear Conversation button in sidebar
    clear_btn = [b for b in at.sidebar.button if "Clear Conversation" in b.label][0]
    clear_btn.click().run()

    assert not at.exception
    # Verify all structures are empty
    cleared_session = at.session_state["agent_session"]
    assert len(cleared_session.dataset_store) == 0
    assert len(cleared_session.chart_registry) == 0
    assert len(cleared_session.agent_messages) == 0
    assert len(cleared_session.ui_messages) == 0

    # UI chat messages cleared and welcome banner displayed
    assert len(at.chat_message) == 0
    assert any("Welcome" in inf.value for inf in at.info)


def test_final_answer_extraction_test_a():
    """Test A: Supply complete successful Graph message sequence and verify final answer extraction."""
    schema_call_id = "call_schema_1"
    sql_call_id = "call_sql_1"

    # Sequence with intermediate tool-calling AIMessages and a final structured AIMessage
    sequence = [
        HumanMessage(content="What are the top 5 product categories by total delivered revenue?"),
        AIMessage(
            content=[{"arguments": "{}", "call_id": schema_call_id, "name": "get_database_schema", "type": "function"}],
            tool_calls=[{"name": "get_database_schema", "args": {}, "id": schema_call_id}],
        ),
        ToolMessage(
            content=json.dumps({"tables": [{"name": "orders"}, {"name": "order_items"}]}),
            tool_call_id=schema_call_id,
            name="get_database_schema",
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "execute_sql", "args": {"query": "SELECT category, SUM(price) FROM ..."}, "id": sql_call_id}],
        ),
        ToolMessage(
            content=json.dumps({
                "status": "success",
                "dataset_id": "ds_top5_cats",
                "columns": ["category", "delivered_revenue"],
                "rows": [["health_beauty", 1233131.72], ["watches_gifts", 1166176.98]],
                "row_count": 2,
            }),
            tool_call_id=sql_call_id,
            name="execute_sql",
        ),
        AIMessage(
            content=[
                {"type": "text", "text": "The top categories are: 1. Health & Beauty ($1,233,131.72), 2. Watches & Gifts ($1,166,176.98)."}
            ]
        ),
    ]

    assistant_text, chart_ids, tool_steps = extract_turn_details(sequence)

    # Verify final assistant response is correctly extracted (not intermediate text or fallback)
    assert assistant_text == "The top categories are: 1. Health & Beauty ($1,233,131.72), 2. Watches & Gifts ($1,166,176.98)."
    assert "Analysis completed." not in assistant_text
    assert assistant_text != DEFAULT_INCOMPLETE_MESSAGE

    # Verify tool executions are preserved
    assert len(tool_steps) == 2
    assert tool_steps[0]["tool"] == "get_database_schema"
    assert tool_steps[0]["status"] == "completed"
    assert tool_steps[1]["tool"] == "execute_sql"
    assert tool_steps[1]["status"] == "success"
    assert "Rows: 2" in tool_steps[1]["summary"]
    assert "Dataset ID: ds_top5_cats" in tool_steps[1]["summary"]


def test_session_execution_test_b(tmp_path):
    """Test B: Run execute_session_turn with a scripted graph and verify single invocation and answer delivery."""
    db_path = _make_test_db(tmp_path)
    session = create_agent_session(db_path=db_path, llm=MockScriptedLLM())

    invoke_count = 0
    expected_answer = "Top categories: Health & Beauty ($1.23M), Watches ($1.16M)."

    class ScriptedSqlGraph:
        def invoke(self, state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            nonlocal invoke_count
            invoke_count += 1
            msgs = list(state.get("messages", []))
            sql_id = "sql_turn_b"
            tool_call = {"name": "execute_sql", "args": {"query": "SELECT * FROM t"}, "id": sql_id}
            sql_tool_msg = ToolMessage(
                content=json.dumps({"status": "success", "dataset_id": "ds_b", "columns": ["c"], "rows": [[1]], "row_count": 1}),
                tool_call_id=sql_id,
                name="execute_sql",
            )
            final_ai = AIMessage(content=[{"type": "text", "text": expected_answer}])
            return {
                "messages": msgs + [
                    AIMessage(content="", tool_calls=[tool_call]),
                    sql_tool_msg,
                    final_ai,
                ],
                "step_count": 2,
            }

    session.graph = ScriptedSqlGraph()

    # User turn
    record = execute_session_turn(session, "List top categories")

    # 1. Graph invoked exactly once per turn
    assert invoke_count == 1

    # 2. Session returns the final answer
    assert record["role"] == "assistant"
    assert record["content"] == expected_answer
    assert "Analysis completed." not in record["content"]

    # 3. UI record contains expected text and tool steps
    assert len(record["tool_executions"]) == 1
    assert record["tool_executions"][0]["tool"] == "execute_sql"

    # 4. Session history includes the correct messages without duplicates
    # Expected: [HumanMessage, AIMessage(tool_call), ToolMessage, AIMessage(final)]
    assert len(session.agent_messages) == 4
    assert isinstance(session.agent_messages[0], HumanMessage)
    assert session.agent_messages[0].content == "List top categories"
    assert isinstance(session.agent_messages[1], AIMessage)
    assert len(session.agent_messages[1].tool_calls) == 1
    assert isinstance(session.agent_messages[2], ToolMessage)
    assert isinstance(session.agent_messages[3], AIMessage)
    assert extract_text_from_message_content(session.agent_messages[3].content) == expected_answer


def test_missing_final_answer_test_c(tmp_path):
    """Test C: Simulate successful SQL execution without a final AIMessage and verify safe incomplete message."""
    db_path = _make_test_db(tmp_path)
    session = create_agent_session(db_path=db_path, llm=MockScriptedLLM())

    class IncompleteGraph:
        def invoke(self, state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            msgs = list(state.get("messages", []))
            sql_id = "sql_incomplete"
            # Model calls tool, tool succeeds, but workflow terminates before producing final AIMessage
            return {
                "messages": msgs + [
                    AIMessage(content="", tool_calls=[{"name": "execute_sql", "args": {"query": "SELECT 1"}, "id": sql_id}]),
                    ToolMessage(
                        content=json.dumps({"status": "success", "dataset_id": "ds_inc", "columns": ["v"], "rows": [[999]], "row_count": 1}),
                        tool_call_id=sql_id,
                        name="execute_sql",
                    ),
                ],
                "step_count": 1,
            }

    session.graph = IncompleteGraph()
    record = execute_session_turn(session, "Compute revenue")

    # 1. Does not falsely report success via 'Analysis completed.'
    assert "Analysis completed." not in record["content"]

    # 2. A safe, sanitized incomplete-execution message is returned
    assert record["content"] == DEFAULT_INCOMPLETE_MESSAGE
    assert "unable to complete" in record["content"].lower() or "not completed" in record["content"].lower()

    # 3. Result is distinguishable from a completed answer
    assert "999" not in record["content"]  # No SQL values fabricated

    # 4. Tool execution was captured
    assert len(record["tool_executions"]) == 1
    assert record["tool_executions"][0]["status"] == "success"


def test_multi_turn_state_and_dataset_reuse_test_d(tmp_path):
    """Test D: Verify multi-turn context retention, dataset reuse across turns, and counter resets."""
    db_path = _make_test_db(tmp_path)
    session = create_agent_session(db_path=db_path, llm=MockScriptedLLM())

    captured_turn_inputs: List[Dict[str, Any]] = []

    class MultiTurnScriptedGraph:
        def __init__(self, dataset_store, chart_registry):
            self.dataset_store = dataset_store
            self.chart_registry = chart_registry
            self.turn_count = 0

        def invoke(self, state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            self.turn_count += 1
            captured_turn_inputs.append(dict(state))
            msgs = list(state.get("messages", []))

            if self.turn_count == 1:
                # Turn 1: SQL retrieves and stores dataset
                self.saved_ds_id = self.dataset_store.save({
                    "status": "success",
                    "columns": ["category", "rev"],
                    "rows": [["bed_bath", 1000], ["health", 800]],
                    "row_count": 2,
                })
                ds_id = self.saved_ds_id

                sql_id = "call_t1_sql"
                return {
                    "messages": msgs + [
                        AIMessage(content="", tool_calls=[{"name": "execute_sql", "args": {"query": "SELECT ..."}, "id": sql_id}]),
                        ToolMessage(
                            content=json.dumps({"status": "success", "dataset_id": ds_id, "columns": ["category", "rev"], "rows": [["bed_bath", 1000]], "row_count": 2}),
                            tool_call_id=sql_id,
                            name="execute_sql",
                        ),
                        AIMessage(content=[{"type": "text", "text": "Top categories are Bed & Bath and Health."}]),
                    ],
                    "step_count": 3,
                    "sql_retry_count": 1,
                }
            else:
                # Turn 2: Visualizes using existing ds_id from Turn 1
                cid = self.chart_registry.save(
                    figure={"data": [{"type": "bar", "x": ["bed_bath", "health"], "y": [1000, 800]}]},
                    dataset_id=self.saved_ds_id,
                    chart_type="bar",
                    title="Revenue by Category",
                )
                chart_id_call = "call_t2_chart"
                return {
                    "messages": msgs + [
                        AIMessage(content="", tool_calls=[{"name": "generate_chart", "args": {"dataset_id": self.saved_ds_id, "chart_type": "bar"}, "id": chart_id_call}]),
                        ToolMessage(
                            content=json.dumps({"status": "success", "chart_id": cid, "chart_type": "bar"}),
                            tool_call_id=chart_id_call,
                            name="generate_chart",
                        ),
                        AIMessage(content=[{"type": "text", "text": f"Here is your bar chart for the top categories (ID: {cid})."}]),
                    ],
                    "step_count": 2,
                    "sql_retry_count": 0,
                }

    graph_inst = MultiTurnScriptedGraph(session.dataset_store, session.chart_registry)
    session.graph = graph_inst

    # Turn 1
    r1 = execute_session_turn(session, "What are the top five categories by revenue?")
    assert r1["content"] == "Top categories are Bed & Bath and Health."
    assert "Analysis completed." not in r1["content"]

    # Verify Turn 1 dataset exists in store
    rec = session.dataset_store.get(graph_inst.saved_ds_id)
    assert rec is not None
    assert rec.row_count == 2

    # Turn 2
    r2 = execute_session_turn(session, "Visualize them.")
    assert "Here is your bar chart" in r2["content"]
    assert len(r2["chart_ids"]) == 1
    chart_id = r2["chart_ids"][0]
    assert chart_id in session.chart_registry

    # Verify both turns received their correct final outputs without duplicates
    assert len(session.ui_messages) == 4
    assert session.ui_messages[0]["content"] == "What are the top five categories by revenue?"
    assert session.ui_messages[1]["content"] == "Top categories are Bed & Bath and Health."
    assert session.ui_messages[2]["content"] == "Visualize them."
    assert "Here is your bar chart" in session.ui_messages[3]["content"]

    # Verify Turn 2 retained Turn 1 context
    turn2_input_msgs = captured_turn_inputs[1]["messages"]
    assert len(turn2_input_msgs) >= 4
    assert turn2_input_msgs[0].content == "What are the top five categories by revenue?"

    # Verify per-turn counters were reset at start of Turn 2 despite Turn 1 ending with step_count=3, sql_retry_count=1
    assert captured_turn_inputs[1]["step_count"] == 0
    assert captured_turn_inputs[1]["sql_retry_count"] == 0
    assert captured_turn_inputs[1]["python_retry_count"] == 0
    assert captured_turn_inputs[1]["pending_python_recovery"] is False

