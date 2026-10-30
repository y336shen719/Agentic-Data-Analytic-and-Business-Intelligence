"""Offline unit tests for LangGraph agent workflow graph (agent/graph.py).

All tests run 100% offline using deterministic fake LLMs and tools, requiring
zero network calls, API keys, or live external database connections.
"""

from __future__ import annotations

from typing import Any, List, Optional
import uuid
import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from agent.graph import DEFAULT_MAX_STEPS, create_agent_graph
from agent.prompts import SYSTEM_PROMPT
from agent.state import create_initial_state
from agent.tool_wrappers import create_sql_tools


class FakeRecordingChatModel(BaseChatModel):
    """Deterministic fake chat model that replays queued responses and records call history."""

    responses: List[AIMessage]
    index: int = 0
    bound_tools: Optional[list] = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        shared = data.get("_shared_calls")
        if shared is None:
            shared = []
        object.__setattr__(self, "_shared_calls", shared)

    @property
    def calls(self) -> List[List[BaseMessage]]:
        return getattr(self, "_shared_calls", [])

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Record a copy of messages passed into the LLM
        self.calls.append(list(messages))
        if self.index < len(self.responses):
            resp = self.responses[self.index]
            self.index += 1
        else:
            resp = self.responses[-1]

        fresh_msg = AIMessage(
            content=resp.content,
            tool_calls=[dict(tc) for tc in resp.tool_calls] if hasattr(resp, "tool_calls") and resp.tool_calls else [],
            id=resp.id if resp.id else str(uuid.uuid4()),
        )
        return ChatResult(generations=[ChatGeneration(message=fresh_msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeRecordingChatModel:
        # Create a bound replica with shared history tracking
        bound = FakeRecordingChatModel(
            responses=self.responses,
            index=self.index,
            bound_tools=list(tools),
            _shared_calls=self.calls,
        )
        return bound

    @property
    def _llm_type(self) -> str:
        return "fake-recording-chat-model"


class LoopingChatModel(BaseChatModel):
    """Fake model that perpetually requests tool execution to test execution limits."""

    tool_name: str = "dummy_calculator"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        call = {
            "name": self.tool_name,
            "args": {"x": 1, "y": 1},
            "id": f"loop_{uuid.uuid4().hex[:8]}",
        }
        msg = AIMessage(content="Requesting another calculation...", tool_calls=[call], id=str(uuid.uuid4()))
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> LoopingChatModel:
        return self

    @property
    def _llm_type(self) -> str:
        return "looping-chat-model"


@tool
def dummy_calculator(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


@tool
def dummy_failing_tool() -> str:
    """A tool that raises an unexpected runtime exception."""
    raise RuntimeError("Simulated tool crash.")


class TestMaxStepsValidation:
    """Tests for max_steps input validation in create_agent_graph."""

    @pytest.mark.parametrize("invalid_val", [0, -1, -10])
    def test_zero_or_negative_max_steps_raises_value_error(self, invalid_val: Any) -> None:
        """Reject non-positive integers."""
        fake_llm = FakeRecordingChatModel(responses=[AIMessage(content="Hi")])
        with pytest.raises(ValueError, match="max_steps must be a strictly positive integer"):
            create_agent_graph(llm=fake_llm, tools=[dummy_calculator], max_steps=invalid_val)

    @pytest.mark.parametrize("bool_val", [True, False])
    def test_boolean_max_steps_raises_value_error(self, bool_val: Any) -> None:
        """Reject booleans even though bool is a subclass of int in Python."""
        fake_llm = FakeRecordingChatModel(responses=[AIMessage(content="Hi")])
        with pytest.raises(ValueError, match="max_steps must be a strictly positive integer"):
            create_agent_graph(llm=fake_llm, tools=[dummy_calculator], max_steps=bool_val)

    @pytest.mark.parametrize("non_int_val", ["12", 3.14, None, [], {}])
    def test_non_integer_max_steps_raises_value_error(self, non_int_val: Any) -> None:
        """Reject non-integer types."""
        fake_llm = FakeRecordingChatModel(responses=[AIMessage(content="Hi")])
        with pytest.raises(ValueError, match="max_steps must be a strictly positive integer"):
            create_agent_graph(llm=fake_llm, tools=[dummy_calculator], max_steps=non_int_val)

    @pytest.mark.parametrize("valid_val", [1, 2, 5, 12, 100])
    def test_strictly_positive_integer_accepted(self, valid_val: int) -> None:
        """Accept positive integers."""
        fake_llm = FakeRecordingChatModel(responses=[AIMessage(content="Hi")])
        graph = create_agent_graph(llm=fake_llm, tools=[dummy_calculator], max_steps=valid_val)
        assert graph is not None


class TestAgentGraphCompilationAndBinding:
    """Tests for graph compilation, default parameters, and tool binding."""

    def test_graph_compilation_with_injected_components(self) -> None:
        """Graph compiles cleanly and returns a CompiledStateGraph instance."""
        fake_llm = FakeRecordingChatModel(responses=[AIMessage(content="Finished")])
        graph = create_agent_graph(llm=fake_llm, tools=[dummy_calculator])

        assert graph is not None
        assert hasattr(graph, "invoke")

    def test_tool_binding_registers_approved_tools_only(self) -> None:
        """LLM receives only the approved tools collection."""
        fake_llm = FakeRecordingChatModel(responses=[AIMessage(content="Done")])
        graph = create_agent_graph(llm=fake_llm, tools=[dummy_calculator])

        state = create_initial_state(messages=[HumanMessage(content="Hi")])
        graph.invoke(state)

        # Verify LLM was invoked and received the system prompt and user message
        assert len(fake_llm.calls) == 1
        assert isinstance(fake_llm.calls[0][0], SystemMessage)
        assert fake_llm.calls[0][0].content == SYSTEM_PROMPT
        assert isinstance(fake_llm.calls[0][1], HumanMessage)


class TestAgentGraphWorkflowExecution:
    """Tests for conditional routing, tool execution, and cycle completions."""

    def test_direct_answer_without_tools_routes_to_end(self) -> None:
        """When LLM returns an answer with no tool calls, graph completes immediately."""
        fake_llm = FakeRecordingChatModel(
            responses=[AIMessage(content="Direct answer with no tools needed.")]
        )
        graph = create_agent_graph(llm=fake_llm, tools=[dummy_calculator])

        state = create_initial_state(messages=[HumanMessage(content="Tell me a joke")])
        final_state = graph.invoke(state)

        # 1 user message + 1 AI response
        assert len(final_state["messages"]) == 2
        assert final_state["messages"][-1].content == "Direct answer with no tools needed."
        assert final_state["step_count"] == 1

    def test_complete_llm_tool_llm_cycle(self) -> None:
        """Verify LLM -> ToolNode -> LLM -> END lifecycle."""
        call_id = "calc_call_101"
        tool_call = {
            "name": "dummy_calculator",
            "args": {"x": 15, "y": 27},
            "id": call_id,
        }
        resp1 = AIMessage(content="", tool_calls=[tool_call])
        resp2 = AIMessage(content="The calculation result is 42.")

        fake_llm = FakeRecordingChatModel(responses=[resp1, resp2])
        graph = create_agent_graph(llm=fake_llm, tools=[dummy_calculator])

        initial_state = create_initial_state(messages=[HumanMessage(content="What is 15 + 27?")])
        final_state = graph.invoke(initial_state)

        assert len(final_state["messages"]) == 4
        assert isinstance(final_state["messages"][0], HumanMessage)
        assert isinstance(final_state["messages"][1], AIMessage)
        assert isinstance(final_state["messages"][2], ToolMessage)
        assert isinstance(final_state["messages"][3], AIMessage)

        tool_msg = final_state["messages"][2]
        assert tool_msg.tool_call_id == call_id
        assert tool_msg.content == "42"

        assert final_state["step_count"] == 2
        assert final_state["messages"][-1].content == "The calculation result is 42."

    def test_multiple_sequential_tool_calls(self) -> None:
        """Verify multi-step iterative tool calling across 2 rounds."""
        call1 = {"name": "dummy_calculator", "args": {"x": 5, "y": 5}, "id": "c1"}
        call2 = {"name": "dummy_calculator", "args": {"x": 10, "y": 20}, "id": "c2"}

        resp1 = AIMessage(content="", tool_calls=[call1])
        resp2 = AIMessage(content="", tool_calls=[call2])
        resp3 = AIMessage(content="Total is 30.")

        fake_llm = FakeRecordingChatModel(responses=[resp1, resp2, resp3])
        graph = create_agent_graph(llm=fake_llm, tools=[dummy_calculator])

        state = create_initial_state(messages=[HumanMessage(content="Calculate in steps")])
        final_state = graph.invoke(state)

        # Expected: HumanMessage, AIMessage, ToolMessage, AIMessage, ToolMessage, AIMessage
        assert len(final_state["messages"]) == 6
        assert final_state["step_count"] == 3
        assert final_state["messages"][-1].content == "Total is 30."

    def test_safe_handling_of_unauthorized_tool_name(self) -> None:
        """When LLM requests an unapproved tool name, ToolNode rejects safely without crashing."""
        call_bad = {"name": "unauthorized_delete_db", "args": {}, "id": "bad_1"}
        resp1 = AIMessage(content="", tool_calls=[call_bad])
        resp2 = AIMessage(content="I apologize, that tool is not available.")

        fake_llm = FakeRecordingChatModel(responses=[resp1, resp2])
        graph = create_agent_graph(llm=fake_llm, tools=[dummy_calculator])

        state = create_initial_state(messages=[HumanMessage(content="Delete data")])
        final_state = graph.invoke(state)

        # ToolMessage must record the error feedback
        tool_msg = final_state["messages"][2]
        assert isinstance(tool_msg, ToolMessage)
        assert "not a valid tool" in str(tool_msg.content)
        assert tool_msg.status == "error"

        # Graph cleanly concludes
        assert final_state["messages"][-1].content == "I apologize, that tool is not available."

    def test_tool_execution_exception_handled_gracefully(self) -> None:
        """When a tool raises an unhandled error, ToolNode catches it and returns error ToolMessage."""
        call_fail = {"name": "dummy_failing_tool", "args": {}, "id": "fail_1"}
        resp1 = AIMessage(content="", tool_calls=[call_fail])
        resp2 = AIMessage(content="I observed the tool error and stopped.")

        fake_llm = FakeRecordingChatModel(responses=[resp1, resp2])
        graph = create_agent_graph(llm=fake_llm, tools=[dummy_failing_tool])

        state = create_initial_state(messages=[HumanMessage(content="Trigger crash")])
        final_state = graph.invoke(state)

        tool_msg = final_state["messages"][2]
        assert isinstance(tool_msg, ToolMessage)
        assert "RuntimeError" in str(tool_msg.content)
        assert tool_msg.status == "error"
        assert final_state["messages"][-1].content == "I observed the tool error and stopped."


class TestExecutionLimitAndPendingToolCalls:
    """Targeted tests for max_steps enforcement, pending tool calls, and message ordering."""

    def test_normal_final_answer_at_exact_step_limit_no_warning(self) -> None:
        """If LLM reaches final answer on the last permitted step, terminate cleanly without warning."""
        resp1 = AIMessage(
            content="",
            tool_calls=[{"name": "dummy_calculator", "args": {"x": 2, "y": 3}, "id": "c1"}],
        )
        resp2 = AIMessage(content="The final computed answer is 5.")

        fake_llm = FakeRecordingChatModel(responses=[resp1, resp2])
        # Limit is exactly 2 LLM steps
        graph = create_agent_graph(llm=fake_llm, tools=[dummy_calculator], max_steps=2)

        state = create_initial_state(messages=[HumanMessage(content="Calculate 2 + 3")])
        final_state = graph.invoke(state)

        assert final_state["step_count"] == 2
        last_msg = final_state["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert last_msg.content == "The final computed answer is 5."
        # No warning message appended
        assert "Execution stopped: maximum" not in last_msg.content

    def test_execution_limit_reached_with_one_pending_tool_call(self) -> None:
        """When limit is reached on an AIMessage with 1 tool call, generate 1 ToolMessage and warning."""
        pending_id = "pending_call_001"
        tool_call = {"name": "dummy_calculator", "args": {"x": 10, "y": 20}, "id": pending_id}
        resp = AIMessage(content="Let me calculate this...", tool_calls=[tool_call])

        fake_llm = FakeRecordingChatModel(responses=[resp])
        graph = create_agent_graph(llm=fake_llm, tools=[dummy_calculator], max_steps=1)

        state = create_initial_state(messages=[HumanMessage(content="Run calc")])
        final_state = graph.invoke(state)

        messages = final_state["messages"]
        # Expected ordering: HumanMessage, AIMessage, ToolMessage, AIMessage(warning)
        assert len(messages) == 4
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert isinstance(messages[2], ToolMessage)
        assert isinstance(messages[3], AIMessage)

        # Matching tool_call_id
        tool_msg = messages[2]
        assert tool_msg.tool_call_id == pending_id
        assert tool_msg.name == "dummy_calculator"
        assert "was not executed because the workflow reached" in tool_msg.content
        assert tool_msg.status == "error"

        # Final explanation AIMessage
        warning_msg = messages[3]
        assert "Execution stopped: maximum LLM invocation limit (1)" in warning_msg.content

    def test_execution_limit_reached_with_multiple_pending_tool_calls(self) -> None:
        """When limit is reached on an AIMessage with multiple tool calls, generate ToolMessage for each."""
        tc1 = {"name": "dummy_calculator", "args": {"x": 1, "y": 2}, "id": "multi_1"}
        tc2 = {"name": "dummy_calculator", "args": {"x": 3, "y": 4}, "id": "multi_2"}
        tc3 = {"name": "dummy_calculator", "args": {"x": 5, "y": 6}, "id": "multi_3"}

        resp = AIMessage(content="Executing multiple steps...", tool_calls=[tc1, tc2, tc3])
        fake_llm = FakeRecordingChatModel(responses=[resp])
        graph = create_agent_graph(llm=fake_llm, tools=[dummy_calculator], max_steps=1)

        state = create_initial_state(messages=[HumanMessage(content="Run multi")])
        final_state = graph.invoke(state)

        messages = final_state["messages"]
        # HumanMessage + AIMessage + 3 ToolMessages + 1 warning AIMessage = 6 messages
        assert len(messages) == 6
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert isinstance(messages[2], ToolMessage)
        assert isinstance(messages[3], ToolMessage)
        assert isinstance(messages[4], ToolMessage)
        assert isinstance(messages[5], AIMessage)

        assert messages[2].tool_call_id == "multi_1"
        assert messages[3].tool_call_id == "multi_2"
        assert messages[4].tool_call_id == "multi_3"

        for tm in [messages[2], messages[3], messages[4]]:
            assert tm.status == "error"
            assert "was not executed because the workflow reached" in tm.content

        assert "maximum LLM invocation limit (1)" in messages[5].content

    def test_no_pending_tools_are_executed_after_limit_reached(self) -> None:
        """Verify that ToolNode never executes pending tools once step limit is reached."""
        spy_executed = []

        @tool
        def spy_tool(val: int) -> int:
            """Spy tool that records executions."""
            spy_executed.append(val)
            return val * 2

        tc = {"name": "spy_tool", "args": {"val": 99}, "id": "spy_call_1"}
        resp = AIMessage(content="Calling spy...", tool_calls=[tc])

        fake_llm = FakeRecordingChatModel(responses=[resp])
        graph = create_agent_graph(llm=fake_llm, tools=[spy_tool], max_steps=1)

        state = create_initial_state(messages=[HumanMessage(content="Call spy tool")])
        final_state = graph.invoke(state)

        # Tool function must NEVER have been called
        assert len(spy_executed) == 0

        # But a ToolMessage was synthesized
        tool_msgs = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "spy_call_1"

    def test_initial_state_already_at_or_above_limit(self) -> None:
        """If state enters graph already at or above limit, LLM is never invoked."""
        fake_llm = FakeRecordingChatModel(responses=[AIMessage(content="Should not be called")])
        graph = create_agent_graph(llm=fake_llm, tools=[dummy_calculator], max_steps=3)

        initial_state = create_initial_state(
            messages=[HumanMessage(content="Initial query")],
            step_count=3,
        )
        final_state = graph.invoke(initial_state)

        # LLM was never called
        assert len(fake_llm.calls) == 0
        assert final_state["step_count"] == 3

        # Final message explains limit was reached
        assert len(final_state["messages"]) == 2
        assert isinstance(final_state["messages"][-1], AIMessage)
        assert "maximum LLM invocation limit (3)" in final_state["messages"][-1].content

    def test_llm_invocation_count_strictly_bounded(self) -> None:
        """In an infinite tool calling loop, LLM invocations equal exactly max_steps."""
        max_steps = 4
        looping_llm = LoopingChatModel()
        graph = create_agent_graph(llm=looping_llm, tools=[dummy_calculator], max_steps=max_steps)

        state = create_initial_state(messages=[HumanMessage(content="Infinite loop test")])
        final_state = graph.invoke(state)

        assert final_state["step_count"] == max_steps
        assert "maximum LLM invocation limit (4)" in final_state["messages"][-1].content

    def test_state_isolation_between_invocations(self) -> None:
        """Successive graph invocations do not leak messages or step counts."""
        fake_llm = FakeRecordingChatModel(responses=[AIMessage(content="Answer")])
        graph = create_agent_graph(llm=fake_llm, tools=[dummy_calculator])

        state1 = create_initial_state(messages=[HumanMessage(content="Question 1")])
        res1 = graph.invoke(state1)

        state2 = create_initial_state(messages=[HumanMessage(content="Question 2")])
        res2 = graph.invoke(state2)

        assert len(res1["messages"]) == 2
        assert res1["messages"][0].content == "Question 1"

        assert len(res2["messages"]) == 2
        assert res2["messages"][0].content == "Question 2"
        assert res1["step_count"] == 1
        assert res2["step_count"] == 1

    def test_last_sql_is_tracked_when_sql_tool_called(self) -> None:
        """When execute_sql is called, last_sql is updated in state."""
        sql_call = {
            "name": "execute_sql",
            "args": {"query": "SELECT COUNT(*) FROM orders;"},
            "id": "sql_1",
        }
        resp1 = AIMessage(content="", tool_calls=[sql_call])
        resp2 = AIMessage(content="Found 99441 orders.")

        fake_llm = FakeRecordingChatModel(responses=[resp1, resp2])

        @tool
        def execute_sql(query: str) -> dict:
            """Fake offline SQL tool."""
            return {
                "status": "success",
                "columns": ["n"],
                "rows": [[99441]],
                "row_count": 1,
                "truncated": False,
                "execution_time_ms": 1.0,
            }

        graph = create_agent_graph(llm=fake_llm, tools=[execute_sql])

        state = create_initial_state(messages=[HumanMessage(content="Count orders")])
        final_state = graph.invoke(state)

        assert final_state["last_sql"] == "SELECT COUNT(*) FROM orders;"
        assert final_state["step_count"] == 2
