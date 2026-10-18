"""Unit tests for Stateful Agent Loop Circuit Breaker."""

from app.guardrails.loop_detector import AgentLoopCircuitBreaker
from app.models import ChatMessage, FunctionCallSpec, ToolCallSpec


def test_detects_repeated_tool_calls_and_intervenes():
    breaker = AgentLoopCircuitBreaker(repeat_trigger=3)
    tc = ToolCallSpec(
        id="call_1",
        function=FunctionCallSpec(name="run_sql", arguments='{"query": "SELECT * FROM users"}'),
    )
    messages = [
        ChatMessage(role="user", content="Find user record"),
        ChatMessage(role="assistant", content="", tool_calls=[tc]),
        ChatMessage(role="assistant", content="", tool_calls=[tc]),
        ChatMessage(role="assistant", content="", tool_calls=[tc]),
    ]

    updated_msgs, meta = breaker.inspect_and_intervene(messages, session_id="sess_1")
    assert meta.loop_detected is True
    assert meta.repeat_count >= 3
    assert meta.action_taken == "injected_system_directive_and_escalated"
    assert len(updated_msgs) == len(messages) + 1
    assert "VERIROUTE CIRCUIT BREAKER ACTIVATED" in (updated_msgs[-1].content or "")
