from typing import Any

from agentfuse.models import PendingCall, ToolResult, hash_args
from agentfuse.parsing import build_window, parse_request, parse_response
from agentfuse.store import Store


def anthropic_request() -> dict[str, Any]:
    return {
        "model": "claude-haiku-4-5",
        "messages": [
            {"role": "user", "content": "find agent frameworks"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "search",
                 "input": {"query": "agent frameworks"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1",
                 "content": "timeout", "is_error": True}]},
        ],
    }


def anthropic_response() -> dict[str, Any]:
    return {
        "content": [
            {"type": "text", "text": "let me search"},
            {"type": "tool_use", "id": "t2", "name": "search",
             "input": {"query": "agent frameworks"}},
        ],
        "usage": {"input_tokens": 500, "output_tokens": 60},
    }


def test_parse_request_takes_only_final_message_results() -> None:
    model, results = parse_request(anthropic_request())
    assert model == "claude-haiku-4-5"
    assert results == (ToolResult(True),)


def test_parse_request_string_content_yields_no_results() -> None:
    _, results = parse_request({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert results == ()


def test_parse_response_extracts_usage_and_tool_calls() -> None:
    tin, tout, calls = parse_response(anthropic_response())
    assert (tin, tout) == (500, 60)
    assert calls[0].name == "search"
    assert calls[0].args_hash == hash_args({"query": "agent frameworks"})


def test_build_window_aggregates_from_store() -> None:
    s = Store(":memory:")
    pending = PendingCall("a1", "r1", "m", 1000.0, ())
    w = build_window(s, pending)
    assert w.events == () and w.run_spend == 0.0 and w.agent_calls_last_minute == 0
