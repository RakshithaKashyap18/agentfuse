from __future__ import annotations

from typing import Any

from agentfuse.models import PendingCall, ToolCall, ToolResult, Window, hash_args
from agentfuse.store import Store

DAY_SECONDS = 86400.0


def parse_request(body: dict[str, Any]) -> tuple[str, tuple[ToolResult, ...]]:
    model = str(body.get("model", "unknown"))
    messages = body.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return model, ()
    content = messages[-1].get("content") if isinstance(messages[-1], dict) else None
    results: list[ToolResult] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                results.append(ToolResult(bool(block.get("is_error", False))))
    return model, tuple(results)


def parse_response(body: dict[str, Any]) -> tuple[int, int, tuple[ToolCall, ...]]:
    usage = body.get("usage", {}) if isinstance(body.get("usage"), dict) else {}
    tin = int(usage.get("input_tokens", 0))
    tout = int(usage.get("output_tokens", 0))
    calls: list[ToolCall] = []
    content = body.get("content", [])
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                calls.append(ToolCall(str(block.get("name", "")),
                                      hash_args(block.get("input", {}))))
    return tin, tout, tuple(calls)


def day_start_ts(now: float) -> float:
    return now - (now % DAY_SECONDS)


def build_window(store: Store, pending: PendingCall) -> Window:
    return Window(
        pending=pending,
        events=store.events_for_run(pending.run),
        agent_calls_last_minute=store.agent_calls_since(pending.agent, pending.ts - 60.0),
        run_spend=store.run_spend(pending.run),
        agent_spend_today=store.agent_spend_since(pending.agent, day_start_ts(pending.ts)),
    )
