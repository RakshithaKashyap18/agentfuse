from __future__ import annotations

import json
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


def parse_sse_response(text: str) -> tuple[int, int, tuple[ToolCall, ...]]:
    """Extract usage and tool calls from an Anthropic streaming (SSE) response body.

    input_tokens come from message_start; the final message_delta carries the
    authoritative output_tokens; tool args arrive as input_json_delta fragments
    that must be reassembled per content-block index before hashing.
    """
    tin = tout = 0
    names: dict[int, str] = {}
    start_inputs: dict[int, Any] = {}
    partials: dict[int, list[str]] = {}
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[5:].strip())
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype == "message_start":
            message = event.get("message", {})
            usage = message.get("usage", {}) if isinstance(message, dict) else {}
            if isinstance(usage, dict):
                tin = int(usage.get("input_tokens", 0))
                tout = int(usage.get("output_tokens", 0))
        elif etype == "content_block_start":
            block = event.get("content_block", {})
            if isinstance(block, dict) and block.get("type") == "tool_use":
                idx = int(event.get("index", 0))
                names[idx] = str(block.get("name", ""))
                start_inputs[idx] = block.get("input", {})
        elif etype == "content_block_delta":
            delta = event.get("delta", {})
            if isinstance(delta, dict) and delta.get("type") == "input_json_delta":
                partials.setdefault(int(event.get("index", 0)), []).append(
                    str(delta.get("partial_json", "")))
        elif etype == "message_delta":
            usage = event.get("usage", {})
            if isinstance(usage, dict) and "output_tokens" in usage:
                tout = int(usage["output_tokens"])
    calls: list[ToolCall] = []
    for idx in sorted(names):
        args: Any = start_inputs.get(idx, {})
        raw_args = "".join(partials.get(idx, ()))
        if raw_args.strip():
            try:
                args = json.loads(raw_args)
            except ValueError:
                pass
        calls.append(ToolCall(names[idx], hash_args(args)))
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
        last_block_ts=store.last_block_ts(pending.run),
    )
