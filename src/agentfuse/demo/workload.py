from __future__ import annotations

from typing import Any

import httpx


async def run_workload(client: httpx.AsyncClient, calls: int = 15) -> dict[str, int]:
    """Drive a scripted 'researcher' agent through the proxy until done or budget."""
    counts = {"ok": 0, "blocked": 0}
    messages: list[dict[str, Any]] = [{"role": "user", "content": "research agent frameworks"}]
    for _ in range(calls):
        resp = await client.post(
            "/anthropic/v1/messages",
            json={"model": "claude-haiku-4-5", "messages": messages},
            headers={"X-Fuse-Agent": "researcher", "X-Fuse-Run": "demo-run"},
        )
        if resp.status_code == 429:
            counts["blocked"] += 1
            guidance = resp.json()["error"]
            # The agent "reads" the block message and feeds it back into context:
            messages.append({"role": "user",
                             "content": f"[agentfuse_blocked] {guidance['message']}"})
            continue
        counts["ok"] += 1
        body = resp.json()
        content = body.get("content", [])
        messages.append({"role": "assistant", "content": content})
        tool_uses = [b for b in content if b.get("type") == "tool_use"]
        if not tool_uses:
            break  # final text answer — run complete
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tool_uses[0]["id"],
             "content": "no results", "is_error": False}]})
    return counts
