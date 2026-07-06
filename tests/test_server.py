from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentfuse.config import FuseConfig
from agentfuse.server import create_app


def fake_upstream(responses: list[dict[str, Any]]) -> FastAPI:
    app = FastAPI()
    state = {"i": 0}

    @app.post("/v1/messages")
    async def messages() -> dict[str, Any]:
        body = responses[min(state["i"], len(responses) - 1)]
        state["i"] += 1
        return body

    return app


def tool_use_response(query: str) -> dict[str, Any]:
    return {"content": [{"type": "tool_use", "id": "t", "name": "search",
                         "input": {"query": query}}],
            "usage": {"input_tokens": 100, "output_tokens": 50}}


def make_client(responses: list[dict[str, Any]], cfg: FuseConfig | None = None) -> TestClient:
    cfg = cfg or FuseConfig(db_path=":memory:", webhook_url="")
    upstream = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake_upstream(responses)),
        base_url="https://fake-upstream",
    )
    return TestClient(create_app(cfg, upstream_client=upstream))


def call(client: TestClient, run: str = "r1", agent: str = "a1") -> httpx.Response:
    return client.post(
        "/anthropic/v1/messages",
        json={"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "go"}]},
        headers={"X-Fuse-Agent": agent, "X-Fuse-Run": run},
    )


def test_forwards_and_meters() -> None:
    client = make_client([tool_use_response("q")])
    r = call(client)
    assert r.status_code == 200
    status = client.get("/api/status").json()
    assert status["spend_by_agent"]["a1"] > 0


def test_loop_breaker_trips_end_to_end() -> None:
    # identical tool_use every turn -> warn at 4, block at 6; a block resets the
    # streak (the run can heal), so relentless looping re-trips the breaker
    client = make_client([tool_use_response("same")] * 20)
    responses = [call(client) for _ in range(15)]
    blocked = [r for r in responses if r.status_code == 429]
    assert len(blocked) >= 2
    body = blocked[0].json()
    assert body["type"] == "error"
    assert "search" in body["error"]["message"]  # model-readable guidance
    incidents = client.get("/api/status").json()["incidents"]
    assert any(i["policy"] == "loop" for i in incidents)


def test_loop_breaker_ignores_volatile_args() -> None:
    # same query every call, but a changing timestamp tries to disguise the loop
    responses: list[dict[str, Any]] = [
        {"content": [{"type": "tool_use", "id": "t", "name": "search",
                      "input": {"query": "same", "timestamp": i}}],
         "usage": {"input_tokens": 100, "output_tokens": 50}}
        for i in range(12)
    ]
    client = make_client(responses)
    codes = [call(client).status_code for _ in range(8)]
    assert 429 in codes


def test_kill_and_reset_endpoints() -> None:
    client = make_client([tool_use_response("q")] * 5)
    assert call(client).status_code == 200
    client.post("/api/runs/r1/kill")
    assert call(client).status_code == 429
    client.post("/api/runs/r1/reset")
    assert call(client).status_code == 200


def test_status_includes_call_rate_history() -> None:
    client = make_client([tool_use_response("q")] * 3)
    for _ in range(3):
        call(client)
    cpm = client.get("/api/status").json()["calls_per_minute"]
    assert sum(row["calls"] for row in cpm) == 3


def test_default_identity_headers() -> None:
    client = make_client([tool_use_response("q")])
    r = client.post("/anthropic/v1/messages",
                    json={"model": "m", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200
    assert "default" in client.get("/api/status").json()["spend_by_agent"]
