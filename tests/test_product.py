from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentfuse.config import FuseConfig
from agentfuse.models import Action, PendingCall, ToolResult, Window, hash_args
from agentfuse.parsing import parse_openai_request, parse_openai_response
from agentfuse.policies.budget import BudgetBreaker
from agentfuse.server import create_app
from tests.test_server import call, make_client, tool_use_response


# --- OpenAI parsing ---

def test_parse_openai_request_trailing_tool_messages() -> None:
    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "tool_calls": [{"id": "t1"}]},
            {"role": "tool", "tool_call_id": "t1", "content": "result"},
        ],
    }
    model, results = parse_openai_request(body)
    assert model == "gpt-4o"
    assert results == (ToolResult(False),)


def test_parse_openai_response_usage_and_tool_calls() -> None:
    body = {
        "choices": [{"message": {
            "tool_calls": [{"type": "function", "function":
                            {"name": "search", "arguments": '{"query": "x"}'}}],
        }}],
        "usage": {"prompt_tokens": 200, "completion_tokens": 30},
    }
    tin, tout, calls = parse_openai_response(body)
    assert (tin, tout) == (200, 30)
    assert calls[0].name == "search"
    assert calls[0].args_hash == hash_args({"query": "x"})


# --- OpenAI endpoint E2E ---

def openai_upstream(responses: list[dict[str, Any]]) -> FastAPI:
    app = FastAPI()
    state = {"i": 0}

    @app.post("/v1/chat/completions")
    async def completions() -> dict[str, Any]:
        body = responses[min(state["i"], len(responses) - 1)]
        state["i"] += 1
        return body

    return app


def openai_tool_response(query: str) -> dict[str, Any]:
    return {"choices": [{"message": {"tool_calls": [
                {"type": "function",
                 "function": {"name": "search", "arguments": f'{{"query": "{query}"}}'}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50}}


def make_openai_client(responses: list[dict[str, Any]]) -> TestClient:
    upstream = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=openai_upstream(responses)),
        base_url="https://fake-openai")
    return TestClient(create_app(FuseConfig(db_path=":memory:"),
                                 openai_upstream_client=upstream))


def openai_call(client: TestClient, run: str = "r1") -> httpx.Response:
    return client.post(
        "/openai/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "go"}]},
        headers={"X-Fuse-Agent": "a1", "X-Fuse-Run": run})


def test_openai_route_forwards_and_meters() -> None:
    client = make_openai_client([openai_tool_response("q")])
    assert openai_call(client).status_code == 200
    assert client.get("/api/status").json()["spend_by_agent"]["a1"] > 0


def test_openai_route_trips_loop_breaker() -> None:
    client = make_openai_client([openai_tool_response("same")] * 12)
    codes = [openai_call(client).status_code for _ in range(8)]
    assert 429 in codes


# --- per-agent budgets ---

def test_budget_per_agent_override() -> None:
    b = BudgetBreaker(None, 50.0, {"coder": 1.0})
    pending_coder = PendingCall("coder", "r1", "m", 0.0, ())
    w = Window(pending_coder, (), 0, 0.0, 2.0)  # coder spent $2 of $1 cap
    assert b.evaluate(w).action is Action.BLOCK
    pending_other = PendingCall("other", "r1", "m", 0.0, ())
    w2 = Window(pending_other, (), 0, 0.0, 2.0)  # others get the $50 default
    assert b.evaluate(w2).action is Action.ALLOW


def test_budget_agents_config(tmp_path: Any) -> None:
    from pathlib import Path

    from agentfuse.config import load_config
    p = Path(str(tmp_path)) / "fuse.toml"
    p.write_text('[budget.agents]\ncoder = 1.5\n')
    assert dict(load_config(p).budget_per_agent) == {"coder": 1.5}


# --- config hot-reload ---

def test_config_endpoint_updates_policies_live() -> None:
    client = make_client([tool_use_response(f"q{i}") for i in range(9)])
    assert call(client).status_code == 200
    r = client.post("/api/config", json={"rate_calls_per_minute": 1})
    assert r.status_code == 200
    assert client.get("/api/config").json()["rate_calls_per_minute"] == 1
    assert call(client).status_code == 429  # cap of 1 already used


def test_config_endpoint_rejects_unknown_keys() -> None:
    client = make_client([tool_use_response("q")])
    assert client.post("/api/config", json={"db_path": "hack"}).status_code == 400


# --- dashboard data ---

def test_calls_per_minute_includes_spend() -> None:
    client = make_client([tool_use_response("q")])
    call(client)
    (row,) = client.get("/api/status").json()["calls_per_minute"]
    assert row["calls"] == 1 and row["spend"] > 0


# --- richer demo ---

def test_demo_also_trips_budget_spiral() -> None:
    from agentfuse.demo import run_demo_inprocess
    result = run_demo_inprocess()
    assert result["spiral"]["blocked"] >= 1
    policies = {i["policy"] for i in result["incidents"]}
    assert "loop" in policies and "budget" in policies
