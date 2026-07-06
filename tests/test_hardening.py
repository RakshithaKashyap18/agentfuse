import json

import httpx
from click.testing import CliRunner
from fastapi import FastAPI

from agentfuse.config import FuseConfig
from agentfuse.models import CallEvent, ToolCall, ToolResult
from agentfuse.store import Store
from tests.test_server import call, make_client, tool_use_response


def ev(i: int, ts: float) -> CallEvent:
    return CallEvent(f"e{i}", ts, "a1", "r1", "m", 10, 10, 1.0,
                     (ToolCall("t", "h"),), (ToolResult(False),), 5.0)


# --- API auth ---

def secured_client() -> object:
    cfg = FuseConfig(db_path=":memory:", api_token="s3cret")
    return make_client([tool_use_response("q")] * 5, cfg)


def test_api_requires_token_when_configured() -> None:
    client = secured_client()
    assert client.get("/api/status").status_code == 401  # type: ignore[attr-defined]
    ok = client.get("/api/status",  # type: ignore[attr-defined]
                    headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
    assert client.get("/api/status?token=s3cret").status_code == 200  # type: ignore[attr-defined]
    assert client.post("/api/runs/r1/kill").status_code == 401  # type: ignore[attr-defined]


def test_proxy_route_unaffected_by_api_token() -> None:
    client = secured_client()
    assert call(client).status_code == 200  # type: ignore[arg-type]


def test_api_open_when_no_token_configured() -> None:
    client = make_client([tool_use_response("q")])
    assert client.get("/api/status").status_code == 200


# --- upstream error visibility ---

def test_upstream_5xx_passes_through_and_records_incident() -> None:
    app = FastAPI()

    @app.post("/v1/messages")
    async def messages() -> object:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content={"type": "error"})

    from fastapi.testclient import TestClient

    from agentfuse.server import create_app
    upstream = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://fake")
    client = TestClient(create_app(FuseConfig(db_path=":memory:"), upstream_client=upstream))
    r = call(client)
    assert r.status_code == 503  # passed through unchanged
    incidents = client.get("/api/status").json()["incidents"]
    assert any(i["policy"] == "upstream" for i in incidents)


def test_upstream_unreachable_returns_502_and_incident() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    from fastapi.testclient import TestClient

    from agentfuse.server import create_app
    upstream = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://fake")
    client = TestClient(create_app(FuseConfig(db_path=":memory:"), upstream_client=upstream))
    r = call(client)
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "upstream_error"
    incidents = client.get("/api/status").json()["incidents"]
    assert any(i["policy"] == "upstream" for i in incidents)


# --- retention ---

def test_store_prune_removes_old_rows() -> None:
    s = Store(":memory:")
    s.add_event(ev(1, ts=100.0))
    s.add_event(ev(2, ts=200.0))
    s.prune(before_ts=150.0)
    assert len(s.events_for_run("r1")) == 1


def test_retention_days_config() -> None:
    assert FuseConfig().retention_days == 0  # keep forever by default


# --- degraded store fallback ---

def test_store_write_failure_buffers_and_flags() -> None:
    s = Store(":memory:")
    s._conn.close()  # simulate storage death
    s.add_event(ev(1, ts=1.0))  # must not raise
    assert s.degraded is True


def test_proxy_survives_dead_store() -> None:
    client = make_client([tool_use_response("q")] * 3)
    app_store = client.app.state.store  # type: ignore[attr-defined]
    app_store._conn.close()
    assert call(client).status_code == 200  # fail-open: still forwarded
    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["degraded"] is True


# --- fuse export ---

def test_export_cli_writes_json_lines(tmp_path: object) -> None:
    from pathlib import Path

    from agentfuse.cli import main
    db = str(Path(str(tmp_path)) / "fuse.db")
    s = Store(db)
    s.add_event(ev(7, ts=42.0))
    s.close()
    result = CliRunner().invoke(main, ["export", "--db", db])
    assert result.exit_code == 0
    row = json.loads(result.output.splitlines()[0])
    assert row["id"] == "e7" and row["run"] == "r1"
