from __future__ import annotations

import logging
import math
import time
import uuid
from importlib import resources
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from agentfuse.alerting import Alerter
from agentfuse.config import FuseConfig
from agentfuse.engine import PolicyEngine
from agentfuse.meter import cost_usd
from agentfuse.models import Action, CallEvent, PendingCall
from agentfuse.parsing import build_window, parse_request, parse_response, parse_sse_response
from agentfuse.policies import default_policies
from agentfuse.store import Store
from agentfuse.streaming import Broadcaster

log = logging.getLogger("agentfuse")
_HOP_HEADERS = {"host", "content-length", "x-fuse-agent", "x-fuse-run"}


def create_app(cfg: FuseConfig, upstream_client: httpx.AsyncClient | None = None) -> FastAPI:
    app = FastAPI(title="AgentFuse")
    app.state.cfg = cfg
    app.state.store = Store(cfg.db_path)
    app.state.engine = PolicyEngine(default_policies(cfg))
    app.state.broadcaster = Broadcaster()
    app.state.default_run = uuid.uuid4().hex[:8]
    client = upstream_client or httpx.AsyncClient(
        base_url=cfg.upstream_anthropic, timeout=120.0
    )
    app.state.alerter = Alerter(cfg.webhook_url, cfg.cooldown_seconds, client)

    @app.post("/anthropic/v1/messages")
    async def proxy_messages(request: Request) -> Response:
        raw = await request.body()
        started = time.time()
        agent = request.headers.get("x-fuse-agent", "default")
        run = request.headers.get("x-fuse-run", app.state.default_run)
        try:
            body: dict[str, Any] = await request.json()
        except Exception:
            body = {}
        model, tool_results = parse_request(body)
        pending = PendingCall(agent, run, model, started, tool_results)
        verdict = app.state.engine.check(build_window(app.state.store, pending))
        if verdict.action >= Action.WARN:
            app.state.store.add_incident(started, run, agent, verdict)
            await app.state.alerter.maybe_fire(verdict, pending)
            await app.state.broadcaster.publish(
                {"kind": "incident", "run": run, "agent": agent,
                 "policy": verdict.policy, "action": verdict.action.name,
                 "message": verdict.message})
        if verdict.action >= Action.BLOCK:
            headers: dict[str, str] = {}
            if verdict.policy == "rate":
                # seconds until the oldest call in the rolling minute ages out
                oldest = app.state.store.oldest_call_ts_since(agent, started - 60.0)
                wait = math.ceil(oldest + 60.0 - started) if oldest > 0 else 60
                headers["Retry-After"] = str(min(60, max(1, wait)))
            return JSONResponse(status_code=429, headers=headers, content={
                "type": "error",
                "error": {"type": "agentfuse_blocked", "message": verdict.message},
            })
        if body.get("stream") is True:
            return await _relay_stream(request, raw, pending)
        upstream = await client.post(
            "/v1/messages", content=raw, headers=_forward_headers(request))
        if upstream.status_code == 200:
            try:
                tin, tout, tool_calls = parse_response(upstream.json(), cfg.loop_volatile_keys)
                event = CallEvent(
                    uuid.uuid4().hex, started, agent, run, model, tin, tout,
                    cost_usd(model, tin, tout), tool_calls, tool_results,
                    (time.time() - started) * 1000.0)
                app.state.store.add_event(event)
                await app.state.broadcaster.publish(
                    {"kind": "call", "run": run, "agent": agent,
                     "cost_usd": event.cost_usd, "ts": event.ts})
            except Exception:
                log.exception("metering failed; response still returned (fail-open)")
        return Response(upstream.content, upstream.status_code,
                        media_type=upstream.headers.get("content-type"))

    async def _relay_stream(request: Request, raw: bytes,
                            pending: PendingCall) -> StreamingResponse:
        # Forward chunks the moment they arrive (the agent must not notice us),
        # buffer a copy, and meter from the assembled body once the stream ends.
        req = client.build_request(
            "POST", "/v1/messages", content=raw, headers=_forward_headers(request))
        upstream = await client.send(req, stream=True)

        async def relay() -> Any:
            chunks: list[bytes] = []
            try:
                async for chunk in upstream.aiter_raw():
                    chunks.append(chunk)
                    yield chunk
            finally:
                await upstream.aclose()
            if upstream.status_code != 200:
                return
            try:
                text = b"".join(chunks).decode("utf-8", errors="replace")
                tin, tout, tool_calls = parse_sse_response(text, cfg.loop_volatile_keys)
                event = CallEvent(
                    uuid.uuid4().hex, pending.ts, pending.agent, pending.run,
                    pending.model, tin, tout, cost_usd(pending.model, tin, tout),
                    tool_calls, pending.tool_results,
                    (time.time() - pending.ts) * 1000.0)
                app.state.store.add_event(event)
                await app.state.broadcaster.publish(
                    {"kind": "call", "run": pending.run, "agent": pending.agent,
                     "cost_usd": event.cost_usd, "ts": event.ts})
            except Exception:
                log.exception("stream metering failed; stream already relayed (fail-open)")

        return StreamingResponse(relay(), status_code=upstream.status_code,
                                 media_type=upstream.headers.get("content-type"))

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        store: Store = app.state.store
        return {"spend_by_agent": store.spend_by_agent(),
                "runs": store.run_states(),
                "incidents": store.recent_incidents(),
                "calls_per_minute": store.calls_per_minute(time.time() - 600.0),
                "killed_runs": sorted(app.state.engine.killed_runs)}

    @app.get("/api/stream")
    async def stream() -> StreamingResponse:
        return StreamingResponse(app.state.broadcaster.subscribe(),
                                 media_type="text/event-stream")

    @app.post("/api/runs/{run}/kill")
    def kill(run: str) -> dict[str, str]:
        app.state.engine.kill(run)
        return {"run": run, "state": "killed"}

    @app.post("/api/runs/{run}/reset")
    def reset(run: str) -> dict[str, str]:
        app.state.engine.reset(run)
        return {"run": run, "state": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return resources.files("agentfuse").joinpath(
            "templates/dashboard.html").read_text(encoding="utf-8")

    return app


def _forward_headers(request: Request) -> dict[str, str]:
    return {k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS}
