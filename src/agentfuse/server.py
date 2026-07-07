from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from collections import defaultdict
from dataclasses import replace
from importlib import resources
from typing import Any, Callable, TypeVar

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from agentfuse.alerting import Alerter
from agentfuse.config import FuseConfig
from agentfuse.engine import PolicyEngine
from agentfuse.meter import cost_usd
from agentfuse.models import Action, CallEvent, PendingCall, Verdict
from agentfuse.parsing import (
    build_window,
    parse_openai_request,
    parse_openai_response,
    parse_request,
    parse_response,
    parse_sse_response,
)
from agentfuse.policies import default_policies
from agentfuse.store import Store
from agentfuse.streaming import Broadcaster

log = logging.getLogger("agentfuse")
_HOP_HEADERS = {"host", "content-length", "x-fuse-agent", "x-fuse-run"}
_MUTABLE_CONFIG = ("budget_per_run", "budget_per_agent_daily", "cooldown_seconds",
                   "loop_threshold", "rate_calls_per_minute", "stall_threshold")
T = TypeVar("T")


def create_app(cfg: FuseConfig, upstream_client: httpx.AsyncClient | None = None,
               openai_upstream_client: httpx.AsyncClient | None = None) -> FastAPI:
    app = FastAPI(title="AgentFuse")
    app.state.cfg = cfg
    app.state.store = Store(cfg.db_path)
    app.state.engine = PolicyEngine(default_policies(cfg))
    app.state.broadcaster = Broadcaster()
    app.state.default_run = uuid.uuid4().hex[:8]
    app.state.run_locks = defaultdict(asyncio.Lock)
    app.state.alert_tasks = set()
    client = upstream_client or httpx.AsyncClient(
        base_url=cfg.upstream_anthropic, timeout=120.0
    )
    openai_client = openai_upstream_client or httpx.AsyncClient(
        base_url=cfg.upstream_openai, timeout=120.0
    )
    app.state.alerter = Alerter(cfg.webhook_url, cfg.cooldown_seconds, client)

    def _authorized(request: Request) -> bool:
        if not cfg.api_token:
            return True
        if request.headers.get("authorization", "") == f"Bearer {cfg.api_token}":
            return True
        return request.query_params.get("token") == cfg.api_token

    def _unauthorized() -> JSONResponse:
        return JSONResponse(status_code=401, content={"error": "unauthorized"})

    def _fire_alert(verdict: Verdict, pending: PendingCall) -> None:
        # fire-and-forget: a slow webhook must not add latency to agent calls
        task = asyncio.get_running_loop().create_task(
            app.state.alerter.maybe_fire(verdict, pending))
        app.state.alert_tasks.add(task)
        task.add_done_callback(app.state.alert_tasks.discard)

    def _record_upstream_incident(ts: float, run: str, agent: str, message: str) -> None:
        app.state.store.add_incident(ts, run, agent, Verdict(Action.WARN, "upstream", message))

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
        # serialize check-then-record per run so parallel calls can't slip past
        # a threshold together; the streaming relay itself happens outside the lock
        async with app.state.run_locks[run]:
            try:
                verdict = app.state.engine.check(build_window(app.state.store, pending))
            except Exception:  # store trouble must not block the call (fail-open)
                log.exception("window build failed; allowing call (fail-open)")
                verdict = Verdict.allow()
            if verdict.action >= Action.WARN:
                app.state.store.add_incident(started, run, agent, verdict)
                _fire_alert(verdict, pending)
                await app.state.broadcaster.publish(
                    {"kind": "incident", "run": run, "agent": agent,
                     "policy": verdict.policy, "action": verdict.action.name,
                     "message": verdict.message})
            if verdict.action >= Action.BLOCK:
                return _blocked_response(verdict, agent, started, openai_shape=False)
            if body.get("stream") is not True:
                return await _forward_and_meter(request, raw, pending)
        return await _relay_stream(request, raw, pending)

    @app.post("/openai/v1/chat/completions")
    async def proxy_chat_completions(request: Request) -> Response:
        raw = await request.body()
        started = time.time()
        agent = request.headers.get("x-fuse-agent", "default")
        run = request.headers.get("x-fuse-run", app.state.default_run)
        try:
            body: dict[str, Any] = await request.json()
        except Exception:
            body = {}
        model, tool_results = parse_openai_request(body)
        pending = PendingCall(agent, run, model, started, tool_results)
        async with app.state.run_locks[run]:
            try:
                verdict = app.state.engine.check(build_window(app.state.store, pending))
            except Exception:
                log.exception("window build failed; allowing call (fail-open)")
                verdict = Verdict.allow()
            if verdict.action >= Action.WARN:
                app.state.store.add_incident(started, run, agent, verdict)
                _fire_alert(verdict, pending)
                await app.state.broadcaster.publish(
                    {"kind": "incident", "run": run, "agent": agent,
                     "policy": verdict.policy, "action": verdict.action.name,
                     "message": verdict.message})
            if verdict.action >= Action.BLOCK:
                return _blocked_response(verdict, agent, started, openai_shape=True)
            if body.get("stream") is True:
                # OpenAI SSE metering not implemented yet: relay untouched
                log.warning("openai stream passthrough: metering skipped")
                req = openai_client.build_request(
                    "POST", "/v1/chat/completions", content=raw,
                    headers=_forward_headers(request))
                try:
                    upstream = await openai_client.send(req, stream=True)
                except httpx.HTTPError as exc:
                    return _upstream_failure(pending, exc)

                async def relay() -> Any:
                    try:
                        async for chunk in upstream.aiter_raw():
                            yield chunk
                    finally:
                        await upstream.aclose()

                return StreamingResponse(relay(), status_code=upstream.status_code,
                                         media_type=upstream.headers.get("content-type"))
            try:
                upstream = await openai_client.post(
                    "/v1/chat/completions", content=raw, headers=_forward_headers(request))
            except httpx.HTTPError as exc:
                return _upstream_failure(pending, exc)
            if upstream.status_code >= 500:
                _record_upstream_incident(started, run, agent,
                                          f"upstream returned {upstream.status_code}")
            if upstream.status_code == 200:
                try:
                    tin, tout, tool_calls = parse_openai_response(
                        upstream.json(), cfg.loop_volatile_keys)
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

    def _blocked_response(verdict: Verdict, agent: str, started: float,
                          openai_shape: bool) -> JSONResponse:
        headers: dict[str, str] = {}
        if verdict.policy == "rate":
            # seconds until the oldest call in the rolling minute ages out
            oldest = app.state.store.oldest_call_ts_since(agent, started - 60.0)
            wait = math.ceil(oldest + 60.0 - started) if oldest > 0 else 60
            headers["Retry-After"] = str(min(60, max(1, wait)))
        error = {"type": "agentfuse_blocked", "message": verdict.message}
        content: dict[str, Any] = (
            {"error": error} if openai_shape else {"type": "error", "error": error})
        return JSONResponse(status_code=429, headers=headers, content=content)

    def _upstream_failure(pending: PendingCall, exc: Exception) -> JSONResponse:
        log.exception("upstream request failed")
        _record_upstream_incident(pending.ts, pending.run, pending.agent,
                                  f"upstream request failed: {exc!r}")
        return JSONResponse(status_code=502, content={
            "type": "error",
            "error": {"type": "upstream_error",
                      "message": "AgentFuse could not reach the upstream API."},
        })

    async def _forward_and_meter(request: Request, raw: bytes,
                                 pending: PendingCall) -> Response:
        started = pending.ts
        try:
            upstream = await client.post(
                "/v1/messages", content=raw, headers=_forward_headers(request))
        except httpx.HTTPError as exc:
            return _upstream_failure(pending, exc)
        if upstream.status_code >= 500:
            _record_upstream_incident(started, pending.run, pending.agent,
                                      f"upstream returned {upstream.status_code}")
        if upstream.status_code == 200:
            try:
                tin, tout, tool_calls = parse_response(upstream.json(), cfg.loop_volatile_keys)
                event = CallEvent(
                    uuid.uuid4().hex, started, pending.agent, pending.run, pending.model,
                    tin, tout, cost_usd(pending.model, tin, tout), tool_calls,
                    pending.tool_results, (time.time() - started) * 1000.0)
                app.state.store.add_event(event)
                if cfg.retention_days > 0:
                    app.state.store.prune(started - cfg.retention_days * 86400.0)
                await app.state.broadcaster.publish(
                    {"kind": "call", "run": pending.run, "agent": pending.agent,
                     "cost_usd": event.cost_usd, "ts": event.ts})
            except Exception:
                log.exception("metering failed; response still returned (fail-open)")
        return Response(upstream.content, upstream.status_code,
                        media_type=upstream.headers.get("content-type"))

    async def _relay_stream(request: Request, raw: bytes,
                            pending: PendingCall) -> Response:
        # Forward chunks the moment they arrive (the agent must not notice us),
        # buffer a copy, and meter from the assembled body once the stream ends.
        req = client.build_request(
            "POST", "/v1/messages", content=raw, headers=_forward_headers(request))
        try:
            upstream = await client.send(req, stream=True)
        except httpx.HTTPError as exc:
            return _upstream_failure(pending, exc)
        if upstream.status_code >= 500:
            _record_upstream_incident(pending.ts, pending.run, pending.agent,
                                      f"upstream returned {upstream.status_code}")

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
                if cfg.retention_days > 0:
                    app.state.store.prune(pending.ts - cfg.retention_days * 86400.0)
                await app.state.broadcaster.publish(
                    {"kind": "call", "run": pending.run, "agent": pending.agent,
                     "cost_usd": event.cost_usd, "ts": event.ts})
            except Exception:
                log.exception("stream metering failed; stream already relayed (fail-open)")

        return StreamingResponse(relay(), status_code=upstream.status_code,
                                 media_type=upstream.headers.get("content-type"))

    @app.get("/api/status")
    def status(request: Request) -> Response:
        if not _authorized(request):
            return _unauthorized()
        store: Store = app.state.store
        failed = False

        def safe(fn: Callable[[], T], default: T) -> T:
            nonlocal failed
            try:
                return fn()
            except Exception:
                failed = True
                return default

        no_spend: dict[str, float] = {}
        no_rows: list[dict[str, object]] = []
        return JSONResponse({
            "spend_by_agent": safe(store.spend_by_agent, no_spend),
            "runs": safe(store.run_states, no_rows),
            "incidents": safe(lambda: store.recent_incidents(), no_rows),
            "calls_per_minute": safe(
                lambda: store.calls_per_minute(time.time() - 600.0), no_rows),
            "killed_runs": sorted(app.state.engine.killed_runs),
            "degraded": store.degraded or failed,
        })

    @app.get("/api/stream")
    async def stream(request: Request) -> Response:
        if not _authorized(request):
            return _unauthorized()
        return StreamingResponse(app.state.broadcaster.subscribe(),
                                 media_type="text/event-stream")

    @app.post("/api/runs/{run}/kill")
    def kill(run: str, request: Request) -> Response:
        if not _authorized(request):
            return _unauthorized()
        app.state.engine.kill(run)
        return JSONResponse({"run": run, "state": "killed"})

    @app.post("/api/runs/{run}/reset")
    def reset(run: str, request: Request) -> Response:
        if not _authorized(request):
            return _unauthorized()
        app.state.engine.reset(run)
        return JSONResponse({"run": run, "state": "ok"})

    @app.get("/api/config")
    def get_config(request: Request) -> Response:
        if not _authorized(request):
            return _unauthorized()
        return JSONResponse({k: getattr(cfg, k) for k in _MUTABLE_CONFIG})

    @app.post("/api/config")
    async def update_config(request: Request) -> Response:
        nonlocal cfg
        if not _authorized(request):
            return _unauthorized()
        try:
            updates = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "invalid JSON body"})
        if (not isinstance(updates, dict) or not updates
                or not set(updates) <= set(_MUTABLE_CONFIG)):
            return JSONResponse(status_code=400, content={
                "error": f"allowed keys: {list(_MUTABLE_CONFIG)}"})
        for key, value in updates.items():
            if value is not None and not isinstance(value, (int, float)):
                return JSONResponse(status_code=400, content={
                    "error": f"{key} must be a number or null"})
        cfg = replace(cfg, **updates)
        app.state.cfg = cfg
        app.state.engine.policies = default_policies(cfg)  # hot-swap thresholds
        return JSONResponse({k: getattr(cfg, k) for k in _MUTABLE_CONFIG})

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return resources.files("agentfuse").joinpath(
            "templates/dashboard.html").read_text(encoding="utf-8")

    return app


def _forward_headers(request: Request) -> dict[str, str]:
    return {k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS}
