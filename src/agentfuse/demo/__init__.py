from __future__ import annotations

import asyncio
from typing import Any

import httpx

from agentfuse.config import FuseConfig
from agentfuse.demo.upstream import make_fake_upstream
from agentfuse.demo.workload import run_workload
from agentfuse.server import create_app


def run_demo_inprocess() -> dict[str, Any]:
    async def _run() -> dict[str, Any]:
        upstream_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=make_fake_upstream()),
            base_url="https://fake-upstream")
        cfg = FuseConfig(db_path=":memory:")
        proxy_app = create_app(cfg, upstream_client=upstream_client)
        proxy_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy_app), base_url="http://fuse")
        counts = await run_workload(proxy_client)
        status = (await proxy_client.get("/api/status")).json()
        return {"counts": counts, "incidents": status["incidents"]}

    return asyncio.run(_run())
