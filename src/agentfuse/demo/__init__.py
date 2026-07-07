from __future__ import annotations

import asyncio
from typing import Any

import httpx

from agentfuse.config import FuseConfig
from agentfuse.demo.upstream import make_fake_upstream
from agentfuse.demo.workload import run_spend_spiral, run_workload
from agentfuse.server import create_app

DEMO_CONFIG = FuseConfig(db_path=":memory:", budget_per_run=0.05)


def run_demo_inprocess() -> dict[str, Any]:
    async def _run() -> dict[str, Any]:
        upstream_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=make_fake_upstream()),
            base_url="https://fake-upstream")
        proxy_app = create_app(DEMO_CONFIG, upstream_client=upstream_client)
        proxy_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy_app), base_url="http://fuse")
        counts = await run_workload(proxy_client)      # act 1: the retry loop
        spiral = await run_spend_spiral(proxy_client)  # act 2: the cost spiral
        status = (await proxy_client.get("/api/status")).json()
        return {"counts": counts, "spiral": spiral, "incidents": status["incidents"]}

    return asyncio.run(_run())
