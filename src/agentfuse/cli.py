from __future__ import annotations

import asyncio
from pathlib import Path

import click
import httpx
import uvicorn

from agentfuse.config import FuseConfig, load_config
from agentfuse.demo import run_demo_inprocess
from agentfuse.demo.upstream import make_fake_upstream
from agentfuse.demo.workload import run_workload
from agentfuse.policies import default_policies
from agentfuse.server import create_app


@click.group()
def main() -> None:
    """AgentFuse — runtime control plane for AI agents."""


@main.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=9000)
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
def serve(host: str, port: int, config_path: Path | None) -> None:
    """Run the proxy + dashboard."""
    app = create_app(load_config(config_path))
    click.echo(f"AgentFuse proxy on http://{host}:{port}  (dashboard at /)")
    uvicorn.run(app, host=host, port=port)


@main.command()
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
def policies(config_path: Path | None) -> None:
    """List active guardrail policies."""
    cfg = load_config(config_path)
    for p in default_policies(cfg):
        click.echo(f"{p.name}: {vars(p)}")


@main.command()
@click.option("--url", default="http://127.0.0.1:9000")
def status(url: str) -> None:
    """Show live spend and incidents from a running proxy."""
    data = httpx.get(f"{url}/api/status", timeout=5.0).json()
    total = sum(data["spend_by_agent"].values())
    click.echo(f"total spend: ${total:.4f}")
    click.echo(f"runs: {len(data['runs'])}  incidents: {len(data['incidents'])}")


@main.command()
@click.option("--headless", is_flag=True, help="Run without a server; print results, exit 0/1.")
@click.option("--port", default=9000)
def demo(headless: bool, port: int) -> None:
    """Run the self-contained pitch demo (no API key needed)."""
    if headless:
        result = run_demo_inprocess()
        counts = result["counts"]
        click.echo(f"calls ok={counts['ok']} blocked={counts['blocked']}")
        for i in result["incidents"]:
            click.echo(f"  [{i['action']}] {i['policy']}: {i['message'][:90]}")
        if counts["blocked"] < 1:
            raise click.ClickException("demo failed: breaker never tripped")
        click.echo("demo OK: breaker tripped and agent recovered")
        return
    # Live mode: real server so the dashboard is watchable while the demo runs.
    import threading
    import time
    import webbrowser

    upstream_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=make_fake_upstream()),
        base_url="https://fake-upstream")
    app = create_app(FuseConfig(db_path=":memory:"), upstream_client=upstream_client)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    time.sleep(1.0)
    url = f"http://127.0.0.1:{port}"
    click.echo(f"Dashboard: {url} — watch the breaker trip.")
    webbrowser.open(url)

    async def _drive() -> None:
        async with httpx.AsyncClient(base_url=url) as client:
            for _ in range(15):
                counts = await run_workload(client, calls=1)
                await asyncio.sleep(1.5)  # slow enough to watch live
                if counts["ok"] == 0 and counts["blocked"] == 0:
                    break

    asyncio.run(_drive())
    click.echo("Demo run finished. Ctrl+C to stop the server.")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
