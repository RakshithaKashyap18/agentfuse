from __future__ import annotations

from pathlib import Path

import click
import httpx
import uvicorn

from agentfuse.config import load_config
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
@click.option("--headless", is_flag=True, help="Run without opening a browser; exit 0/1.")
def demo(headless: bool) -> None:
    """Run the self-contained pitch demo (no API key needed)."""
    raise click.ClickException("demo arrives in Task 13")


if __name__ == "__main__":
    main()
