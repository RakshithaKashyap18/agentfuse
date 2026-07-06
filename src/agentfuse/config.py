from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FuseConfig:
    upstream_anthropic: str = "https://api.anthropic.com"
    budget_per_run: float | None = 5.0
    budget_per_agent_daily: float | None = 50.0
    loop_threshold: int = 4
    stall_threshold: int = 5
    rate_calls_per_minute: int = 30
    webhook_url: str = ""
    cooldown_seconds: int = 600
    db_path: str = "./fuse.db"


def load_config(path: Path | None) -> FuseConfig:
    if path is None or not path.exists():
        return FuseConfig()
    data = tomllib.loads(path.read_text())
    upstream = data.get("upstream", {})
    budget = data.get("budget", {})
    policies = data.get("policies", {})
    alerting = data.get("alerting", {})
    storage = data.get("storage", {})
    return FuseConfig(
        upstream_anthropic=upstream.get("anthropic", FuseConfig.upstream_anthropic),
        budget_per_run=budget.get("per_run", FuseConfig.budget_per_run),
        budget_per_agent_daily=budget.get("per_agent_daily", FuseConfig.budget_per_agent_daily),
        loop_threshold=policies.get("loop", {}).get("threshold", FuseConfig.loop_threshold),
        stall_threshold=policies.get("stall", {}).get("threshold", FuseConfig.stall_threshold),
        rate_calls_per_minute=policies.get("rate", {}).get(
            "calls_per_minute", FuseConfig.rate_calls_per_minute
        ),
        webhook_url=alerting.get("webhook_url", FuseConfig.webhook_url),
        cooldown_seconds=alerting.get("cooldown_seconds", FuseConfig.cooldown_seconds),
        db_path=storage.get("db_path", FuseConfig.db_path),
    )
