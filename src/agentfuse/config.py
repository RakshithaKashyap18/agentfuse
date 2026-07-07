from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


DEFAULT_VOLATILE_KEYS: tuple[str, ...] = (
    "timestamp", "ts", "request_id", "nonce", "trace_id", "idempotency_key",
)


@dataclass(frozen=True)
class FuseConfig:
    upstream_anthropic: str = "https://api.anthropic.com"
    upstream_openai: str = "https://api.openai.com"
    budget_per_run: float | None = 5.0
    budget_per_agent_daily: float | None = 50.0
    budget_per_agent: tuple[tuple[str, float], ...] = ()  # per-agent daily overrides
    loop_threshold: int = 4
    loop_volatile_keys: tuple[str, ...] = DEFAULT_VOLATILE_KEYS
    stall_threshold: int = 5
    rate_calls_per_minute: int = 30
    webhook_url: str = ""
    cooldown_seconds: int = 600
    db_path: str = "./fuse.db"
    retention_days: int = 0  # 0 = keep events forever
    api_token: str = ""  # empty = /api/* endpoints are open


def load_config(path: Path | None) -> FuseConfig:
    if path is None or not path.exists():
        return FuseConfig()
    data = tomllib.loads(path.read_text())
    upstream = data.get("upstream", {})
    budget = data.get("budget", {})
    policies = data.get("policies", {})
    alerting = data.get("alerting", {})
    storage = data.get("storage", {})
    server = data.get("server", {})
    return FuseConfig(
        upstream_anthropic=upstream.get("anthropic", FuseConfig.upstream_anthropic),
        upstream_openai=upstream.get("openai", FuseConfig.upstream_openai),
        budget_per_run=budget.get("per_run", FuseConfig.budget_per_run),
        budget_per_agent_daily=budget.get("per_agent_daily", FuseConfig.budget_per_agent_daily),
        budget_per_agent=tuple(sorted(budget.get("agents", {}).items())),
        loop_threshold=policies.get("loop", {}).get("threshold", FuseConfig.loop_threshold),
        loop_volatile_keys=tuple(
            policies.get("loop", {}).get("volatile_keys", FuseConfig.loop_volatile_keys)
        ),
        stall_threshold=policies.get("stall", {}).get("threshold", FuseConfig.stall_threshold),
        rate_calls_per_minute=policies.get("rate", {}).get(
            "calls_per_minute", FuseConfig.rate_calls_per_minute
        ),
        webhook_url=alerting.get("webhook_url", FuseConfig.webhook_url),
        cooldown_seconds=alerting.get("cooldown_seconds", FuseConfig.cooldown_seconds),
        db_path=storage.get("db_path", FuseConfig.db_path),
        retention_days=storage.get("retention_days", FuseConfig.retention_days),
        api_token=server.get("api_token", FuseConfig.api_token),
    )
