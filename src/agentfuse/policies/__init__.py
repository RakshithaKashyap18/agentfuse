from __future__ import annotations

from agentfuse.config import FuseConfig
from agentfuse.policies.base import Policy
from agentfuse.policies.budget import BudgetBreaker
from agentfuse.policies.loop import LoopBreaker
from agentfuse.policies.rate import RateLimiter
from agentfuse.policies.stall import StallDetector


def default_policies(cfg: FuseConfig) -> list[Policy]:
    return [
        LoopBreaker(cfg.loop_threshold),
        BudgetBreaker(cfg.budget_per_run, cfg.budget_per_agent_daily),
        StallDetector(cfg.stall_threshold),
        RateLimiter(cfg.rate_calls_per_minute),
    ]
