from __future__ import annotations

from agentfuse.models import Action, Verdict, Window


class RateLimiter:
    name = "rate"

    def __init__(self, calls_per_minute: int) -> None:
        self.calls_per_minute = calls_per_minute

    def evaluate(self, window: Window) -> Verdict:
        if window.agent_calls_last_minute >= self.calls_per_minute:
            return Verdict(
                Action.BLOCK, self.name,
                f"AgentFuse blocked this call: rate cap of {self.calls_per_minute} calls/minute "
                f"reached. Wait before retrying.",
            )
        return Verdict.allow(self.name)
