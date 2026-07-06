from __future__ import annotations

from agentfuse.models import Action, Verdict, Window


class BudgetBreaker:
    name = "budget"

    def __init__(self, per_run: float | None, per_agent_daily: float | None) -> None:
        self.per_run = per_run
        self.per_agent_daily = per_agent_daily

    def evaluate(self, window: Window) -> Verdict:
        checks = (
            ("run", window.run_spend, self.per_run),
            ("agent (today)", window.agent_spend_today, self.per_agent_daily),
        )
        worst = Verdict.allow(self.name)
        for scope, spent, limit in checks:
            if limit is None or limit <= 0:
                continue
            if spent >= limit:
                return Verdict(
                    Action.BLOCK, self.name,
                    f"AgentFuse blocked this call: {scope} budget exhausted "
                    f"(${spent:.2f} of ${limit:.2f}). Stop and summarize what you have so far.",
                )
            if spent >= 0.8 * limit and worst.action is Action.ALLOW:
                worst = Verdict(
                    Action.WARN, self.name,
                    f"{scope} spend at ${spent:.2f} of ${limit:.2f} budget (>=80%).",
                )
        return worst
