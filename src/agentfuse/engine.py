from __future__ import annotations

import logging

from agentfuse.models import Action, Verdict, Window
from agentfuse.policies.base import Policy

log = logging.getLogger("agentfuse")


class PolicyEngine:
    def __init__(self, policies: list[Policy]) -> None:
        self.policies = policies
        self.killed_runs: set[str] = set()

    def kill(self, run: str) -> None:
        self.killed_runs.add(run)

    def reset(self, run: str) -> None:
        self.killed_runs.discard(run)

    def check(self, window: Window) -> Verdict:
        if window.pending.run in self.killed_runs:
            return Verdict(
                Action.KILL, "kill-switch",
                "AgentFuse: this run has been killed by an operator. No further calls "
                "will be forwarded until the run is reset.",
            )
        worst = Verdict.allow()
        for policy in self.policies:
            try:
                verdict = policy.evaluate(window)
            except Exception:  # fail open: a control plane must not become the outage
                log.exception("policy %s raised; skipping (fail-open)", policy.name)
                continue
            if verdict.action > worst.action:
                worst = verdict
        return worst
