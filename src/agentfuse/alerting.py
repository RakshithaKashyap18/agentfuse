from __future__ import annotations

import logging

import httpx

from agentfuse.models import Action, PendingCall, Verdict

log = logging.getLogger("agentfuse")


class Alerter:
    def __init__(self, webhook_url: str, cooldown_seconds: int,
                 client: httpx.AsyncClient) -> None:
        self.webhook_url = webhook_url
        self.cooldown_seconds = cooldown_seconds
        self.client = client
        self._last_fired: dict[tuple[str, str], float] = {}

    async def maybe_fire(self, v: Verdict, pending: PendingCall) -> bool:
        if not self.webhook_url or v.action < Action.WARN:
            return False
        key = (v.policy, pending.run)
        last = self._last_fired.get(key)
        if last is not None and pending.ts - last < self.cooldown_seconds:
            return False
        payload = {
            "text": f":rotating_light: AgentFuse {v.action.name} — "
                    f"[{pending.agent}/{pending.run}] {v.policy}: {v.message}",
            "policy": v.policy,
            "action": v.action.name,
            "agent": pending.agent,
            "run": pending.run,
            "message": v.message,
        }
        try:
            await self.client.post(self.webhook_url, json=payload)
        except httpx.HTTPError:
            log.exception("webhook alert failed")
            return False
        self._last_fired[key] = pending.ts
        return True
