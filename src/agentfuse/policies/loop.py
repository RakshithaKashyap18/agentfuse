from __future__ import annotations

from agentfuse.models import Action, Verdict, Window


class LoopBreaker:
    name = "loop"

    def __init__(self, threshold: int) -> None:
        self.threshold = threshold

    def evaluate(self, window: Window) -> Verdict:
        calls = [
            tc
            for ev in window.events
            if ev.ts >= window.last_block_ts
            for tc in ev.tool_calls
        ]
        if not calls:
            return Verdict.allow(self.name)
        last = calls[-1]
        streak = 0
        for tc in reversed(calls):
            if tc != last:
                break
            streak += 1
        if streak >= self.threshold + 2:
            return Verdict(
                Action.BLOCK, self.name,
                f"AgentFuse blocked this call: you have called `{last.name}` with the same "
                f"arguments {streak} times in a row. Try a different approach, different "
                f"arguments, or a different tool.",
            )
        if streak >= self.threshold:
            return Verdict(
                Action.WARN, self.name,
                f"`{last.name}` called with identical arguments {streak} times in a row.",
            )
        return Verdict.allow(self.name)
