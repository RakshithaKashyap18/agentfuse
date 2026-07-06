from __future__ import annotations

from agentfuse.models import Action, Verdict, Window


class StallDetector:
    name = "stall"

    def __init__(self, threshold: int) -> None:
        self.threshold = threshold

    def evaluate(self, window: Window) -> Verdict:
        results = [tr for ev in window.events for tr in ev.tool_results]
        results.extend(window.pending.tool_results)
        streak = 0
        for tr in reversed(results):
            if not tr.is_error:
                break
            streak += 1
        if streak >= self.threshold + 3:
            return Verdict(
                Action.BLOCK, self.name,
                f"AgentFuse blocked this call: the last {streak} tool results were all errors. "
                f"The current approach is not working — report the blocker instead of retrying.",
            )
        if streak >= self.threshold:
            return Verdict(Action.WARN, self.name, f"{streak} consecutive tool errors.")
        return Verdict.allow(self.name)
