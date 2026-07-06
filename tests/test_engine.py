from agentfuse.engine import PolicyEngine
from agentfuse.models import Action, PendingCall, Verdict, Window


def make_window() -> Window:
    return Window(PendingCall("a1", "r1", "m", 0.0, ()), (), 0, 0.0, 0.0)


class Fixed:
    def __init__(self, name: str, action: Action) -> None:
        self.name = name
        self.action = action

    def evaluate(self, window: Window) -> Verdict:
        return Verdict(self.action, self.name, f"{self.name} fired")


class Exploding:
    name = "boom"

    def evaluate(self, window: Window) -> Verdict:
        raise RuntimeError("bug in policy")


def test_most_severe_verdict_wins() -> None:
    eng = PolicyEngine([Fixed("a", Action.WARN), Fixed("b", Action.BLOCK)])
    v = eng.check(make_window())
    assert v.action is Action.BLOCK and v.policy == "b"


def test_failing_policy_is_skipped_fail_open() -> None:
    eng = PolicyEngine([Exploding(), Fixed("ok", Action.ALLOW)])
    assert eng.check(make_window()).action is Action.ALLOW


def test_killed_run_blocks_everything_until_reset() -> None:
    eng = PolicyEngine([])
    eng.kill("r1")
    assert eng.check(make_window()).action is Action.KILL
    eng.reset("r1")
    assert eng.check(make_window()).action is Action.ALLOW
