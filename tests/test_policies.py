from agentfuse.models import (
    Action, CallEvent, PendingCall, ToolCall, ToolResult, Window,
)
from agentfuse.policies.budget import BudgetBreaker
from agentfuse.policies.loop import LoopBreaker
from agentfuse.policies.rate import RateLimiter
from agentfuse.policies.stall import StallDetector


def ev(i: int, tool_calls: tuple[ToolCall, ...] = (), cost: float = 0.0,
       tool_results: tuple[ToolResult, ...] = ()) -> CallEvent:
    return CallEvent(str(i), float(i), "a1", "r1", "claude-haiku-4-5",
                     100, 100, cost, tool_calls, tool_results, 10.0)


def win(events: tuple[CallEvent, ...] = (), pending_results: tuple[ToolResult, ...] = (),
        rate: int = 0, run_spend: float = 0.0, agent_spend: float = 0.0) -> Window:
    pending = PendingCall("a1", "r1", "claude-haiku-4-5", 999.0, pending_results)
    return Window(pending, events, rate, run_spend, agent_spend)


SAME = ToolCall("search", "abc123")
OTHER = ToolCall("search", "zzz999")


def test_loop_allows_below_threshold() -> None:
    w = win(events=tuple(ev(i, (SAME,)) for i in range(3)))
    assert LoopBreaker(4).evaluate(w).action is Action.ALLOW


def test_loop_warns_at_threshold_blocks_at_plus_two() -> None:
    w4 = win(events=tuple(ev(i, (SAME,)) for i in range(4)))
    assert LoopBreaker(4).evaluate(w4).action is Action.WARN
    w6 = win(events=tuple(ev(i, (SAME,)) for i in range(6)))
    v = LoopBreaker(4).evaluate(w6)
    assert v.action is Action.BLOCK
    assert "search" in v.message  # model-readable: names the looping tool


def test_loop_streak_resets_on_different_args() -> None:
    events = tuple(ev(i, (SAME,)) for i in range(5)) + (ev(9, (OTHER,)),)
    assert LoopBreaker(4).evaluate(win(events=events)).action is Action.ALLOW


def test_budget_warns_at_80_blocks_at_100() -> None:
    assert BudgetBreaker(5.0, None).evaluate(win(run_spend=3.9)).action is Action.ALLOW
    assert BudgetBreaker(5.0, None).evaluate(win(run_spend=4.0)).action is Action.WARN
    assert BudgetBreaker(5.0, None).evaluate(win(run_spend=5.0)).action is Action.BLOCK


def test_budget_disabled_when_none() -> None:
    assert BudgetBreaker(None, None).evaluate(win(run_spend=999.0)).action is Action.ALLOW


def test_stall_counts_trailing_errors_including_pending() -> None:
    err = (ToolResult(True),)
    events = tuple(ev(i, tool_results=err) for i in range(4))
    w = win(events=events, pending_results=err)  # 5 trailing errors total
    assert StallDetector(5).evaluate(w).action is Action.WARN
    events8 = tuple(ev(i, tool_results=err) for i in range(7))
    assert StallDetector(5).evaluate(win(events=events8, pending_results=err)).action is Action.BLOCK


def test_stall_reset_by_success() -> None:
    err, ok = ToolResult(True), ToolResult(False)
    events = tuple(ev(i, tool_results=(err,)) for i in range(6)) + (ev(9, tool_results=(ok,)),)
    assert StallDetector(5).evaluate(win(events=events)).action is Action.ALLOW


def test_rate_blocks_over_cap() -> None:
    assert RateLimiter(30).evaluate(win(rate=29)).action is Action.ALLOW
    assert RateLimiter(30).evaluate(win(rate=30)).action is Action.BLOCK
