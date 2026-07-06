from agentfuse.models import Action, CallEvent, ToolCall, ToolResult, Verdict
from agentfuse.store import Store


def ev(i: int, run: str = "r1", agent: str = "a1", ts: float = 0.0,
       cost: float = 1.0) -> CallEvent:
    return CallEvent(f"e{i}", ts, agent, run, "claude-haiku-4-5", 10, 10, cost,
                     (ToolCall("search", "h1"),), (ToolResult(False),), 5.0)


def test_round_trip_preserves_tool_calls() -> None:
    s = Store(":memory:")
    s.add_event(ev(1))
    (got,) = s.events_for_run("r1")
    assert got.tool_calls == (ToolCall("search", "h1"),)
    assert got.tool_results == (ToolResult(False),)


def test_spend_and_rate_queries() -> None:
    s = Store(":memory:")
    s.add_event(ev(1, ts=100.0, cost=2.0))
    s.add_event(ev(2, ts=160.0, cost=3.0))
    assert s.run_spend("r1") == 5.0
    assert s.agent_calls_since("a1", since_ts=150.0) == 1
    assert s.agent_spend_since("a1", since_ts=0.0) == 5.0
    assert s.spend_by_agent() == {"a1": 5.0}


def test_incidents_recorded_and_listed() -> None:
    s = Store(":memory:")
    s.add_incident(1.0, "r1", "a1", Verdict(Action.BLOCK, "loop", "blocked"))
    (inc,) = s.recent_incidents()
    assert inc["policy"] == "loop" and inc["action"] == "BLOCK"


def test_calls_per_minute_buckets_recent_events() -> None:
    s = Store(":memory:")
    s.add_event(ev(1, ts=60.0))
    s.add_event(ev(2, ts=61.0))
    s.add_event(ev(3, ts=125.0))
    s.add_event(ev(4, ts=10.0))  # before since_ts: excluded
    assert s.calls_per_minute(since_ts=60.0) == [
        {"minute": 1, "calls": 2},
        {"minute": 2, "calls": 1},
    ]


def test_last_block_ts_ignores_warns_and_other_runs() -> None:
    s = Store(":memory:")
    assert s.last_block_ts("r1") == 0.0
    s.add_incident(1.0, "r1", "a1", Verdict(Action.WARN, "loop", "w"))
    s.add_incident(2.0, "r1", "a1", Verdict(Action.BLOCK, "loop", "b"))
    s.add_incident(3.0, "r2", "a1", Verdict(Action.BLOCK, "loop", "b"))
    assert s.last_block_ts("r1") == 2.0
