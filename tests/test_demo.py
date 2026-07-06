from agentfuse.demo import run_demo_inprocess


def test_demo_trips_loop_breaker_and_recovers() -> None:
    result = run_demo_inprocess()
    counts = result["counts"]
    incidents = result["incidents"]
    assert counts["blocked"] >= 1                      # breaker tripped
    assert counts["ok"] > counts["blocked"]            # and the run still made progress
    policies = {i["policy"] for i in incidents}
    assert "loop" in policies                          # it was the loop breaker
    actions = {i["action"] for i in incidents}
    assert "BLOCK" in actions and "WARN" in actions    # escalation happened
