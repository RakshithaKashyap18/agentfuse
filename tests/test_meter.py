import pytest

from agentfuse.meter import cost_usd


def test_haiku_pricing() -> None:
    # 1M in @ $1 + 1M out @ $5
    assert cost_usd("claude-haiku-4-5", 1_000_000, 1_000_000) == pytest.approx(6.0)


def test_longest_prefix_wins_and_unknown_falls_back() -> None:
    assert cost_usd("claude-opus-4-8", 1_000_000, 0) == pytest.approx(15.0)
    assert cost_usd("totally-unknown", 1_000_000, 0) == pytest.approx(3.0)  # default


def test_zero_tokens_zero_cost() -> None:
    assert cost_usd("claude-sonnet-5", 0, 0) == 0.0
