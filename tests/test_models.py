from agentfuse.models import Action, Verdict, hash_args


def test_action_severity_ordering() -> None:
    assert Action.ALLOW < Action.WARN < Action.BLOCK < Action.KILL


def test_hash_args_is_order_insensitive() -> None:
    assert hash_args({"a": 1, "b": 2}) == hash_args({"b": 2, "a": 1})
    assert hash_args({"a": 1}) != hash_args({"a": 2})
    assert len(hash_args({"a": 1})) == 12


def test_verdict_allow_helper() -> None:
    v = Verdict.allow()
    assert v.action is Action.ALLOW and v.message == ""
