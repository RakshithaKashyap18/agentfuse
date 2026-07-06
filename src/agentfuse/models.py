from __future__ import annotations

import enum
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass


class Action(enum.IntEnum):
    ALLOW = 0
    WARN = 1
    BLOCK = 2
    KILL = 3


@dataclass(frozen=True)
class ToolCall:
    name: str
    args_hash: str


@dataclass(frozen=True)
class ToolResult:
    is_error: bool


@dataclass(frozen=True)
class CallEvent:
    id: str
    ts: float
    agent: str
    run: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    tool_calls: tuple[ToolCall, ...]
    tool_results: tuple[ToolResult, ...]
    latency_ms: float


@dataclass(frozen=True)
class PendingCall:
    agent: str
    run: str
    model: str
    ts: float
    tool_results: tuple[ToolResult, ...]


@dataclass(frozen=True)
class Window:
    pending: PendingCall
    events: tuple[CallEvent, ...]  # completed calls for this run, oldest first
    agent_calls_last_minute: int
    run_spend: float
    agent_spend_today: float
    # ts of this run's most recent BLOCK/KILL incident; streak-based policies only
    # count events at/after it, so a block resets the streak and the run can heal.
    last_block_ts: float = 0.0


@dataclass(frozen=True)
class Verdict:
    action: Action
    policy: str
    message: str

    @staticmethod
    def allow(policy: str = "") -> "Verdict":
        return Verdict(Action.ALLOW, policy, "")


def _drop_volatile(value: object, volatile: frozenset[str]) -> object:
    if isinstance(value, dict):
        return {k: _drop_volatile(v, volatile) for k, v in value.items() if k not in volatile}
    if isinstance(value, list):
        return [_drop_volatile(v, volatile) for v in value]
    return value


def hash_args(args: object, volatile_keys: Iterable[str] = ()) -> str:
    """Hash normalized JSON args; volatile keys (timestamps, request ids, ...)
    are dropped recursively first so a loop can't disguise itself with them."""
    volatile = frozenset(volatile_keys)
    if volatile:
        args = _drop_volatile(args, volatile)
    canon = json.dumps(args, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:12]
