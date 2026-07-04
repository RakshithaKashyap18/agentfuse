# AgentFuse Implementation Plan — 7-Day Schedule

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build AgentFuse v1 — a reverse proxy for the Anthropic Messages API that meters agent spend in real time and trips circuit breakers (loop / budget / stall / rate) before damage is done, with a live dashboard and a zero-API-key demo.

**Architecture:** A FastAPI app exposes the proxy endpoint; every call is parsed into events, checked against a pure-function policy engine (fail-open), stored in SQLite, and broadcast over SSE to a single-file HTML dashboard. `fuse demo` wires the proxy to a built-in fake upstream via an in-process httpx transport, so the whole pitch demo runs with no API key.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, httpx, Click, stdlib sqlite3/tomllib, single-file HTML dashboard (Chart.js via CDN). Dev: pytest, pytest-asyncio, pytest-cov, mypy --strict, ruff.

**Spec:** `docs/superpowers/specs/2026-07-03-agentfuse-design.md`

## Global Constraints

- Python 3.11+ (uses `tomllib`, `X | None` unions)
- Package name `agentfuse`, import name `agentfuse`, CLI entry point `fuse`
- `mypy --strict` clean, `ruff check` clean, all code type-annotated
- Policy engine failures must **fail open** (forward the call, log the error)
- Blocked calls return HTTP 429 with a **model-readable** JSON error body
- Runtime deps limited to: fastapi, uvicorn, httpx, click
- Identity headers: `X-Fuse-Agent` (default `"default"`), `X-Fuse-Run` (default: one generated run id per proxy process)
- Streaming (`"stream": true`) requests are v1-out-of-scope: proxy them through untouched with metering skipped (log a `stream_passthrough` note); full support is a Day-7 stretch goal

---

## Day 1 — Scaffold, models, meter, config

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `src/agentfuse/__init__.py`, `tests/__init__.py`, `.gitignore`, `.github/workflows/ci.yml`

**Interfaces:**
- Produces: installable package `agentfuse`, `pytest` / `mypy` / `ruff` runnable

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "agentfuse"
version = "0.1.0"
description = "Runtime control plane for AI agents: live cost metering and circuit breakers"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
dependencies = [
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "httpx>=0.27",
    "click>=8.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5",
    "mypy>=1.10",
    "ruff>=0.4",
]

[project.scripts]
fuse = "agentfuse.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/agentfuse"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.mypy]
strict = true
files = ["src/agentfuse"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]
```

- [ ] **Step 2: Create empty `src/agentfuse/__init__.py`** containing `__version__ = "0.1.0"`, empty `tests/__init__.py`, and `.gitignore` with:

```
.venv/
__pycache__/
*.egg-info/
.coverage
fuse.db
dist/
```

- [ ] **Step 3: Create `.github/workflows/ci.yml`**

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e ".[dev]"
      - run: ruff check src tests
      - run: mypy
      - run: pytest --cov=agentfuse --cov-fail-under=85
```

- [ ] **Step 4: Create venv, install, verify**

Run: `python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"` then `.venv/Scripts/pytest --collect-only`
Expected: installs cleanly; pytest reports "no tests ran" without errors.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "chore: project scaffold with CI, mypy strict, ruff"
```

### Task 2: Core models

**Files:**
- Create: `src/agentfuse/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `Action` (IntEnum ALLOW=0 WARN=1 BLOCK=2 KILL=3), frozen dataclasses `ToolCall(name, args_hash)`, `ToolResult(is_error)`, `CallEvent(id, ts, agent, run, model, input_tokens, output_tokens, cost_usd, tool_calls, tool_results, latency_ms)`, `PendingCall(agent, run, model, ts, tool_results)`, `Window(pending, events, agent_calls_last_minute, run_spend, agent_spend_today)`, `Verdict(action, policy, message)` with `Verdict.allow()`, and `hash_args(args) -> str`.

- [ ] **Step 1: Write failing tests in `tests/test_models.py`**

```python
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
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_models.py -v` → FAIL (module not found).

- [ ] **Step 3: Implement `src/agentfuse/models.py`**

```python
from __future__ import annotations

import enum
import hashlib
import json
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


@dataclass(frozen=True)
class Verdict:
    action: Action
    policy: str
    message: str

    @staticmethod
    def allow(policy: str = "") -> "Verdict":
        return Verdict(Action.ALLOW, policy, "")


def hash_args(args: object) -> str:
    canon = json.dumps(args, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:12]
```

- [ ] **Step 4: Run tests** — `pytest tests/test_models.py -v` → PASS. Also `mypy` → clean.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: core event and verdict models"`

### Task 3: Cost meter

**Files:**
- Create: `src/agentfuse/meter.py`
- Test: `tests/test_meter.py`

**Interfaces:**
- Produces: `cost_usd(model: str, input_tokens: int, output_tokens: int) -> float`; `PRICES: dict[str, tuple[float, float]]` (dollars per million input/output tokens, keyed by model-name prefix, longest prefix wins, unknown model falls back to `DEFAULT_PRICE`).

- [ ] **Step 1: Failing tests in `tests/test_meter.py`**

```python
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
```

- [ ] **Step 2: Run** — `pytest tests/test_meter.py -v` → FAIL.

- [ ] **Step 3: Implement `src/agentfuse/meter.py`**

```python
from __future__ import annotations

PRICES: dict[str, tuple[float, float]] = {
    "claude-opus": (15.0, 75.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku": (1.0, 5.0),
}
DEFAULT_PRICE: tuple[float, float] = (3.0, 15.0)


def price_for(model: str) -> tuple[float, float]:
    best = ""
    for prefix in PRICES:
        if model.startswith(prefix) and len(prefix) > len(best):
            best = prefix
    return PRICES[best] if best else DEFAULT_PRICE


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p_in, p_out = price_for(model)
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000
```

- [ ] **Step 4: Run** — `pytest tests/test_meter.py -v` → PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: token cost meter with prefix-matched price table"`

### Task 4: Config loader

**Files:**
- Create: `src/agentfuse/config.py`, `fuse.example.toml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `FuseConfig` dataclass with fields `upstream_anthropic: str`, `budget_per_run: float | None`, `budget_per_agent_daily: float | None`, `loop_threshold: int`, `stall_threshold: int`, `rate_calls_per_minute: int`, `webhook_url: str`, `cooldown_seconds: int`, `db_path: str`; `load_config(path: Path | None) -> FuseConfig` (missing file or `None` → all defaults).

- [ ] **Step 1: Failing tests in `tests/test_config.py`**

```python
from pathlib import Path

from agentfuse.config import FuseConfig, load_config


def test_defaults_when_no_file() -> None:
    cfg = load_config(None)
    assert cfg == FuseConfig()
    assert cfg.budget_per_run == 5.0
    assert cfg.loop_threshold == 4


def test_partial_toml_overrides(tmp_path: Path) -> None:
    p = tmp_path / "fuse.toml"
    p.write_text('[budget]\nper_run = 2.5\n[policies.loop]\nthreshold = 7\n')
    cfg = load_config(p)
    assert cfg.budget_per_run == 2.5
    assert cfg.loop_threshold == 7
    assert cfg.rate_calls_per_minute == 30  # untouched default
```

- [ ] **Step 2: Run** — `pytest tests/test_config.py -v` → FAIL.

- [ ] **Step 3: Implement `src/agentfuse/config.py`**

```python
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FuseConfig:
    upstream_anthropic: str = "https://api.anthropic.com"
    budget_per_run: float | None = 5.0
    budget_per_agent_daily: float | None = 50.0
    loop_threshold: int = 4
    stall_threshold: int = 5
    rate_calls_per_minute: int = 30
    webhook_url: str = ""
    cooldown_seconds: int = 600
    db_path: str = "./fuse.db"


def load_config(path: Path | None) -> FuseConfig:
    if path is None or not path.exists():
        return FuseConfig()
    data = tomllib.loads(path.read_text())
    upstream = data.get("upstream", {})
    budget = data.get("budget", {})
    policies = data.get("policies", {})
    alerting = data.get("alerting", {})
    storage = data.get("storage", {})
    return FuseConfig(
        upstream_anthropic=upstream.get("anthropic", FuseConfig.upstream_anthropic),
        budget_per_run=budget.get("per_run", FuseConfig.budget_per_run),
        budget_per_agent_daily=budget.get("per_agent_daily", FuseConfig.budget_per_agent_daily),
        loop_threshold=policies.get("loop", {}).get("threshold", FuseConfig.loop_threshold),
        stall_threshold=policies.get("stall", {}).get("threshold", FuseConfig.stall_threshold),
        rate_calls_per_minute=policies.get("rate", {}).get(
            "calls_per_minute", FuseConfig.rate_calls_per_minute
        ),
        webhook_url=alerting.get("webhook_url", FuseConfig.webhook_url),
        cooldown_seconds=alerting.get("cooldown_seconds", FuseConfig.cooldown_seconds),
        db_path=storage.get("db_path", FuseConfig.db_path),
    )
```

- [ ] **Step 4: Write `fuse.example.toml`** — copy the Configuration block from the spec verbatim.
- [ ] **Step 5: Run** — `pytest tests/test_config.py -v` → PASS; `mypy` clean. Commit: `git add -A && git commit -m "feat: TOML config loader with defaults"`

---

## Day 2 — Policy engine (the product's brain)

### Task 5: Policy protocol + four policies

**Files:**
- Create: `src/agentfuse/policies/__init__.py`, `base.py`, `loop.py`, `budget.py`, `stall.py`, `rate.py`
- Test: `tests/test_policies.py`

**Interfaces:**
- Consumes: `Window`, `Verdict`, `Action`, `ToolCall`, `ToolResult` from `agentfuse.models`
- Produces: `Policy` Protocol (`name: str`, `evaluate(window: Window) -> Verdict`); classes `LoopBreaker(threshold: int)`, `BudgetBreaker(per_run: float | None, per_agent_daily: float | None)`, `StallDetector(threshold: int)`, `RateLimiter(calls_per_minute: int)`; `default_policies(cfg: FuseConfig) -> list[Policy]` in `policies/__init__.py`.

- [ ] **Step 1: Failing tests in `tests/test_policies.py`** (helper builds Windows; test each policy below/at/above threshold)

```python
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
```

- [ ] **Step 2: Run** — `pytest tests/test_policies.py -v` → FAIL.

- [ ] **Step 3: Implement.** `src/agentfuse/policies/base.py`:

```python
from __future__ import annotations

from typing import Protocol

from agentfuse.models import Verdict, Window


class Policy(Protocol):
    name: str

    def evaluate(self, window: Window) -> Verdict: ...
```

`src/agentfuse/policies/loop.py`:

```python
from __future__ import annotations

from agentfuse.models import Action, Verdict, Window


class LoopBreaker:
    name = "loop"

    def __init__(self, threshold: int) -> None:
        self.threshold = threshold

    def evaluate(self, window: Window) -> Verdict:
        calls = [tc for ev in window.events for tc in ev.tool_calls]
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
```

`src/agentfuse/policies/budget.py`:

```python
from __future__ import annotations

from agentfuse.models import Action, Verdict, Window


class BudgetBreaker:
    name = "budget"

    def __init__(self, per_run: float | None, per_agent_daily: float | None) -> None:
        self.per_run = per_run
        self.per_agent_daily = per_agent_daily

    def evaluate(self, window: Window) -> Verdict:
        checks = (
            ("run", window.run_spend, self.per_run),
            ("agent (today)", window.agent_spend_today, self.per_agent_daily),
        )
        worst = Verdict.allow(self.name)
        for scope, spent, limit in checks:
            if limit is None or limit <= 0:
                continue
            if spent >= limit:
                return Verdict(
                    Action.BLOCK, self.name,
                    f"AgentFuse blocked this call: {scope} budget exhausted "
                    f"(${spent:.2f} of ${limit:.2f}). Stop and summarize what you have so far.",
                )
            if spent >= 0.8 * limit and worst.action is Action.ALLOW:
                worst = Verdict(
                    Action.WARN, self.name,
                    f"{scope} spend at ${spent:.2f} of ${limit:.2f} budget (>=80%).",
                )
        return worst
```

`src/agentfuse/policies/stall.py`:

```python
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
```

`src/agentfuse/policies/rate.py`:

```python
from __future__ import annotations

from agentfuse.models import Action, Verdict, Window


class RateLimiter:
    name = "rate"

    def __init__(self, calls_per_minute: int) -> None:
        self.calls_per_minute = calls_per_minute

    def evaluate(self, window: Window) -> Verdict:
        if window.agent_calls_last_minute >= self.calls_per_minute:
            return Verdict(
                Action.BLOCK, self.name,
                f"AgentFuse blocked this call: rate cap of {self.calls_per_minute} calls/minute "
                f"reached. Wait before retrying.",
            )
        return Verdict.allow(self.name)
```

`src/agentfuse/policies/__init__.py`:

```python
from __future__ import annotations

from agentfuse.config import FuseConfig
from agentfuse.policies.base import Policy
from agentfuse.policies.budget import BudgetBreaker
from agentfuse.policies.loop import LoopBreaker
from agentfuse.policies.rate import RateLimiter
from agentfuse.policies.stall import StallDetector


def default_policies(cfg: FuseConfig) -> list[Policy]:
    return [
        LoopBreaker(cfg.loop_threshold),
        BudgetBreaker(cfg.budget_per_run, cfg.budget_per_agent_daily),
        StallDetector(cfg.stall_threshold),
        RateLimiter(cfg.rate_calls_per_minute),
    ]
```

- [ ] **Step 4: Run** — `pytest tests/test_policies.py -v` → PASS; `mypy` clean.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: four guardrail policies with warn/block escalation"`

### Task 6: Policy engine (verdict precedence + kill switch + fail-open)

**Files:**
- Create: `src/agentfuse/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `Policy`, `Window`, `Verdict`, `Action`
- Produces: `PolicyEngine(policies: list[Policy])` with `check(window: Window) -> Verdict` (most severe wins; killed runs always get KILL; a raising policy is skipped — fail open), `kill(run: str) -> None`, `reset(run: str) -> None`, `killed_runs: set[str]`.

- [ ] **Step 1: Failing tests in `tests/test_engine.py`**

```python
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
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement `src/agentfuse/engine.py`**

```python
from __future__ import annotations

import logging

from agentfuse.models import Action, Verdict, Window
from agentfuse.policies.base import Policy

log = logging.getLogger("agentfuse")


class PolicyEngine:
    def __init__(self, policies: list[Policy]) -> None:
        self.policies = policies
        self.killed_runs: set[str] = set()

    def kill(self, run: str) -> None:
        self.killed_runs.add(run)

    def reset(self, run: str) -> None:
        self.killed_runs.discard(run)

    def check(self, window: Window) -> Verdict:
        if window.pending.run in self.killed_runs:
            return Verdict(
                Action.KILL, "kill-switch",
                "AgentFuse: this run has been killed by an operator. No further calls "
                "will be forwarded until the run is reset.",
            )
        worst = Verdict.allow()
        for policy in self.policies:
            try:
                verdict = policy.evaluate(window)
            except Exception:  # fail open: a control plane must not become the outage
                log.exception("policy %s raised; skipping (fail-open)", policy.name)
                continue
            if verdict.action > worst.action:
                worst = verdict
        return worst
```

- [ ] **Step 4: Run** — `pytest tests/test_engine.py -v` → PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: policy engine with precedence, kill switch, fail-open"`

---

## Day 3 — Storage and request/response parsing

### Task 7: SQLite store

**Files:**
- Create: `src/agentfuse/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `CallEvent`, `Verdict`
- Produces: `Store(db_path: str)` with `add_event(e: CallEvent) -> None`, `events_for_run(run: str) -> tuple[CallEvent, ...]`, `agent_calls_since(agent: str, since_ts: float) -> int`, `run_spend(run: str) -> float`, `agent_spend_since(agent: str, since_ts: float) -> float`, `add_incident(ts: float, run: str, agent: str, v: Verdict) -> None`, `recent_incidents(limit: int = 50) -> list[dict[str, object]]`, `spend_by_agent() -> dict[str, float]`, `run_states() -> list[dict[str, object]]` (run, agent, calls, spend, worst incident action), `close() -> None`. Thread-safe via internal `threading.Lock`; `":memory:"` supported.

- [ ] **Step 1: Failing tests in `tests/test_store.py`**

```python
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
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement `src/agentfuse/store.py`**

```python
from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

from agentfuse.models import CallEvent, ToolCall, ToolResult, Verdict

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY, ts REAL, agent TEXT, run TEXT, model TEXT,
    input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL,
    tool_calls TEXT, tool_results TEXT, latency_ms REAL
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run, ts);
CREATE INDEX IF NOT EXISTS idx_events_agent ON events(agent, ts);
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, run TEXT, agent TEXT,
    policy TEXT, action TEXT, message TEXT
);
"""


class Store:
    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def add_event(self, e: CallEvent) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (e.id, e.ts, e.agent, e.run, e.model, e.input_tokens, e.output_tokens,
                 e.cost_usd,
                 json.dumps([[tc.name, tc.args_hash] for tc in e.tool_calls]),
                 json.dumps([tr.is_error for tr in e.tool_results]),
                 e.latency_ms),
            )
            self._conn.commit()

    def events_for_run(self, run: str) -> tuple[CallEvent, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE run = ? ORDER BY ts", (run,)
            ).fetchall()
        return tuple(
            CallEvent(
                r["id"], r["ts"], r["agent"], r["run"], r["model"],
                r["input_tokens"], r["output_tokens"], r["cost_usd"],
                tuple(ToolCall(n, h) for n, h in json.loads(r["tool_calls"])),
                tuple(ToolResult(bool(x)) for x in json.loads(r["tool_results"])),
                r["latency_ms"],
            )
            for r in rows
        )

    def _scalar(self, sql: str, params: tuple[Any, ...]) -> float:
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return float(row[0] or 0)

    def agent_calls_since(self, agent: str, since_ts: float) -> int:
        return int(self._scalar(
            "SELECT COUNT(*) FROM events WHERE agent = ? AND ts >= ?", (agent, since_ts)))

    def run_spend(self, run: str) -> float:
        return self._scalar("SELECT SUM(cost_usd) FROM events WHERE run = ?", (run,))

    def agent_spend_since(self, agent: str, since_ts: float) -> float:
        return self._scalar(
            "SELECT SUM(cost_usd) FROM events WHERE agent = ? AND ts >= ?", (agent, since_ts))

    def add_incident(self, ts: float, run: str, agent: str, v: Verdict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO incidents (ts, run, agent, policy, action, message) "
                "VALUES (?,?,?,?,?,?)",
                (ts, run, agent, v.policy, v.action.name, v.message),
            )
            self._conn.commit()

    def recent_incidents(self, limit: int = 50) -> list[dict[str, object]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, run, agent, policy, action, message FROM incidents "
                "ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def spend_by_agent(self) -> dict[str, float]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT agent, SUM(cost_usd) AS s FROM events GROUP BY agent").fetchall()
        return {r["agent"]: float(r["s"]) for r in rows}

    def run_states(self) -> list[dict[str, object]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT e.run, MAX(e.agent) AS agent, COUNT(*) AS calls, "
                "SUM(e.cost_usd) AS spend, "
                "(SELECT i.action FROM incidents i WHERE i.run = e.run "
                " ORDER BY i.id DESC LIMIT 1) AS last_action "
                "FROM events e GROUP BY e.run ORDER BY MAX(e.ts) DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run** — `pytest tests/test_store.py -v` → PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: SQLite event and incident store"`

### Task 8: Anthropic request/response parsing + window builder

**Files:**
- Create: `src/agentfuse/parsing.py`
- Test: `tests/test_parsing.py`

**Interfaces:**
- Consumes: `models`, `meter.cost_usd`, `Store`
- Produces: `parse_request(body: dict[str, Any]) -> tuple[str, tuple[ToolResult, ...]]` (model, tool_results **from the final message only** — messages carry full history, so earlier ones were already counted on prior calls); `parse_response(body: dict[str, Any]) -> tuple[int, int, tuple[ToolCall, ...]]` (input_tokens, output_tokens, tool_calls); `build_window(store: Store, pending: PendingCall) -> Window`; `day_start_ts(now: float) -> float` (UTC midnight).

- [ ] **Step 1: Failing tests in `tests/test_parsing.py`**

```python
from typing import Any

from agentfuse.models import PendingCall, ToolResult, hash_args
from agentfuse.parsing import build_window, parse_request, parse_response
from agentfuse.store import Store


def anthropic_request() -> dict[str, Any]:
    return {
        "model": "claude-haiku-4-5",
        "messages": [
            {"role": "user", "content": "find agent frameworks"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "search",
                 "input": {"query": "agent frameworks"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1",
                 "content": "timeout", "is_error": True}]},
        ],
    }


def anthropic_response() -> dict[str, Any]:
    return {
        "content": [
            {"type": "text", "text": "let me search"},
            {"type": "tool_use", "id": "t2", "name": "search",
             "input": {"query": "agent frameworks"}},
        ],
        "usage": {"input_tokens": 500, "output_tokens": 60},
    }


def test_parse_request_takes_only_final_message_results() -> None:
    model, results = parse_request(anthropic_request())
    assert model == "claude-haiku-4-5"
    assert results == (ToolResult(True),)


def test_parse_request_string_content_yields_no_results() -> None:
    _, results = parse_request({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert results == ()


def test_parse_response_extracts_usage_and_tool_calls() -> None:
    tin, tout, calls = parse_response(anthropic_response())
    assert (tin, tout) == (500, 60)
    assert calls[0].name == "search"
    assert calls[0].args_hash == hash_args({"query": "agent frameworks"})


def test_build_window_aggregates_from_store() -> None:
    s = Store(":memory:")
    pending = PendingCall("a1", "r1", "m", 1000.0, ())
    w = build_window(s, pending)
    assert w.events == () and w.run_spend == 0.0 and w.agent_calls_last_minute == 0
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement `src/agentfuse/parsing.py`**

```python
from __future__ import annotations

from typing import Any

from agentfuse.models import PendingCall, ToolCall, ToolResult, Window, hash_args
from agentfuse.store import Store

DAY_SECONDS = 86400.0


def parse_request(body: dict[str, Any]) -> tuple[str, tuple[ToolResult, ...]]:
    model = str(body.get("model", "unknown"))
    messages = body.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return model, ()
    content = messages[-1].get("content") if isinstance(messages[-1], dict) else None
    results: list[ToolResult] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                results.append(ToolResult(bool(block.get("is_error", False))))
    return model, tuple(results)


def parse_response(body: dict[str, Any]) -> tuple[int, int, tuple[ToolCall, ...]]:
    usage = body.get("usage", {}) if isinstance(body.get("usage"), dict) else {}
    tin = int(usage.get("input_tokens", 0))
    tout = int(usage.get("output_tokens", 0))
    calls: list[ToolCall] = []
    content = body.get("content", [])
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                calls.append(ToolCall(str(block.get("name", "")),
                                      hash_args(block.get("input", {}))))
    return tin, tout, tuple(calls)


def day_start_ts(now: float) -> float:
    return now - (now % DAY_SECONDS)


def build_window(store: Store, pending: PendingCall) -> Window:
    return Window(
        pending=pending,
        events=store.events_for_run(pending.run),
        agent_calls_last_minute=store.agent_calls_since(pending.agent, pending.ts - 60.0),
        run_spend=store.run_spend(pending.run),
        agent_spend_today=store.agent_spend_since(pending.agent, day_start_ts(pending.ts)),
    )
```

- [ ] **Step 4: Run** — `pytest tests/test_parsing.py -v` → PASS; full suite `pytest -q` green; `mypy` clean.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: Anthropic payload parsing and policy window builder"`

---

## Day 4 — The proxy server, SSE, alerting

### Task 9: Alerting + SSE broadcaster

**Files:**
- Create: `src/agentfuse/alerting.py`, `src/agentfuse/streaming.py`
- Test: `tests/test_alerting.py`

**Interfaces:**
- Produces: `Alerter(webhook_url: str, cooldown_seconds: int, client: httpx.AsyncClient)` with `async maybe_fire(v: Verdict, pending: PendingCall) -> bool` (fires on WARN+; keyed cooldown on `(policy, run)`; returns whether it fired; no-op when url empty; network errors swallowed+logged). `Broadcaster` with `async publish(event: dict[str, object]) -> None` and `subscribe() -> AsyncIterator[str]` (yields `data: <json>\n\n` SSE frames; each subscriber gets its own `asyncio.Queue`; queue is removed on generator exit).

- [ ] **Step 1: Failing tests in `tests/test_alerting.py`**

```python
import httpx

from agentfuse.alerting import Alerter
from agentfuse.models import Action, PendingCall, Verdict


def pending() -> PendingCall:
    return PendingCall("a1", "r1", "m", 100.0, ())


def make_alerter(cooldown: int = 600) -> tuple[Alerter, list[str]]:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.content.decode())
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return Alerter("https://hooks.example/x", cooldown, client), seen


async def test_fires_on_warn_and_respects_cooldown() -> None:
    alerter, seen = make_alerter()
    v = Verdict(Action.WARN, "loop", "looping")
    assert await alerter.maybe_fire(v, pending()) is True
    assert await alerter.maybe_fire(v, pending()) is False  # cooldown
    assert len(seen) == 1 and "looping" in seen[0]


async def test_allow_never_fires_and_empty_url_noop() -> None:
    alerter, seen = make_alerter()
    assert await alerter.maybe_fire(Verdict.allow(), pending()) is False
    silent = Alerter("", 600, httpx.AsyncClient())
    assert await silent.maybe_fire(Verdict(Action.BLOCK, "b", "m"), pending()) is False
    assert seen == []
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement.** `src/agentfuse/alerting.py`:

```python
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
```

`src/agentfuse/streaming.py`:

```python
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator


class Broadcaster:
    def __init__(self) -> None:
        self._queues: set[asyncio.Queue[str]] = set()

    async def publish(self, event: dict[str, object]) -> None:
        frame = f"data: {json.dumps(event)}\n\n"
        for q in list(self._queues):
            q.put_nowait(frame)

    async def subscribe(self) -> AsyncIterator[str]:
        q: asyncio.Queue[str] = asyncio.Queue()
        self._queues.add(q)
        try:
            while True:
                yield await q.get()
        finally:
            self._queues.discard(q)
```

- [ ] **Step 4: Run** — `pytest tests/test_alerting.py -v` → PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: webhook alerter with cooldown and SSE broadcaster"`

### Task 10: FastAPI app — proxy route + REST API

**Files:**
- Create: `src/agentfuse/server.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: everything above
- Produces: `create_app(cfg: FuseConfig, upstream_client: httpx.AsyncClient | None = None) -> FastAPI`. Injectable `upstream_client` is how tests and the demo swap in a fake upstream (via `httpx.ASGITransport`). Routes:
  - `POST /anthropic/v1/messages` — the proxy (verdict check → 429 or forward → meter → store → broadcast)
  - `GET /api/status` — `{"spend_by_agent": ..., "runs": ..., "incidents": ...}`
  - `GET /api/stream` — SSE
  - `POST /api/runs/{run}/kill`, `POST /api/runs/{run}/reset`
  - `GET /` — dashboard HTML (Task 11)
  - App state on `app.state`: `store`, `engine`, `broadcaster`, `alerter`, `cfg`, `default_run` (uuid4 hex generated at startup).

- [ ] **Step 1: Failing tests in `tests/test_server.py`** — a fake upstream ASGI app scripted per-test:

```python
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentfuse.config import FuseConfig
from agentfuse.server import create_app


def fake_upstream(responses: list[dict[str, Any]]) -> FastAPI:
    app = FastAPI()
    state = {"i": 0}

    @app.post("/v1/messages")
    async def messages() -> dict[str, Any]:
        body = responses[min(state["i"], len(responses) - 1)]
        state["i"] += 1
        return body

    return app


def tool_use_response(query: str) -> dict[str, Any]:
    return {"content": [{"type": "tool_use", "id": "t", "name": "search",
                         "input": {"query": query}}],
            "usage": {"input_tokens": 100, "output_tokens": 50}}


def make_client(responses: list[dict[str, Any]], cfg: FuseConfig | None = None) -> TestClient:
    cfg = cfg or FuseConfig(db_path=":memory:", webhook_url="")
    upstream = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake_upstream(responses)),
        base_url="https://fake-upstream",
    )
    return TestClient(create_app(cfg, upstream_client=upstream))


def call(client: TestClient, run: str = "r1", agent: str = "a1") -> httpx.Response:
    return client.post(
        "/anthropic/v1/messages",
        json={"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "go"}]},
        headers={"X-Fuse-Agent": agent, "X-Fuse-Run": run},
    )


def test_forwards_and_meters() -> None:
    client = make_client([tool_use_response("q")])
    r = call(client)
    assert r.status_code == 200
    status = client.get("/api/status").json()
    assert status["spend_by_agent"]["a1"] > 0


def test_loop_breaker_trips_end_to_end() -> None:
    # identical tool_use every turn -> warn at 4, block at 6
    client = make_client([tool_use_response("same")] * 10)
    codes = [call(client).status_code for _ in range(8)]
    assert 429 in codes
    blocked = next(r for r in [call(client)] if r.status_code == 429)
    body = blocked.json()
    assert body["type"] == "error"
    assert "search" in body["error"]["message"]  # model-readable guidance
    incidents = client.get("/api/status").json()["incidents"]
    assert any(i["policy"] == "loop" for i in incidents)


def test_kill_and_reset_endpoints() -> None:
    client = make_client([tool_use_response("q")] * 5)
    assert call(client).status_code == 200
    client.post("/api/runs/r1/kill")
    assert call(client).status_code == 429
    client.post("/api/runs/r1/reset")
    assert call(client).status_code == 200


def test_default_identity_headers() -> None:
    client = make_client([tool_use_response("q")])
    r = client.post("/anthropic/v1/messages",
                    json={"model": "m", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200
    assert "default" in client.get("/api/status").json()["spend_by_agent"]
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement `src/agentfuse/server.py`**

```python
from __future__ import annotations

import logging
import time
import uuid
from importlib import resources
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from agentfuse.alerting import Alerter
from agentfuse.config import FuseConfig
from agentfuse.engine import PolicyEngine
from agentfuse.models import Action, CallEvent, PendingCall
from agentfuse.parsing import build_window, parse_request, parse_response
from agentfuse.meter import cost_usd
from agentfuse.policies import default_policies
from agentfuse.store import Store
from agentfuse.streaming import Broadcaster

log = logging.getLogger("agentfuse")
_HOP_HEADERS = {"host", "content-length", "x-fuse-agent", "x-fuse-run"}


def create_app(cfg: FuseConfig, upstream_client: httpx.AsyncClient | None = None) -> FastAPI:
    app = FastAPI(title="AgentFuse")
    app.state.cfg = cfg
    app.state.store = Store(cfg.db_path)
    app.state.engine = PolicyEngine(default_policies(cfg))
    app.state.broadcaster = Broadcaster()
    app.state.default_run = uuid.uuid4().hex[:8]
    client = upstream_client or httpx.AsyncClient(
        base_url=cfg.upstream_anthropic, timeout=120.0
    )
    app.state.alerter = Alerter(cfg.webhook_url, cfg.cooldown_seconds, client)

    @app.post("/anthropic/v1/messages")
    async def proxy_messages(request: Request) -> Response:
        raw = await request.body()
        started = time.time()
        agent = request.headers.get("x-fuse-agent", "default")
        run = request.headers.get("x-fuse-run", app.state.default_run)
        try:
            body: dict[str, Any] = await request.json()
        except Exception:
            body = {}
        if body.get("stream") is True:
            log.warning("stream_passthrough: metering skipped for streaming request")
            upstream = await client.post(
                "/v1/messages", content=raw, headers=_forward_headers(request))
            return Response(upstream.content, upstream.status_code,
                            media_type=upstream.headers.get("content-type"))
        model, tool_results = parse_request(body)
        pending = PendingCall(agent, run, model, started, tool_results)
        verdict = app.state.engine.check(build_window(app.state.store, pending))
        if verdict.action >= Action.WARN:
            app.state.store.add_incident(started, run, agent, verdict)
            await app.state.alerter.maybe_fire(verdict, pending)
            await app.state.broadcaster.publish(
                {"kind": "incident", "run": run, "agent": agent,
                 "policy": verdict.policy, "action": verdict.action.name,
                 "message": verdict.message})
        if verdict.action >= Action.BLOCK:
            return JSONResponse(status_code=429, content={
                "type": "error",
                "error": {"type": "agentfuse_blocked", "message": verdict.message},
            })
        upstream = await client.post(
            "/v1/messages", content=raw, headers=_forward_headers(request))
        if upstream.status_code == 200:
            try:
                tin, tout, tool_calls = parse_response(upstream.json())
                event = CallEvent(
                    uuid.uuid4().hex, started, agent, run, model, tin, tout,
                    cost_usd(model, tin, tout), tool_calls, tool_results,
                    (time.time() - started) * 1000.0)
                app.state.store.add_event(event)
                await app.state.broadcaster.publish(
                    {"kind": "call", "run": run, "agent": agent,
                     "cost_usd": event.cost_usd, "ts": event.ts})
            except Exception:
                log.exception("metering failed; response still returned (fail-open)")
        return Response(upstream.content, upstream.status_code,
                        media_type=upstream.headers.get("content-type"))

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        store: Store = app.state.store
        return {"spend_by_agent": store.spend_by_agent(),
                "runs": store.run_states(),
                "incidents": store.recent_incidents(),
                "killed_runs": sorted(app.state.engine.killed_runs)}

    @app.get("/api/stream")
    async def stream() -> StreamingResponse:
        return StreamingResponse(app.state.broadcaster.subscribe(),
                                 media_type="text/event-stream")

    @app.post("/api/runs/{run}/kill")
    def kill(run: str) -> dict[str, str]:
        app.state.engine.kill(run)
        return {"run": run, "state": "killed"}

    @app.post("/api/runs/{run}/reset")
    def reset(run: str) -> dict[str, str]:
        app.state.engine.reset(run)
        return {"run": run, "state": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return resources.files("agentfuse").joinpath(
            "templates/dashboard.html").read_text(encoding="utf-8")

    return app


def _forward_headers(request: Request) -> dict[str, str]:
    return {k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS}
```

Note: create `src/agentfuse/templates/dashboard.html` as a placeholder file containing `<!-- dashboard: Task 11 -->` so the route imports cleanly; Task 11 replaces it.

- [ ] **Step 4: Run** — `pytest tests/test_server.py -v` → PASS; full suite green; `mypy` clean.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: proxy server with enforcement, REST API, SSE"`

---

## Day 5 — Dashboard + CLI

### Task 11: Single-file dashboard

**Files:**
- Create (replace placeholder): `src/agentfuse/templates/dashboard.html`
- Test: `tests/test_dashboard.py`
- Modify: `pyproject.toml` — ensure templates ship in the wheel:

```toml
[tool.hatch.build.targets.wheel.force-include]
"src/agentfuse/templates" = "agentfuse/templates"
```

**Interfaces:**
- Consumes: `GET /api/status` JSON shape and SSE frames (`kind: "call" | "incident"`) from Task 10.

- [ ] **Step 1: Failing test in `tests/test_dashboard.py`**

```python
from tests.test_server import make_client, tool_use_response


def test_dashboard_serves_html_with_panels() -> None:
    client = make_client([tool_use_response("q")])
    html = client.get("/").text
    for anchor in ("id=\"spend\"", "id=\"runs\"", "id=\"incidents\"", "EventSource"):
        assert anchor in html
```

- [ ] **Step 2: Run** → FAIL (placeholder has no panels).

- [ ] **Step 3: Write `dashboard.html`** — dark single-page layout, no build step:

```html
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>AgentFuse</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body { font-family: system-ui, sans-serif; background: #0f1117; color: #e5e7eb;
         margin: 0; padding: 1.5rem; }
  h1 { font-size: 1.2rem; margin: 0 0 1rem; } h1 span { color: #f59e0b; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
          gap: 1rem; }
  .card { background: #1a1d27; border: 1px solid #2a2e3d; border-radius: 10px;
          padding: 1rem; }
  .big { font-size: 2.2rem; font-weight: 700; color: #34d399; }
  table { width: 100%; border-collapse: collapse; font-size: .85rem; }
  td, th { padding: .35rem .5rem; text-align: left; border-bottom: 1px solid #2a2e3d; }
  .badge { padding: .1rem .5rem; border-radius: 999px; font-size: .75rem; }
  .OK { background: #064e3b; } .WARN { background: #78350f; }
  .BLOCK, .KILL { background: #7f1d1d; }
  button { background: #2a2e3d; color: #e5e7eb; border: 0; border-radius: 6px;
           padding: .2rem .6rem; cursor: pointer; }
  #incidents li { margin-bottom: .5rem; list-style: none; font-size: .85rem; }
  #incidents { padding: 0; max-height: 300px; overflow-y: auto; }
</style>
</head>
<body>
<h1>⚡ Agent<span>Fuse</span> — live control plane</h1>
<div class="grid">
  <div class="card"><h3>Total spend</h3><div class="big" id="spend">$0.0000</div>
    <div id="spend-by-agent"></div></div>
  <div class="card"><h3>Calls / minute</h3><canvas id="rate-chart" height="140"></canvas></div>
  <div class="card"><h3>Breaker board</h3><table id="runs"><thead>
    <tr><th>run</th><th>agent</th><th>calls</th><th>spend</th><th>state</th><th></th></tr>
    </thead><tbody></tbody></table></div>
  <div class="card"><h3>Incident feed</h3><ul id="incidents"></ul></div>
</div>
<script>
const buckets = new Map(); // minute -> call count
const chart = new Chart(document.getElementById("rate-chart"), {
  type: "bar",
  data: { labels: [], datasets: [{ label: "calls", data: [], backgroundColor: "#60a5fa" }] },
  options: { plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } },
});

async function refresh() {
  const s = await (await fetch("/api/status")).json();
  const total = Object.values(s.spend_by_agent).reduce((a, b) => a + b, 0);
  document.getElementById("spend").textContent = "$" + total.toFixed(4);
  document.getElementById("spend-by-agent").innerHTML = Object.entries(s.spend_by_agent)
    .map(([a, v]) => `${a}: $${v.toFixed(4)}`).join("<br>");
  const tbody = document.querySelector("#runs tbody");
  tbody.innerHTML = s.runs.map(r => {
    const killed = s.killed_runs.includes(r.run);
    const state = killed ? "KILL" : (r.last_action || "OK");
    const action = killed
      ? `<button onclick="post('/api/runs/${r.run}/reset')">reset</button>`
      : `<button onclick="post('/api/runs/${r.run}/kill')">kill</button>`;
    return `<tr><td>${r.run}</td><td>${r.agent}</td><td>${r.calls}</td>
      <td>$${(r.spend || 0).toFixed(4)}</td>
      <td><span class="badge ${state}">${state}</span></td><td>${action}</td></tr>`;
  }).join("");
  const feed = document.getElementById("incidents");
  feed.innerHTML = s.incidents.map(i =>
    `<li><span class="badge ${i.action}">${i.action}</span>
     <b>${i.policy}</b> [${i.agent}/${i.run}] — ${i.message}</li>`).join("");
}

async function post(url) { await fetch(url, { method: "POST" }); refresh(); }

const es = new EventSource("/api/stream");
es.onmessage = (m) => {
  const e = JSON.parse(m.data);
  if (e.kind === "call") {
    const minute = new Date(e.ts * 1000).toISOString().slice(11, 16);
    buckets.set(minute, (buckets.get(minute) || 0) + 1);
    const entries = [...buckets.entries()].slice(-10);
    chart.data.labels = entries.map(x => x[0]);
    chart.data.datasets[0].data = entries.map(x => x[1]);
    chart.update("none");
  }
  refresh();
};
refresh();
</script>
</body>
</html>
```

- [ ] **Step 4: Run** — `pytest tests/test_dashboard.py -v` → PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: live single-file dashboard with breaker board and SSE"`

### Task 12: CLI

**Files:**
- Create: `src/agentfuse/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `create_app`, `load_config`, `default_policies`
- Produces: Click group `main` with commands: `serve` (`--host 127.0.0.1 --port 9000 --config PATH`), `policies` (`--config PATH`, prints each policy name + its config values), `status` (`--url http://127.0.0.1:9000`, fetches `/api/status`, prints spend + incident count), `demo` (Task 13 fills the body; create the stub now raising `click.ClickException("demo arrives in Task 13")`).

- [ ] **Step 1: Failing tests in `tests/test_cli.py`**

```python
from click.testing import CliRunner

from agentfuse.cli import main


def test_policies_lists_all_four() -> None:
    result = CliRunner().invoke(main, ["policies"])
    assert result.exit_code == 0
    for name in ("loop", "budget", "stall", "rate"):
        assert name in result.output


def test_serve_is_registered() -> None:
    result = CliRunner().invoke(main, ["serve", "--help"])
    assert result.exit_code == 0 and "--port" in result.output
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement `src/agentfuse/cli.py`**

```python
from __future__ import annotations

from pathlib import Path

import click
import httpx
import uvicorn

from agentfuse.config import load_config
from agentfuse.policies import default_policies
from agentfuse.server import create_app


@click.group()
def main() -> None:
    """AgentFuse — runtime control plane for AI agents."""


@main.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=9000)
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
def serve(host: str, port: int, config_path: Path | None) -> None:
    """Run the proxy + dashboard."""
    app = create_app(load_config(config_path))
    click.echo(f"AgentFuse proxy on http://{host}:{port}  (dashboard at /)")
    uvicorn.run(app, host=host, port=port)


@main.command()
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
def policies(config_path: Path | None) -> None:
    """List active guardrail policies."""
    cfg = load_config(config_path)
    for p in default_policies(cfg):
        click.echo(f"{p.name}: {vars(p)}")


@main.command()
@click.option("--url", default="http://127.0.0.1:9000")
def status(url: str) -> None:
    """Show live spend and incidents from a running proxy."""
    data = httpx.get(f"{url}/api/status", timeout=5.0).json()
    total = sum(data["spend_by_agent"].values())
    click.echo(f"total spend: ${total:.4f}")
    click.echo(f"runs: {len(data['runs'])}  incidents: {len(data['incidents'])}")


@main.command()
@click.option("--headless", is_flag=True, help="Run without opening a browser; exit 0/1.")
def demo(headless: bool) -> None:
    """Run the self-contained pitch demo (no API key needed)."""
    raise click.ClickException("demo arrives in Task 13")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run** — `pytest tests/test_cli.py -v` → PASS. Manual smoke: `fuse serve` then open http://127.0.0.1:9000 — dashboard renders.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: fuse CLI (serve, policies, status)"`

---

## Day 6 — The pitch demo

### Task 13: Fake upstream + scripted workload + `fuse demo`

**Files:**
- Create: `src/agentfuse/demo/__init__.py`, `src/agentfuse/demo/upstream.py`, `src/agentfuse/demo/workload.py`
- Modify: `src/agentfuse/cli.py` (replace `demo` stub body)
- Test: `tests/test_demo.py`

**Interfaces:**
- Produces:
  - `upstream.make_fake_upstream() -> FastAPI` — POST `/v1/messages` returns scripted responses: calls 1–8 return the **same** `tool_use` (`search`, `{"query": "agent frameworks"}`); once the workload signals recovery (its request contains the string `agentfuse_blocked` in any message), it returns a different tool once, then a final text-only answer.
  - `workload.run_workload(proxy_url: str, calls: int = 15) -> dict[str, int]` — drives a "researcher" agent loop against the proxy with `X-Fuse-Agent: researcher`, `X-Fuse-Run: demo-run`; on 429 it appends the block message as a user message (this is the "agent reads the guidance" moment) and continues; returns `{"ok": n, "blocked": n}`.
  - `run_demo_inprocess() -> dict[str, object]` in `demo/__init__.py` — builds the proxy app with an ASGI-transport client into the fake upstream, runs the workload against it via a second ASGI-transport client, returns `{"counts": ..., "incidents": ...}`. Used by both the headless test and `fuse demo`.

- [ ] **Step 1: Failing E2E test in `tests/test_demo.py`**

```python
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
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement.** `src/agentfuse/demo/upstream.py`:

```python
from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, Request

LOOP_RESPONSE: dict[str, Any] = {
    "content": [{"type": "tool_use", "id": "t1", "name": "search",
                 "input": {"query": "agent frameworks"}}],
    "usage": {"input_tokens": 400, "output_tokens": 80},
}
PIVOT_RESPONSE: dict[str, Any] = {
    "content": [{"type": "tool_use", "id": "t2", "name": "fetch_docs",
                 "input": {"url": "https://docs.example/agents"}}],
    "usage": {"input_tokens": 450, "output_tokens": 90},
}
FINAL_RESPONSE: dict[str, Any] = {
    "content": [{"type": "text",
                 "text": "Summary: three agent frameworks compared. Done."}],
    "usage": {"input_tokens": 500, "output_tokens": 200},
}


def make_fake_upstream() -> FastAPI:
    app = FastAPI()
    state = {"recovered": False, "pivoted": False}

    @app.post("/v1/messages")
    async def messages(request: Request) -> dict[str, Any]:
        body = json.loads(await request.body())
        if "agentfuse_blocked" in json.dumps(body):
            state["recovered"] = True
        if not state["recovered"]:
            return LOOP_RESPONSE          # keeps looping until the breaker speaks
        if not state["pivoted"]:
            state["pivoted"] = True
            return PIVOT_RESPONSE         # the guidance worked: new strategy
        return FINAL_RESPONSE

    return app
```

`src/agentfuse/demo/workload.py`:

```python
from __future__ import annotations

from typing import Any

import httpx


async def run_workload(client: httpx.AsyncClient, calls: int = 15) -> dict[str, int]:
    """Drive a scripted 'researcher' agent through the proxy until done or budget."""
    counts = {"ok": 0, "blocked": 0}
    messages: list[dict[str, Any]] = [{"role": "user", "content": "research agent frameworks"}]
    for _ in range(calls):
        resp = await client.post(
            "/anthropic/v1/messages",
            json={"model": "claude-haiku-4-5", "messages": messages},
            headers={"X-Fuse-Agent": "researcher", "X-Fuse-Run": "demo-run"},
        )
        if resp.status_code == 429:
            counts["blocked"] += 1
            guidance = resp.json()["error"]
            # The agent "reads" the block message and feeds it back into context:
            messages.append({"role": "user",
                             "content": f"[agentfuse_blocked] {guidance['message']}"})
            continue
        counts["ok"] += 1
        body = resp.json()
        content = body.get("content", [])
        messages.append({"role": "assistant", "content": content})
        tool_uses = [b for b in content if b.get("type") == "tool_use"]
        if not tool_uses:
            break  # final text answer — run complete
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tool_uses[0]["id"],
             "content": "no results", "is_error": False}]})
    return counts
```

`src/agentfuse/demo/__init__.py`:

```python
from __future__ import annotations

import asyncio
from typing import Any

import httpx

from agentfuse.config import FuseConfig
from agentfuse.demo.upstream import make_fake_upstream
from agentfuse.demo.workload import run_workload
from agentfuse.server import create_app


def run_demo_inprocess() -> dict[str, Any]:
    async def _run() -> dict[str, Any]:
        upstream_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=make_fake_upstream()),
            base_url="https://fake-upstream")
        cfg = FuseConfig(db_path=":memory:")
        proxy_app = create_app(cfg, upstream_client=upstream_client)
        proxy_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy_app), base_url="http://fuse")
        counts = await run_workload(proxy_client)
        status = (await proxy_client.get("/api/status")).json()
        return {"counts": counts, "incidents": status["incidents"]}

    return asyncio.run(_run())
```

- [ ] **Step 4: Replace the `demo` stub in `cli.py`**

```python
@main.command()
@click.option("--headless", is_flag=True, help="Run without a server; print results, exit 0/1.")
@click.option("--port", default=9000)
def demo(headless: bool, port: int) -> None:
    """Run the self-contained pitch demo (no API key needed)."""
    if headless:
        result = run_demo_inprocess()
        counts = result["counts"]
        click.echo(f"calls ok={counts['ok']} blocked={counts['blocked']}")
        for i in result["incidents"]:
            click.echo(f"  [{i['action']}] {i['policy']}: {i['message'][:90]}")
        if counts["blocked"] < 1:
            raise click.ClickException("demo failed: breaker never tripped")
        click.echo("demo OK: breaker tripped and agent recovered")
        return
    # Live mode: real server so the dashboard is watchable while the demo runs.
    import threading
    import time
    import webbrowser

    import uvicorn

    upstream_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=make_fake_upstream()),
        base_url="https://fake-upstream")
    app = create_app(FuseConfig(db_path=":memory:"), upstream_client=upstream_client)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    time.sleep(1.0)
    url = f"http://127.0.0.1:{port}"
    click.echo(f"Dashboard: {url} — watch the breaker trip.")
    webbrowser.open(url)

    async def _drive() -> None:
        async with httpx.AsyncClient(base_url=url) as client:
            for _ in range(15):
                counts = await run_workload(client, calls=1)
                await asyncio.sleep(1.5)  # slow enough to watch live
                if counts["ok"] == 0 and counts["blocked"] == 0:
                    break

    asyncio.run(_drive())
    click.echo("Demo run finished. Ctrl+C to stop the server.")
    while True:
        time.sleep(3600)
```

Add the needed imports to `cli.py`: `import asyncio`, `from agentfuse.config import FuseConfig`, `from agentfuse.demo import run_demo_inprocess`, `from agentfuse.demo.upstream import make_fake_upstream`, `from agentfuse.demo.workload import run_workload`.

Note: live mode drives one call per loop iteration so the dashboard visibly ticks; the workload rebuilds context each iteration, so the loop breaker trips on the stored run history (same `X-Fuse-Run`), matching headless behavior.

- [ ] **Step 5: Run** — `pytest tests/test_demo.py -v` → PASS. Manual: `fuse demo` opens the dashboard, spend ticks up, incident feed shows WARN then BLOCK, run recovers.
- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: zero-API-key pitch demo with fake upstream and scripted workload"`

---

## Day 7 — Polish, README, quality gates, release

### Task 14: README with the pitch + quality pass

**Files:**
- Create: `README.md`, `LICENSE` (MIT)
- Modify: anything the quality pass flags

- [ ] **Step 1: Write `README.md`** with these sections (this is the startup pitch — spend real time here):
  - Title + tagline: *"Stop your agent before it burns $500 in a retry loop."*
  - **Why this exists** — observability tools read yesterday's exhaust; AgentFuse sits in the request path and intervenes during the run. Include the positioning table from the spec.
  - **60-second demo** — `pip install agentfuse && fuse demo` (no API key).
  - **Quick start** — one-line integration: `Anthropic(base_url="http://localhost:9000/anthropic")` + the two identity headers.
  - **The four breakers** — loop / budget / stall / rate, each with its default and the model-readable block message it sends (screenshot or verbatim example).
  - **Blocking that heals** — show the demo transcript excerpt where the agent receives the guidance message and pivots.
  - **Dashboard** — screenshot (take one from `fuse demo`).
  - **Configuration** — `fuse.example.toml` contents.
  - **Design decisions** — fail-open engine, why 429-with-guidance instead of silent drop, why a proxy instead of an SDK.
  - **Roadmap** — replay debugger, chaos testing, OpenAI-compatible endpoint, streaming metering (the platform slide).
- [ ] **Step 2: Full quality pass**

Run: `ruff check src tests && mypy && pytest --cov=agentfuse --cov-report=term-missing`
Expected: all clean, coverage ≥ 85%. Fix anything that isn't.

- [ ] **Step 3: Manual end-to-end check** — `fuse demo` on a clean venv (`pip install -e .` only, no dev deps) to prove runtime deps are complete.
- [ ] **Step 4: Push to GitHub, verify CI green.**

```bash
git add -A && git commit -m "docs: README with pitch narrative and design decisions"
git remote add origin <your-repo-url> && git push -u origin main
```

- [ ] **Step 5: Tag** — `git tag v0.1.0 && git push --tags`. Optional stretch (only if Days 1–6 landed early): PyPI publish via `hatch build && twine upload dist/*`; streaming SSE metering; OpenAI-compatible `/openai/v1/chat/completions` route.

---

## Self-review notes (already applied)

- Spec coverage: proxy ✅(T10) metering ✅(T3,T10) four policies ✅(T5) verdict enforcement + model-readable 429 ✅(T10) kill/reset ✅(T6,T10) dashboard ✅(T11) SSE ✅(T9,T11) webhook alerts ✅(T9) config ✅(T4) SQLite ✅(T7) CLI ✅(T12) demo ✅(T13) error handling: fail-open ✅(T6,T10), upstream errors passed through ✅(T10 returns upstream status verbatim), malformed body → `body={}` and forwarded ✅(T10). Streaming: explicitly passthrough-only per Global Constraints (spec marks full support as stretch).
- The spec's "SQLite write failures degrade to in-memory buffering with a dashboard banner" is deliberately deferred to post-v1: the fail-open try/except around metering in T10 already guarantees a store failure cannot break proxying, which is the load-bearing requirement. Noted here so it's a conscious cut, not an omission.
- Type consistency: `Verdict.allow()`, `Window` field names, and `make_client`/`tool_use_response` test helpers are defined once (T2, T10) and reused verbatim in T11's test import.
