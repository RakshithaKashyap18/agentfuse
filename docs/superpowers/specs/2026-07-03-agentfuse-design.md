# AgentFuse — Design Spec

**Date:** 2026-07-03
**Status:** Approved for planning

## One-liner

AgentFuse is a runtime control plane for AI agents: a reverse proxy that sits between agents and their LLM API, meters every call in real time, detects dangerous patterns (retry loops, cost spirals, stalls), and trips a circuit breaker before the damage is done.

**Tagline:** *Stop your agent before it burns $500 in a retry loop.*

## Positioning

Existing agent-observability tools (including batch trace-triage tools) read exhaust after the fact: they tell you what went wrong yesterday. AgentFuse sits in the request path and intervenes while it is happening.

| Axis | Trace-triage tools | AgentFuse |
|---|---|---|
| When | After the run (batch) | During the run (real time) |
| Role | Observe and rank | Meter and intervene |
| Integration | Upload trace files | One-line `base_url` change |
| Output | Morning report | Tripped breaker + live dashboard |

**Platform story (roadmap, not v1):** because every agent action flows through the interception layer, the same component later enables record-and-replay debugging (the events are already logged) and chaos testing (the interceptor injects faults instead of passing traffic through).

## V1 scope

In scope:
1. LLM reverse proxy for the Anthropic Messages API (an OpenAI-compatible endpoint is a stretch goal, not required for v1)
2. Real-time cost metering per agent / per run
3. Policy engine with four built-in guardrails and verdict escalation
4. Live dashboard (embedded HTML, SSE updates)
5. Webhook alerts
6. `fuse demo` — self-contained, no-API-key demo that shows a breaker tripping
7. CLI, TOML config, SQLite persistence

Out of scope for v1 (roadmap only): replay debugger UI, chaos/fault injection, multi-tenant auth, distributed deployment, per-user billing.

## How it attaches

The agent (any framework, any language) points its LLM client at the AgentFuse proxy:

```python
client = Anthropic(base_url="http://localhost:9000/anthropic")
```

Identity comes from two optional HTTP headers, defaulting when absent:
- `X-Fuse-Agent` — logical agent name (default: `"default"`)
- `X-Fuse-Run` — run/session identifier (default: one run per proxy process start)

The proxy forwards requests to the real upstream (API key passed through from the client request), streams responses back, and extracts per call: model, input/output token counts, `tool_use` blocks in responses, `tool_result` blocks in requests (including `is_error`), and latency. Tool activity is inferred entirely from LLM traffic — no SDK or agent-code changes required.

## Architecture

```
Agent ──► AgentFuse proxy ──► Anthropic / OpenAI API
              │
        meter.py    (tokens → dollars, price table, per-agent/run aggregation)
        policies/   (verdict per call: allow / warn / block / kill)
        store.py    (SQLite event log)
              │
        server.py   (dashboard + SSE + REST)   alerting.py (webhooks)   cli.py
```

### Modules

| Module | Responsibility |
|---|---|
| `proxy.py` | Reverse proxy endpoints; request/response parsing; event extraction; verdict enforcement |
| `meter.py` | Model price table; cost per call; live aggregation per agent, run, and day |
| `policies/base.py` | `Policy` protocol: `evaluate(window: EventWindow) -> Verdict`; policy registry |
| `policies/loop.py` | Loop breaker |
| `policies/budget.py` | Budget breaker |
| `policies/stall.py` | Stall detector |
| `policies/rate.py` | Rate limiter |
| `store.py` | SQLite event log; queries for dashboard and policy windows |
| `server.py` | FastAPI app: proxy routes, dashboard, SSE stream, REST API |
| `alerting.py` | Webhook POST on `warn`-and-above verdicts, per-policy cooldown |
| `config.py` | `fuse.toml` loader with defaults |
| `cli.py` | `fuse serve`, `fuse demo`, `fuse status`, `fuse policies` |
| `demo/` | Fake upstream LLM + scripted two-agent workload |
| `templates/dashboard.html` | Single-file embedded dashboard |

### Built-in policies (v1)

1. **Loop breaker** — the same tool called with near-identical arguments N times consecutively within a run. Near-identical = normalized JSON of the arguments hashed after dropping volatile keys (configurable); default N = 4. Escalation: warn at N, block at N+2.
2. **Budget breaker** — per-run and per-agent dollar budgets computed from metered tokens. Warn at 80% of budget, block at 100%. Default: $5 per run (configurable; no budget configured = policy inactive).
3. **Stall detector** — K consecutive `tool_result` blocks with `is_error: true` within a run. Default K = 5. Verdict: warn, then block at K+3.
4. **Rate limiter** — LLM calls per minute per agent exceeds a cap. Default: 30/min. Verdict: block (returns retry-after guidance).

### Verdicts and enforcement

`allow` → forward normally. `warn` → forward, log incident, fire webhook. `block` → do not forward; return an HTTP 429 response whose body is a structured, model-readable message explaining *why* and *what to try instead* (e.g. "AgentFuse blocked this call: you have called `search` with the same arguments 6 times. Try a different query or a different tool."). `kill` → block this and all subsequent calls for the run until manually reset via CLI/dashboard.

The model-readable block message is a deliberate feature: blocking can heal the agent, not just stop it. The demo shows an agent recovering after receiving one.

Verdict precedence: the most severe verdict across all policies wins. Every verdict above `allow` is recorded as an incident.

## Dashboard

Single-file HTML (Tailwind + Chart.js via CDN), served by FastAPI, live via Server-Sent Events at `/api/stream`. Panels:
- **Live spend ticker** — total and per-agent dollars, updating per call
- **Calls/min chart** — rolling window per agent
- **Breaker status board** — per run: OK / warned / blocked / killed, with reset button
- **Incident feed** — most recent verdicts above `allow`, with policy, agent, and the message sent to the model

## The demo (`fuse demo`)

One command, zero API keys. Starts: (1) the proxy pointed at (2) a built-in fake upstream that returns scripted LLM responses, and (3) a scripted two-agent workload ("researcher" + "coder") that deliberately enters a tool retry loop and a cost spiral. The user opens the dashboard and watches, within ~60 seconds: spend climbing, loop detected, breaker trips, the agent receives the guidance message and switches strategy, run completes. This is the pitch demo and also serves as the end-to-end integration test.

## Configuration (`fuse.toml`)

```toml
[upstream]
anthropic = "https://api.anthropic.com"

[budget]
per_run = 5.00          # dollars; omit to disable
per_agent_daily = 50.00

[policies.loop]
threshold = 4

[policies.stall]
threshold = 5

[policies.rate]
calls_per_minute = 30

[alerting]
webhook_url = ""        # Slack-compatible
cooldown_seconds = 600

[storage]
db_path = "./fuse.db"   # ":memory:" for ephemeral
```

All sections optional; sensible defaults throughout.

## Error handling

- Upstream errors (5xx, timeouts) are passed through to the agent unchanged and logged as events — AgentFuse never masks upstream failures.
- If the policy engine itself raises, the call is **forwarded** (fail-open) and the error logged; a control plane must not become the outage.
- Malformed/unparseable request bodies are forwarded as-is with metering skipped and a parse-failure event logged.
- SQLite write failures degrade to in-memory buffering with a dashboard warning banner.

## Testing

- **Policy unit tests** — each policy against synthetic event windows (below/at/above thresholds, escalation, reset).
- **Meter tests** — price table math, streaming token accumulation, aggregation.
- **Proxy integration tests** — FastAPI test client + fake upstream: pass-through fidelity (headers, streaming), event extraction, verdict enforcement (429 body shape), fail-open behavior.
- **Demo as E2E test** — `fuse demo --headless` runs the scripted workload and asserts the loop breaker tripped and the run completed.
- Quality gates: pytest with coverage, `mypy --strict`, `ruff check` clean, CI via GitHub Actions.

## Tech stack

Python 3.11+, FastAPI + uvicorn, httpx (upstream calls, streaming), SQLite via stdlib `sqlite3`, Click (CLI), single-file HTML dashboard (no frontend build step). Packaging with `pyproject.toml`; PyPI-publishable as `agentfuse` with `fuse` console entry point.
