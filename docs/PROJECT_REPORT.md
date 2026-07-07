# AgentFuse — Project Report

**Version:** 0.4.0 · **Date:** 2026-07-06
**Repository:** <https://github.com/RakshithaKashyap18/agentfuse>
**Package:** <https://pypi.org/project/agentfuse1/> (`pip install agentfuse1`)
**License:** MIT · **Language:** Python 3.11+ (tested on 3.11 / 3.12 / 3.13)

---

## 1. Executive summary

AgentFuse is a **runtime control plane for AI agents**: a reverse proxy that sits
between an agent and its LLM API, meters every call in real time, detects dangerous
patterns — retry loops, cost spirals, stalls, runaway call rates — and trips a
circuit breaker *before* the damage is done.

> *Tagline: Stop your agent before it burns $500 in a retry loop.*

Where observability tools read yesterday's exhaust and tell you what went wrong in a
morning report, AgentFuse sits in the request path and **intervenes while it is
happening**. Integration is a one-line `base_url` change; no SDK and no agent-code
changes are required.

| Axis | Trace-triage tools | AgentFuse |
|---|---|---|
| When | After the run (batch) | During the run (real time) |
| Role | Observe and rank | Meter and intervene |
| Integration | Upload trace files | One-line `base_url` change |
| Output | Morning report | Tripped breaker + live dashboard |

## 2. The problem

AI agents call an LLM in a loop and use tools to act. Their known failure mode is
**not knowing when they are stuck**: an agent can call the same search with the same
arguments fifty times overnight, retry a failing tool forever, or silently burn
through a budget. Every iteration is a paid API call, and each retry resends the
whole growing conversation, so cost accelerates. Teams usually discover this from
the bill.

## 3. How it works

The agent points its LLM client at AgentFuse instead of the provider:

```python
client = Anthropic(base_url="http://localhost:9000/anthropic")
# optional identity headers: X-Fuse-Agent (name), X-Fuse-Run (session id)
```

Every call then passes through a five-stage pipeline:

1. **Identify** — agent and run come from two optional headers, with defaults.
2. **Check** — a policy engine evaluates the run's recent history against four
   breakers and returns a verdict: `allow`, `warn`, `block`, or `kill`.
3. **Enforce** — *allow* forwards; *warn* forwards but records an incident and
   fires a webhook; *block* returns HTTP 429 whose body is a **model-readable
   explanation** ("you have called `search` with the same arguments 6 times in a
   row — try a different approach…"). Agents read that guidance and usually pivot,
   so blocking can *heal* a run, not just stop it. A block also resets the
   detection streak, giving the agent a fresh chance — and re-trips if it keeps
   looping. *kill* is a manual per-run switch, active until reset.
4. **Meter** — tokens are counted from the response and converted to dollars via
   a price table; the event is persisted to SQLite.
5. **Broadcast** — every event streams over Server-Sent Events to a live dashboard.

Two design principles run throughout: **fail open** (a bug or dead disk inside
AgentFuse must never take the agent down — the call is forwarded and the error
logged) and **never fail silently** (a blocked agent always gets an explanation it
can act on; a rate-limited one gets a `Retry-After` header).

## 4. Architecture

```
 Agent ──────────► AgentFuse proxy ──────────► Anthropic / OpenAI API
 (any framework)        │
                  server.py     FastAPI app: proxy routes, REST API, SSE, auth
                  parsing.py    Anthropic/OpenAI + SSE payload parsing, window builder
                  engine.py     verdict precedence, kill switch, fail-open
                  policies/     loop.py · budget.py · stall.py · rate.py
                  meter.py      tokens → dollars (prefix-matched price table)
                  store.py      SQLite event/incident log, degraded-mode buffer
                  streaming.py  SSE broadcaster          alerting.py  webhooks
                  config.py     fuse.toml loader         cli.py       fuse CLI
                  demo/         fake upstream + scripted two-agent workload
                  templates/dashboard.html   single-file live dashboard
```

Roughly 900 lines of strictly-typed Python. Policies are pure functions of an
immutable `Window` (the run's recent history), which keeps the product's brain
trivially testable.

## 5. Feature inventory

### The four circuit breakers

| Breaker | Watches for | Default | Escalation |
|---|---|---|---|
| **Loop** | same tool called with *near-identical* arguments N times in a row — volatile keys (timestamps, request ids, nonces…) are ignored so a loop can't disguise itself | N = 4 | warn at 4, block at 6, streak resets on block (heal), re-trips if looping continues |
| **Budget** | dollar spend per run and per agent per day, with per-agent overrides | $5/run, $50/agent/day | warn at 80%, block at 100% |
| **Stall** | consecutive tool-result errors | 5 | warn at 5, block at 8 |
| **Rate** | LLM calls per minute per agent | 30/min | block at cap with `Retry-After` header (seconds until the window frees up) |

Verdict precedence: the most severe verdict across all policies wins; every verdict
above `allow` is recorded as an incident. A policy that throws is skipped (fail-open).

### Metering

- Real-time token counting and dollar conversion per call, per agent, per run
- **Streaming supported** (Anthropic): SSE chunks relay to the agent byte-exact and
  untouched while AgentFuse assembles its own copy — usage read from
  `message_start`/`message_delta`, tool calls reassembled from `input_json_delta`
  fragments
- Prefix-matched price table with sensible fallback for unknown models

### Provider endpoints

- `POST /anthropic/v1/messages` — Anthropic Messages API (full support incl. streaming)
- `POST /openai/v1/chat/completions` — OpenAI-compatible; same breakers and
  metering (OpenAI streaming currently relays unmetered — on the roadmap)

### Live dashboard (`GET /`)

Single HTML file, no build step, updates over SSE:

- Total + per-agent **spend ticker**
- **Calls & spend per minute** chart (bars + spend line), backfilled from history
  so it survives page reloads
- **Breaker board** — one row per run with state badge (OK/WARN/BLOCK/KILL) and
  kill/reset buttons
- **Incident feed** with per-agent filter, showing the exact message each model saw
- **Degraded-storage banner** when SQLite is failing

### Operations & hardening

- **API auth** — optional bearer token guards all `/api/*` endpoints
  (`[server] api_token`; dashboard passes `?token=`); the agent-facing proxy
  route is unaffected
- **Config hot-reload** — `GET/POST /api/config` inspects and changes budgets and
  thresholds on a running proxy, no restart
- **Kill switch** — `POST /api/runs/{run}/kill` and `/reset`
- **Upstream visibility** — provider 5xx and connection failures are recorded as
  incidents; an unreachable upstream returns a clean 502
- **Degraded-store fallback** — SQLite write failures buffer events in memory,
  flush on recovery, and never interrupt proxying; policy checks fail open if the
  store is unreadable
- **Webhook alerts** (Slack-compatible) on warn-and-above, per-(policy, run)
  cooldown, fired in the background so a slow webhook never delays agent calls
- **Retention** — `[storage] retention_days` prunes old rows
- **Per-run serialization** — parallel calls from one run can't slip past a
  threshold together

### CLI (`fuse`)

| Command | Purpose |
|---|---|
| `fuse serve` | run the proxy + dashboard (`--host/--port/--config`) |
| `fuse demo` | zero-API-key pitch demo (`--headless` for CI) |
| `fuse status` | spend and incident counts from a running proxy |
| `fuse policies` | show active guardrails and their settings |
| `fuse export` | dump all metered events as JSON lines for billing/analysis |

### The demo (zero API key, ~60 seconds)

`fuse demo` wires the proxy to a built-in fake LLM and plays **two acts**:
a "researcher" agent walks into a retry loop (warn → warn → block → reads the
guidance → pivots tools → finishes the job), then a "coder" agent burns expensive
calls until the budget breaker warns at 80% and cuts it off at 100%. Headless mode
exits nonzero if either breaker fails to trip, making the pitch demo double as the
end-to-end test.

## 6. Quality & engineering practice

- **71 tests**, ~87% coverage, enforced at ≥85% in CI
- `mypy --strict` and `ruff` clean; `py.typed` shipped for downstream type checkers
- Built test-first throughout (every feature started as a failing test)
- CI on GitHub Actions across Python 3.11 / 3.12 / 3.13, on every push and PR
- Publishing to PyPI via **trusted publishing** (OIDC) — no stored tokens; every
  `v*` tag builds and uploads automatically
- Every change lands through a PR with green CI; version tags v0.1.0 → v0.4.0 each
  have a GitHub Release

## 7. Release history

| Version | Highlights |
|---|---|
| v0.1.0 | Core product: proxy, meter, four breakers, engine, SQLite, SSE dashboard, alerts, CLI, demo |
| v0.1.1 | Dashboard chart backfills from history (survives reloads) |
| v0.2.0 | Streaming metering (byte-exact SSE passthrough + assembly) |
| v0.3.0 | Near-identical loop matching via configurable volatile keys |
| v0.3.1 | `Retry-After` header on rate blocks |
| v0.4.0 | **First PyPI release.** OpenAI endpoint, per-agent budgets, config hot-reload, API auth, upstream visibility, degraded-store fallback, retention + export, dashboard upgrades, two-act demo |

## 8. Design decisions

- **A proxy, not an SDK** — tool activity is inferred entirely from LLM traffic,
  so any framework in any language is supported with zero agent-code changes.
- **429 with guidance instead of a silent drop** — agents retry silent failures
  forever; an explanation in the response body gives the model something to act
  on. The demo exists to prove this heals runs.
- **Fail-open everywhere** — a control plane must not become the outage: policy
  crashes are skipped, store failures buffer, metering failures never block a
  response.
- **Pure-function policies over an immutable window** — the enforcement brain is
  deterministic and unit-testable without a server.

## 9. Roadmap

- **OpenAI streaming metering** — the OpenAI endpoint currently relays SSE unmetered
- **Replay debugger** — the event log is already a recording; add a UI to step through a run
- **Chaos testing** — inject tool failures/latency at the interception layer to test agent resilience
- **Fleet mode** — multiple proxies reporting to one dashboard; org-level budgets

The strategic bet: because every agent action already flows through the
interception layer, the same component that meters and blocks today can record,
replay, and inject faults tomorrow — the proxy position is the platform.
