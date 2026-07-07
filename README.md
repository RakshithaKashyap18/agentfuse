# ⚡ AgentFuse

**Stop your agent before it burns $500 in a retry loop.**

AgentFuse is a runtime control plane for AI agents: a reverse proxy that sits between
your agents and the Anthropic API, meters every call in real time, detects dangerous
patterns — retry loops, cost spirals, stalls — and trips a circuit breaker *before*
the damage is done.

## Why this exists

Agent observability tools read yesterday's exhaust. They ingest trace files after the
run is over and tell you, in the morning report, that your researcher agent called
`search` with the same arguments forty times overnight and spent $500 doing it.

AgentFuse sits in the request path and intervenes while it is happening:

| Axis | Trace-triage tools | AgentFuse |
|---|---|---|
| When | After the run (batch) | During the run (real time) |
| Role | Observe and rank | Meter and intervene |
| Integration | Upload trace files | One-line `base_url` change |
| Output | Morning report | Tripped breaker + live dashboard |

## 60-second demo — no API key needed

```bash
pip install agentfuse1
fuse demo
```

`fuse demo` starts the proxy against a built-in fake upstream and plays two acts.
Act one: a scripted "researcher" agent walks into a retry loop — spend ticks up,
the loop breaker warns, trips, the agent reads the guidance, pivots, and finishes
the job. Act two: a "coder" agent burns expensive calls until the budget breaker
warns at 80% and cuts it off at 100%. In CI, `fuse demo --headless` runs both arcs
and exits nonzero if either breaker fails to trip.

## Quick start

Run the proxy, then point your existing agent at it — no SDK, no code changes beyond
`base_url`:

```bash
fuse serve                     # proxy on :9000, dashboard at http://127.0.0.1:9000/
```

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:9000/anthropic")
```

Identity comes from two optional headers (any framework or language that can set a
header is supported):

- `X-Fuse-Agent` — logical agent name (default `"default"`)
- `X-Fuse-Run` — run/session id (default: one generated run per proxy process)

Your API key passes through to the real upstream untouched; AgentFuse never stores it.

Streaming (`"stream": true`) is fully supported: response chunks are relayed to the
agent the moment they arrive, while AgentFuse assembles its own copy of the stream to
meter tokens and feed the breakers — the agent notices nothing.

OpenAI-based agents get the same protection: point them at
`base_url="http://localhost:9000/openai/v1"` and calls to `chat/completions` are
metered and breaker-checked too (OpenAI streaming passes through unmetered for now).

## The four breakers

| Breaker | Watches for | Default | Escalation |
|---|---|---|---|
| **Loop** | same tool called with near-identical arguments N times in a row (volatile keys like timestamps and request ids are ignored, so a loop can't disguise itself) | N = 4 | warn at 4, block at 6 |
| **Budget** | per-run and per-agent-daily dollar spend | $5 / run, $50 / agent / day | warn at 80%, block at 100% |
| **Stall** | K consecutive `tool_result` errors | K = 5 | warn at 5, block at 8 |
| **Rate** | LLM calls per minute per agent | 30 / min | block at cap, with a `Retry-After` header saying exactly how long to wait |

A blocked call returns HTTP 429 with a **model-readable** error body — written for the
agent, not just the operator:

```json
{
  "type": "error",
  "error": {
    "type": "agentfuse_blocked",
    "message": "AgentFuse blocked this call: you have called `search` with the same arguments 6 times in a row. Try a different approach, different arguments, or a different tool."
  }
}
```

There is also a kill switch: `POST /api/runs/{run}/kill` (or the dashboard button)
blocks everything for a run until you reset it.

## Blocking that heals

The block message is a feature, not just a stop sign. From the demo transcript:

```
calls ok=8 blocked=1
  [WARN]  loop: `search` called with identical arguments 4 times in a row.
  [WARN]  loop: `search` called with identical arguments 5 times in a row.
  [BLOCK] loop: AgentFuse blocked this call: you have called `search` with the
          same arguments 6 times in a row. Try a different approach, different
          arguments, or a different tool.
demo OK: breaker tripped and agent recovered
```

The agent reads the guidance, feeds it back into its own context, switches from
`search` to `fetch_docs`, and completes the run. A block also resets the streak, so a
healed run keeps working — and an agent that ignores the guidance and keeps looping
re-trips the breaker.

## Dashboard

`fuse serve` (and `fuse demo`) serve a live dashboard at `/` — a single HTML file,
no build step, updated over Server-Sent Events:

- **Live spend ticker** — total and per-agent dollars, updating per call
- **Calls/min chart** — rolling call-rate window
- **Breaker board** — per run: OK / WARN / BLOCK / KILL, with kill and reset buttons
- **Incident feed** — every verdict above allow, with the exact message the model saw

## Configuration

All sections optional; sensible defaults throughout. See `fuse.example.toml`:

```toml
[upstream]
anthropic = "https://api.anthropic.com"

[budget]
per_run = 5.00          # dollars; omit to disable
per_agent_daily = 50.00

[policies.loop]
threshold = 4
# keys ignored when comparing tool arguments
volatile_keys = ["timestamp", "ts", "request_id", "nonce", "trace_id", "idempotency_key"]

[policies.stall]
threshold = 5

[policies.rate]
calls_per_minute = 30

[alerting]
webhook_url = ""        # Slack-compatible
cooldown_seconds = 600

[storage]
db_path = "./fuse.db"   # ":memory:" for ephemeral
retention_days = 0      # 0 = keep events forever; >0 prunes older rows

[server]
api_token = ""          # if set, /api/* requires Bearer token (dashboard: /?token=...)
```

Run with `fuse serve --config fuse.toml`. Inspect the active policies with
`fuse policies`; check a running proxy with `fuse status`; dump all metered
events as JSON lines with `fuse export` (for billing or analysis).

Thresholds and budgets can be changed on a running proxy without a restart:
`GET /api/config` shows the live values, `POST /api/config` with a JSON subset
(e.g. `{"budget_per_run": 2.5}`) hot-swaps the policies.

## Design decisions

- **Fail-open engine.** If a policy raises, the call is forwarded and the error
  logged. A control plane must not become the outage.
- **429 with guidance instead of a silent drop.** Agents retry silent failures
  forever; an explanation in the response body gives the model something to act on.
  The demo exists to prove this works.
- **A proxy, not an SDK.** Tool activity is inferred entirely from LLM traffic —
  `tool_use` blocks in responses, `tool_result` blocks in requests. Any framework,
  any language, zero agent-code changes.
- **Upstream errors pass through unchanged.** AgentFuse never masks a real API
  failure; malformed bodies are forwarded with metering skipped. Upstream 5xx
  and connection failures are recorded as incidents so the dashboard shows
  *why* your agent is failing.
- **Storage can die without taking the proxy down.** If SQLite writes fail,
  events buffer in memory, the dashboard shows a degraded banner, and calls
  keep flowing; policy checks fail open.

## Roadmap

The proxy position is the platform: every agent action already flows through it.

- **Replay debugger** — the event log is already a recording; add a UI to step
  through a run.
- **Chaos testing** — inject tool failures and latency at the interception layer to
  test agent resilience before production does.
- **OpenAI streaming metering** — the OpenAI endpoint currently passes SSE through
  unmetered, like the Anthropic endpoint did before v0.2.

## Development

```bash
pip install -e ".[dev]"
ruff check src tests && mypy && pytest --cov=agentfuse
```

A full project report — problem, architecture, complete feature inventory, release
history, and design decisions — lives at
[`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md).

MIT licensed.
