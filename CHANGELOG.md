# Changelog

## v0.3.2 (unreleased)

- PyPI publish workflow via GitHub Actions trusted publishing (no stored tokens)
- Fixed wheel build: redundant force-include duplicated the dashboard template
- `py.typed` marker so downstream type checkers see AgentFuse's annotations
- CI now tests Python 3.11, 3.12, and 3.13

## v0.3.1 (2026-07-06)

- Rate-limit 429s carry a `Retry-After` header: seconds until the oldest call
  in the rolling minute ages out, honored automatically by HTTP clients

## v0.3.0 (2026-07-06)

- Near-identical loop matching: volatile keys (timestamps, request ids, nonces,
  trace ids, idempotency keys) are dropped from tool arguments before hashing,
  so a loop can't disguise itself; configurable via `[policies.loop] volatile_keys`

## v0.2.0 (2026-07-06)

- Streaming metering: `"stream": true` calls are policy-checked, relayed
  byte-exact, and metered from the assembled SSE body (previously unmetered
  and unprotected)

## v0.1.1 (2026-07-06)

- Dashboard calls/min chart backfills from stored history instead of rendering
  empty after a page reload

## v0.1.0 (2026-07-06)

Initial release: Anthropic Messages API reverse proxy with real-time cost
metering, four circuit breakers (loop / budget / stall / rate) with
warn-then-block escalation and model-readable 429 guidance, fail-open policy
engine with per-run kill switch, SQLite persistence, SSE dashboard,
Slack-compatible webhook alerts, `fuse` CLI, and a zero-API-key demo.
