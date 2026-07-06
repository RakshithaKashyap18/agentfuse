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

    def last_block_ts(self, run: str) -> float:
        return self._scalar(
            "SELECT MAX(ts) FROM incidents WHERE run = ? AND action IN ('BLOCK', 'KILL')",
            (run,))

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
