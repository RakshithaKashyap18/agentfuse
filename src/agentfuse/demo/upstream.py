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
