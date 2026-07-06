from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from agentfuse.models import hash_args
from agentfuse.parsing import parse_sse_response

SSE_BODY = (
    'event: message_start\n'
    'data: {"type":"message_start","message":{"id":"m1","usage":'
    '{"input_tokens":500,"output_tokens":1}}}\n'
    '\n'
    'event: content_block_start\n'
    'data: {"type":"content_block_start","index":0,"content_block":'
    '{"type":"text","text":""}}\n'
    '\n'
    'event: content_block_delta\n'
    'data: {"type":"content_block_delta","index":0,"delta":'
    '{"type":"text_delta","text":"let me search"}}\n'
    '\n'
    'event: content_block_start\n'
    'data: {"type":"content_block_start","index":1,"content_block":'
    '{"type":"tool_use","id":"t1","name":"search","input":{}}}\n'
    '\n'
    'event: content_block_delta\n'
    'data: {"type":"content_block_delta","index":1,"delta":'
    '{"type":"input_json_delta","partial_json":"{\\"query\\":"}}\n'
    '\n'
    'event: content_block_delta\n'
    'data: {"type":"content_block_delta","index":1,"delta":'
    '{"type":"input_json_delta","partial_json":"\\"agent frameworks\\"}"}}\n'
    '\n'
    'event: message_delta\n'
    'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},'
    '"usage":{"output_tokens":60}}\n'
    '\n'
    'event: message_stop\n'
    'data: {"type":"message_stop"}\n'
    '\n'
)


def test_parse_sse_extracts_usage_and_assembled_tool_calls() -> None:
    tin, tout, calls = parse_sse_response(SSE_BODY)
    assert (tin, tout) == (500, 60)
    (call,) = calls
    assert call.name == "search"
    assert call.args_hash == hash_args({"query": "agent frameworks"})


def test_parse_sse_garbage_yields_zeros() -> None:
    assert parse_sse_response("not sse at all") == (0, 0, ())


def sse_upstream() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/messages")
    async def messages() -> StreamingResponse:
        async def gen() -> Any:
            for chunk in SSE_BODY.encode().splitlines(keepends=True):
                yield chunk

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def stream_call(client: Any, run: str = "r1") -> Any:
    return client.post(
        "/anthropic/v1/messages",
        json={"model": "claude-haiku-4-5", "stream": True,
              "messages": [{"role": "user", "content": "go"}]},
        headers={"X-Fuse-Agent": "a1", "X-Fuse-Run": run},
    )


def make_streaming_client() -> Any:
    import httpx
    from fastapi.testclient import TestClient

    from agentfuse.config import FuseConfig
    from agentfuse.server import create_app

    upstream = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=sse_upstream()),
        base_url="https://fake-upstream",
    )
    return TestClient(create_app(FuseConfig(db_path=":memory:"), upstream_client=upstream))


def test_streaming_passthrough_is_byte_exact_and_metered() -> None:
    client = make_streaming_client()
    r = stream_call(client)
    assert r.status_code == 200
    assert r.text == SSE_BODY  # agent sees the stream untouched
    status = client.get("/api/status").json()
    # haiku: 500 in @ $1/M + 60 out @ $5/M
    assert status["spend_by_agent"]["a1"] > 0


def test_streaming_calls_are_protected_by_breakers() -> None:
    client = make_streaming_client()
    # the SSE body carries the same search tool_use every call -> loop trips
    codes = [stream_call(client).status_code for _ in range(8)]
    assert 429 in codes
    incidents = client.get("/api/status").json()["incidents"]
    assert any(i["policy"] == "loop" for i in incidents)
