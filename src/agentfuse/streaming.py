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
