from __future__ import annotations

from typing import Protocol

from agentfuse.models import Verdict, Window


class Policy(Protocol):
    name: str

    def evaluate(self, window: Window) -> Verdict: ...
