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
