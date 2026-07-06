from __future__ import annotations

PRICES: dict[str, tuple[float, float]] = {
    "claude-opus": (15.0, 75.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku": (1.0, 5.0),
}
DEFAULT_PRICE: tuple[float, float] = (3.0, 15.0)


def price_for(model: str) -> tuple[float, float]:
    best = ""
    for prefix in PRICES:
        if model.startswith(prefix) and len(prefix) > len(best):
            best = prefix
    return PRICES[best] if best else DEFAULT_PRICE


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p_in, p_out = price_for(model)
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000
