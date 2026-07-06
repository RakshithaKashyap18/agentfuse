# AgentFuse

Runtime control plane for AI agents: a reverse proxy that meters every LLM call in
real time, detects dangerous patterns (retry loops, cost spirals, stalls), and trips
a circuit breaker before the damage is done.

*Stop your agent before it burns $500 in a retry loop.*
