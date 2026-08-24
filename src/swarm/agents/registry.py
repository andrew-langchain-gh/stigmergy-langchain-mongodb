from __future__ import annotations

from swarm.agent_base import AgentSpec
from swarm.agents import phase1, phase2

AGENTS: dict[str, AgentSpec] = {
    spec.name: spec
    for spec in (
        phase1.LOGS,
        phase1.METRICS,
        phase1.DEPLOYS,
        phase1.IMPACT,
        phase2.ENGINEERING,
        phase2.RISK,
        phase2.COMMS,
        phase2.SCHEDULING,
    )
}

SHORT_NAMES = {name.removesuffix("-agent"): name for name in AGENTS}


def get_spec(name: str) -> AgentSpec:
    resolved = AGENTS.get(name) or AGENTS.get(SHORT_NAMES.get(name, ""))
    if resolved is None:
        raise SystemExit(f"Unknown agent {name!r}. Known: {', '.join(sorted(SHORT_NAMES))}")
    return resolved
