"""Entry point for a single agent process: `python -m swarm.agent_main <name>`."""

from __future__ import annotations

import sys

from swarm.agent_base import run_agent
from swarm.agents import get_spec


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m swarm.agent_main <agent-name>")
    run_agent(get_spec(sys.argv[1]))


if __name__ == "__main__":
    main()
