"""Entry point for the exogenous watchdog process."""

from __future__ import annotations

import asyncio

from swarm.watchdog import run_watchdog

if __name__ == "__main__":
    try:
        asyncio.run(run_watchdog())
    except KeyboardInterrupt:
        pass
