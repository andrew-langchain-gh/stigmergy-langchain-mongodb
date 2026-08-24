"""Load the fabricated INC-1042 dataset and build the vector index.

The evidence is fixed so the demo tells the same story every run. What stays live is
each agent's retrieval and reasoning over it — the dataset constrains the conclusions,
not the path taken to reach them.

Note what this does *not* insert: the incident document. That arrives later, from
`swarm trigger`, standing in for a dumb threshold monitor. Nothing agentic starts the
incident.
"""

from __future__ import annotations

import json

from swarm import db
from swarm.config import DATA_DIR, settings


def _load(name: str):
    with open(DATA_DIR / f"{name}.json") as handle:
        return json.load(handle)


def _ticket_text(ticket: dict) -> str:
    return f"{ticket['subject']}. {ticket['body']}"


async def seed(*, reset: bool = True) -> dict[str, int]:
    """Populate evidence collections and the vector-indexed ticket corpus."""
    client = db.async_client()
    database = db.db(client)

    if reset:
        for name in (
            db.INCIDENTS,
            db.BLACKBOARD,
            db.AGENT_STATUS,
            db.EVIDENCE_LOGS,
            db.EVIDENCE_METRICS,
            db.EVIDENCE_DEPLOYS,
            db.EVIDENCE_TICKETS,
            db.CHECKPOINTS,
            db.CHECKPOINT_WRITES,
        ):
            await database[name].delete_many({})

    logs, metrics, deploys, tickets = (
        _load("logs"),
        _load("metrics"),
        _load("deploys"),
        _load("tickets"),
    )

    await database[db.EVIDENCE_LOGS].insert_many(logs)
    await database[db.EVIDENCE_METRICS].insert_many(metrics)
    await database[db.EVIDENCE_DEPLOYS].insert_many(deploys)
    await database[db.EVIDENCE_TICKETS].insert_many(tickets)
    await db.bootstrap_indexes(client)
    await client.close()

    # The customer-impact agent reaches tickets only through BaseStore.search(), so the
    # corpus has to live in the store with an Atlas Vector Search index over it.
    sync = db.sync_client()
    if reset:
        sync[settings.db_name][db.STORE].delete_many({})
    store = db.make_store(sync)
    for ticket in tickets:
        store.put(
            db.TICKET_NAMESPACE,
            ticket["ticket_id"],
            {"text": _ticket_text(ticket), **ticket},
        )
    sync.close()

    from swarm.procedural import seed_procedures

    lessons = seed_procedures()

    return {
        "procedures": lessons,
        "logs": len(logs),
        "metrics": len(metrics),
        "deploys": len(deploys),
        "tickets": len(tickets),
    }
