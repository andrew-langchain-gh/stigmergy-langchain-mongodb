"""Private procedural memory, one namespace per agent.

The closing image of the talk: **swarm coordination via shared blackboard memory, plus
individual improvement via private procedural memory — same MongoDB deployment, different
namespaces.**

The blackboard lives in `blackboard`, where every agent reads every trace. Procedures
live in the `store` collection under `("procedures", <agent>)`, where nobody but the
owning agent looks. An agent rewrites its own operating instructions after an incident,
so the risk agent's objections get systematically sharper over successive incidents
without anything about the coordination layer changing.

Narrated from a slide rather than demoed live — improvement across incidents is not
something a single run can show — but it is real, and `swarm procedures` will print it.
"""

from __future__ import annotations

from swarm import db

SEED_PROCEDURES: dict[str, list[str]] = {
    "risk-agent": [
        "Before objecting to a rollback, check whether the causing change bundled a "
        "security fix. INC-1042: reverting config revision 47 would have reintroduced "
        "SEC-3391. Bundled changes make rollback riskier than its ETA suggests.",
        "Reserve blocking=true for objections you can tie to a specific artifact. A "
        "blocking objection is a veto, and vetoing everything makes the swarm stall.",
    ],
    "metrics-agent": [
        "When a gauge steps to a round number instantaneously, suspect a configuration "
        "limit rather than organic exhaustion, and say so in the open question.",
        "Always check request volume before blaming load. INC-1042 looked like a traffic "
        "spike until requests_per_min turned out to be flat.",
    ],
    "impact-agent": [
        "Never keyword-search tickets for internal terms. Customers say 'the page spins', "
        "not 'connection pool timeout'. Search by symptom.",
        "Check customer_tier on every matched ticket. A single platinum account creates a "
        "contractual deadline that constrains remediation.",
    ],
}


def namespace(agent: str) -> tuple[str, str]:
    return (db.PROCEDURES_NAMESPACE, agent)


def seed_procedures() -> int:
    client = db.sync_client()
    store = db.make_store(client)
    count = 0
    for agent, lessons in SEED_PROCEDURES.items():
        for i, lesson in enumerate(lessons):
            store.put(namespace(agent), f"lesson-{i}", {"text": lesson, "agent": agent})
            count += 1
    client.close()
    return count


def read_procedures(agent: str) -> list[str]:
    client = db.sync_client()
    store = db.make_store(client)
    items = store.search(namespace(agent))
    client.close()
    return [item.value["text"] for item in items]
