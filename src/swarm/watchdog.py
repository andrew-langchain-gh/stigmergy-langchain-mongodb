"""The exogenous process: fires the incident, and enforces both escape valves.

This is deliberately not an agent. Two jobs the swarm cannot honestly do for itself:

* **The trigger.** A dumb threshold monitor, standing in for PagerDuty or Datadog. If one
  of the "real" agents opened the incident, the cold start would be agentic and the
  no-orchestrator claim would be a cheat.
* **The timeouts.** Stigmergic convergence does not naturally terminate. If nothing has
  converged after N seconds, the best-corroborated candidate is force-promoted and
  flagged `confidence: low`. Phase 2 gets the symmetric treatment. This gap is real, and
  the talk names it rather than hiding it.

Everything here is keyed on a **run sequence number** stamped onto the incident. One
incident is one run, and the swarm can work several in succession without any agent
carrying budget, message history or convergence state across the boundary.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pymongo import AsyncMongoClient, ReturnDocument

from swarm import blackboard as bb
from swarm import control, db
from swarm.config import DATA_DIR, INCIDENT_ID, settings

RUN_COUNTER_ID = "run_counter"


async def next_run_seq(client: AsyncMongoClient) -> int:
    """Monotonic run number. Survives `reset`, so cycle 12 is still called 12."""
    doc = await db.db(client)[db.CONTROL].find_one_and_update(
        {"_id": RUN_COUNTER_ID},
        {"$inc": {"n": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return int(doc["n"])


async def current_run_seq(client: AsyncMongoClient) -> int:
    doc = await db.incidents(client).find_one({"incident_id": INCIDENT_ID}, {"run_seq": 1})
    return int((doc or {}).get("run_seq", 0))


async def trigger(client: AsyncMongoClient | None = None) -> dict:
    """One insert. It fans out to all eight agents at once, with no direct calls."""
    with open(DATA_DIR / "incident.json") as handle:
        incident = json.load(handle)
    owned = client is None
    client = client or db.async_client()
    try:
        incident["run_seq"] = await next_run_seq(client)
        await db.incidents(client).delete_many({"incident_id": INCIDENT_ID})
        await db.incidents(client).insert_one(dict(incident))
    finally:
        if owned:
            await client.close()
    return incident


async def reset_board(client: AsyncMongoClient, *, keep_agent_rows: bool = False) -> None:
    """Clear one run's traces. Evidence and procedural memory are untouched.

    `keep_agent_rows` is for continuous mode: deleting the agent rows would blank the
    dashboard's agent panel until each agent next happened to write a heartbeat, which on
    an unattended screen reads as a crash. Zeroing them in place keeps all eight visible.
    """
    database = db.db(client)
    await database[db.BLACKBOARD].delete_many({})
    await database[db.INCIDENTS].delete_many({})
    await database[db.CHECKPOINTS].delete_many({})
    await database[db.CHECKPOINT_WRITES].delete_many({})

    if keep_agent_rows:
        await database[db.AGENT_STATUS].update_many(
            {},
            {
                "$set": {"cycles": 0, "idle_cycles": 0, "state": "waiting"},
                "$unset": {"resume_token": "", "last_wake": ""},
            },
        )
    else:
        await database[db.AGENT_STATUS].delete_many({})

    # Leave the gate mode alone but drop unspent tokens from the previous run.
    await database[db.CONTROL].update_one(
        {"_id": control.GATE_ID}, {"$set": {"tokens": 0, "steps_taken": 0}}
    )


async def _enforce_escape_valves(
    client: AsyncMongoClient, doc: dict[str, Any] | None, elapsed: int, phase1_forced: bool
) -> bool:
    """Returns the new `phase1_forced` flag."""
    in_remediation = bool(doc and doc.get("phase") == "remediation")

    if not in_remediation and elapsed >= settings.phase1_timeout and not phase1_forced:
        forced = await bb.force_promote_best_candidate(client)
        if forced:
            print(
                f"[watchdog] phase 1 timed out at {elapsed}s; force-promoted "
                f"{forced['hypothesis_id']} with confidence=low"
            )
        else:
            print(f"[watchdog] phase 1 timed out at {elapsed}s; no candidate to promote")
        return True

    if in_remediation and elapsed >= settings.phase1_timeout + settings.phase2_timeout:
        forced = await bb.force_select_least_objected(client)
        if forced:
            print(
                f"[watchdog] phase 2 timed out at {elapsed}s; force-selected the "
                f"least-objected option with "
                f"{forced.get('unresolved_objection_count')} unresolved objections"
            )
    return phase1_forced


async def run_watchdog() -> None:
    """Enforce the escape valves, per run, for as long as this process lives.

    It does not exit when an incident resolves — in continuous mode another run follows,
    and its timeouts have to be measured from *its* own trigger rather than from process
    start. The run sequence number on the incident is what tells them apart.
    """
    client = db.async_client()
    interval = 5
    run_seq = await current_run_seq(client)
    elapsed = 0
    phase1_forced = False
    announced = False

    try:
        while True:
            await asyncio.sleep(interval)

            seen = await current_run_seq(client)
            if seen != run_seq:
                run_seq, elapsed, phase1_forced, announced = seen, 0, False, False
                print(f"[watchdog] now watching run {run_seq}")
                continue

            elapsed += interval
            doc = await bb.focal_document(client)

            if doc and doc.get("status") == "resolved":
                if not announced:
                    print(f"[watchdog] run {run_seq} resolved after {elapsed}s")
                    announced = True
                continue

            phase1_forced = await _enforce_escape_valves(client, doc, elapsed, phase1_forced)
    finally:
        await client.close()
