"""The presenter's step gate.

In `step` mode an agent may not take a turn until it claims a token, and tokens are only
minted when the presenter asks for one. So "next" means: exactly one agent, whichever one
gets there first, takes exactly one turn.

The gate is a single document and tokens are claimed with a conditional `find_one_and_update`
— the same primitive the convergence rules use. If six agents are waiting and one token is
granted, MongoDB decides who gets it. Nothing in the swarm arbitrates, and there is still
no process telling any agent what to do; the presenter is throttling the whole swarm, not
scheduling it.
"""

from __future__ import annotations

from typing import Any

from pymongo import AsyncMongoClient, ReturnDocument

from swarm import db

GATE_ID = "gate"
RUN = "run"
STEP = "step"

DEFAULT_GATE = {"_id": GATE_ID, "mode": RUN, "tokens": 0, "steps_taken": 0}


def _collection(client: AsyncMongoClient):
    return db.db(client)[db.CONTROL]


async def ensure_gate(client: AsyncMongoClient) -> None:
    await _collection(client).update_one(
        {"_id": GATE_ID}, {"$setOnInsert": DEFAULT_GATE}, upsert=True
    )


async def get_gate(client: AsyncMongoClient) -> dict[str, Any]:
    doc = await _collection(client).find_one({"_id": GATE_ID})
    return doc or dict(DEFAULT_GATE)


async def set_mode(client: AsyncMongoClient, mode: str) -> dict[str, Any]:
    """Switching to run mode clears any unspent tokens; they only mean something in step mode."""
    await ensure_gate(client)
    update: dict[str, Any] = {"$set": {"mode": mode}}
    if mode == RUN:
        update["$set"]["tokens"] = 0
    return await _collection(client).find_one_and_update(
        {"_id": GATE_ID}, update, return_document=ReturnDocument.AFTER
    )


async def grant(client: AsyncMongoClient, n: int = 1) -> dict[str, Any]:
    """Mint n step tokens. Also flips into step mode, so `next` works without setup."""
    await ensure_gate(client)
    return await _collection(client).find_one_and_update(
        {"_id": GATE_ID},
        {"$inc": {"tokens": n}, "$set": {"mode": STEP}},
        return_document=ReturnDocument.AFTER,
    )


async def claim_token(client: AsyncMongoClient) -> bool:
    """Atomically spend one token. Exactly one caller can win any given token."""
    doc = await _collection(client).find_one_and_update(
        {"_id": GATE_ID, "tokens": {"$gt": 0}},
        {"$inc": {"tokens": -1, "steps_taken": 1}},
        return_document=ReturnDocument.AFTER,
    )
    return doc is not None
