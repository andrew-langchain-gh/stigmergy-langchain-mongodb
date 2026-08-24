"""Change streams: how a write by one agent becomes a wake-up for another.

No agent ever calls another agent. An agent writes to the blackboard; MongoDB fans that
write out to every subscriber. Adding a ninth agent needs no change anywhere in this
file, and the dashboard is just another subscriber with no privileged access.

Resume tokens are persisted per agent, so an agent that dies and restarts replays what
it missed rather than silently skipping it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pymongo import AsyncMongoClient

from swarm import db


async def load_resume_token(client: AsyncMongoClient, agent: str) -> dict[str, Any] | None:
    doc = await db.agent_status(client).find_one({"agent": agent})
    return (doc or {}).get("resume_token")


async def save_resume_token(client: AsyncMongoClient, agent: str, token: dict[str, Any]) -> None:
    await db.agent_status(client).update_one(
        {"agent": agent}, {"$set": {"resume_token": token}}, upsert=True
    )


#: An agent reacts to *writes*. A delete carries no post-image, so it would arrive as
#: "None wrote a None" and burn a turn on a document that no longer exists — and clearing
#: the board between runs deletes every document at once. Filtering here rather than at
#: each call site means no agent can accidentally treat a wipe as a wake-up.
WRITES_ONLY = [{"$match": {"operationType": {"$in": ["insert", "update", "replace"]}}}]


def ignore_own_writes(agent: str) -> list[dict[str, Any]]:
    """Without this the swarm feedback-loops: my write wakes me, and I write again."""
    return [{"$match": {"fullDocument.last_touched_by": {"$ne": agent}}}]


async def watch_blackboard(
    client: AsyncMongoClient,
    agent: str,
    *,
    extra_stages: list[dict[str, Any]] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield blackboard change events relevant to `agent`, resuming where it left off."""
    pipeline = WRITES_ONLY + ignore_own_writes(agent) + (extra_stages or [])
    token = await load_resume_token(client, agent)

    kwargs: dict[str, Any] = {"full_document": "updateLookup"}
    if token:
        kwargs["resume_after"] = token

    try:
        stream = await db.blackboard(client).watch(pipeline, **kwargs)
    except Exception:
        # A token can go stale if the oplog rolled past it; start fresh rather than die.
        stream = await db.blackboard(client).watch(pipeline, full_document="updateLookup")

    async with stream:
        async for event in stream:
            await save_resume_token(client, agent, event["_id"])
            yield event


async def watch_incidents(client: AsyncMongoClient, agent: str) -> AsyncIterator[dict[str, Any]]:
    """One insert into `incidents` fans out to all agents at once. No coordinator."""
    async with await db.incidents(client).watch(
        [{"$match": {"operationType": "insert"}}], full_document="updateLookup"
    ) as stream:
        async for event in stream:
            yield event
