"""Read-only access to each agent's own data source.

Every agent sees a different slice of the incident. None of them can see the whole
picture alone — which is the property that makes the federated premise honest rather
than imposed.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain.tools import tool
from pymongo import AsyncMongoClient

from swarm import db
from swarm.config import settings

# Real observability queries take seconds. Keeping that latency here is what gives the
# presenter a window to kill an agent *during* a tool call for the checkpoint beat.
QUERY_LATENCY_S = 1.5


def _dump(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        row.pop("_id", None)
    return json.dumps(rows, indent=2, default=str)


def build_logs_tool(client: AsyncMongoClient):
    @tool
    async def query_logs(service: str, since: str = "14:20", until: str = "14:50") -> str:
        """Search application logs for a service in a HH:MM time window.

        Returns log lines with level, logger, message, occurrence count and stack traces.
        """
        await asyncio.sleep(QUERY_LATENCY_S)
        rows = (
            await db.db(client)[db.EVIDENCE_LOGS]
            .find({"service": service, "timestamp": {"$gte": since, "$lte": until + ":99"}})
            .sort("timestamp", 1)
            .to_list(length=None)
        )
        return _dump(rows)

    return query_logs


def build_metrics_tool(client: AsyncMongoClient):
    @tool
    async def query_metrics(service: str, since: str = "14:20", until: str = "14:50") -> str:
        """Fetch the per-minute metric series for a service in a HH:MM time window.

        Includes p50/p99 latency, error rate, request volume, and database connection
        pool gauges (active, max, and time spent waiting for a connection).
        """
        await asyncio.sleep(QUERY_LATENCY_S)
        rows = (
            await db.db(client)[db.EVIDENCE_METRICS]
            .find({"service": service, "timestamp": {"$gte": since, "$lte": until}})
            .sort("timestamp", 1)
            .to_list(length=None)
        )
        return _dump(rows)

    return query_metrics


def build_deploys_tool(client: AsyncMongoClient):
    @tool
    async def query_deploys(since: str = "13:00", until: str = "15:00", service: str = "") -> str:
        """List deploys, config pushes and feature-flag changes in a HH:MM time window.

        Leave `service` empty to see changes across all services — a change to a
        neighbouring service is often the cause.
        """
        await asyncio.sleep(QUERY_LATENCY_S)
        query: dict[str, Any] = {"timestamp": {"$gte": since, "$lte": until}}
        if service:
            query["service"] = service
        rows = (
            await db.db(client)[db.EVIDENCE_DEPLOYS]
            .find(query)
            .sort("timestamp", 1)
            .to_list(length=None)
        )
        return _dump(rows)

    return query_deploys


def build_ticket_search_tool(store):
    @tool
    async def search_support_tickets(query: str, limit: int = 6) -> str:
        """Semantically search customer support tickets by meaning, not keywords.

        Customers describe symptoms in their own words and almost never name the service
        or the error. Describe the *symptom* you are looking for rather than internal
        terminology. Each result carries a relevance score — low-scoring results are
        probably unrelated tickets that happened to be open at the time.
        """
        await asyncio.sleep(QUERY_LATENCY_S)
        hits = await store.asearch(db.TICKET_NAMESPACE, query=query, limit=limit)
        rows = []
        for hit in hits:
            value = dict(hit.value)
            value.pop("text", None)
            rows.append({"relevance": round(hit.score, 3), **value})
        return json.dumps(rows, indent=2, default=str)

    return search_support_tickets


def build_keyword_ticket_tool(client: AsyncMongoClient):
    """The deliberate foil for the semantic-search beat.

    Kept in the customer-impact agent's toolbox so the audience can watch an exact-field
    search return nothing on the same corpus that semantic search reads fine.
    """

    @tool
    async def keyword_search_tickets(term: str) -> str:
        """Search support tickets for an exact keyword or phrase (case-insensitive)."""
        await asyncio.sleep(QUERY_LATENCY_S)
        rows = (
            await db.db(client)[db.EVIDENCE_TICKETS]
            .find({"$or": [
                {"subject": {"$regex": term, "$options": "i"}},
                {"body": {"$regex": term, "$options": "i"}},
            ]})
            .to_list(length=None)
        )
        return _dump(rows) if rows else f"No tickets matched the keyword {term!r}."

    return keyword_search_tickets
