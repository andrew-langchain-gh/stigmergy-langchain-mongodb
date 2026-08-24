"""MongoDB handles and collection names.

Every piece of system state lives in one of these collections. Nothing important
lives in an agent's process memory — that is what makes the swarm inspectable by
reading the database, and what makes killing an agent survivable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymongo import AsyncMongoClient, MongoClient
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase

from swarm.config import settings

if TYPE_CHECKING:
    from langgraph.checkpoint.mongodb import MongoDBSaver
    from langgraph.store.mongodb import MongoDBStore

INCIDENTS = "incidents"
BLACKBOARD = "blackboard"
AGENT_STATUS = "agent_status"
CONTROL = "control"

EVIDENCE_LOGS = "evidence_logs"
EVIDENCE_METRICS = "evidence_metrics"
EVIDENCE_DEPLOYS = "evidence_deploys"
EVIDENCE_TICKETS = "evidence_tickets"

STORE = "store"
CHECKPOINTS = "checkpoints"
CHECKPOINT_WRITES = "checkpoint_writes"

TICKET_NAMESPACE = ("evidence", "tickets")
PROCEDURES_NAMESPACE = "procedures"
VECTOR_INDEX_NAME = "store_vector_index"


def sync_client() -> MongoClient:
    """MongoDBStore and MongoDBSaver both take a *synchronous* client.

    Their `a*` methods wrap the blocking calls in an executor rather than using an
    async driver, so there is no AsyncMongoDBSaver / AsyncMongoDBStore to reach for.
    Change streams and our own blackboard writes use the async client below.
    """
    return MongoClient(settings.mongodb_uri)


def async_client() -> AsyncMongoClient:
    return AsyncMongoClient(settings.mongodb_uri)


def db(client: AsyncMongoClient) -> AsyncDatabase:
    return client[settings.db_name]


def blackboard(client: AsyncMongoClient) -> AsyncCollection:
    return db(client)[BLACKBOARD]


def incidents(client: AsyncMongoClient) -> AsyncCollection:
    return db(client)[INCIDENTS]


def agent_status(client: AsyncMongoClient) -> AsyncCollection:
    return db(client)[AGENT_STATUS]


def make_store(client: MongoClient, *, with_vectors: bool = True) -> "MongoDBStore":
    """The shared long-term store: ticket corpus plus per-agent procedural memory.

    One deployment, one collection, different namespaces — swarm-facing evidence under
    `("evidence", "tickets")`, private agent procedures under `("procedures", <agent>)`.
    """
    from langchain_openai import OpenAIEmbeddings
    from langgraph.store.mongodb import MongoDBStore, create_vector_index_config

    index_config = None
    if with_vectors:
        index_config = create_vector_index_config(
            dims=settings.embedding_dims,
            embed=OpenAIEmbeddings(model=settings.embedding_model),
            fields=["text"],
            name=VECTOR_INDEX_NAME,
        )
    return MongoDBStore(
        collection=client[settings.db_name][STORE],
        index_config=index_config,
        auto_index_timeout=90,
    )


def make_checkpointer(client: MongoClient) -> "MongoDBSaver":
    from langgraph.checkpoint.mongodb import MongoDBSaver

    return MongoDBSaver(
        client=client,
        db_name=settings.db_name,
        checkpoint_collection_name=CHECKPOINTS,
        writes_collection_name=CHECKPOINT_WRITES,
    )


async def enable_post_images(client: AsyncMongoClient) -> bool:
    """Ask MongoDB to record a post-image for every blackboard write.

    Without this, a change stream can only do `updateLookup`, which fetches the document
    as it is *now* rather than as it was after the event — so two quick writes can report
    the same state twice and an intermediate version is lost. Post-images give the exact
    state each write produced, which is what makes an honest revision history possible.
    """
    try:
        await db(client).command(
            {"collMod": BLACKBOARD, "changeStreamPreAndPostImages": {"enabled": True}}
        )
        return True
    except Exception:  # noqa: BLE001 — older/standalone servers simply cannot do this
        return False


async def bootstrap_indexes(client: AsyncMongoClient) -> None:
    """Ordinary indexes. The vector index is created by MongoDBStore at seed time."""
    database = db(client)
    await enable_post_images(client)
    await database[BLACKBOARD].create_index([("incident_id", 1), ("doc_type", 1)])
    await database[BLACKBOARD].create_index([("hypothesis_id", 1)], sparse=True)
    await database[BLACKBOARD].create_index([("observation_id", 1)], sparse=True)
    await database[BLACKBOARD].create_index([("question_id", 1)], sparse=True)
    await database[BLACKBOARD].create_index([("status", 1)])
    await database[AGENT_STATUS].create_index([("agent", 1)], unique=True)
    await database[EVIDENCE_LOGS].create_index([("timestamp", 1)])
    await database[EVIDENCE_METRICS].create_index([("timestamp", 1)])
    await database[EVIDENCE_DEPLOYS].create_index([("timestamp", 1)])
    await database[EVIDENCE_TICKETS].create_index([("timestamp", 1)])
