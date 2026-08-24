"""Typed write operations against the shared blackboard, plus the convergence rules.

Two things worth reading closely:

1. **Convergence is a rule, not a role.** `check_phase1_convergence` is a plain function
   any agent calls after touching a hypothesis. It is deliberately never exposed to an
   LLM as a tool — no agent ever *decides* to promote. A dedicated "triage agent" would
   just be an orchestrator wearing a disguise.

2. **Race safety comes from MongoDB, not from application locking.** Both transitions are
   single-document conditional `find_one_and_update` calls. If four agents cross the
   threshold in the same tick, the database guarantees exactly one of them observes the
   transition. This is the job you would normally reach for a coordinator to do.
"""

from __future__ import annotations

import uuid
from typing import Any

from pymongo import AsyncMongoClient, ReturnDocument

from swarm import db
from swarm.config import INCIDENT_ID
from swarm.schema import (
    Constraint,
    EvidenceType,
    Hypothesis,
    Objection,
    Observation,
    OpenQuestion,
    Option,
    to_doc,
    utcnow,
)

MIN_EVIDENCE_TYPES = 3


def _short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


# --------------------------------------------------------------------------------------
# Phase 1 — stigmergic traces
# --------------------------------------------------------------------------------------


async def post_observation(
    client: AsyncMongoClient,
    *,
    posted_by: str,
    evidence_type: EvidenceType | str,
    summary: str,
    detail: str = "",
    source_refs: list[str] | None = None,
) -> str:
    obs = Observation(
        observation_id=_short_id("obs"),
        incident_id=INCIDENT_ID,
        posted_by=posted_by,
        evidence_type=EvidenceType(evidence_type),
        summary=summary,
        detail=detail,
        source_refs=source_refs or [],
    )
    await db.blackboard(client).insert_one(to_doc(obs) | {"last_touched_by": posted_by})
    return obs.observation_id


async def create_hypothesis(
    client: AsyncMongoClient, *, created_by: str, statement: str
) -> str:
    hyp = Hypothesis(
        hypothesis_id=_short_id("H"),
        incident_id=INCIDENT_ID,
        created_by=created_by,
        statement=statement,
    )
    await db.blackboard(client).insert_one(to_doc(hyp) | {"last_touched_by": created_by})
    return hyp.hypothesis_id


async def link_observation(
    client: AsyncMongoClient,
    *,
    hypothesis_id: str,
    observation_id: str,
    agent: str,
    contradicting: bool = False,
) -> bool:
    """Attach an observation to a hypothesis, keeping the evidence-type set current."""
    coll = db.blackboard(client)
    obs = await coll.find_one({"observation_id": observation_id})
    if obs is None:
        return False

    if contradicting:
        update: dict[str, Any] = {"$addToSet": {"contradicting_observations": observation_id}}
    else:
        update = {
            "$addToSet": {
                "supporting_observations": observation_id,
                "evidence_types_covered": obs["evidence_type"],
            }
        }
    update["$set"] = {"last_touched_by": agent}
    result = await coll.update_one({"hypothesis_id": hypothesis_id}, update)
    return result.matched_count == 1


async def post_open_question(
    client: AsyncMongoClient,
    *,
    asked_by: str,
    question: str,
    context: str = "",
    hypothesis_id: str | None = None,
) -> str:
    """Leave a question in shared memory. Nobody is assigned to it.

    Whichever agent's domain happens to match picks it up purely because the trace is
    sitting there. This *is* the stigmergy mechanism: the trace is the task assignment.
    """
    question_doc = OpenQuestion(
        question_id=_short_id("q"),
        incident_id=INCIDENT_ID,
        asked_by=asked_by,
        question=question,
        context=context,
        hypothesis_id=hypothesis_id,
    )
    await db.blackboard(client).insert_one(to_doc(question_doc) | {"last_touched_by": asked_by})
    if hypothesis_id:
        await db.blackboard(client).update_one(
            {"hypothesis_id": hypothesis_id},
            {"$inc": {"open_question_count": 1}, "$set": {"last_touched_by": asked_by}},
        )
    return question_doc.question_id


async def answer_open_question(
    client: AsyncMongoClient, *, question_id: str, answered_by: str, answer: str
) -> str | None:
    """Answer a question. Atomic so two agents cannot both claim the same one."""
    coll = db.blackboard(client)
    doc = await coll.find_one_and_update(
        {"question_id": question_id, "status": "open"},
        {
            "$set": {
                "status": "answered",
                "answered_by": answered_by,
                "answer": answer,
                "last_touched_by": answered_by,
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if doc is None:
        return None
    if doc.get("hypothesis_id"):
        await coll.update_one(
            {"hypothesis_id": doc["hypothesis_id"]},
            {"$inc": {"open_question_count": -1}, "$set": {"last_touched_by": answered_by}},
        )
    return doc.get("hypothesis_id")


# --------------------------------------------------------------------------------------
# The hinge — phase 1 to phase 2
# --------------------------------------------------------------------------------------


async def check_phase1_convergence(
    client: AsyncMongoClient, hypothesis_id: str, agent: str = "rule"
) -> dict[str, Any] | None:
    """Promote a hypothesis iff the rule holds. Returns the document only to the winner.

    The rule, expressed entirely in the query filter so it is evaluated atomically:

    * at least 3 distinct evidence types corroborating (`evidence_types_covered.2` exists)
    * zero unresolved contradicting observations
    * zero open questions still referencing it

    The very same write that flips `status` seeds phase 2 by adding `options` and
    `constraints` to *this* document. Nothing centralised ever decided to change modes.
    """
    return await db.blackboard(client).find_one_and_update(
        {
            "hypothesis_id": hypothesis_id,
            "doc_type": "hypothesis",
            "status": "candidate",
            f"evidence_types_covered.{MIN_EVIDENCE_TYPES - 1}": {"$exists": True},
            "contradicting_observations": {"$size": 0},
            "open_question_count": 0,
        },
        {
            "$set": {
                "status": "confirmed",
                "phase": "remediation",
                "options": [],
                "constraints": [],
                "constraint_count": 0,
                "promoted_at": utcnow(),
                # Recorded separately from last_touched_by, which later writes overwrite.
                # This is the agent whose write happened to cross the threshold — worth
                # being able to name, since the point is that it could have been any of them.
                "promoted_by": agent,
                "last_touched_by": agent,
            }
        },
        return_document=ReturnDocument.AFTER,
    )


# --------------------------------------------------------------------------------------
# Phase 2 — negotiation by editing
# --------------------------------------------------------------------------------------


async def propose_option(
    client: AsyncMongoClient, *, hypothesis_id: str, proposed_by: str, action: str, eta: str, rationale: str = ""
) -> str | None:
    option = Option(
        option_id=_short_id("opt"), proposed_by=proposed_by, action=action, eta=eta, rationale=rationale
    )
    result = await db.blackboard(client).update_one(
        {"hypothesis_id": hypothesis_id, "phase": "remediation", "status": "confirmed"},
        {"$push": {"options": to_doc(option)}, "$set": {"last_touched_by": proposed_by}},
    )
    return option.option_id if result.modified_count == 1 else None


async def attach_objection(
    client: AsyncMongoClient,
    *,
    hypothesis_id: str,
    option_id: str,
    by: str,
    objection: str,
    severity: str,
    blocking: bool,
) -> str | None:
    """Disagreement enters the artifact as structure, not as a chat turn."""
    obj = Objection(
        objection_id=_short_id("obj"),
        by=by,
        objection=objection,
        severity=severity,  # type: ignore[arg-type]
        blocking=blocking,
    )
    result = await db.blackboard(client).update_one(
        {"hypothesis_id": hypothesis_id, "options.option_id": option_id},
        {"$push": {"options.$.objections": to_doc(obj)}, "$set": {"last_touched_by": by}},
    )
    return obj.objection_id if result.modified_count == 1 else None


async def withdraw_objection(
    client: AsyncMongoClient, *, hypothesis_id: str, objection_id: str, reason: str
) -> bool:
    result = await db.blackboard(client).update_one(
        {"hypothesis_id": hypothesis_id},
        {
            "$set": {
                "options.$[o].objections.$[j].withdrawn": True,
                "options.$[o].objections.$[j].withdrawn_reason": reason,
                "last_touched_by": "risk",
            }
        },
        array_filters=[{"o.objections.objection_id": objection_id}, {"j.objection_id": objection_id}],
    )
    return result.modified_count == 1


async def attach_constraint(
    client: AsyncMongoClient,
    *,
    hypothesis_id: str,
    by: str,
    type: str,
    detail: str,
    deadline: str | None = None,
) -> str | None:
    constraint = Constraint(
        constraint_id=_short_id("c"),
        by=by,
        type=type,  # type: ignore[arg-type]
        detail=detail,
        deadline=deadline,
    )
    result = await db.blackboard(client).update_one(
        {"hypothesis_id": hypothesis_id, "phase": "remediation"},
        {
            "$push": {"constraints": to_doc(constraint)},
            "$inc": {"constraint_count": 1},
            "$set": {"last_touched_by": by},
        },
    )
    return constraint.constraint_id if result.modified_count == 1 else None


async def select_option(
    client: AsyncMongoClient,
    *,
    hypothesis_id: str,
    option_id: str,
    selected_by: str,
    constraints_seen: int,
) -> dict[str, Any] | None:
    """Resolve the incident, iff the option carries no live blocking objection.

    Two guarantees ride on this single filter:

    * `$elemMatch` scopes the objection test to *this* option, not to every option on the
      document — a whole-document test would let one option's veto block a different one.
    * `constraint_count` is an optimistic-concurrency check. If comms attached a new
      constraint after scheduling read the document, this write fails rather than
      selecting against a stale view.
    """
    return await db.blackboard(client).find_one_and_update(
        {
            "hypothesis_id": hypothesis_id,
            "phase": "remediation",
            "status": "confirmed",
            "constraint_count": constraints_seen,
            "options": {
                "$elemMatch": {
                    "option_id": option_id,
                    "objections": {
                        "$not": {"$elemMatch": {"blocking": True, "withdrawn": {"$ne": True}}}
                    },
                }
            },
        },
        {
            "$set": {
                "options.$.status": "selected",
                "status": "resolved",
                "selected_by": selected_by,
                "resolved_at": utcnow(),
                "last_touched_by": selected_by,
            }
        },
        return_document=ReturnDocument.AFTER,
    )


# --------------------------------------------------------------------------------------
# Escape valves — exogenous, and named out loud in the talk rather than hidden
# --------------------------------------------------------------------------------------


async def force_promote_best_candidate(client: AsyncMongoClient) -> dict[str, Any] | None:
    """Stigmergic convergence does not naturally terminate; it needs a designed timeout."""
    coll = db.blackboard(client)
    candidates = await coll.find(
        {"doc_type": "hypothesis", "status": "candidate", "incident_id": INCIDENT_ID}
    ).to_list(length=None)
    if not candidates:
        return None
    best = max(candidates, key=lambda h: len(h.get("evidence_types_covered", [])))
    return await coll.find_one_and_update(
        {"hypothesis_id": best["hypothesis_id"], "status": "candidate"},
        {
            "$set": {
                "status": "confirmed",
                "phase": "remediation",
                "options": [],
                "constraints": [],
                "constraint_count": 0,
                "confidence": "low",
                "forced": True,
                "promoted_at": utcnow(),
                "promoted_by": "watchdog (timeout)",
                "last_touched_by": "watchdog",
            }
        },
        return_document=ReturnDocument.AFTER,
    )


async def force_select_least_objected(client: AsyncMongoClient) -> dict[str, Any] | None:
    coll = db.blackboard(client)
    doc = await coll.find_one(
        {"doc_type": "hypothesis", "phase": "remediation", "status": "confirmed"}
    )
    if not doc or not doc.get("options"):
        return None

    def live_blocking(option: dict[str, Any]) -> int:
        return sum(
            1
            for o in option.get("objections", [])
            if o.get("blocking") and not o.get("withdrawn")
        )

    best = min(doc["options"], key=live_blocking)
    index = doc["options"].index(best)
    return await coll.find_one_and_update(
        {"hypothesis_id": doc["hypothesis_id"], "status": "confirmed"},
        {
            "$set": {
                f"options.{index}.status": "selected",
                "status": "resolved",
                "forced": True,
                "unresolved_objection_count": live_blocking(best),
                "resolved_at": utcnow(),
                "last_touched_by": "watchdog",
            }
        },
        return_document=ReturnDocument.AFTER,
    )


# --------------------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------------------


async def candidate_hypotheses(client: AsyncMongoClient) -> list[dict[str, Any]]:
    return await (
        db.blackboard(client)
        .find(
            {"doc_type": "hypothesis", "incident_id": INCIDENT_ID, "status": "candidate"},
            {"_id": 0, "hypothesis_id": 1, "statement": 1, "evidence_types_covered": 1},
        )
        .to_list(length=None)
    )


async def question_asker(client: AsyncMongoClient, question_id: str) -> str | None:
    doc = await db.blackboard(client).find_one({"question_id": question_id}, {"asked_by": 1})
    return (doc or {}).get("asked_by")


async def snapshot(client: AsyncMongoClient) -> dict[str, Any]:
    coll = db.blackboard(client)
    docs = await coll.find({"incident_id": INCIDENT_ID}, {"_id": 0}).to_list(length=None)
    return {
        "observations": [d for d in docs if d["doc_type"] == "observation"],
        "hypotheses": [d for d in docs if d["doc_type"] == "hypothesis"],
        "open_questions": [d for d in docs if d["doc_type"] == "open_question"],
    }


async def focal_document(client: AsyncMongoClient) -> dict[str, Any] | None:
    """The one document the audience watches morph across both phases."""
    coll = db.blackboard(client)
    doc = await coll.find_one(
        {"doc_type": "hypothesis", "incident_id": INCIDENT_ID, "phase": "remediation"},
        {"_id": 0},
        sort=[("promoted_at", -1)],
    )
    if doc:
        return doc
    return await coll.find_one(
        {"doc_type": "hypothesis", "incident_id": INCIDENT_ID},
        {"_id": 0},
        sort=[("created_at", -1)],
    )
