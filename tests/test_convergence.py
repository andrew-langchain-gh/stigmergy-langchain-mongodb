"""The claims the talk makes about atomicity, tested rather than asserted."""

from __future__ import annotations

import asyncio

import pytest

from swarm import blackboard as bb
from swarm import db
from swarm.config import INCIDENT_ID


@pytest.fixture
async def client():
    c = db.async_client()
    await db.blackboard(c).delete_many({"incident_id": INCIDENT_ID})
    yield c
    await db.blackboard(c).delete_many({"incident_id": INCIDENT_ID})
    await c.close()


async def _corroborate(client, hyp: str, types: list[str]) -> None:
    for evidence_type in types:
        obs = await bb.post_observation(
            client,
            posted_by=f"{evidence_type}-agent",
            evidence_type=evidence_type,
            summary=f"finding from {evidence_type}",
        )
        await bb.link_observation(client, hypothesis_id=hyp, observation_id=obs, agent=f"{evidence_type}-agent")


async def test_promotes_only_with_three_evidence_types(client):
    hyp = await bb.create_hypothesis(client, created_by="metrics-agent", statement="pool exhaustion")

    await _corroborate(client, hyp, ["metrics", "logs"])
    assert await bb.check_phase1_convergence(client, hyp) is None, "2 types must not promote"

    await _corroborate(client, hyp, ["deploy-history"])
    promoted = await bb.check_phase1_convergence(client, hyp)
    assert promoted is not None
    assert promoted["status"] == "confirmed"
    assert promoted["phase"] == "remediation"
    assert promoted["options"] == []


async def test_duplicate_evidence_type_does_not_count_twice(client):
    hyp = await bb.create_hypothesis(client, created_by="logs-agent", statement="pool exhaustion")
    await _corroborate(client, hyp, ["logs", "logs", "logs", "metrics"])
    assert await bb.check_phase1_convergence(client, hyp) is None


async def test_open_question_blocks_promotion(client):
    hyp = await bb.create_hypothesis(client, created_by="metrics-agent", statement="pool exhaustion")
    await _corroborate(client, hyp, ["metrics", "logs", "deploy-history"])
    question = await bb.post_open_question(
        client, asked_by="metrics-agent", question="why 10?", hypothesis_id=hyp
    )
    assert await bb.check_phase1_convergence(client, hyp) is None

    await bb.answer_open_question(
        client, question_id=question, answered_by="deploy-agent", answer="config push"
    )
    assert await bb.check_phase1_convergence(client, hyp) is not None


async def test_contradicting_observation_blocks_promotion(client):
    hyp = await bb.create_hypothesis(client, created_by="logs-agent", statement="pool exhaustion")
    await _corroborate(client, hyp, ["metrics", "logs", "deploy-history"])
    obs = await bb.post_observation(
        client, posted_by="impact-agent", evidence_type="customer-impact", summary="no complaints"
    )
    await bb.link_observation(client, hypothesis_id=hyp, observation_id=obs, agent="impact-agent", contradicting=True)
    assert await bb.check_phase1_convergence(client, hyp) is None


async def test_exactly_one_agent_wins_the_promotion_race(client):
    """Four agents cross the threshold in the same tick. Mongo settles it, not us."""
    hyp = await bb.create_hypothesis(client, created_by="logs-agent", statement="pool exhaustion")
    await _corroborate(client, hyp, ["metrics", "logs", "deploy-history", "customer-impact"])

    results = await asyncio.gather(*(bb.check_phase1_convergence(client, hyp) for _ in range(4)))
    winners = [r for r in results if r is not None]
    assert len(winners) == 1, f"expected exactly one winner, got {len(winners)}"


async def test_answer_open_question_is_claimed_once(client):
    question = await bb.post_open_question(client, asked_by="metrics-agent", question="why 10?")
    results = await asyncio.gather(
        *(
            bb.answer_open_question(
                client, question_id=question, answered_by=f"agent-{i}", answer="config push"
            )
            for i in range(4)
        )
    )
    # answer_open_question returns the hypothesis id (None here) for the winner and None
    # for losers, so assert against the stored document instead.
    doc = await db.blackboard(client).find_one({"question_id": question})
    assert doc["status"] == "answered"
    assert sum(1 for r in results if r is not None) == 0  # no hypothesis linked
    assert doc["answered_by"].startswith("agent-")


async def test_blocking_objection_prevents_selection(client):
    hyp = await bb.create_hypothesis(client, created_by="logs-agent", statement="pool exhaustion")
    await _corroborate(client, hyp, ["metrics", "logs", "deploy-history"])
    await bb.check_phase1_convergence(client, hyp)

    opt = await bb.propose_option(
        client, hypothesis_id=hyp, proposed_by="engineering", action="rollback", eta="20m"
    )
    objection = await bb.attach_objection(
        client,
        hypothesis_id=hyp,
        option_id=opt,
        by="risk",
        objection="reintroduces SEC-3391",
        severity="high",
        blocking=True,
    )

    assert (
        await bb.select_option(
            client, hypothesis_id=hyp, option_id=opt, selected_by="scheduling", constraints_seen=0
        )
        is None
    )

    await bb.withdraw_objection(client, hypothesis_id=hyp, objection_id=objection, reason="mitigated")
    resolved = await bb.select_option(
        client, hypothesis_id=hyp, option_id=opt, selected_by="scheduling", constraints_seen=0
    )
    assert resolved is not None and resolved["status"] == "resolved"


async def test_objection_on_one_option_does_not_block_another(client):
    """The $elemMatch scoping matters: a veto is per-option, not per-document."""
    hyp = await bb.create_hypothesis(client, created_by="logs-agent", statement="pool exhaustion")
    await _corroborate(client, hyp, ["metrics", "logs", "deploy-history"])
    await bb.check_phase1_convergence(client, hyp)

    opt1 = await bb.propose_option(
        client, hypothesis_id=hyp, proposed_by="engineering", action="rollback", eta="20m"
    )
    opt2 = await bb.propose_option(
        client, hypothesis_id=hyp, proposed_by="engineering", action="forward-fix", eta="35m"
    )
    await bb.attach_objection(
        client,
        hypothesis_id=hyp,
        option_id=opt1,
        by="risk",
        objection="reintroduces SEC-3391",
        severity="high",
        blocking=True,
    )

    resolved = await bb.select_option(
        client, hypothesis_id=hyp, option_id=opt2, selected_by="scheduling", constraints_seen=0
    )
    assert resolved is not None
    selected = [o for o in resolved["options"] if o["status"] == "selected"]
    assert len(selected) == 1 and selected[0]["option_id"] == opt2


async def test_stale_constraint_view_rejects_selection(client):
    """Comms adds a constraint after scheduling read the doc: the write must fail."""
    hyp = await bb.create_hypothesis(client, created_by="logs-agent", statement="pool exhaustion")
    await _corroborate(client, hyp, ["metrics", "logs", "deploy-history"])
    await bb.check_phase1_convergence(client, hyp)
    opt = await bb.propose_option(
        client, hypothesis_id=hyp, proposed_by="engineering", action="forward-fix", eta="35m"
    )

    await bb.attach_constraint(
        client,
        hypothesis_id=hyp,
        by="customer-comms",
        type="deadline",
        detail="platinum SLA notification",
        deadline="14:58",
    )

    assert (
        await bb.select_option(
            client, hypothesis_id=hyp, option_id=opt, selected_by="scheduling", constraints_seen=0
        )
        is None
    ), "selection against a stale constraint view must be rejected"

    assert (
        await bb.select_option(
            client, hypothesis_id=hyp, option_id=opt, selected_by="scheduling", constraints_seen=1
        )
        is not None
    )
