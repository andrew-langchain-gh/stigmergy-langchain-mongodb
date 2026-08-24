"""Phase-2 blackboard writes: negotiation by editing.

Each role gets a different, narrow slice of these. Risk cannot propose options,
engineering cannot veto them, and only scheduling can select one. Disagreement therefore
has to enter the document as *structure* — an objection with a severity and a blocking
flag, attached to a specific option — rather than as a natural-language turn.

That constraint is the point. Agents arguing in prose would produce a transcript nobody
can query; agents editing a typed artifact produce a document whose current state is the
answer, and whose conflicts a database query can find.
"""

from __future__ import annotations

import json

from langchain.tools import tool
from pymongo import AsyncMongoClient

from swarm import blackboard as bb


def _read_proposal_tool(client: AsyncMongoClient):
    @tool
    async def read_proposal() -> str:
        """Read the current remediation proposal: options, objections and constraints."""
        doc = await bb.focal_document(client)
        return json.dumps(doc, indent=2, default=str) if doc else "No proposal open yet."

    return read_proposal


def build_engineering_tools(client: AsyncMongoClient, agent: str) -> list:
    @tool
    async def propose_option(action: str, eta: str, rationale: str = "") -> str:
        """Add a candidate remediation to the proposal.

        Propose genuinely different approaches rather than variations on one — the point
        of multiple options is to give risk and scheduling a real choice. Give an honest
        ETA; it is what the deadline constraint gets checked against.
        """
        doc = await bb.focal_document(client)
        if not doc:
            return "No proposal is open yet."
        opt_id = await bb.propose_option(
            client,
            hypothesis_id=doc["hypothesis_id"],
            proposed_by=agent,
            action=action,
            eta=eta,
            rationale=rationale,
        )
        return f"Proposed option {opt_id}." if opt_id else "Could not add option."

    return [_read_proposal_tool(client), propose_option]


def build_risk_tools(client: AsyncMongoClient, agent: str) -> list:
    @tool
    async def attach_objection(
        option_id: str, objection: str, severity: str, blocking: bool
    ) -> str:
        """Attach a risk objection to one specific option.

        `severity` is one of low, medium, high. Set `blocking` true only when the option
        must not ship as written — a blocking objection makes that option unselectable
        until it is withdrawn, so it is a veto, not a comment. Ground every objection in
        something on the blackboard rather than in generic caution.
        """
        doc = await bb.focal_document(client)
        if not doc:
            return "No proposal is open yet."
        obj_id = await bb.attach_objection(
            client,
            hypothesis_id=doc["hypothesis_id"],
            option_id=option_id,
            by=agent,
            objection=objection,
            severity=severity,
            blocking=blocking,
        )
        return f"Attached objection {obj_id} to {option_id}." if obj_id else "Unknown option."

    @tool
    async def withdraw_objection(objection_id: str, reason: str) -> str:
        """Withdraw one of your objections once it has been addressed or mitigated."""
        doc = await bb.focal_document(client)
        if not doc:
            return "No proposal is open yet."
        ok = await bb.withdraw_objection(
            client, hypothesis_id=doc["hypothesis_id"], objection_id=objection_id, reason=reason
        )
        return f"Withdrew {objection_id}." if ok else "Unknown objection."

    return [_read_proposal_tool(client), attach_objection, withdraw_objection]


def build_comms_tools(client: AsyncMongoClient, agent: str) -> list:
    @tool
    async def attach_constraint(type: str, detail: str, deadline: str = "") -> str:
        """Attach a proposal-level constraint that any chosen option must satisfy.

        `type` is one of deadline, policy, capacity. Constraints apply to the proposal as
        a whole rather than to a single option. Use `deadline` (HH:MM) for contractual
        notification windows.
        """
        doc = await bb.focal_document(client)
        if not doc:
            return "No proposal is open yet."
        cid = await bb.attach_constraint(
            client,
            hypothesis_id=doc["hypothesis_id"],
            by=agent,
            type=type,
            detail=detail,
            deadline=deadline or None,
        )
        return f"Attached constraint {cid}." if cid else "Could not attach constraint."

    return [_read_proposal_tool(client), attach_constraint]


def build_scheduling_tools(client: AsyncMongoClient, agent: str) -> list:
    @tool
    async def select_option(option_id: str, justification: str) -> str:
        """Select the option to execute, resolving the incident.

        You are the only agent that can do this. The write is conditional: it succeeds
        only if the option carries no live blocking objection and no constraint has been
        added since you last read the proposal. If it fails, re-read and reconsider
        rather than retrying blindly — the document changed underneath you, which is
        information.
        """
        doc = await bb.focal_document(client)
        if not doc:
            return "No proposal is open yet."
        resolved = await bb.select_option(
            client,
            hypothesis_id=doc["hypothesis_id"],
            option_id=option_id,
            selected_by=agent,
            constraints_seen=doc.get("constraint_count", 0),
        )
        if resolved:
            return f"Selected {option_id}. Incident resolved. Justification: {justification}"
        return (
            f"Selection of {option_id} was rejected by the database. Either it still has a "
            "live blocking objection, or a constraint was added since you read the proposal. "
            "Re-read the proposal."
        )

    return [_read_proposal_tool(client), select_option]
