"""Phase-1 blackboard writes: the stigmergic traces.

Note what is absent: there is no `promote_hypothesis` tool and no `assign_to_agent` tool.
Promotion is a rule that fires mechanically after a write (see
`blackboard.check_phase1_convergence`), and work is never assigned — an open question is
picked up by whichever agent's domain happens to match the trace sitting in shared memory.
"""

from __future__ import annotations

import json

from langchain.tools import tool
from pymongo import AsyncMongoClient

from swarm import blackboard as bb


def build_phase1_tools(client: AsyncMongoClient, agent: str, evidence_type: str) -> list:
    @tool
    async def post_observation(summary: str, detail: str = "", source_refs: str = "") -> str:
        """Post a raw finding to the shared blackboard.

        Observations are timestamped facts from your own data source. They claim no
        correctness and no causation — state what you saw, not what it means.

        `summary` is one line: the finding itself.
        `detail` is the supporting specifics another agent would need to act on it
        without re-reading your source — exact values, timestamps, transitions, and
        anything you noticed but did not fold into the summary. Always write it.
        `source_refs` is a comma-separated list of the source record ids you read.
        """
        refs = [r.strip() for r in source_refs.split(",") if r.strip()]
        obs_id = await bb.post_observation(
            client,
            posted_by=agent,
            evidence_type=evidence_type,
            summary=summary,
            detail=detail,
            source_refs=refs,
        )
        return f"Posted observation {obs_id}."

    @tool
    async def create_hypothesis(statement: str, competes_with: str = "") -> str:
        """Create a causal claim that explains observations.

        Only for a genuinely *new* explanation. If a hypothesis already says roughly what
        yours would, link your observation to that one instead — corroboration is what
        moves an investigation forward, and a near-duplicate splits the evidence so that
        neither version ever accumulates enough support.

        To argue against an existing hypothesis rather than restate it, pass its id as
        `competes_with` and make the statement say what you think it gets wrong.
        """
        existing = await bb.candidate_hypotheses(client)
        if existing and not competes_with:
            listing = "\n".join(
                f"  {h['hypothesis_id']}: {h['statement']}" for h in existing
            )
            return (
                "Not created — hypotheses already exist for this incident:\n"
                f"{listing}\n\n"
                "If one of these covers your explanation, link your observation to it with "
                "link_observation_to_hypothesis. Only if you genuinely disagree with one, "
                "call create_hypothesis again with competes_with set to its id."
            )
        hyp_id = await bb.create_hypothesis(client, created_by=agent, statement=statement)
        return f"Created hypothesis {hyp_id}."

    @tool
    async def link_observation_to_hypothesis(
        hypothesis_id: str, observation_id: str, contradicting: bool = False
    ) -> str:
        """Attach one of your observations to a hypothesis as supporting or contradicting evidence.

        This is the write that can trip the convergence rule, because it changes how many
        distinct evidence types corroborate the hypothesis.
        """
        ok = await bb.link_observation(
            client,
            hypothesis_id=hypothesis_id,
            observation_id=observation_id,
            agent=agent,
            contradicting=contradicting,
        )
        if not ok:
            return f"Could not link — check that {hypothesis_id} and {observation_id} exist."

        promoted = await bb.check_phase1_convergence(client, hypothesis_id, agent=agent)
        if promoted:
            return (
                f"Linked {observation_id} to {hypothesis_id}. That link satisfied the "
                f"convergence rule and {hypothesis_id} is now confirmed; remediation has opened."
            )
        return f"Linked {observation_id} to {hypothesis_id}."

    @tool
    async def post_open_question(question: str, context: str = "", hypothesis_id: str = "") -> str:
        """Leave a question you cannot answer from your own data source.

        Do not address it to anyone — you have no way to call another agent and no way to
        know who could answer. State the question precisely enough that an agent looking
        at a different data source can recognise it as theirs.
        """
        qid = await bb.post_open_question(
            client,
            asked_by=agent,
            question=question,
            context=context,
            hypothesis_id=hypothesis_id or None,
        )
        return f"Posted open question {qid}."

    @tool
    async def answer_open_question(question_id: str, answer: str) -> str:
        """Answer someone else's open question from your own data source.

        Only answer if your data actually settles it. An unanswered question is far less
        damaging than a confidently wrong one, because open questions block promotion.
        """
        asker = await bb.question_asker(client, question_id)
        if asker == agent:
            return (
                "That is your own question, so answering it resolves nothing — you already "
                "had that data when you asked. Post an observation instead, and leave the "
                "question for an agent whose data source can actually settle it."
            )
        hypothesis_id = await bb.answer_open_question(
            client, question_id=question_id, answered_by=agent, answer=answer
        )
        result = f"Answered {question_id}."
        if hypothesis_id:
            promoted = await bb.check_phase1_convergence(client, hypothesis_id, agent=agent)
            if promoted:
                result += (
                    f" That cleared the last open question on {hypothesis_id}, which satisfied "
                    "the convergence rule; it is now confirmed and remediation has opened."
                )
        return result

    @tool
    async def read_blackboard() -> str:
        """Read everything on the shared blackboard for this incident.

        Other agents' observations, hypotheses and open questions. This is the only way
        you learn what anyone else has found.
        """
        snap = await bb.snapshot(client)
        return json.dumps(snap, indent=2, default=str)

    return [
        read_blackboard,
        post_observation,
        create_hypothesis,
        link_observation_to_hypothesis,
        post_open_question,
        answer_open_question,
    ]
