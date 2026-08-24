"""Document shapes for the shared blackboard.

Three phase-1 trace types (observation, hypothesis, open_question) and the phase-2
structures that grow onto a *confirmed hypothesis* rather than into a new collection.
One `_id`'s revision history therefore spans the whole incident: bare trigger →
corroborated hypothesis → negotiated plan.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EvidenceType(StrEnum):
    LOGS = "logs"
    METRICS = "metrics"
    DEPLOY_HISTORY = "deploy-history"
    CUSTOMER_IMPACT = "customer-impact"


class DocType(StrEnum):
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    OPEN_QUESTION = "open_question"


class HypothesisStatus(StrEnum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    RESOLVED = "resolved"


class Observation(BaseModel):
    doc_type: Literal[DocType.OBSERVATION] = DocType.OBSERVATION
    observation_id: str
    incident_id: str
    posted_by: str
    evidence_type: EvidenceType
    summary: str
    detail: str = ""
    source_refs: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class Objection(BaseModel):
    objection_id: str
    by: str
    objection: str
    severity: Literal["low", "medium", "high"]
    blocking: bool
    withdrawn: bool = False
    withdrawn_reason: str | None = None


class Option(BaseModel):
    option_id: str
    proposed_by: str
    action: str
    eta: str
    rationale: str = ""
    objections: list[Objection] = Field(default_factory=list)
    status: Literal["proposed", "selected", "rejected"] = "proposed"


class Constraint(BaseModel):
    constraint_id: str
    by: str
    type: Literal["deadline", "policy", "capacity"]
    detail: str
    deadline: str | None = None


class Hypothesis(BaseModel):
    """Phase-1 claim that may grow phase-2 fields in place.

    `evidence_types_covered`, `contradicting_observations` and `open_question_count` are
    maintained as denormalised counters *on this document* so the convergence rule can be
    expressed as a single-document conditional write. That is the whole trick: the rule
    is one `find_one_and_update`, and MongoDB's atomicity settles ties.
    """

    doc_type: Literal[DocType.HYPOTHESIS] = DocType.HYPOTHESIS
    hypothesis_id: str
    incident_id: str
    created_by: str
    statement: str
    status: HypothesisStatus = HypothesisStatus.CANDIDATE
    supporting_observations: list[str] = Field(default_factory=list)
    evidence_types_covered: list[EvidenceType] = Field(default_factory=list)
    contradicting_observations: list[str] = Field(default_factory=list)
    open_question_count: int = 0
    created_at: datetime = Field(default_factory=utcnow)

    # Phase 2 — absent until the convergence write adds them.
    phase: Literal["investigation", "remediation"] = "investigation"
    root_cause_summary: str | None = None
    options: list[Option] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    constraint_count: int = 0
    confidence: Literal["low", "normal"] = "normal"
    forced: bool = False


class OpenQuestion(BaseModel):
    doc_type: Literal[DocType.OPEN_QUESTION] = DocType.OPEN_QUESTION
    question_id: str
    incident_id: str
    asked_by: str
    question: str
    context: str = ""
    hypothesis_id: str | None = None
    status: Literal["open", "answered"] = "open"
    answered_by: str | None = None
    answer: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


def to_doc(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="python")
