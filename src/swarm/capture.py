"""Persist every version of the focal document to disk.

The dashboard already reconstructs the full revision history from change-stream
post-images, but it holds it in a list that dies with the process. This is the same
subscriber, writing each version to a file instead — so a specific version can go on a
slide, and two adjacent versions can be diffed.

Like the dashboard, this has no connection to any agent and no privileged access. It is
one more reader on the same bus. Two consequences worth knowing before you rely on it:

* It only sees writes it was running to witness, so start it before `swarm trigger`.
* It needs `changeStreamPreAndPostImages` on the blackboard. `updateLookup` would re-read
  the document as it is *now* and silently collapse two quick writes into one version,
  which would make the exported history a lie.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pymongo import AsyncMongoClient

from swarm import blackboard as bb
from swarm import db
from swarm.config import INCIDENT_ID, settings
from swarm.revisions import as_json, describe_change, json_diff_lines

TRIGGER_LABEL = "exogenous trigger"


@dataclass
class Revision:
    """One version of one document, exactly as a single write left it."""

    index: int
    at: datetime
    by: str
    what: str
    doc: dict[str, Any]
    previous: dict[str, Any] | None


@dataclass
class Chain:
    """Every version of one hypothesis document, oldest first."""

    hypothesis_id: str
    revisions: list[Revision] = field(default_factory=list)

    @property
    def latest(self) -> dict[str, Any]:
        return self.revisions[-1].doc

    def reached_remediation(self) -> bool:
        return any(r.doc.get("phase") == "remediation" for r in self.revisions)

    def append(self, doc: dict[str, Any], at: datetime) -> Revision:
        previous = self.revisions[-1].doc if self.revisions else None
        revision = Revision(
            index=len(self.revisions) + 1,
            at=at,
            by=doc.get("last_touched_by", "?"),
            what=describe_change(previous, doc),
            doc=doc,
            previous=previous,
        )
        self.revisions.append(revision)
        return revision


def _slug(text: str) -> str:
    """Filename-safe version of a change label, keeping it readable in a file listing."""
    text = text.replace("→", "to")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return (text[:60].rstrip("-") or "touched")


def _stem(index: int, what: str, by: str) -> str:
    return f"v{index:02d}-{_slug(what)}--{by.removesuffix('-agent')}"


def _evidence_lookup(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every observation and question by id, so a diff can be annotated with its content."""
    lookup: dict[str, dict[str, Any]] = {}
    for obs in snapshot.get("observations", []):
        lookup[obs["observation_id"]] = obs
    for question in snapshot.get("open_questions", []):
        lookup[question["question_id"]] = question
    return lookup


def _added_ids(diff: list[str]) -> list[str]:
    """The obs_/q_ ids this write *added*, in the order the diff introduces them."""
    seen: list[str] = []
    for line in diff:
        if not line.startswith("+"):
            continue
        for match in re.finditer(r"\b((?:obs|q)_[0-9a-f]{6})\b", line):
            if match.group(1) not in seen:
                seen.append(match.group(1))
    return seen


def _question_event(diff: list[str], by: str, lookup: dict[str, dict[str, Any]]):
    """The focal document tracks questions only as a count, never by id.

    So the stigmergy moment — one agent asks, a *different* agent answers — reaches the
    diff as `open_question_count` 0->1 and 1->0 and nothing else. This recovers which
    question moved, by matching the write's author against who asked or who answered.
    """
    before = after = None
    for line in diff:
        match = re.match(r'([+-])\s*"open_question_count":\s*(\d+)', line)
        if not match:
            continue
        if match.group(1) == "-":
            before = int(match.group(2))
        else:
            after = int(match.group(2))
    if before is None or after is None or before == after:
        return None

    questions = [q for q in lookup.values() if "question_id" in q]
    if after > before:
        matches = [q for q in questions if q.get("asked_by") == by]
        return ("opened", matches[0]) if matches else None
    matches = [q for q in questions if q.get("answered_by") == by]
    return ("answered", matches[0]) if matches else None


def _render_observation(identifier: str, record: dict[str, Any]) -> list[str]:
    lines = [
        f"### `{identifier}` — `{record['posted_by']}` · {record['evidence_type']}",
        "",
        f"**{record.get('summary', '')}**",
    ]
    if record.get("detail"):
        lines += ["", record["detail"]]
    if record.get("source_refs"):
        refs = ", ".join(f"`{r}`" for r in record["source_refs"])
        lines += ["", f"*source refs:* {refs}"]
    return lines


def _resolved_markdown(
    diff: list[str], lookup: dict[str, dict[str, Any]], by: str = ""
) -> list[str]:
    """The point of the slide: what the ids this write added actually say.

    The document stores ids (and for questions, only a count) because that is the data
    model. On a slide that is unreadable, so this inlines the content next to the diff
    that introduced it.
    """
    sections: list[list[str]] = []

    for identifier in _added_ids(diff):
        record = lookup.get(identifier)
        if record is None:
            continue
        if "observation_id" in record:
            sections.append(_render_observation(identifier, record))

    event = _question_event(diff, by, lookup)
    if event is not None:
        kind, question = event
        identifier = question["question_id"]
        block = [
            f"### `{identifier}` — asked by `{question['asked_by']}`",
            "",
            f"**{question.get('question', '')}**",
        ]
        if kind == "opened":
            block += [
                "",
                "*Open at this point in the run.* It is addressed to nobody: no agent is "
                "named, no work is routed. It is a mark left in the environment.",
            ]
        else:
            block += [
                "",
                f"*Answered by* `{question['answered_by']}` — who was never asked. "
                "It picked the question up because the trace was sitting there.",
                "",
                str(question.get("answer") or ""),
            ]
        sections.append(block)

    if not sections:
        return []

    lines = ["", "## What this write's ids refer to", ""]
    for block in sections:
        lines += block + [""]
    return lines


def _write_version(
    directory: Path,
    stem: str,
    doc: dict[str, Any],
    revision: Revision | None,
    t_plus: float | None,
    lookup: dict[str, dict[str, Any]] | None = None,
) -> None:
    """The JSON a slide shows, plus a markdown companion explaining what changed."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.json").write_text(as_json(doc) + "\n")

    if revision is None:
        return

    offset = f"T+{t_plus:.1f}s" if t_plus is not None else "—"
    diff = json_diff_lines(revision.previous, revision.doc)
    body = [
        f"# v{revision.index:02d} — {revision.what}",
        "",
        f"- **written by** `{revision.by}`",
        f"- **at** {revision.at.strftime('%H:%M:%S')} ({offset} after the trigger)",
        f"- **document** `{revision.doc.get('hypothesis_id')}` "
        f"status `{revision.doc.get('status')}` phase `{revision.doc.get('phase', 'investigation')}`",
        "",
        "## What this write changed",
        "",
        "```diff",
        *(diff or ["(no textual change — the write touched a field back to its own value)"]),
        "```",
        *_resolved_markdown(diff, lookup or {}, revision.by),
        "",
        "## The document after this write",
        "",
        "```json",
        as_json(revision.doc),
        "```",
    ]
    (directory / f"{stem}.md").write_text("\n".join(body) + "\n")


def _index_markdown(chain: Chain, trigger: dict[str, Any] | None, started: datetime) -> str:
    lines = [
        f"# {INCIDENT_ID} — revision history of `{chain.hypothesis_id}`",
        "",
        "One `_id`. Every row is one agent's ordinary write. Nothing in this list was",
        "scheduled, assigned or coordinated by anything.",
        "",
        "| # | T+ | written by | what the write did | file |",
        "|---|---|---|---|---|",
    ]
    if trigger is not None:
        lines.append(
            f"| 00 | T+0.0s | `threshold-monitor` | {TRIGGER_LABEL} "
            f"(a **different** document, in `incidents`) | `versions/v00-{_slug(TRIGGER_LABEL)}"
            "--threshold-monitor.json` |"
        )
    for revision in chain.revisions:
        offset = (revision.at - started).total_seconds()
        stem = _stem(revision.index, revision.what, revision.by)
        lines.append(
            f"| {revision.index:02d} | T+{offset:.1f}s | `{revision.by}` | "
            f"{revision.what} | `versions/{stem}.json` |"
        )
    lines += [
        "",
        "## Final state",
        "",
        "```json",
        as_json(chain.latest),
        "```",
    ]
    return "\n".join(lines) + "\n"


def _evidence_markdown(snapshot: dict[str, Any]) -> str:
    """The ids inside the document are opaque. This is what they resolve to."""
    lines = [
        f"# {INCIDENT_ID} — what the ids in the document refer to",
        "",
        "The focal document references observations and questions by id. On a slide those",
        "ids are unreadable, so this resolves each one to the agent that wrote it and what",
        "it said.",
        "",
        "## Observations",
        "",
        "| id | posted by | evidence type | summary |",
        "|---|---|---|---|",
    ]
    for obs in sorted(snapshot["observations"], key=lambda o: o.get("created_at") or datetime.min):
        summary = str(obs.get("summary", "")).replace("|", "\\|")
        lines.append(
            f"| `{obs['observation_id']}` | `{obs['posted_by']}` | "
            f"{obs['evidence_type']} | {summary} |"
        )

    lines += [
        "",
        "## Open questions",
        "",
        "Every row where **asked by** and **answered by** differ is the stigmergy mechanism:",
        "nobody routed the question, it was picked up because the trace was sitting there.",
        "",
        "| id | asked by | question | answered by | answer |",
        "|---|---|---|---|---|",
    ]
    for question in sorted(
        snapshot["open_questions"], key=lambda q: q.get("created_at") or datetime.min
    ):
        text = str(question.get("question", "")).replace("|", "\\|")
        answer = str(question.get("answer") or "").replace("|", "\\|")
        answered_by = question.get("answered_by")
        lines.append(
            f"| `{question['question_id']}` | `{question['asked_by']}` | {text} | "
            f"{'`' + answered_by + '`' if answered_by else '_(unanswered)_'} | {answer} |"
        )

    others = [h for h in snapshot["hypotheses"]]
    if len(others) > 1:
        lines += [
            "",
            "## All hypotheses opened during this run",
            "",
            "| id | created by | status | evidence types | statement |",
            "|---|---|---|---|---|",
        ]
        for hyp in others:
            statement = str(hyp.get("statement", "")).replace("|", "\\|")
            lines.append(
                f"| `{hyp['hypothesis_id']}` | `{hyp['created_by']}` | {hyp['status']} | "
                f"{len(hyp.get('evidence_types_covered', []))} | {statement} |"
            )

    return "\n".join(lines) + "\n"


def _quality(chain: Chain, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Is this run usable for the deck? Recorded so the choice is mechanical.

    The demo's conclusion is stable but its route is not, so a run is judged on whether
    every beat the talk narrates actually happened in it.
    """
    final = chain.latest
    options = final.get("options", [])
    vetoed = {
        option["option_id"]
        for option in options
        for objection in option.get("objections", [])
        if objection.get("blocking") and not objection.get("withdrawn")
    }
    selected = [o["option_id"] for o in options if o.get("status") == "selected"]
    labels = [r.what for r in chain.revisions]
    cross_answered = [
        q
        for q in snapshot["open_questions"]
        if q.get("answered_by") and q["answered_by"] != q["asked_by"]
    ]

    checks = {
        "not_forced": not final.get("forced", False),
        "normal_confidence": final.get("confidence") == "normal",
        "three_evidence_types": len(final.get("evidence_types_covered", [])) >= 3,
        "two_or_more_options": len(options) >= 2,
        "live_blocking_objection": bool(vetoed),
        "veto_held": bool(selected) and not (set(selected) & vetoed),
        "constraint_attached": len(final.get("constraints", [])) >= 1,
        "cross_agent_answer": bool(cross_answered),
        "resolved": final.get("status") == "resolved",
        "phase_flip_captured": any("PHASE → remediation" in label for label in labels),
    }
    return {
        "usable_for_deck": all(checks.values()),
        "checks": checks,
        "vetoed_options": sorted(vetoed),
        "selected_options": selected,
        "cross_agent_answers": [
            {"question_id": q["question_id"], "asked_by": q["asked_by"], "answered_by": q["answered_by"]}
            for q in cross_answered
        ],
    }


async def capture(out_dir: Path, *, timeout: int = 600) -> dict[str, Any]:
    """Watch the blackboard and write every version of the focal document to `out_dir`."""
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    client = db.async_client()
    post_images = await db.enable_post_images(client)
    started = datetime.now()

    trigger = await db.incidents(client).find_one({"incident_id": INCIDENT_ID}, {"_id": 0})
    chains: dict[str, Chain] = {}
    resolved = asyncio.Event()

    async def follow_incidents() -> None:
        """The trigger is a different document, and may land after we start watching."""
        nonlocal trigger, started
        if trigger is not None:
            return
        async with await db.incidents(client).watch(
            [{"$match": {"operationType": "insert"}}], full_document="updateLookup"
        ) as stream:
            async for event in stream:
                doc = event.get("fullDocument") or {}
                if doc.get("incident_id") != INCIDENT_ID:
                    continue
                doc.pop("_id", None)
                trigger = doc
                started = datetime.now()
                _write_version(
                    out_dir / "versions",
                    f"v00-{_slug(TRIGGER_LABEL)}--threshold-monitor",
                    trigger,
                    None,
                    None,
                )
                return

    async def follow_blackboard() -> None:
        try:
            stream = await db.blackboard(client).watch(full_document="whenAvailable")
        except Exception:  # noqa: BLE001 — a standalone server cannot do post-images
            stream = await db.blackboard(client).watch(full_document="updateLookup")

        async with stream:
            async for event in stream:
                doc = event.get("fullDocument") or {}
                if doc.get("doc_type") != "hypothesis":
                    continue
                doc.pop("_id", None)
                hyp_id = doc.get("hypothesis_id")
                if not hyp_id:
                    continue

                chain = chains.setdefault(hyp_id, Chain(hypothesis_id=hyp_id))
                revision = chain.append(doc, datetime.now())
                # Write immediately: an interrupted run still yields everything up to here.
                _write_version(
                    out_dir / "_raw" / hyp_id,
                    _stem(revision.index, revision.what, revision.by),
                    doc,
                    revision,
                    (revision.at - started).total_seconds(),
                )
                if doc.get("status") == "resolved":
                    resolved.set()
                    return

    async def deadline() -> None:
        await asyncio.sleep(timeout)
        resolved.set()

    watchers = [
        asyncio.create_task(follow_incidents()),
        asyncio.create_task(follow_blackboard()),
        asyncio.create_task(deadline()),
    ]
    try:
        await resolved.wait()
    finally:
        for task in watchers:
            task.cancel()
        await asyncio.gather(*watchers, return_exceptions=True)

    summary = await _finalize(client, out_dir, chains, trigger, started, post_images)
    await client.close()
    return summary


async def _finalize(
    client: AsyncMongoClient,
    out_dir: Path,
    chains: dict[str, Chain],
    trigger: dict[str, Any] | None,
    started: datetime,
    post_images: bool,
) -> dict[str, Any]:
    """Promote the focal chain to `versions/`, park the rest, and write the indexes."""
    if not chains:
        (out_dir / "run.json").write_text(
            json.dumps({"error": "no hypothesis writes witnessed"}, indent=2) + "\n"
        )
        return {"versions": 0, "out_dir": str(out_dir)}

    # The focal document is the one that spans both phases; failing that, the longest-lived.
    focal = max(
        chains.values(), key=lambda c: (c.reached_remediation(), len(c.revisions))
    )

    # Before the versions, not after: each diff annotates the ids it introduced.
    snapshot = await bb.snapshot(client)
    lookup = _evidence_lookup(snapshot)

    versions = out_dir / "versions"
    if trigger is not None:
        _write_version(
            versions, f"v00-{_slug(TRIGGER_LABEL)}--threshold-monitor", trigger, None, None
        )
    for revision in focal.revisions:
        _write_version(
            versions,
            _stem(revision.index, revision.what, revision.by),
            revision.doc,
            revision,
            (revision.at - started).total_seconds(),
            lookup,
        )

    for hyp_id, chain in chains.items():
        if hyp_id == focal.hypothesis_id:
            continue
        for revision in chain.revisions:
            _write_version(
                out_dir / "other-hypotheses" / hyp_id,
                _stem(revision.index, revision.what, revision.by),
                revision.doc,
                revision,
                (revision.at - started).total_seconds(),
                lookup,
            )

    (out_dir / "index.md").write_text(_index_markdown(focal, trigger, started))
    (out_dir / "evidence-index.md").write_text(_evidence_markdown(snapshot))

    final = focal.latest
    quality = _quality(focal, snapshot)
    run = {
        "incident_id": INCIDENT_ID,
        "hypothesis_id": focal.hypothesis_id,
        "captured_at": started.isoformat(timespec="seconds"),
        "duration_s": round((focal.revisions[-1].at - started).total_seconds(), 1),
        "versions": len(focal.revisions),
        "trigger_captured": trigger is not None,
        "post_images_enabled": post_images,
        "model": settings.model,
        "pace_ms": settings.pace_ms,
        "max_cycles": settings.max_cycles,
        "final": {
            "status": final.get("status"),
            "phase": final.get("phase"),
            "forced": final.get("forced", False),
            "confidence": final.get("confidence"),
            "evidence_types_covered": final.get("evidence_types_covered", []),
            "supporting_observations": len(final.get("supporting_observations", [])),
            "options": len(final.get("options", [])),
            "objections": sum(len(o.get("objections", [])) for o in final.get("options", [])),
            "constraints": len(final.get("constraints", [])),
            "promoted_by": final.get("promoted_by"),
            "selected_by": final.get("selected_by"),
        },
        "board": {
            "observations": len(snapshot["observations"]),
            "hypotheses": len(snapshot["hypotheses"]),
            "open_questions": len(snapshot["open_questions"]),
        },
        "other_hypotheses": sorted(k for k in chains if k != focal.hypothesis_id),
        "quality": quality,
        "revisions": [
            {
                "index": r.index,
                "t_plus_s": round((r.at - started).total_seconds(), 1),
                "by": r.by,
                "what": r.what,
                "file": f"versions/{_stem(r.index, r.what, r.by)}.json",
            }
            for r in focal.revisions
        ],
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n")
    return run
