"""What one write did to the focal document.

Shared by the dashboard and by `swarm capture` on purpose: the labels an audience reads off
the revision panel on stage must be the same labels that end up in the exported files, or a
slide and the live demo would describe the same write differently.
"""

from __future__ import annotations

import difflib
import json
from typing import Any


def blocking_count(doc: dict[str, Any]) -> int:
    return sum(
        1
        for option in doc.get("options", [])
        for objection in option.get("objections", [])
        if objection.get("blocking") and not objection.get("withdrawn")
    )


def describe_change(prev: dict[str, Any] | None, cur: dict[str, Any]) -> str:
    """What this particular write did to the document.

    This is the line that makes the revision list a narrative rather than a log: you can
    point at the exact write that added the third evidence type, or flipped the status, or
    attached the veto.
    """
    if prev is None:
        return "hypothesis created"

    bits: list[str] = []
    if prev.get("status") != cur.get("status"):
        bits.append(f"status {prev.get('status')} → {cur.get('status')}")
    if prev.get("phase") != cur.get("phase"):
        bits.append(f"PHASE → {cur.get('phase')}")

    for label, key in (("evidence", "evidence_types_covered"), ("obs", "supporting_observations")):
        before, after = len(prev.get(key, [])), len(cur.get(key, []))
        if after != before:
            bits.append(f"{label} {before} → {after}")

    if prev.get("open_question_count", 0) != cur.get("open_question_count", 0):
        bits.append(f"open q {prev.get('open_question_count', 0)} → {cur.get('open_question_count', 0)}")

    before, after = len(prev.get("options", [])), len(cur.get("options", []))
    if after > before:
        bits.append(f"+{after - before} option")

    def objections(doc: dict[str, Any]) -> int:
        return sum(len(o.get("objections", [])) for o in doc.get("options", []))

    if objections(cur) > objections(prev):
        blocking = blocking_count(cur) > blocking_count(prev)
        bits.append("+BLOCKING objection" if blocking else "+objection")
    elif blocking_count(cur) < blocking_count(prev):
        bits.append("objection withdrawn")

    before, after = len(prev.get("constraints", [])), len(cur.get("constraints", []))
    if after > before:
        bits.append(f"+{after - before} constraint")

    def selected(doc: dict[str, Any]) -> list[str]:
        return [o["option_id"] for o in doc.get("options", []) if o["status"] == "selected"]

    if selected(cur) and not selected(prev):
        bits.append("OPTION SELECTED")

    return ", ".join(bits) or "touched"


def as_json(doc: dict[str, Any] | None) -> str:
    """Pretty JSON, with datetimes stringified — the form the document goes on a slide in."""
    return json.dumps(doc, indent=2, default=str, ensure_ascii=False) if doc else "null"


def json_diff_lines(prev: dict[str, Any] | None, cur: dict[str, Any]) -> list[str]:
    """Unified diff between two versions, so a slide can show the change and not just the state."""
    before = as_json(prev).splitlines() if prev is not None else []
    return list(
        difflib.unified_diff(
            before,
            as_json(cur).splitlines(),
            fromfile="previous version",
            tofile="this version",
            lineterm="",
            n=3,
        )
    )
