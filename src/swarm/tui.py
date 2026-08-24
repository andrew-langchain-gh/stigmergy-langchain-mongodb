"""The audience-facing dashboard.

Worth saying out loud while this is on screen: the dashboard has no connection to any
agent. It opens a change stream and renders what the database tells it, exactly like the
agents do. It is the ninth subscriber to the same bus, with no privileged access and no
IPC — which is the cleanest available proof that the database really is the coordination
mechanism, and not just a place the agents happen to persist things.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import termios
import tty
from collections import deque
from contextlib import contextmanager
from datetime import datetime

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from swarm import blackboard as bb
from swarm import control, db
from swarm.config import INCIDENT_ID, settings
from swarm.revisions import describe_change

HEADER_H = 4
AGENTS_H = 12  # 2 phase headers + 8 agent rows + panel border
REVISIONS_H = 12
SIDE_W = 52

DOC_ABBREV = {"observation": "obs", "hypothesis": "hyp", "open_question": "q"}

STATE_STYLE = {
    "thinking": "bold yellow",
    "idle": "dim",
    "converged": "green",
    "waiting": "cyan",
    "resuming": "bold magenta",
    "stopped": "bold red",
    "budget-exhausted": "red",
}


def _agent_panel(rows: list[dict], alive_pids: set[int]) -> Panel:
    table = Table.grid(padding=(0, 1), expand=True)
    table.add_column("agent", width=19)
    table.add_column("state", width=16)
    table.add_column("cy", justify="right", width=3)
    table.add_column("idle", justify="right", width=4)

    for phase, label in ((1, "PHASE 1 — stigmergy"), (2, "PHASE 2 — negotiation")):
        table.add_row(Text(label, style="bold white on grey23"), "", "", "")
        for row in [r for r in rows if r.get("phase") == phase]:
            state = row.get("state", "?")
            pid = row.get("pid")
            dead = state != "stopped" and pid not in alive_pids
            shown = "DEAD" if dead else state
            style = "bold red blink" if dead else STATE_STYLE.get(state, "white")
            table.add_row(
                Text(row["agent"], style="bold" if not dead else "red"),
                Text(shown, style=style),
                str(row.get("cycles", 0)),
                str(row.get("idle_cycles", 0)),
            )
    return Panel(table, title="agents", border_style="grey37")


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# statement, action, objection, constraint — generous first, tightened only if needed.
CLIP_PROFILES = [
    (240, 150, 240, 180),
    (200, 120, 170, 140),
    (150, 95, 120, 100),
    (110, 75, 90, 75),
    (80, 55, 65, 55),
    (55, 40, 45, 40),
]


def _compact(doc: dict, profile: tuple[int, int, int, int] = CLIP_PROFILES[1]) -> dict:
    """A curated view, not a filtered one.

    The raw document runs past 70 lines, so on any projector the objections and
    constraints — the whole payload of phase 2 — fall below the fold. This keeps what
    carries the story and drops opaque ids, timestamps and denormalised counters. Prose
    is clipped because the presenter narrates it; `swarm show` prints the real thing.
    """
    stmt_n, action_n, obj_n, cons_n = profile
    out: dict = {"statement": _clip(doc.get("statement", ""), stmt_n)}

    if doc.get("confidence") == "low":
        out["confidence"] = "low"
    if doc.get("forced"):
        out["forced"] = True

    # Nested arrays of scalars cost a line each and read no better than one joined
    # string, so they get flattened. It is still a JSON document, just a legible one.
    covered = doc.get("evidence_types_covered", [])
    out["evidence_types_covered"] = f"{len(covered)} — {', '.join(covered)}"
    if doc.get("open_question_count"):
        out["open_questions_blocking"] = doc["open_question_count"]

    options = []
    for option in doc.get("options", []):
        slim: dict = {
            "action": _clip(option["action"], action_n),
            "eta": option["eta"],
            "status": option["status"],
        }
        # Objections are the whole payload of phase 2 — kept, and kept legible.
        objections = []
        for o in option.get("objections", []):
            flag = "WITHDRAWN" if o.get("withdrawn") else (
                "BLOCKING" if o["blocking"] else "non-blocking"
            )
            objections.append(
                f"{o['by']} [{flag}/{o['severity']}] {_clip(o['objection'], obj_n)}"
            )
        if objections:
            slim["objections"] = objections
        options.append(slim)
    if options:
        out["options"] = options

    constraints = [
        f"{c['by']} [{c['type']}"
        + (f" by {c['deadline']}" if c.get("deadline") else "")
        + f"] {_clip(c['detail'], cons_n)}"
        for c in doc.get("constraints", [])
    ]
    if constraints:
        out["constraints"] = constraints

    for key in ("promoted_by", "selected_by"):
        if doc.get(key):
            out[key] = doc[key]
    return out


def _terse(doc: dict) -> dict:
    """Last resort for a short terminal: collapse each section to one line."""
    covered = doc.get("evidence_types_covered", [])
    options = []
    for option in doc.get("options", []):
        live = sum(
            1 for o in option.get("objections", []) if o["blocking"] and not o.get("withdrawn")
        )
        veto = f"  [{live} BLOCKING]" if live else ""
        options.append(
            f"{option['status'].upper()}: {_clip(option['action'], 60)} ({option['eta']}){veto}"
        )
    out: dict = {
        "statement": _clip(doc.get("statement", ""), 90),
        "evidence_types_covered": f"{len(covered)} — {', '.join(covered)}",
        "options": options,
    }
    if doc.get("constraints"):
        deadlines = [c["deadline"] for c in doc["constraints"] if c.get("deadline")]
        out["constraints"] = f"{len(doc['constraints'])}" + (
            f" (deadline {deadlines[0]})" if deadlines else ""
        )
    if doc.get("forced"):
        out["forced"] = True
    return out


def _rendered_rows(renderable, width: int) -> int:
    probe = Console(width=width, height=400, no_color=True)
    with probe.capture() as captured:
        probe.print(renderable)
    return len(captured.get().rstrip("\n").splitlines())


def _document_panel(
    doc: dict | None,
    rows: int = 0,
    width: int = 0,
    revision: tuple[int, int] | None = None,
) -> Panel:
    """Render the focal document, tightened until it actually fits the space given.

    A Layout crops whatever overflows, and what overflows is the bottom of the document —
    the objections, the constraints, the resolution. Exactly the payoff. So rather than
    hoping the terminal is tall enough, measure and clip until it fits.
    """
    if not doc:
        body: Group | Syntax = Group(
            Text("\n  No hypothesis on the blackboard yet.", style="dim"),
            Text("  Run `swarm trigger` to insert the incident.\n", style="dim"),
        )
        title = "focal document"
        border = "grey37"
        return Panel(body, title=title, border_style=border)

    status = doc.get("status", "?")
    phase = doc.get("phase", "investigation")
    title = f"{doc.get('hypothesis_id')}  ·  phase={phase}  ·  status={status}"
    border = {"candidate": "yellow", "confirmed": "cyan", "resolved": "green"}.get(
        status, "grey37"
    )
    if revision is not None:
        current, total = revision
        title = f"revision {current}/{total}  ·  phase={phase}  ·  status={status}"
        border = "magenta"

    def render(payload: dict) -> Syntax:
        return Syntax(
            # ensure_ascii=False, or an em-dash the model wrote renders as a literal
            # "—" on the projector.
            json.dumps(payload, indent=2, default=str, ensure_ascii=False),
            "json",
            theme="ansi_dark",
            word_wrap=True,
        )

    chosen = render(_compact(doc, CLIP_PROFILES[0]))
    if rows > 0 and width > 0:
        # A Panel costs 2 columns of border plus 2 of padding, and 2 rows of border.
        inner_width = max(width - 4, 20)
        candidates = [_compact(doc, p) for p in CLIP_PROFILES] + [_terse(doc)]
        for payload in candidates:
            candidate = render(payload)
            if _rendered_rows(candidate, inner_width) + 2 <= rows:
                chosen = candidate
                break
        else:
            chosen = render(_terse(doc))
            title += "  ·  truncated, `swarm show` for full"

    return Panel(chosen, title=title, border_style=border)


def _revisions_panel(revisions: list[dict], selected: int | None, height: int) -> Panel:
    """Every version of the focal document, oldest first, newest at the bottom."""
    if not revisions:
        return Panel(
            Text("\n  no writes to the focal document yet", style="dim"),
            title="revision history",
            border_style="grey37",
        )

    live = selected is None
    index = len(revisions) - 1 if live else selected
    rows = height - 2
    # Keep the selected revision in view while scrolling through a long history.
    start = max(0, min(index - rows // 2, len(revisions) - rows)) if len(revisions) > rows else 0

    table = Table.grid(padding=(0, 1), expand=True)
    table.add_column(width=4)
    table.add_column(width=8)
    table.add_column(width=18)
    table.add_column(ratio=1)

    for i in range(start, min(start + rows, len(revisions))):
        rev = revisions[i]
        current = i == index
        marker = "▶" if current else " "
        style = "bold white on grey27" if current else "grey70"
        table.add_row(
            Text(f"{marker}{i + 1:>3}", style=style),
            Text(rev["at"].strftime("%H:%M:%S"), style=style),
            Text(rev["by"].removesuffix("-agent"), style=style),
            Text(rev["what"], style=style),
        )

    suffix = "LIVE" if live else f"scrubbing — {index + 1}/{len(revisions)}"
    return Panel(
        table,
        title=f"revision history  ·  {len(revisions)} versions  ·  {suffix}",
        border_style="grey37" if live else "magenta",
    )


def _ticker_panel(events: deque[str], height: int) -> Panel:
    """Newest last, cropped to whatever vertical space the left column has."""
    visible = list(events)[-height:] if height > 0 else []
    return Panel(
        Text("\n".join(visible) or "waiting for change events…", style="grey70"),
        title="change stream",
        border_style="grey37",
    )


def _counts_line(snapshot: dict) -> Text:
    open_q = sum(1 for q in snapshot["open_questions"] if q["status"] == "open")
    return Text.assemble(
        ("  observations ", "dim"), (str(len(snapshot["observations"])), "bold white"),
        ("   hypotheses ", "dim"), (str(len(snapshot["hypotheses"])), "bold white"),
        ("   open questions ", "dim"),
        (str(open_q), "bold yellow" if open_q else "bold white"),
        ("   model ", "dim"), (settings.model, "bold white"),
    )


def _gate_line(gate: dict, held: int) -> Text:
    stepping = gate.get("mode") == control.STEP
    if stepping:
        return Text.assemble(
            ("  STEP ", "bold black on yellow"),
            ("  step ", "dim"), (str(gate.get("steps_taken", 0)), "bold white"),
            ("   tokens ", "dim"), (str(gate.get("tokens", 0)), "bold white"),
            ("   agents waiting ", "dim"),
            (str(held), "bold yellow" if held else "dim"),
            ("      n", "bold white"), (" next  ", "dim"),
            ("1-9", "bold white"), (" burst  ", "dim"),
            ("r", "bold white"), (" run  ", "dim"),
            ("←/→", "bold white"), (" history  ", "dim"),
            ("=", "bold white"), (" live  ", "dim"),
            ("q", "bold white"), (" quit", "dim"),
        )
    return Text.assemble(
        ("  RUN ", "bold black on green"),
        ("   steps taken ", "dim"), (str(gate.get("steps_taken", 0)), "bold white"),
        ("      h", "bold white"), (" hold/step  ", "dim"),
        ("n", "bold white"), (" next  ", "dim"),
        ("←/→", "bold white"), (" history  ", "dim"),
        ("=", "bold white"), (" live  ", "dim"),
        ("q", "bold white"), (" quit", "dim"),
    )


@contextmanager
def _cbreak(stream):
    """Read single keypresses without waiting for Enter, and always restore the terminal.

    If stdin is not a tty (piped, or run under a supervisor) this is a no-op and the
    dashboard simply has no keyboard — `swarm next` from another terminal still works.
    """
    if not stream.isatty():
        yield False
        return
    fd = stream.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


async def run_dashboard() -> None:
    client = db.async_client()
    console = Console()
    # Keep more history than fits; the panel crops to the space it actually has, so a
    # taller terminal simply shows more of the stream.
    events: deque[str] = deque(maxlen=200)

    # Every version of the focal document, rebuilt from change-stream post-images. The
    # dashboard still writes nothing — this is derived entirely from what the database
    # pushes to it. `cursor` is None while following live, or an index while scrubbing.
    revisions: list[dict] = []
    cursor: int | None = None

    def alive(pid: int | None) -> bool:
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    async def render() -> Layout:
        rows = await db.agent_status(client).find().sort("agent", 1).to_list(length=None)
        alive_pids = {r["pid"] for r in rows if alive(r.get("pid"))}
        doc = await bb.focal_document(client)
        snapshot = await bb.snapshot(client)
        gate = await control.get_gate(client)
        incident = await db.incidents(client).find_one({"incident_id": INCIDENT_ID}, {"run_seq": 1})
        run_seq = int((incident or {}).get("run_seq", 0))
        held = sum(1 for r in rows if r.get("state") == "held" and r.get("pid") in alive_pids)

        ticker_height = max(console.size.height - HEADER_H - AGENTS_H - 2, 3)
        revisions_height = min(REVISIONS_H, max(6, (console.size.height - HEADER_H) // 3))
        doc_height = console.size.height - HEADER_H - revisions_height

        # While scrubbing, show the document as it was at that revision rather than as it
        # is now — that is the whole point of the history panel.
        shown = doc if cursor is None else revisions[cursor]["doc"]

        layout = Layout()
        layout.split_column(
            Layout(
                Panel(
                    Group(_counts_line(snapshot), _gate_line(gate, held)),
                    title=(
                        f"{INCIDENT_ID} — shared memory coordination"
                        + (f"  ·  run {run_seq}" if run_seq > 1 else "")
                    ),
                    border_style="yellow" if gate.get("mode") == control.STEP else "grey37",
                ),
                size=HEADER_H,
            ),
            Layout(name="body"),
        )
        layout["body"].split_row(Layout(name="side", size=SIDE_W), Layout(name="doc"))
        layout["side"].split_column(
            Layout(_agent_panel(rows, alive_pids), size=AGENTS_H),
            Layout(_ticker_panel(events, ticker_height)),
        )
        layout["doc"].split_column(
            Layout(
                _document_panel(
                    shown,
                    rows=doc_height,
                    width=console.size.width - SIDE_W,
                    revision=None if cursor is None else (cursor + 1, len(revisions)),
                )
            ),
            Layout(_revisions_panel(revisions, cursor, revisions_height), size=revisions_height),
        )
        return layout

    async def follow() -> None:
        swept = False
        # whenAvailable uses the post-image recorded by the server, so each event carries
        # the document exactly as that write left it. updateLookup would re-read the
        # current document and silently collapse rapid successive versions into one.
        await db.enable_post_images(client)
        try:
            stream = await db.blackboard(client).watch(full_document="whenAvailable")
        except Exception:  # noqa: BLE001
            stream = await db.blackboard(client).watch(full_document="updateLookup")

        async with stream:
            async for event in stream:
                # A reset deletes the whole board, which would otherwise scroll one
                # anonymous line per document past the ticker. Report the sweep once.
                if event["operationType"] not in ("insert", "update", "replace"):
                    if event["operationType"] in ("drop", "invalidate") or not swept:
                        events.append(f"{datetime.now().strftime('%H:%M:%S')} "
                                      f"{'—' * 11} board cleared")
                        swept = True
                    continue
                swept = False

                doc = event.get("fullDocument") or {}
                stamp = datetime.now()
                who = doc.get("last_touched_by", "?")
                glyph = "+" if event["operationType"] == "insert" else "~"
                kind = DOC_ABBREV.get(doc.get("doc_type", ""), "?")
                ident = (
                    doc.get("observation_id")
                    or doc.get("hypothesis_id")
                    or doc.get("question_id")
                    or ""
                )
                events.append(
                    f"{stamp.strftime('%H:%M:%S')} {who.removesuffix('-agent'):<11} "
                    f"{glyph}{kind} {ident[-6:]}"
                )

                if doc.get("doc_type") != "hypothesis":
                    continue
                # A `swarm reset` replaces the incident, so start the history over rather
                # than splicing two different documents together.
                if revisions and revisions[-1]["doc"].get("hypothesis_id") != doc.get(
                    "hypothesis_id"
                ):
                    revisions.clear()
                previous = revisions[-1]["doc"] if revisions else None
                revisions.append(
                    {
                        "at": stamp,
                        "by": who,
                        "what": describe_change(previous, doc),
                        "doc": doc,
                    }
                )

    quit_requested = asyncio.Event()

    def scrub(delta: int | None, *, to: int | None = None) -> None:
        """Move through the revision history. `None` snaps back to following live."""
        nonlocal cursor
        if not revisions:
            cursor = None
            return
        if to is not None:
            cursor = max(0, min(to, len(revisions) - 1))
            return
        if delta is None:
            cursor = None
            return
        start = len(revisions) - 1 if cursor is None else cursor
        target = max(0, min(start + delta, len(revisions) - 1))
        # Stepping forward off the end returns to live rather than sticking on the last one.
        cursor = None if target == len(revisions) - 1 and delta > 0 else target

    # Every arrow steps one revision — the list is vertical, so up/down has to mean "the
    # next one", not "five along". Bulk movement lives on PageUp/PageDown and Home/End.
    ARROWS = {"A": -1, "B": +1, "D": -1, "C": +1}
    TILDES = {"5": -10, "6": +10}

    async def keyboard(enabled: bool) -> None:
        """n/space step, digits burst, r run, h hold; arrows scrub one revision at a
        time, PageUp/PageDown jump 10, Home first, = or End back to live, q quit."""
        if not enabled:
            return
        loop = asyncio.get_running_loop()
        chunks: asyncio.Queue[str] = asyncio.Queue()
        fd = sys.stdin.fileno()

        def on_readable() -> None:
            # os.read, not sys.stdin.read: the buffered text wrapper can block trying to
            # fill its buffer, which silently swallows single keypresses.
            try:
                data = os.read(fd, 64)
            except (OSError, ValueError):
                return
            if data:
                chunks.put_nowait(data.decode(errors="ignore"))

        loop.add_reader(fd, on_readable)
        try:
            while not quit_requested.is_set():
                pending = await chunks.get()
                while pending:
                    # Consume CSI sequences whole. Arrows are 3 bytes (ESC [ A) but
                    # PageUp/Home are tilde-terminated and longer (ESC [ 5 ~), and a
                    # partially-consumed sequence would leave "[" or "~" to be misread
                    # as a plain keypress.
                    if pending.startswith("\x1b[") and len(pending) >= 3:
                        body = pending[2:]
                        terminator = next(
                            (i for i, ch in enumerate(body) if ch.isalpha() or ch == "~"), None
                        )
                        if terminator is None:
                            break  # incomplete sequence; wait for the rest
                        code, pending = body[: terminator + 1], body[terminator + 1 :]
                        if code in ARROWS:
                            scrub(ARROWS[code])
                        elif code.endswith("~") and code[:-1] in TILDES:
                            scrub(TILDES[code[:-1]])
                        elif code in ("H", "1~"):  # Home
                            scrub(None, to=0)
                        elif code in ("F", "4~"):  # End
                            scrub(None)
                        continue

                    key, pending = pending[0], pending[1:]
                    if key in ("n", " ", "\n", "\r"):
                        await control.grant(client, 1)
                    elif key.isdigit() and key != "0":
                        await control.grant(client, int(key))
                    elif key == "r":
                        await control.set_mode(client, control.RUN)
                    elif key in ("h", "s"):
                        await control.set_mode(client, control.STEP)
                    elif key == "[":
                        scrub(-1)
                    elif key == "]":
                        scrub(+1)
                    elif key in ("=", "l", "\x1b"):
                        scrub(None)
                    elif key in ("q", "\x03"):
                        quit_requested.set()
        finally:
            loop.remove_reader(sys.stdin.fileno())

    await control.ensure_gate(client)
    with _cbreak(sys.stdin) as keys_enabled:
        with Live(await render(), console=console, refresh_per_second=4, screen=True) as live:
            tasks = [asyncio.create_task(follow()), asyncio.create_task(keyboard(keys_enabled))]

            def report(task: asyncio.Task) -> None:
                # A background task that dies silently would leave the dashboard looking
                # fine while the keyboard or the change stream is dead. Say so instead.
                if not task.cancelled() and task.exception():
                    print(f"[tui] background task failed: {task.exception()!r}", file=sys.stderr)

            for task in tasks:
                task.add_done_callback(report)
            try:
                while not quit_requested.is_set():
                    live.update(await render())
                    await asyncio.sleep(0.25)
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
            finally:
                for task in tasks:
                    task.cancel()
                await client.close()
