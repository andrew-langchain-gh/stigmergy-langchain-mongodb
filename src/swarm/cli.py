"""Presenter controls.

`run` supervises eight real OS processes. That matters: the resilience beats depend on
`kill` sending a real SIGKILL to a real pid, not on simulating a failure in-process.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from swarm import blackboard as bb
from swarm import control, db, watchdog
from swarm.agents import AGENTS, get_spec
from swarm.config import INCIDENT_ID, settings

app = typer.Typer(add_completion=False, help="Shared-memory agent coordination demo.")
console = Console()


def _agent_command(name: str) -> list[str]:
    return [sys.executable, "-m", "swarm.agent_main", name]


def _spawn_swarm(names: list[str], *, with_watchdog: bool, quiet: bool = False):
    """Start one OS process per agent, plus the watchdog. Returns (procs, watchdog_proc)."""
    procs: dict[str, subprocess.Popen] = {}
    for name in names:
        kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL} if quiet else {}
        procs[name] = subprocess.Popen(_agent_command(name), **kwargs)
        if not quiet:
            console.print(f"[green]started[/green] {name} pid={procs[name].pid}")

    watchdog_proc = None
    if with_watchdog:
        kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL} if quiet else {}
        watchdog_proc = subprocess.Popen(
            [sys.executable, "-m", "swarm.watchdog_main"], **kwargs
        )
        if not quiet:
            console.print(f"[green]started[/green] watchdog pid={watchdog_proc.pid}")
    return procs, watchdog_proc


def _stop_swarm(procs: dict[str, subprocess.Popen], watchdog_proc) -> None:
    everything = list(procs.values()) + ([watchdog_proc] if watchdog_proc else [])
    for proc in everything:
        proc.terminate()
    for proc in everything:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@app.command()
def seed() -> None:
    """Load the INC-1042 dataset and build the vector index."""
    from swarm.seed import seed as run_seed

    counts = asyncio.run(run_seed())
    console.print(f"[green]seeded[/green] {counts}")


@app.command()
def doctor() -> None:
    """Check every dependency the live demo relies on, before an audience is watching."""
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        mark = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        console.print(f"  {mark}  {label}" + (f"  [dim]{detail}[/dim]" if detail else ""))

    console.print("\n[bold]Environment[/bold]")
    check("ANTHROPIC_API_KEY set", bool(os.environ.get("ANTHROPIC_API_KEY")))
    check("OPENAI_API_KEY set", bool(os.environ.get("OPENAI_API_KEY")))
    check("LangSmith tracing", os.environ.get("LANGSMITH_TRACING") == "true",
          os.environ.get("LANGSMITH_PROJECT") or "no project set")

    console.print("\n[bold]MongoDB[/bold]")
    client = db.sync_client()
    try:
        hello = client.admin.command("hello")
        check("reachable", True, settings.mongodb_uri.split("@")[-1])
        check("replica set (change streams)", bool(hello.get("setName")), hello.get("setName", ""))
        database = client[settings.db_name]
        indexes = list(database[db.STORE].list_search_indexes())
        vector = [i for i in indexes if i["name"] == db.VECTOR_INDEX_NAME]
        check(
            "vector index queryable",
            bool(vector) and vector[0].get("queryable", False),
            vector[0].get("status", "missing") if vector else "missing — run `swarm seed`",
        )
        for coll, expected in [
            (db.EVIDENCE_LOGS, 17),
            (db.EVIDENCE_METRICS, 30),
            (db.EVIDENCE_DEPLOYS, 4),
            (db.EVIDENCE_TICKETS, 8),
        ]:
            n = database[coll].count_documents({})
            check(f"{coll} loaded", n == expected, f"{n}/{expected}")
    finally:
        client.close()

    console.print("\n[bold]Semantic search[/bold]")
    try:
        sync = db.sync_client()
        store = db.make_store(sync)
        hits = asyncio.run(store.asearch(db.TICKET_NAMESPACE, query="checkout failing", limit=3))
        check("embedding round-trip", len(hits) > 0, f"{len(hits)} hits")
        sync.close()
    except Exception as exc:  # noqa: BLE001
        check("embedding round-trip", False, str(exc)[:80])

    console.print()
    if ok:
        console.print("[bold green]All checks passed — safe to present.[/bold green]\n")
    else:
        console.print("[bold red]Some checks failed.[/bold red]\n")
        raise typer.Exit(1)


@app.command()
def trigger() -> None:
    """Insert the single incident document. This is the only thing that starts the swarm."""
    incident = asyncio.run(watchdog.trigger())
    console.print(f"[yellow]trigger[/yellow] {incident['incident_id']}: {incident['trigger']}")


@app.command()
def run(
    agents: str = typer.Option("", help="Comma-separated subset; default is all eight."),
    with_watchdog: bool = typer.Option(True, help="Run the exogenous timeout watchdog."),
    step: bool = typer.Option(False, help="Start in step mode — nothing moves until you say so."),
) -> None:
    """Supervise the swarm: one OS process per agent."""
    names = [n.strip() for n in agents.split(",") if n.strip()] or list(AGENTS)
    names = [get_spec(n).name for n in names]

    _gate_action(lambda c: control.set_mode(c, control.STEP if step else control.RUN))
    if step:
        console.print("[yellow]STEP mode[/yellow] — agents will park until released\n")

    procs, watchdog_proc = _spawn_swarm(names, with_watchdog=with_watchdog)

    console.print("\n[dim]Ctrl-C to stop the swarm.[/dim]\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]stopping swarm[/yellow]")
        _stop_swarm(procs, watchdog_proc)


@app.command()
def loop(
    cycles: int = typer.Option(0, help="How many incidents to work. 0 runs until Ctrl-C."),
    dwell: int = typer.Option(20, help="Seconds to leave the resolved document on screen."),
    timeout: int = typer.Option(300, help="Give up on a stuck run after N seconds and recycle."),
    with_watchdog: bool = typer.Option(True, help="Run the exogenous timeout watchdog."),
    quiet: bool = typer.Option(False, help="Silence agent output; print only cycle lines."),
) -> None:
    """Work the incident over and over: resolve, hold, reset, re-trigger.

    For running the demo unattended on a second screen while you present. The agents are
    started once and stay up across cycles — each new trigger carries a new run number,
    and every agent resets its own budget, convergence state and LangGraph thread when it
    sees one. Pair it with `swarm dashboard` on the screen the audience can see.
    """
    names = list(AGENTS)
    _gate_action(lambda c: control.set_mode(c, control.RUN))

    console.print("[cyan]continuous mode[/cyan] — Ctrl-C to stop\n")
    procs, watchdog_proc = _spawn_swarm(names, with_watchdog=with_watchdog, quiet=quiet)
    console.print()

    completed = 0
    try:
        while cycles == 0 or completed < cycles:
            outcome = asyncio.run(_one_cycle(dwell=dwell, timeout=timeout))
            completed += 1
            style = "green" if outcome["resolved"] and not outcome["forced"] else "yellow"
            console.print(
                f"[{style}]cycle {outcome['run_seq']}[/{style}] "
                f"{outcome['status']} in {outcome['seconds']}s · "
                f"{outcome['evidence']} evidence types, {outcome['options']} options, "
                f"{outcome['objections']} objections"
                + ("  [red]forced by watchdog[/red]" if outcome["forced"] else "")
                + (
                    f"  [yellow]{outcome['hypotheses']} competing hypotheses[/yellow]"
                    if outcome["hypotheses"] > 1
                    else ""
                )
            )
            if any(p.poll() is not None for p in procs.values()):
                dead = [n for n, p in procs.items() if p.poll() is not None]
                console.print(f"[red]agent process died:[/red] {', '.join(dead)} — stopping")
                break
    except KeyboardInterrupt:
        console.print("\n[yellow]stopping[/yellow]")
    finally:
        _stop_swarm(procs, watchdog_proc)
        console.print(f"[dim]{completed} cycle(s) completed[/dim]")


async def _one_cycle(*, dwell: int, timeout: int) -> dict:
    """Reset, trigger, wait for resolution, then hold the final state on screen."""
    client = db.async_client()
    try:
        # Close the board for the changeover. The step gate already exists to stop agents
        # taking turns, so reuse it: with no tokens minted, every agent parks before its
        # next turn and nothing can be mid-write while the wipe and re-trigger happen.
        await control.set_mode(client, control.STEP)
        if not await _await_quiet(client, limit=90):
            console.print("[yellow]an agent is still mid-turn — resetting anyway[/yellow]")

        await watchdog.reset_board(client, keep_agent_rows=True)
        incident = await watchdog.trigger(client)
        run_seq = incident["run_seq"]
        await control.set_mode(client, control.RUN)
        started = time.monotonic()

        doc = None
        while time.monotonic() - started < timeout:
            await asyncio.sleep(1)
            doc = await bb.focal_document(client)
            if doc and doc.get("status") == "resolved":
                break

        seconds = round(time.monotonic() - started, 1)
        resolved = bool(doc and doc.get("status") == "resolved")
        hypotheses = await db.blackboard(client).count_documents(
            {"incident_id": INCIDENT_ID, "doc_type": "hypothesis"}
        )
        outcome = {
            "run_seq": run_seq,
            "resolved": resolved,
            "status": "resolved" if resolved else "TIMED OUT",
            "seconds": seconds,
            "forced": bool(doc and doc.get("forced")),
            "evidence": len((doc or {}).get("evidence_types_covered", [])),
            "options": len((doc or {}).get("options", [])),
            "objections": sum(
                len(o.get("objections", [])) for o in (doc or {}).get("options", [])
            ),
            # More than one means the swarm split its attention; the focal document is then
            # only part of the story, so it is worth showing rather than hiding.
            "hypotheses": hypotheses,
        }
        if resolved and dwell:
            await asyncio.sleep(dwell)
        return outcome
    finally:
        await client.close()


async def _await_quiet(client, *, limit: int) -> bool:
    """Block until no agent reports `thinking`. False if `limit` seconds passed first."""
    for _ in range(limit):
        rows = await db.agent_status(client).find({}, {"state": 1}).to_list(length=None)
        if not any(r.get("state") == "thinking" for r in rows):
            return True
        await asyncio.sleep(1)
    return False


@app.command()
def start(name: str) -> None:
    """Restart one agent. It replays what it missed and finishes any pending step."""
    spec = get_spec(name)
    proc = subprocess.Popen(_agent_command(spec.name))
    console.print(f"[green]started[/green] {spec.name} pid={proc.pid}")
    console.print("[dim]leave this terminal open; the agent runs here[/dim]")
    proc.wait()


@app.command()
def kill(name: str) -> None:
    """SIGKILL one agent by the pid it recorded in MongoDB. No graceful shutdown."""
    spec = get_spec(name)
    client = db.sync_client()
    doc = client[settings.db_name][db.AGENT_STATUS].find_one({"agent": spec.name})
    client.close()
    if not doc or not doc.get("pid"):
        console.print(f"[red]no pid recorded for {spec.name}[/red]")
        raise typer.Exit(1)
    try:
        os.kill(doc["pid"], signal.SIGKILL)
        console.print(f"[red]SIGKILL[/red] {spec.name} pid={doc['pid']} (state was {doc.get('state')})")
    except ProcessLookupError:
        console.print(f"[yellow]{spec.name} pid={doc['pid']} was already gone[/yellow]")


@app.command()
def status() -> None:
    """Read swarm state straight out of MongoDB — no agent is consulted."""
    client = db.sync_client()
    rows = list(client[settings.db_name][db.AGENT_STATUS].find().sort("agent", 1))
    doc = client[settings.db_name][db.BLACKBOARD].find_one(
        {"doc_type": "hypothesis", "phase": "remediation"}
    )
    client.close()

    def alive(pid: int | None) -> bool:
        if not pid:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    table = Table(title="agents")
    for col in ("agent", "phase", "state", "run", "cycles", "idle", "pid"):
        table.add_column(col)
    for row in rows:
        state = row.get("state", "")
        # A SIGKILLed agent never got to update its own row, so trust the pid over the
        # state field — the same reason the dashboard probes rather than asking.
        if state != "stopped" and not alive(row.get("pid")):
            state = "[bold red]DEAD[/bold red]"
        table.add_row(
            row["agent"],
            str(row.get("phase", "")),
            state,
            str(row.get("run_seq", "")),
            str(row.get("cycles", 0)),
            str(row.get("idle_cycles", 0)),
            str(row.get("pid", "")),
        )
    console.print(table)
    if doc:
        console.print(f"focal document: {doc['hypothesis_id']} status={doc['status']}")


@app.command()
def show() -> None:
    """Print the focal document — the one whose history spans both phases."""
    client = db.async_client()

    async def _get():
        doc = await bb.focal_document(client)
        await client.close()
        return doc

    doc = asyncio.run(_get())
    payload = json.dumps(doc, indent=2, default=str, ensure_ascii=False) if doc else "null"
    if sys.stdout.isatty():
        console.print_json(payload)
    else:
        print(payload)  # keep it pipe-safe: `swarm show | jq ...`


@app.command()
def capture(
    out: str = typer.Option(..., help="Directory to write the version files into."),
    timeout: int = typer.Option(600, help="Give up waiting for a resolution after N seconds."),
) -> None:
    """Write every version of the focal document to disk, one file per write.

    Start this *before* `swarm trigger`. Like the dashboard it is only a change-stream
    subscriber, so it can only export writes it was running to witness. It exits on its own
    once the incident resolves.
    """
    from pathlib import Path

    from swarm.capture import capture as run_capture

    console.print(f"[cyan]capturing[/cyan] → {out}  [dim](waiting for writes; ^C to stop early)[/dim]")
    summary = asyncio.run(run_capture(Path(out), timeout=timeout))

    if not summary.get("versions"):
        console.print("[red]nothing captured[/red] — was the swarm running, and did you trigger it?")
        raise typer.Exit(1)

    quality = summary["quality"]
    verdict = (
        "[bold green]usable for the deck[/bold green]"
        if quality["usable_for_deck"]
        else "[bold yellow]not clean[/bold yellow] — "
        + ", ".join(k for k, ok in quality["checks"].items() if not ok)
    )
    console.print(
        f"[green]captured[/green] {summary['versions']} versions of "
        f"{summary['hypothesis_id']} in {summary['duration_s']}s → {out}\n{verdict}"
    )


@app.command()
def dashboard() -> None:
    """The audience-facing view. Reads change streams only; never talks to an agent."""
    from swarm.tui import run_dashboard

    asyncio.run(run_dashboard())


def _gate_action(coro_factory) -> dict:
    async def _run():
        client = db.async_client()
        try:
            return await coro_factory(client)
        finally:
            await client.close()

    return asyncio.run(_run())


@app.command()
def next(count: int = typer.Argument(1, help="How many agent turns to release.")) -> None:
    """Release one agent turn. Whichever agent claims the token first takes it."""
    gate = _gate_action(lambda c: control.grant(c, count))
    console.print(
        f"[green]+{count}[/green] step token(s) — "
        f"{gate['tokens']} unspent, {gate['steps_taken']} taken so far"
    )


@app.command()
def hold() -> None:
    """Enter step mode: agents park before each turn until you release them."""
    _gate_action(lambda c: control.set_mode(c, control.STEP))
    console.print("[yellow]STEP mode[/yellow] — `swarm next` (or `n` in the dashboard) to advance")


@app.command()
def free() -> None:
    """Leave step mode and let the swarm run at full speed."""
    _gate_action(lambda c: control.set_mode(c, control.RUN))
    console.print("[green]RUN mode[/green] — agents proceed on their own")


@app.command()
def timeline() -> None:
    """Replay what happened, in order, straight from the blackboard's own timestamps."""
    client = db.sync_client()
    docs = list(client[settings.db_name][db.BLACKBOARD].find({"incident_id": INCIDENT_ID}))
    client.close()

    rows = []
    for doc in docs:
        stamp = doc.get("created_at")
        kind = doc["doc_type"]
        if kind == "observation":
            who, what = doc["posted_by"], f"[{doc['evidence_type']}] {doc['summary']}"
        elif kind == "open_question":
            who, what = doc["asked_by"], f"ASKED: {doc['question']}"
        else:
            who, what = doc["created_by"], f"HYPOTHESIS: {doc['statement']}"
        rows.append((stamp, who, kind, what))
        if kind == "open_question" and doc.get("answered_by"):
            rows.append((stamp, doc["answered_by"], "answer", f"ANSWERED: {doc['answer']}"))
        if kind == "hypothesis" and doc.get("promoted_at"):
            rows.append(
                (doc["promoted_at"], doc.get("promoted_by", "rule"), "promotion",
                 f"CONVERGENCE RULE FIRED -> {doc['hypothesis_id']} confirmed, remediation open")
            )
        if kind == "hypothesis" and doc.get("resolved_at"):
            rows.append(
                (doc["resolved_at"], doc.get("selected_by", "watchdog"), "resolution",
                 "OPTION SELECTED -> incident resolved")
            )

    rows.sort(key=lambda r: r[0])
    if not rows:
        console.print("[dim]nothing on the blackboard yet[/dim]")
        return
    start = rows[0][0]
    for stamp, who, kind, what in rows:
        offset = (stamp - start).total_seconds()
        style = {"promotion": "bold cyan", "resolution": "bold green",
                 "open_question": "yellow", "answer": "bold yellow"}.get(kind, "white")
        line = Text.assemble(
            (f"T+{offset:6.1f}s  ", "dim"), (f"{who:<19}", "bold"), (what, style)
        )
        console.print(line, no_wrap=True, overflow="ellipsis")


@app.command()
def procedures(agent: str = typer.Option("", help="One agent, or all if omitted.")) -> None:
    """Show private procedural memory — same deployment as the blackboard, own namespace."""
    from swarm.procedural import SEED_PROCEDURES, read_procedures

    names = [get_spec(agent).name] if agent else sorted(SEED_PROCEDURES)
    for name in names:
        lessons = read_procedures(name)
        console.print(f"\n[bold]{name}[/bold]  [dim]namespace ('procedures', '{name}')[/dim]")
        for lesson in lessons or ["[dim](none recorded)[/dim]"]:
            console.print(f"  • {lesson}")
    console.print()


@app.command()
def reset() -> None:
    """Clear the blackboard, agent status and checkpoints. Evidence stays loaded."""
    _gate_action(lambda c: watchdog.reset_board(c, keep_agent_rows=False))
    console.print("[green]reset[/green] — run `swarm trigger` to start again")


if __name__ == "__main__":
    app()
