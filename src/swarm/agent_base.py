"""The runtime each of the eight agents runs.

Two distinct resilience mechanisms live here, and the talk contrasts them back to back:

* **Swarm-level.** No agent holds state anybody else needs. Kill one and the blackboard
  keeps converging, because coordination happens through the database rather than
  through connections between agents. That is architectural — nothing here implements it.
* **Agent-level.** Each agent is a LangGraph graph checkpointed into MongoDB under its
  own `thread_id`, run with `durability="sync"`. Kill one mid-run and it comes back at
  the pending step instead of restarting its investigation. That is `MongoDBSaver`.

Same action (kill a process), two mechanisms, two different jobs.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import signal
from dataclasses import dataclass, field
from typing import Any

from langchain.agents import create_agent
from pymongo import AsyncMongoClient

from swarm import blackboard as bb
from swarm import control, db, watch, watchdog
from swarm.config import INCIDENT_ID, settings

IDLE_CYCLES_TO_CONVERGE = 3
TICK_SECONDS = 20


@dataclass
class AgentSpec:
    name: str
    phase: int
    system_prompt: str
    evidence_type: str = ""
    build_tools: Any = None
    # Change-stream stages narrowing what this agent considers relevant. This is the only
    # "routing" in the system, and it is a filter the agent sets on itself — not an
    # address anybody else writes to.
    interests: list[dict[str, Any]] = field(default_factory=list)


class Agent:
    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec
        self.name = spec.name
        self.client: AsyncMongoClient = db.async_client()
        self.sync = db.sync_client()
        self.store = db.make_store(self.sync)
        self.checkpointer = db.make_checkpointer(self.sync)
        self.cycles = 0
        self.idle_cycles = 0
        self.run_seq = 0
        self._stopping = False
        self._was_engaged = False
        self._last_signature: tuple | None = None
        # One agent, one LangGraph thread. Two concurrent ainvoke calls against the same
        # thread_id would interleave writes into the same checkpoint, so wake-ups from
        # different sources are serialised here.
        self._turn = asyncio.Lock()

        tools = spec.build_tools(self.client, self.store)
        self.graph = create_agent(
            model=settings.model,
            tools=tools,
            system_prompt=self._prompt_with_procedures(spec),
            checkpointer=self.checkpointer,
            store=self.store,
        )
        self.config = self._config_for(0)

    def _config_for(self, run_seq: int) -> dict[str, Any]:
        """One LangGraph thread per run, not per agent.

        A thread that outlived the incident would carry the previous investigation's
        messages into the next one — the agent would "remember" a resolved outage that no
        longer exists on the board. Keying the thread on the run number gives each
        incident a clean thread while leaving the *checkpointing* behaviour, and so the
        resilience beat, exactly as it was.
        """
        return {"configurable": {"thread_id": f"{INCIDENT_ID}:r{run_seq}:{self.name}"}}

    def _adopt_run(self, run_seq: int) -> None:
        """Switch to a new run: fresh budget, fresh convergence state, fresh thread.

        Per-incident state is per-incident. The cycle budget exists to stop one agent
        write-amplifying over a single investigation, so carrying it across incidents
        would just starve the swarm on the second one.
        """
        # run_seq 0 means "no incident open" — a momentary state during a changeover, not
        # a run to adopt.
        if not run_seq or run_seq == self.run_seq:
            return
        previous = self.run_seq
        self.run_seq = run_seq
        self.cycles = 0
        self.idle_cycles = 0
        self._was_engaged = False
        self._last_signature = None
        # A new dict rather than a mutation: an in-flight ainvoke still holds the old one.
        self.config = self._config_for(run_seq)
        if previous:
            self.log(f"run {previous} → {run_seq}: budget and convergence state reset")

    @staticmethod
    def _prompt_with_procedures(spec: AgentSpec) -> str:
        """Private procedural memory, read from a namespace only this agent looks at.

        Same MongoDB deployment as the shared blackboard, different namespace: swarm
        coordination is public, individual improvement is private.
        """
        from swarm.procedural import read_procedures

        try:
            lessons = read_procedures(spec.name)
        except Exception:  # noqa: BLE001 — never let memory lookup stop an agent starting
            lessons = []
        if not lessons:
            return spec.system_prompt
        body = "\n".join(f"- {lesson}" for lesson in lessons)
        return (
            f"{spec.system_prompt}\n"
            f"\nLessons you recorded for yourself after previous incidents:\n{body}\n"
        )

    # -- status -------------------------------------------------------------------

    def log(self, message: str) -> None:
        print(f"[{self.name}] {message}", flush=True)

    async def heartbeat(self, **fields: Any) -> None:
        """All agent state lives in Mongo, so `swarm status` is just a query."""
        await db.agent_status(self.client).update_one(
            {"agent": self.name},
            {
                "$set": {
                    "agent": self.name,
                    "phase": self.spec.phase,
                    "pid": os.getpid(),
                    "cycles": self.cycles,
                    "idle_cycles": self.idle_cycles,
                    "run_seq": self.run_seq,
                    **fields,
                }
            },
            upsert=True,
        )

    # -- the react loop -----------------------------------------------------------

    async def _write_count(self) -> int:
        return await db.blackboard(self.client).count_documents({"last_touched_by": self.name})

    async def _unanswered_questions_from_others(self) -> list[dict[str, Any]]:
        if self.spec.phase != 1:
            return []
        return await (
            db.blackboard(self.client)
            .find(
                {
                    "doc_type": "open_question",
                    "incident_id": INCIDENT_ID,
                    "status": "open",
                    "asked_by": {"$ne": self.name},
                },
                {"_id": 0, "question_id": 1, "question": 1},
            )
            .to_list(length=None)
        )

    async def engaged(self) -> bool:
        """Is the incident currently in the phase this agent works in?

        Phase-2 agents must not burn their cycle budget commenting on an investigation,
        and — the subtler half — a phase-2 agent that idled its way to `converged` during
        phase 1 has to become active again when remediation opens. Convergence is
        per-phase, so crossing into your phase resets it.
        """
        doc = await bb.focal_document(self.client)
        in_remediation = bool(doc and doc.get("phase") == "remediation")
        return in_remediation if self.spec.phase == 2 else not in_remediation

    async def _artifact_signature(self) -> tuple:
        """A cheap fingerprint of the negotiated artifact's *shape*."""
        doc = await bb.focal_document(self.client)
        if not doc:
            return ()
        options = doc.get("options", [])
        live_objections = sum(
            1
            for option in options
            for objection in option.get("objections", [])
            if not objection.get("withdrawn")
        )
        return (doc.get("status"), len(options), live_objections, doc.get("constraint_count", 0))

    async def _gate(self) -> bool:
        engaged = await self.engaged()
        if engaged and not self._was_engaged:
            self.idle_cycles = 0
            self.log("my phase is now open — convergence counter reset")
        self._was_engaged = engaged
        if not engaged:
            return False

        # In phase 2, writing nothing is often deliberate waiting rather than having
        # nothing to say — scheduling in particular is supposed to hold off until risk
        # and comms have weighed in. So a change to the artifact's structure clears the
        # convergence counter: new structure means there is genuinely something new to
        # consider, even for an agent that has been quiet for several turns.
        if self.spec.phase == 2:
            signature = await self._artifact_signature()
            if signature != self._last_signature:
                if self._last_signature is not None and self.idle_cycles:
                    self.log("artifact changed shape — convergence counter reset")
                self._last_signature = signature
                self.idle_cycles = 0

        if self.idle_cycles >= IDLE_CYCLES_TO_CONVERGE:
            return False
        return True

    async def think(self, wake_reason: str) -> None:
        """One turn: pace, look at the board, act, then let the rule decide the rest."""
        async with self._turn:
            await self._think(wake_reason)

    async def _await_turn(self) -> str | None:
        """Block until the presenter grants a step. Returns the mode, or None if stopping.

        An agent parked here reports `held`, so the dashboard can show who is queued up
        waiting for a turn — which is itself worth pointing at on stage: they are all
        ready to act on the same board, and none of them is being dispatched.
        """
        announced = False
        while not self._stopping:
            gate = await control.get_gate(self.client)
            if gate.get("mode") != control.STEP:
                return gate.get("mode", control.RUN)
            if await control.claim_token(self.client):
                return control.STEP
            if not announced:
                await self.heartbeat(state="held")
                announced = True
            await asyncio.sleep(0.2)
        return None

    async def _think(self, wake_reason: str) -> None:
        # Whichever stream woke us, the run that is actually open decides which budget and
        # which thread this turn belongs to. Checking here rather than only on the incident
        # stream means a turn queued before a changeover cannot land against the old run.
        self._adopt_run(await watchdog.current_run_seq(self.client))

        if self.cycles >= settings.max_cycles:
            await self.heartbeat(state="budget-exhausted")
            return

        mode = await self._await_turn()
        if mode is None:
            return
        # In step mode the presenter is the clock, so skip the artificial pacing delay.
        if mode != control.STEP:
            await asyncio.sleep(random.uniform(0.5, 1.0) * settings.pace_ms / 1000)

        before = await self._write_count()
        self.cycles += 1
        await self.heartbeat(state="thinking", last_wake=wake_reason)
        self.log(f"cycle {self.cycles}: {wake_reason.splitlines()[0][:80]}")

        snapshot = await bb.snapshot(self.client)
        prompt = (
            f"{wake_reason}\n\n"
            f"Current shared blackboard for {INCIDENT_ID}:\n"
            f"{json.dumps(snapshot, indent=2, default=str)}\n\n"
            "Decide whether there is anything useful for you to contribute right now. "
            "If there is nothing new in your domain, say so briefly and write nothing."
        )

        await self.graph.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            self.config,
            durability="sync",
        )

        after = await self._write_count()
        if after > before:
            self.idle_cycles = 0
        else:
            self.idle_cycles += 1

        state = "converged" if self.idle_cycles >= IDLE_CYCLES_TO_CONVERGE else "idle"
        await self.heartbeat(state=state)
        wrote = after - before
        self.log(f"  -> {wrote} write(s); idle_cycles={self.idle_cycles} state={state}")

    async def resume_if_interrupted(self) -> bool:
        """Restart path: finish the step that was pending when this process was killed.

        LangGraph checkpoints at node boundaries, so this re-issues the *pending tool
        call* rather than resuming mid-tool — the investigation so far is not repeated.
        """
        state = await self.graph.aget_state(self.config)
        if not state.next:
            self.log("no pending work in my checkpoint; starting clean")
            return False
        done = len(state.values.get("messages", []))
        self.log(
            f"found a checkpoint with pending work at node {state.next} "
            f"({done} messages already done) — resuming rather than restarting"
        )
        await self.heartbeat(state="resuming")
        await self.graph.ainvoke(None, self.config, durability="sync")
        after = await self.graph.aget_state(self.config)
        self.log(
            f"resumed and completed; thread now has "
            f"{len(after.values.get('messages', []))} messages"
        )
        await self.heartbeat(state="idle")
        return True

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._request_stop)

        # Adopt whatever run is already open *before* looking for pending work, so the
        # checkpoint we resume from is this run's thread and not a previous one's.
        incident = await db.incidents(self.client).find_one({"incident_id": INCIDENT_ID})
        self.run_seq = int((incident or {}).get("run_seq", 0))
        self.config = self._config_for(self.run_seq)

        await self.heartbeat(state="waiting", alive=True)
        resumed = await self.resume_if_interrupted()

        if incident and not resumed and await self._gate():
            await self.think(self._incident_wake(incident))

        async def on_incident() -> None:
            async for event in watch.watch_incidents(self.client, self.name):
                if self._stopping:
                    return
                doc = event["fullDocument"]
                self._adopt_run(int(doc.get("run_seq", 0)))
                if not await self._gate():
                    continue
                await self.think(self._incident_wake(doc))

        async def on_blackboard() -> None:
            async for event in watch.watch_blackboard(
                self.client, self.name, extra_stages=self.spec.interests
            ):
                if self._stopping:
                    return
                if not await self._gate():
                    continue
                doc = event.get("fullDocument") or {}
                await self.think(
                    "A change landed on the shared blackboard: "
                    f"{doc.get('last_touched_by')} wrote a {doc.get('doc_type')}."
                )

        async def on_tick() -> None:
            """Re-read the board periodically, not only when something changes.

            Purely event-driven agents deadlock in one specific way: an open question
            nobody answers produces no further writes, so no change event fires, so
            nobody ever reconsiders it, and the board sits there until the watchdog
            times out. A trace left in shared memory has to stay actionable after the
            moment it was written.

            This is not a scheduler and not a coordinator — each agent ticks itself, and
            only acts when it can see concrete work of its own. Its idle-cycle budget
            still applies, so a tick loop cannot run away.
            """
            while not self._stopping:
                await asyncio.sleep(TICK_SECONDS)
                if not await self._gate():
                    continue
                pending = await self._unanswered_questions_from_others()
                if not pending:
                    continue
                listed = "; ".join(q["question"][:140] for q in pending[:3])
                await self.think(
                    "Nothing new has landed, but these open questions are still "
                    f"unanswered on the blackboard: {listed}. If your own data source can "
                    "settle any of them, answer it. If not, leave it and say so."
                )

        try:
            await asyncio.gather(on_incident(), on_blackboard(), on_tick())
        except asyncio.CancelledError:
            pass
        finally:
            await self.heartbeat(state="stopped", alive=False)
            await self.client.close()
            self.sync.close()

    def _request_stop(self) -> None:
        self._stopping = True
        for task in asyncio.all_tasks():
            task.cancel()

    @staticmethod
    def _incident_wake(incident: dict[str, Any]) -> str:
        # `run_seq` is our own bookkeeping, not part of the symptom report. The trigger is
        # supposed to be thin; an agent has no business reasoning about which run it is.
        hidden = {"_id", "run_seq"}
        body = {k: v for k, v in incident.items() if k not in hidden}
        return (
            "A new incident was opened by an automated threshold monitor:\n"
            f"{json.dumps(body, indent=2, default=str)}\n\n"
            "This is a symptom report only — it contains no diagnosis. Investigate it "
            "using your own data source."
        )


def run_agent(spec: AgentSpec) -> None:
    asyncio.run(Agent(spec).run())
